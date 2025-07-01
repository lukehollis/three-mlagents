# --------------------------------------------------------------
# Crawler Example – PyBullet rag-doll with 4 arms + 4 forearms
# --------------------------------------------------------------
# This is a **simplified** but structurally correct port of the Unity
# Crawler task. It builds a small URDF rag-doll at runtime (torso + 8
# limb links), exposes a 172-float observation and 20 continuous action
# vector, trains a minimal PPO agent, and streams a reduced state over
# WebSocket for visualisation. No try/except is used to comply with the
# repository rules.

import os
import math
import asyncio
from datetime import datetime
import uuid
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket
import pybullet as p
import pybullet_data

# -----------------------------------------------------------------------------------
# URDF helper – write a small crawler robot to /tmp so PyBullet can load it
# -----------------------------------------------------------------------------------

_URDF_PATH = "/tmp/crawler_ragdoll.urdf"

_RAGDOLL_URDF = """
<?xml version=\"1.0\" ?>
<robot name=\"crawler\">
  <link name=\"torso\">
    <inertial>
      <mass value=\"1.0\" />
      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\" />
    </inertial>
    <visual>
      <geometry><box size=\"0.4 0.2 0.2\" /></geometry>
    </visual>
    <collision>
      <geometry><box size=\"0.4 0.2 0.2\" /></geometry>
    </collision>
  </link>

{{links}}
{{joints}}
</robot>
"""

_LINK_TMPL = """
  <link name=\"{name}\">
    <inertial><mass value=\"0.2\"/><inertia ixx=\"0.01\" ixy=\"0\" ixz=\"0\" iyy=\"0.01\" iyz=\"0\" izz=\"0.01\"/></inertial>
    <visual><geometry><capsule radius=\"0.05\" length=\"0.3\"/></geometry></visual>
    <collision><geometry><capsule radius=\"0.05\" length=\"0.3\"/></geometry></collision>
  </link>
"""

_JOINT_TMPL = """
  <joint name=\"{jname}\" type=\"revolute\">
    <parent link=\"{parent}\" />
    <child  link=\"{child}\" />
    <origin xyz=\"{px} {py} {pz}\" rpy=\"0 0 0\" />
    <axis xyz=\"{ax} {ay} {az}\" />
    <limit lower=\"-1.5708\" upper=\"1.5708\" effort=\"20\" velocity=\"5\" />
  </joint>
"""


def _write_ragdoll_urdf():
    limbs = []
    joints = []
    limb_names = []
    for i in range(4):
        up_name = f"upper_{i}"
        lo_name = f"lower_{i}"
        limb_names.extend([up_name, lo_name])
        limbs.append(_LINK_TMPL.format(name=up_name))
        limbs.append(_LINK_TMPL.format(name=lo_name))
        # attach upper to torso – hinge around local Z so legs swing forward/back
        sign = 1.0 if i < 2 else -1.0  # left/right side
        px = 0.2 * sign
        py = 0.05 if i % 2 == 0 else -0.05
        joints.append(_JOINT_TMPL.format(jname=f"joint_torso_{up_name}", parent="torso", child=up_name, px=px, py=py, pz=0, ax=0, ay=0, az=1))
        # attach lower to upper – hinge around local Z
        joints.append(_JOINT_TMPL.format(jname=f"joint_{up_name}_{lo_name}", parent=up_name, child=lo_name, px=0, py=0, pz=-0.3, ax=0, ay=0, az=1))

    with open(_URDF_PATH, "w") as fh:
        fh.write(_RAGDOLL_URDF.replace("{{links}}", "".join(limbs)).replace("{{joints}}", "".join(joints)))


_write_ragdoll_urdf()

# -----------------------------------------------------------------------------------
# Environment definition
# -----------------------------------------------------------------------------------

OBS_SIZE = 172
ACTION_SIZE = 20
MAX_EPISODE_STEPS = 1000
TARGET_SPEED = 1.0  # m/s


class CrawlerEnv:
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.robot_id: int | None = None
        self.joints: list[int] = []
        self.step_counter = 0
        self.prev_base_pos = (0.0, 0.0, 0.0)
        self.dt = p.getPhysicsEngineParameters(physicsClientId=self.client)["fixedTimeStep"]
        self.reset()

    # --------------------------------------------------
    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(_URDF_PATH, [0, 0, 0.5], useFixedBase=False, physicsClientId=self.client)
        self.joints = list(range(p.getNumJoints(self.robot_id, physicsClientId=self.client)))[:ACTION_SIZE]
        self.step_counter = 0
        self.prev_base_pos = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)[0]
        return self._get_obs()

    # --------------------------------------------------
    def _get_obs(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_lin, base_ang = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        obs = []
        obs.extend(base_pos)
        obs.extend(base_ori)
        obs.extend(base_lin)
        obs.extend(base_ang)
        for j in self.joints:
            jstate = p.getJointState(self.robot_id, j, physicsClientId=self.client)
            obs.append(jstate[0])  # position
            obs.append(jstate[1])  # velocity
        # zero-pad to 172
        if len(obs) < OBS_SIZE:
            obs.extend([0.0] * (OBS_SIZE - len(obs)))
        return np.array(obs[:OBS_SIZE], dtype=np.float32)

    # --------------------------------------------------
    def step(self, action: np.ndarray):
        # action expected in [-1,1]
        for idx, j in enumerate(self.joints):
            tgt = float(action[idx]) * math.pi * 0.5  # scale to ±90°
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=tgt, force=5, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        self.step_counter += 1
        # reward: geometric product of speed alignment and heading alignment
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        forward_vel = (base_pos[0] - self.prev_base_pos[0]) / self.dt
        speed_rew = max(0.0, 1.0 - abs(forward_vel - TARGET_SPEED) / TARGET_SPEED)
        ori_mat = p.getMatrixFromQuaternion(base_ori)
        body_x_axis = (ori_mat[0], ori_mat[3], ori_mat[6])
        heading_rew = (body_x_axis[0] + 1.0) * 0.5
        upright_rew = max(0.0, body_x_axis[2])
        yaw_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)[1][2]
        yaw_pen = 1.0 / (1.0 + 5.0 * abs(yaw_vel))
        reward = speed_rew * heading_rew * upright_rew * yaw_pen
        self.prev_base_pos = base_pos
        done = self.step_counter >= MAX_EPISODE_STEPS or base_pos[2] < 0.1  # fell over
        return self._get_obs(), reward, done


# -----------------------------------------------------------------------------------
# PPO agent (very compact, single file)
# -----------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.actor_mean = nn.Linear(128, ACTION_SIZE)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -1.5))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        h = self.shared(obs)
        return self.actor_mean(h), self.log_std.expand_as(self.actor_mean(h)), self.critic(h)


POLICIES_DIR = "policies"
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.001
LR = 3e-4


def _export_model_onnx(model: nn.Module, path: str):
    dummy = torch.zeros((1, OBS_SIZE), dtype=torch.float32)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})


async def train_crawler(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"crawler_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[CrawlerEnv] = [CrawlerEnv() for _ in range(8)]
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor([e.reset() for e in envs], dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    while ep_counter < 2000:  # episodes
        # collect rollout
        with torch.no_grad():
            mean, log_std, value = model(obs)
            std = log_std.exp()
            actions = mean + std * torch.randn_like(mean)
            probs = (-0.5 * ((actions - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        actions_np = actions.clamp(-1, 1).cpu().numpy()
        step_obs = []
        rewards = []
        dones = []
        for idx, env in enumerate(envs):
            nobs, rew, dn = env.step(actions_np[idx])
            step_obs.append(nobs)
            rewards.append(rew)
            dones.append(dn)
        next_obs = torch.tensor(step_obs, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        step_buffer.append({"obs": obs, "actions": actions, "logp": probs, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        # stream first env every 2 steps for smoother preview
        if len(step_buffer) % 2 == 0:
            e0 = envs[0]
            base_pos, base_ori = p.getBasePositionAndOrientation(e0.robot_id, physicsClientId=e0.client)
            joint_angles = [p.getJointState(e0.robot_id, j, physicsClientId=e0.client)[0] for j in e0.joints]
            await websocket.send_json({
                "type": "train_step",
                "state": {
                    "basePos": [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])],
                    "baseOri": [float(x) for x in base_ori],
                    "jointAngles": [float(a) for a in joint_angles],
                },
                "episode": ep_counter + 1,
            })

        # finish episode if any env done
        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                envs[i].reset()

        # once buffer big enough, update policy
        if len(step_buffer) * len(envs) >= BATCH_SIZE:
            # convert buffer to tensors
            obs_cat = torch.cat([b["obs"] for b in step_buffer])
            act_cat = torch.cat([b["actions"] for b in step_buffer])
            logp_cat = torch.cat([b["logp"] for b in step_buffer])
            rew_cat = torch.cat([b["reward"] for b in step_buffer])
            done_cat = torch.cat([b["done"] for b in step_buffer])
            val_cat = torch.cat([b["value"] for b in step_buffer])
            # compute advantages via GAE
            adv = torch.zeros_like(rew_cat)
            gae = 0.0
            for t in reversed(range(rew_cat.shape[0])):
                delta = rew_cat[t] + GAMMA * (1.0 - done_cat[t]) * val_cat[t] - val_cat[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - done_cat[t]) * gae
                adv[t] = gae
            returns = adv + val_cat
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # PPO update
            for _ in range(EPOCHS):
                idx = torch.randperm(obs_cat.shape[0])
                for start in range(0, obs_cat.shape[0], MINI_BATCH):
                    mb_idx = idx[start:start + MINI_BATCH]
                    mb_obs = obs_cat[mb_idx]
                    mb_act = act_cat[mb_idx]
                    mb_logp_old = logp_cat[mb_idx]
                    mb_adv = adv[mb_idx]
                    mb_ret = returns[mb_idx]
                    mean, log_std, value = model(mb_obs)
                    std = log_std.exp()
                    logp = (-0.5 * ((mb_act - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                    ratio = (logp - mb_logp_old).exp()
                    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean() - ENT_COEF * (-logp.mean())
                    value_loss = ((mb_ret - value) ** 2).mean()
                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # gather simple metrics for frontend
            avg_reward = float(rew_cat.mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            # send progress every update (include reward/loss)
            await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": timestamp, "session_uuid": session_uuid})


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}


def infer_action_crawler(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("crawler_policy_") and f.endswith(".onnx")]
        files.sort(reverse=True)
        model_filename = files[0]
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess
    # pad or truncate to OBS_SIZE so it matches network input
    if len(obs) < OBS_SIZE:
        obs_adj = obs + [0.0] * (OBS_SIZE - len(obs))
    else:
        obs_adj = obs[:OBS_SIZE]
    inp = np.array([obs_adj], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    return out[0].tolist()  # return continuous 20-D vector


async def run_crawler(websocket: WebSocket, model_filename: str | None = None):
    """Stream live inference rollout of a single CrawlerEnv to the websocket."""
    env = CrawlerEnv()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act_vec = infer_action_crawler(obs.tolist(), model_filename)
        nobs, _, done = env.step(np.array(act_vec, dtype=np.float32))
        base_pos, base_ori = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client)
        joint_angles = [p.getJointState(env.robot_id, j, physicsClientId=env.client)[0] for j in env.joints]
        await websocket.send_json({
            "type": "run_step",
            "state": {
                "basePos": [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])],
                "baseOri": [float(x) for x in base_ori],
                "jointAngles": [float(a) for a in joint_angles],
            },
            "episode": episode + 1,
        })
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0)  # cooperative yield 