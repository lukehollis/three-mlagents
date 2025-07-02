# --------------------------------------------------------------
# Crawler Example – Gym Ant-v5 wrapper for training and visualization
# --------------------------------------------------------------
# This version follows the Gymnasium MuJoCo Ant environment exactly. It keeps
# the websocket interface identical to crawler.py so the frontend can drive
# training / inference while the underlying physics and reward match the
# reference implementation documented at
# https://mgoulao.github.io/gym-docs/environments/mujoco/ant/
#
# Key points:
#   • Action space  : Box(-1, 1, (8,))  – raw torques
#   • Observation   : 111-D vector as in Gym Ant (x,y excluded)
#   • Reward        : healthy + forward − ctrl_cost − contact_cost (handled by Gym)
#   • Termination   : unhealthy or 1000 steps (handled by Gym)
#
# NOTE: This file purposefully avoids extra shaping or privileged information.

import math
import os
from datetime import datetime
import uuid
from typing import List

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from fastapi import WebSocket

import gymnasium as gym
# Importing gymnasium_robotics registers classic MuJoCo environments (Ant, Hopper, etc.)
import gymnasium_robotics  # noqa: F401  # register envs
import mujoco  # for mj_name2id helper

# -----------------------------------------------------------------------------------
# Environment wrapper (adds a small helper for Three.js visualisation)
# -----------------------------------------------------------------------------------

class AntEnvWrapper:
    """Thin wrapper around gym.make('Ant-v5') with helper for rendering state."""

    def __init__(self):
        self.env = gym.make("Ant-v5", exclude_current_positions_from_observation=True)
        self.obs, _ = self.env.reset()

        # Keep reference to the underlying MuJoCo data for viz
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        # Cache torso body index (API differs between mujoco-py and mujoco >=2.3)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action: np.ndarray):
        self.obs, reward, done, truncated, _ = self.env.step(action)
        # Gymnasium returns both done & truncated; treat either as episode end
        return self.obs, reward, done or truncated

    # ------------------------------------------------------------------
    # Helper for frontend rendering – returns minimal pose information.
    # ------------------------------------------------------------------
    def get_state_for_viz(self):
        # Base position and orientation (quaternion) from MuJoCo data
        pos = self.data.xpos[self.torso_id].copy()
        quat = self.data.xquat[self.torso_id].copy()
        # Joint angles (8 joints) – order matches env.joint_names
        joint_qpos = []
        for jid in range(self.model.njnt):
            addr = self.model.jnt_qposadr[jid]
            joint_qpos.append(float(self.data.qpos[addr]))
        return {
            "basePos": pos.tolist(),
            "baseOri": quat.tolist(),
            "jointAngles": joint_qpos,
        }

# -----------------------------------------------------------------------------------
# PPO agent (size adapted to Ant)
# -----------------------------------------------------------------------------------

ACTION_SIZE = 8

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_mean = nn.Linear(128, ACTION_SIZE)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -0.5))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        h = self.shared(obs)
        return self.actor_mean(h), self.log_std.expand_as(self.actor_mean(h)), self.critic(h)

# Training hyper-parameters (kept identical to crawler.py for now)
POLICIES_DIR = "policies"
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.001
LR = 3e-4

# ------------------------------------------------------------------
# Helper to export the trained policy to ONNX so the browser can run it.
# ------------------------------------------------------------------

def _export_model_onnx(model: nn.Module, path: str):
    dummy_input_dim = model.shared[0].in_features  # first Linear input size
    dummy = torch.zeros((1, dummy_input_dim), dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )

# -----------------------------------------------------------------------------------
# Training loop (multi-env vectorised via simple Python list)
# -----------------------------------------------------------------------------------

async def train_ant(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"ant_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    # Create parallel environments (8 as before)
    envs: List[AntEnvWrapper] = [AntEnvWrapper() for _ in range(8)]

    OBS_SIZE = envs[0].obs.shape[0]

    model = ActorCritic(OBS_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor([e.reset() for e in envs], dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []

    while ep_counter < 2000:
        with torch.no_grad():
            mean, log_std, value = model(obs)
            std = log_std.exp()
            dist = Normal(mean, std)
            actions = dist.sample()
            logp = dist.log_prob(actions).sum(-1, keepdim=True)
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
        step_buffer.append({"obs": obs, "actions": actions, "logp": logp, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        # Stream first env for live preview
        if len(step_buffer) % 8 == 0:
            state = envs[0].get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter + 1})
            await asyncio.sleep(0.00)

        # Episode management
        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                envs[i].reset()

        # PPO update when buffer full
        if len(step_buffer) * len(envs) >= BATCH_SIZE:
            # GAE advantage calculation
            with torch.no_grad():
                _, _, next_value = model(obs)
                next_value = next_value.squeeze()

            num_steps = len(step_buffer)
            num_envs = len(envs)
            
            values = torch.cat([b["value"] for b in step_buffer]).view(num_steps, num_envs)
            rewards = torch.cat([b["reward"] for b in step_buffer]).view(num_steps, num_envs)
            dones = torch.cat([b["done"] for b in step_buffer]).view(num_steps, num_envs)
            all_values = torch.cat([values, next_value.unsqueeze(0)], dim=0)

            advantages = torch.zeros(num_steps, num_envs)
            gae = 0.0
            for t in reversed(range(num_steps)):
                delta = rewards[t] + GAMMA * (1.0 - dones[t]) * all_values[t + 1] - all_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
                advantages[t] = gae
            
            adv = advantages.flatten().unsqueeze(1)
            val_cat = values.flatten().unsqueeze(1)
            returns = adv + val_cat
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            obs_cat = torch.cat([b["obs"] for b in step_buffer])
            act_cat = torch.cat([b["actions"] for b in step_buffer])
            logp_cat = torch.cat([b["logp"] for b in step_buffer])

            # PPO epochs
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
                    dist = Normal(mean, std)
                    
                    logp_new = dist.log_prob(mb_act).sum(-1, keepdim=True)
                    entropy_bonus = dist.entropy().sum(-1).mean()

                    ratio = (logp_new - mb_logp_old).exp()
                    
                    policy_loss1 = ratio * mb_adv
                    policy_loss2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                    
                    value_loss = ((mb_ret - value) ** 2).mean()
                    loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            avg_reward = float(torch.cat([b["reward"] for b in step_buffer]).mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": ts, "session_uuid": session_uuid})

# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}


def infer_action_ant(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("ant_policy_") and f.endswith(".onnx")]
        files.sort(reverse=True)
        model_filename = files[0]
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess
    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    return out[0].tolist()


async def run_ant(websocket: WebSocket, model_filename: str | None = None):
    env = AntEnvWrapper()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act_vec = infer_action_ant(obs, model_filename)
        nobs, _, done = env.step(np.array(act_vec, dtype=np.float32))
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.03)
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0)  # cooperative yield 