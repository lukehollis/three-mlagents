import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# Bicycle Environment
# -----------------------------------------------------------------------------------

class BicycleEnv:
    """A simple bicycle riding environment."""

    def __init__(self):
        # Physics constants
        self.g = 9.8
        self.h = 0.8  # height of center of mass
        self.L = 1.0  # wheelbase
        self.v = 5.0  # constant velocity
        self.dt = 0.02

        # State variables
        self.x = 0.0
        self.z = 0.0
        self.theta = 0.0  # heading angle
        self.phi = 0.0  # lean angle
        self.phi_dot = 0.0  # lean rate
        self.delta = 0.0  # steering angle

        self.max_phi = np.pi / 4  # fail if lean angle > 45 degrees
        self.max_delta = np.pi / 6 # max steering angle

        self.steps = 0
        self.reset()

    def reset(self):
        self.x = 0.0
        self.z = 0.0
        self.theta = 0.0
        # Start with a small random lean
        self.phi = np.random.uniform(-0.1, 0.1)
        self.phi_dot = np.random.uniform(-0.1, 0.1)
        self.delta = 0.0
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1

        # Action: 0=steer left, 1=no steer, 2=steer right
        steer_change = 0.0
        if action == 0:
            steer_change = -0.05
        elif action == 2:
            steer_change = 0.05
        
        self.delta += steer_change
        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)

        # Physics update
        # Simplified model: phi_ddot = gravity_torque - centrifugal_torque
        phi_ddot = (self.g / self.h) * np.sin(self.phi) - (self.v**2 / (self.L * self.h)) * np.tan(self.delta) * np.cos(self.phi)
        self.phi_dot += phi_ddot * self.dt
        self.phi += self.phi_dot * self.dt
        
        # Decay steering towards zero
        self.delta *= 0.95

        # Update position
        self.theta += (self.v / self.L) * np.tan(self.delta) * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.z += self.v * np.sin(self.theta) * self.dt

        # Termination
        done = False
        reward = 1.0  # Reward for surviving
        if abs(self.phi) > self.max_phi:
            reward = -10.0
            done = True
        
        if self.steps > 2000:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([
            self.phi,
            self.phi_dot,
            self.delta,
            np.cos(self.theta),
            np.sin(self.theta)
        ])

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "pos": [self.x, self.z],
            "theta": self.theta,
            "phi": self.phi,
            "delta": self.delta,
            "wheelbase": self.L,
            "bounds": [100, 100] # For camera
        }

# -----------------------------------------------------------------------------------
# PPO agent (adapted for discrete actions)
# -----------------------------------------------------------------------------------

ACTION_SIZE = 3  # 0: left, 1: stay, 2: right

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_logits = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        h = self.shared(obs)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

POLICIES_DIR = "policies"
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

def _export_model_onnx(model: nn.Module, path: str):
    dummy_input_dim = model.shared[0].in_features
    dummy = torch.zeros((1, dummy_input_dim), dtype=torch.float32)
    # The output of this model is a distribution, which can't be exported.
    # We export a modified version that just gives actor logits.
    class ExportableModel(nn.Module):
        def __init__(self, actor_critic_model):
            super().__init__()
            self.model = actor_critic_model
        def forward(self, obs):
            dist, _ = self.model(obs)
            return dist.logits
    
    exportable_model = ExportableModel(model)
    torch.onnx.export(
        exportable_model, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )


# -----------------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------------

async def train_bicycle(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"bicycle_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[BicycleEnv] = [BicycleEnv() for _ in range(8)]
    OBS_SIZE = envs[0]._get_obs().shape[0]

    model = ActorCritic(OBS_SIZE, ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor(np.array([e.reset() for e in envs]), dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []

    while ep_counter < 4000:
        with torch.no_grad():
            dist, value = model(obs)
            actions = dist.sample()
            logp = dist.log_prob(actions).unsqueeze(1)
        
        actions_np = actions.cpu().numpy()

        step_obs, rewards, dones = [], [], []
        for idx, env in enumerate(envs):
            nobs, rew, dn = env.step(actions_np[idx])
            step_obs.append(nobs)
            rewards.append(rew)
            dones.append(dn)
        
        next_obs = torch.tensor(np.array(step_obs), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        step_buffer.append({"obs": obs, "actions": actions.unsqueeze(1), "logp": logp, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        if len(step_buffer) % 8 == 0:
            state = envs[0].get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter + 1})
            await asyncio.sleep(0.00)

        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                obs[i] = torch.tensor(envs[i].reset(), dtype=torch.float32)

        if len(step_buffer) * len(envs) >= BATCH_SIZE:
            with torch.no_grad():
                _, next_value = model(obs)
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

            for _ in range(EPOCHS):
                idx = torch.randperm(obs_cat.shape[0])
                for start in range(0, obs_cat.shape[0], MINI_BATCH):
                    mb_idx = idx[start:start + MINI_BATCH]
                    mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret = obs_cat[mb_idx], act_cat[mb_idx], logp_cat[mb_idx], adv[mb_idx], returns[mb_idx]

                    dist, value = model(mb_obs)
                    logp_new = dist.log_prob(mb_act.squeeze(1)).unsqueeze(1)
                    entropy_bonus = dist.entropy().mean()
                    ratio = (logp_new - mb_logp_old).exp()
                    
                    policy_loss1 = ratio * mb_adv
                    policy_loss2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                    
                    value_loss = (value - mb_ret).pow(2).mean()
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

def infer_action_bicycle(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("bicycle_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No bicycle policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    action = np.argmax(out, axis=1)[0]
    return int(action)


async def run_bicycle(websocket: WebSocket, model_filename: str | None = None):
    env = BicycleEnv()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_bicycle(list(obs), model_filename)
        nobs, _, done = env.step(act)
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.03)
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0) 