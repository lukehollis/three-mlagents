# --------------------------------------------------------------
# Worm Example – Gym Swimmer-v5 wrapper for training and visualization
# --------------------------------------------------------------
# This version follows the Gymnasium MuJoCo Swimmer environment. It keeps
# the websocket interface identical to other examples so the frontend can drive
# training / inference while the underlying physics and reward match the
# reference implementation documented at
# https://gymnasium.farama.org/environments/mujoco/swimmer/
#
# Key points:
#   • Action space  : Box(-1, 1, (2,))  – raw torques for 2 joints
#   • Observation   : 8-D vector (angles, velocities)
#   • Reward        : forward_reward - ctrl_cost (handled by Gym)
#   • Termination   : 1000 steps (handled by Gym)
#
# NOTE: This file purposefully avoids extra shaping or privileged information.

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
from torch.distributions.normal import Normal
from fastapi import WebSocket

import gymnasium as gym
# Importing gymnasium_robotics registers classic MuJoCo environments
import gymnasium_robotics  # noqa: F401  # register envs
import mujoco

# -----------------------------------------------------------------------------------
# Environment wrapper (adds a small helper for Three.js visualisation)
# -----------------------------------------------------------------------------------

class WormEnvWrapper:
    """Thin wrapper around gym.make('Swimmer-v5') with helper for rendering state."""

    def __init__(self):
        # With n links, there are n-1 joints. Default is 3 links.
        self.env = gym.make("Swimmer-v5", exclude_current_positions_from_observation=True)
        self.obs, _ = self.env.reset()

        # Keep reference to the underlying MuJoCo data for viz
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        # Cache body part names. Default swimmer has 'torso', 'mid', 'back'
        self.body_names = ["torso", "mid", "back"]


    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action: np.ndarray):
        self.obs, reward, done, truncated, _ = self.env.step(action)
        # Gymnasium returns both done & truncated; treat either as episode end
        return self.obs, reward, done or truncated

    # ------------------------------------------------------------------
    # Helper for frontend rendering – returns pose info for each segment.
    # ------------------------------------------------------------------
    def get_state_for_viz(self) -> Dict[str, Any]:
        segments = []
        for name in self.body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()

            # Find the first capsule geom associated with this body to get its size
            geom_id = -1
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == body_id and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    geom_id = i
                    break
            
            # Default size, fallback if no capsule geom is found
            size = np.array([0.1, 0.1])
            if geom_id != -1:
                # MuJoCo capsule size is [radius, half-height]
                size = self.model.geom_size[geom_id].copy()

            segments.append({
                "name": name, 
                "pos": pos.tolist(), 
                "quat": quat.tolist(),
                "size": size.tolist()
            })
        return {"segments": segments}


# -----------------------------------------------------------------------------------
# PPO agent (size adapted to Swimmer)
# -----------------------------------------------------------------------------------

ACTION_SIZE = 2  # Swimmer-v5 has 2 actions by default

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_mean = nn.Sequential(nn.Linear(128, ACTION_SIZE), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -0.5))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        h = self.shared(obs)
        mean = self.actor_mean(h)
        return mean, self.log_std.expand_as(mean), self.critic(h)

# Training hyper-parameters (kept identical to other examples for now)
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

async def train_worm(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"worm_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    # Create parallel environments (8 like other examples)
    envs: List[WormEnvWrapper] = [WormEnvWrapper() for _ in range(8)]

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
                    
                    # Value clipping for more stable training
                    mb_val_old = val_cat[mb_idx]
                    v_clipped = mb_val_old + torch.clamp(value - mb_val_old, -CLIP_EPS, CLIP_EPS)
                    v_loss_clipped = (v_clipped - mb_ret)**2
                    v_loss_unclipped = (value - mb_ret)**2
                    value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    
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


def infer_action_worm(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("worm_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No worm policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    return out[0].tolist()


async def run_worm(websocket: WebSocket, model_filename: str | None = None):
    env = WormEnvWrapper()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act_vec = infer_action_worm(list(obs), model_filename)
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