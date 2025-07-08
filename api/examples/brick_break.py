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
# BrickBreak Environment
# -----------------------------------------------------------------------------------

class BrickBreakEnv:
    """A simple BrickBreak game environment."""

    def __init__(self, width=40, height=40, paddle_width=8, ball_radius=1, brick_rows=5, brick_cols=8):
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.ball_radius = ball_radius
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.brick_width = width / brick_cols
        self.brick_height = 2

        self.paddle_x = 0
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.bricks = np.ones((brick_rows, brick_cols))
        
        self.reset()

    def reset(self):
        self.paddle_x = self.width / 2
        self.ball_pos = np.array([self.width / 2, self.height / 4])
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        self.ball_vel = np.array([np.cos(angle), np.sin(angle)]) * 1.5
        self.bricks = np.ones((self.brick_rows, self.brick_cols))
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        # Action: 0=left, 1=stay, 2=right
        if action == 0:
            self.paddle_x -= 3
        elif action == 2:
            self.paddle_x += 3
        
        self.paddle_x = np.clip(self.paddle_x, self.paddle_width / 2, self.width - self.paddle_width / 2)

        # Update ball position
        self.ball_pos += self.ball_vel

        # Collisions
        reward = 0.0
        
        # Wall collisions
        if self.ball_pos[0] <= self.ball_radius or self.ball_pos[0] >= self.width - self.ball_radius:
            self.ball_vel[0] *= -1
        if self.ball_pos[1] >= self.height - self.ball_radius:
            self.ball_vel[1] *= -1

        # Paddle collision
        if self.ball_vel[1] < 0 and \
           self.ball_pos[1] - self.ball_radius <= 2 and \
           self.ball_pos[0] >= self.paddle_x - self.paddle_width / 2 and \
           self.ball_pos[0] <= self.paddle_x + self.paddle_width / 2:
            self.ball_vel[1] *= -1
            offset = (self.ball_pos[0] - self.paddle_x) / (self.paddle_width / 2)
            self.ball_vel[0] += offset * 0.5
            reward = 0.1

        # Brick collisions
        brick_y_start = self.height - self.brick_rows * self.brick_height - 10
        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if self.bricks[r, c] == 1:
                    brick_x = c * self.brick_width
                    brick_y = brick_y_start + r * self.brick_height
                    if self.ball_pos[0] >= brick_x and self.ball_pos[0] <= brick_x + self.brick_width and \
                       self.ball_pos[1] >= brick_y and self.ball_pos[1] <= brick_y + self.brick_height:
                        self.bricks[r, c] = 0
                        self.ball_vel[1] *= -1
                        reward = 1.0
                        break
            if reward == 1.0:
                break

        # Termination
        done = False
        if self.ball_pos[1] < self.ball_radius:
            reward = -1.0
            done = True
        
        if np.sum(self.bricks) == 0:
            reward = 10.0
            done = True
        
        if self.steps > 2000:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.concatenate([
            self.ball_pos / np.array([self.width, self.height]),
            self.ball_vel,
            [self.paddle_x / self.width],
            self.bricks.flatten()
        ])

    def get_state_for_viz(self) -> Dict[str, Any]:
        brick_list = []
        brick_y_start = self.height - self.brick_rows * self.brick_height - 10
        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if self.bricks[r, c] == 1:
                    brick_list.append({
                        "pos": [c * self.brick_width + self.brick_width / 2, brick_y_start + r * self.brick_height + self.brick_height / 2],
                        "size": [self.brick_width * 0.9, self.brick_height * 0.8]
                    })
        return {
            "ball": {"pos": self.ball_pos.tolist(), "radius": self.ball_radius},
            "paddle": {"pos": [self.paddle_x, 1], "size": [self.paddle_width, 2]},
            "bricks": brick_list,
            "bounds": [self.width, self.height]
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

async def train_brick_break(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"brickbreak_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[BrickBreakEnv] = [BrickBreakEnv() for _ in range(8)]
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

def infer_action_brick_break(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("brickbreak_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No brickbreak policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    action = np.argmax(out, axis=1)[0]
    return int(action)


async def run_brick_break(websocket: WebSocket, model_filename: str | None = None):
    env = BrickBreakEnv()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_brick_break(list(obs), model_filename)
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