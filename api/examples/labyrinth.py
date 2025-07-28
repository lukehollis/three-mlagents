import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any
from collections import deque
import random

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from fastapi import WebSocket


ACTION_SIZE = 4  # 0=up, 1=down, 2=left, 3=right
POLICIES_DIR = "policies"
BATCH_SIZE = 8192
MINI_BATCH = 512
EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 2e-4
EPISODES = 25000
LABYRINTH_WIDTH = 129
LABYRINTH_HEIGHT = 65

# -----------------------------------------------------------------------------------
# Labyrinth Environment
# -----------------------------------------------------------------------------------

class LabyrinthEnv:
    """A 2D ASCII labyrinth environment with a pursuer (Minotaur) and an exit."""

    def __init__(self, training_mode: bool = True):
        self.width = LABYRINTH_WIDTH
        self.height = LABYRINTH_HEIGHT
        self.grid = np.full((self.height, self.width), '#')
        self.training_mode = training_mode
        self.reset()

    def _generate_labyrinth(self):
        # Iterative maze generation to handle large sizes without recursion limits.
        grid = np.full((self.height, self.width), '#', dtype='<U1')
        
        def is_valid(y, x):
            return 1 <= y < self.height - 1 and 1 <= x < self.width - 1

        stack = []
        start_y, start_x = 1, 1
        grid[start_y, start_x] = ' '
        stack.append((start_y, start_x))

        while stack:
            y, x = stack[-1]
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            
            carved_new_path = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                nwy, nwx = y + dy // 2, x + dx // 2
                
                if is_valid(ny, nx) and grid[ny, nx] == '#':
                    grid[nwy, nwx] = ' '
                    grid[ny, nx] = ' '
                    stack.append((ny, nx))
                    carved_new_path = True
                    break
            
            if not carved_new_path:
                stack.pop()
        return grid
    
    def _get_random_empty_cell(self):
        while True:
            y = random.randint(1, self.height - 2)
            x = random.randint(1, self.width - 2)
            if self.grid[y, x] == ' ':
                return (y, x)

    def reset(self):
        self.grid = self._generate_labyrinth()
        
        self.theseus_pos = self._get_random_empty_cell()
        
        self.minotaur_pos = self._get_random_empty_cell()
        while np.linalg.norm(np.array(self.theseus_pos) - np.array(self.minotaur_pos)) < 40:
            self.minotaur_pos = self._get_random_empty_cell()

        self.exit_pos = self._get_random_empty_cell()
        while np.linalg.norm(np.array(self.theseus_pos) - np.array(self.exit_pos)) < 40:
            self.exit_pos = self._get_random_empty_cell()
            
        self.grid[self.exit_pos] = 'E'
        self.steps = 0
        self.minotaur_turn_counter = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        py, px = self.theseus_pos
        ny, nx = py, px

        if action == 0: ny -= 1
        elif action == 1: ny += 1
        elif action == 2: nx -= 1
        elif action == 3: nx += 1

        reward = -0.05  # Time penalty

        if self.grid[ny, nx] == '#':
            reward -= 1.0 # Hit wall
        else:
            self.theseus_pos = (ny, nx)
            reward += 0.01 # Movement incentive

        done = False
        if self.theseus_pos == self.exit_pos:
            reward = 50.0; done = True
        elif self.theseus_pos == self.minotaur_pos:
            reward = -50.0; done = True
        elif self.steps >= 2500:
            reward = -10.0; done = True

        if done:
            return self._get_obs(), reward, done
        
        self.minotaur_turn_counter += 1
        if self.minotaur_turn_counter % 2 == 0:
            my, mx = self.minotaur_pos
            ty, tx = self.theseus_pos
            
            dy, dx = np.sign(ty - my), np.sign(tx - mx)
            
            # Smarter greedy move: try axis with largest distance first, fallback to other.
            if abs(ty - my) > abs(tx - mx):
                # Try vertical first
                if dy != 0 and self.grid[my + dy, mx] != '#':
                    self.minotaur_pos = (my + dy, mx)
                elif dx != 0 and self.grid[my, mx + dx] != '#': # Fallback to horizontal
                    self.minotaur_pos = (my, mx + dx)
            else:
                # Try horizontal first
                if dx != 0 and self.grid[my, mx + dx] != '#':
                    self.minotaur_pos = (my, mx + dx)
                elif dy != 0 and self.grid[my + dy, mx] != '#': # Fallback to vertical
                    self.minotaur_pos = (my + dy, mx)

            if self.theseus_pos == self.minotaur_pos:
                reward = -50.0; done = True
        
        # Reward for getting closer to exit
        dist_to_exit_prev = abs(py - self.exit_pos[0]) + abs(px - self.exit_pos[1])
        dist_to_exit_new = abs(self.theseus_pos[0] - self.exit_pos[0]) + abs(self.theseus_pos[1] - self.exit_pos[1])
        reward += 0.1 * (dist_to_exit_prev - dist_to_exit_new)
        
        # Penalty for getting closer to minotaur
        dist_to_mino_prev = abs(py - self.minotaur_pos[0]) + abs(px - self.minotaur_pos[1])
        dist_to_mino_new = abs(self.theseus_pos[0] - self.minotaur_pos[0]) + abs(self.theseus_pos[1] - self.minotaur_pos[1])
        reward -= 0.05 * (dist_to_mino_prev - dist_to_mino_new)

        return self._get_obs(), reward, done

    def _get_obs(self):
        # Flattened grid representation
        obs_grid = np.zeros((self.height, self.width), dtype=np.float32)
        obs_grid[self.grid == '#'] = -1.0
        obs_grid[self.grid == 'E'] = 1.0
        
        ty, tx = self.theseus_pos
        obs_grid[ty, tx] = 0.8
        
        my, mx = self.minotaur_pos
        obs_grid[my, mx] = -0.8
        
        return obs_grid.flatten()

    def get_state_for_viz(self) -> Dict[str, Any]:
        grid_viz = self.grid.copy()
        grid_viz[self.theseus_pos] = 'T'
        grid_viz[self.minotaur_pos] = 'M'
        return {"grid": grid_viz.tolist(), "steps": self.steps}

# -----------------------------------------------------------------------------------
# PPO agent (adapted for discrete actions)
# -----------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, height, width, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, height, width)
            conv_out_size = self.conv(dummy_input).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU()
        )
        self.actor_logits = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        obs_reshaped = obs.view(batch_size, 1, LABYRINTH_HEIGHT, LABYRINTH_WIDTH)
        h_conv = self.conv(obs_reshaped)
        h = self.shared(h_conv)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value


def _export_model_onnx(model: nn.Module, path: str):
    dummy_input_dim = LABYRINTH_HEIGHT * LABYRINTH_WIDTH
    dummy = torch.zeros((1, dummy_input_dim), dtype=torch.float32)
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

async def train_labyrinth(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"labyrinth_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[LabyrinthEnv] = [LabyrinthEnv() for _ in range(4)]
    
    model = ActorCritic(LABYRINTH_HEIGHT, LABYRINTH_WIDTH, ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor(np.array([e.reset() for e in envs]), dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    
    total_steps = 0

    while ep_counter < EPISODES:
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
        
        total_steps += len(envs)
        next_obs = torch.tensor(np.array(step_obs), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        step_buffer.append({"obs": obs, "actions": actions.unsqueeze(1), "logp": logp, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        if len(step_buffer) % 16 == 0:
            state = envs[0].get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter + 1})
            await asyncio.sleep(0.01)

        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                obs[i] = torch.tensor(envs[i].reset(), dtype=torch.float32)

        if total_steps >= BATCH_SIZE:
            with torch.no_grad():
                _, next_value = model(obs)
                next_value = next_value.squeeze()

            num_steps = len(step_buffer)
            num_envs = len(envs)
            
            values = torch.cat([b["value"] for b in step_buffer]).view(num_steps, num_envs)
            rewards_cat = torch.cat([b["reward"] for b in step_buffer]).view(num_steps, num_envs)
            dones_cat = torch.cat([b["done"] for b in step_buffer]).view(num_steps, num_envs)
            all_values = torch.cat([values, next_value.unsqueeze(0)], dim=0)

            advantages = torch.zeros(num_steps, num_envs)
            gae = 0.0
            for t in reversed(range(num_steps)):
                delta = rewards_cat[t] + GAMMA * (1.0 - dones_cat[t]) * all_values[t + 1] - all_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones_cat[t]) * gae
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
                    value_clipped = val_cat[mb_idx] + torch.clamp(value - val_cat[mb_idx], -CLIP_EPS, CLIP_EPS)
                    value_loss_clipped = (value_clipped - mb_ret).pow(2).mean()
                    value_loss = 0.5 * torch.max(value_loss, value_loss_clipped)

                    loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            avg_reward = float(rewards_cat.mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            total_steps = 0
            await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": ts, "session_uuid": session_uuid})

async def run_simulation(websocket: WebSocket):
    env = LabyrinthEnv(training_mode=False)
    from starlette.websockets import WebSocketState, WebSocketDisconnect

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            state = env.get_state_for_viz()
            await websocket.send_json({"type": "state", "state": state})
            await asyncio.sleep(0.5)
            env.reset()
    except WebSocketDisconnect:
        print("Labyrinth simulation client disconnected.")


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}

def infer_action_labyrinth(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("labyrinth_policy_") and f.endswith(".onnx")]
        if not files:
            return random.randint(0, ACTION_SIZE - 1)
        files.sort(reverse=True)
        model_filename = files[0]

    model_path = os.path.join(POLICIES_DIR, model_filename)

    if not os.path.exists(model_path):
        return random.randint(0, ACTION_SIZE - 1)
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    action = np.argmax(out, axis=1)[0]
    return int(action)

async def run_labyrinth(websocket: WebSocket, model_filename: str | None = None):
    env = LabyrinthEnv(training_mode=False)
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_labyrinth(list(obs), model_filename)
        nobs, _, done = env.step(act)
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.1)
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0.01) 