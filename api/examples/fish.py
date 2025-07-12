import asyncio
import random
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from policies.minefarm_policy import ActorCritic # Reusing policy

logger = logging.getLogger(__name__)

# --- Constants ---
GRID_SIZE_X = 64 
GRID_SIZE_Y = 64
GRID_SIZE_Z = 64
NUM_FISH = 20
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 250, "color": [0.8, 0.8, 0.2]}, # Prioritize eating, increased reward
    "coral_a": {"value": 0, "color": [0.9, 0.3, 0.3]},
    "coral_b": {"value": 0, "color": [0.3, 0.9, 0.3]},
    "rock": {"value": 0, "color": [0.5, 0.5, 0.5]},
    "sand": {"value": 0, "color": [0.8, 0.7, 0.5]},
}
DISCRETE_ACTIONS = [
    "move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "move_z-",
]
ACTION_MAP_MOVE = {
    "move_x+": np.array([1, 0, 0]), "move_x-": np.array([-1, 0, 0]),
    "move_y+": np.array([0, 1, 0]), "move_y-": np.array([0, -1, 0]),
    "move_z+": np.array([0, 0, 1]), "move_z-": np.array([0, 0, -1]),
}

# --- Fish Class ---
class Fish:
    def __init__(self, fish_id: int, pos: np.ndarray):
        self.id = fish_id
        self.pos = pos.astype(np.float32)
        self.energy = 250.0 # More energy to explore
        self.color = [random.random(), 0.5, 1.0 - random.random()]
        self.velocity = np.zeros(3, dtype=np.float32)

# --- Environment Class ---
class FishEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.step_count = 0
        self.fish: List[Fish] = []
        self.trained_policy: ActorCritic = None
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self._spawn_scene()
        self.step_count = 0
        self.fish = [Fish(i, np.array([
            random.randint(0, GRID_SIZE_X - 1),
            random.randint(10, GRID_SIZE_Y - 10),
            random.randint(0, GRID_SIZE_Z - 1)
        ])) for i in range(NUM_FISH)]
        
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        food_locations = np.argwhere(self.grid == food_idx)
        food_tree = cKDTree(food_locations) if len(food_locations) > 0 else None
        return np.array([get_fish_state_vector(f, self.grid, food_tree) for f in self.fish])

    def _spawn_scene(self):
        # Sandy bottom
        self.grid[:, 0:2, :] = list(ENTITY_TYPES.keys()).index("sand") + 1
        # Rocks and corals
        for _ in range(30):
            px, pz, base_y = random.randint(0, GRID_SIZE_X - 1), random.randint(0, GRID_SIZE_Z - 1), 2
            if random.choice(['rock', 'coral']) == 'rock':
                for y_off in range(random.randint(1, 4)): self.grid[px, base_y + y_off, pz] = list(ENTITY_TYPES.keys()).index("rock") + 1
            else:
                coral_type = random.choice([list(ENTITY_TYPES.keys()).index("coral_a") + 1, list(ENTITY_TYPES.keys()).index("coral_b") + 1])
                for y_off in range(random.randint(2, 6)): self.grid[px, base_y + y_off, pz] = coral_type
        # Food
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        for _ in range(NUM_FISH):
            self.grid[random.randint(0, GRID_SIZE_X - 1), random.randint(2, GRID_SIZE_Y - 1), random.randint(0, GRID_SIZE_Z - 1)] = food_idx

    def _execute_actions(self, fish_actions: List[Tuple[str, Any]], food_tree: cKDTree):
        randomized_order = list(zip(self.fish, fish_actions))
        random.shuffle(randomized_order)
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        dones = []
        rewards = []

        for fish, (action, data) in randomized_order:
            fish.energy -= 0.2
            
            moved = False
            ate_food = False
            old_pos = fish.pos.copy()
            current_reward = -0.1 # Time penalty

            if action in ACTION_MAP_MOVE: 
                fish.energy -= 0.3
                
                if data is not None:
                    target_pos_int = np.round(data).astype(int)
                    target_pos_int = np.clip(target_pos_int, [0, 2, 0], [GRID_SIZE_X - 1, GRID_SIZE_Y - 1, GRID_SIZE_Z - 1])
                    
                    target_cell_val = self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]]

                    if target_cell_val == 0: # water
                        fish.pos = data
                        moved = True
                    elif target_cell_val == food_idx:
                        fish.pos = data
                        fish.energy += ENTITY_TYPES["food"]["value"]
                        self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] = 0
                        moved = True
                        ate_food = True
                        current_reward += ENTITY_TYPES["food"]["value"]
            
            if moved and food_tree is not None:
                dist_before, _ = food_tree.query(old_pos)
                if not ate_food: # No distance reward if food was eaten, the big reward is enough
                    dist_after, _ = food_tree.query(fish.pos)
                    if np.isfinite(dist_before):
                        distance_delta = dist_before - dist_after
                        current_reward += distance_delta * 4.0 # Movement reward

            if moved:
                new_velocity = fish.pos - old_pos
                fish.velocity = fish.velocity * 0.7 + new_velocity * 0.3
            else:
                fish.velocity *= 0.9
            
            is_done = ate_food or fish.energy <= 0
            dones.append(is_done)
            rewards.append(current_reward)

            if is_done:
                fish.pos = np.array([random.randint(0, GRID_SIZE_X-1), random.randint(10, GRID_SIZE_Y-10), random.randint(0, GRID_SIZE_Z-1)], dtype=np.float32)
                fish.energy = 250.0
                if ate_food:
                     self.grid[random.randint(0,GRID_SIZE_X-1), random.randint(2,GRID_SIZE_Y-1), random.randint(0,GRID_SIZE_Z-1)] = food_idx
        
        return dones, rewards

    def step(self, actions_np: np.ndarray):
        self.step_count += 1
        fish_actions = []
        
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        food_locations = np.argwhere(self.grid == food_idx)
        food_tree = cKDTree(food_locations) if len(food_locations) > 0 else None

        for i, fish in enumerate(self.fish):
            action_name = DISCRETE_ACTIONS[actions_np[i]]
            action_data = fish.pos + ACTION_MAP_MOVE[action_name] if action_name in ACTION_MAP_MOVE else fish.pos
            fish_actions.append((action_name, action_data))

        dones, rewards = self._execute_actions(fish_actions, food_tree)
        
        final_rewards = np.array(rewards, dtype=np.float32)
        next_obs = np.array([get_fish_state_vector(f, self.grid, food_tree) for f in self.fish])
        return next_obs, final_rewards, np.array(dones)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {"grid": self.grid.tolist(), "agents": [{"id": f.id, "pos": f.pos.tolist(), "energy": f.energy, "color": f.color, "velocity": f.velocity.tolist()} for f in self.fish], "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z], "resource_types": ENTITY_TYPES}

def get_fish_state_vector(fish: Fish, grid: np.ndarray, food_tree: cKDTree) -> np.ndarray:
    pos = np.round(fish.pos).astype(int)
    view = np.zeros((5, 5, 5))
    x_s, x_e = max(0, pos[0]-2), min(GRID_SIZE_X, pos[0]+3)
    y_s, y_e = max(0, pos[1]-2), min(GRID_SIZE_Y, pos[1]+3)
    z_s, z_e = max(0, pos[2]-2), min(GRID_SIZE_Z, pos[2]+3)
    grid_slice = grid[int(x_s):int(x_e), int(y_s):int(y_e), int(z_s):int(z_e)]
    vx_s, vy_s, vz_s = max(0, 2 - (pos[0] - x_s)), max(0, 2 - (pos[1] - y_s)), max(0, 2 - (pos[2] - z_s))
    x_len, y_len, z_len = min(grid_slice.shape[0], 5-vx_s), min(grid_slice.shape[1], 5-vy_s), min(grid_slice.shape[2], 5-vz_s)
    if x_len>0 and y_len>0 and z_len>0: view[int(vx_s):int(vx_s)+x_len, int(vy_s):int(vy_s)+y_len, int(vz_s):int(vz_s)+z_len] = grid_slice[:x_len, :y_len, :z_len]
    view_flat = view.flatten()

    energy_vec = np.array([fish.energy / 100.0])
    
    food_vec = np.zeros(3)
    if food_tree is not None:
        dist, idx = food_tree.query(fish.pos)
        if np.isfinite(dist) and dist > 1e-6: # Add a small epsilon to avoid division by zero
            nearest_food_pos = food_tree.data[idx]
            direction_to_food = nearest_food_pos - fish.pos
            food_vec = direction_to_food / dist # Normalize to a unit vector

    return np.concatenate([view_flat, energy_vec, food_vec])

# PPO Hyperparameters & Training Loop
BATCH_SIZE, MINI_BATCH, EPOCHS, GAMMA, GAE_LAMBDA, CLIP_EPS, ENT_COEF, LR = 1024, 128, 4, 0.99, 0.95, 0.2, 0.05, 3e-4

async def train_fish(websocket: WebSocket, env: FishEnv):
    obs = env.reset()
    model = ActorCritic(obs.shape[1], len(DISCRETE_ACTIONS))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    step_buffer, ep_counter, total_steps = [], 0, 0
    obs_t = torch.tensor(obs, dtype=torch.float32)

    while ep_counter < 75000:
        with torch.no_grad():
            dist, value = model(obs_t)
            actions_t = dist.sample()
            logp_t = dist.log_prob(actions_t)
        next_obs, rewards, dones = env.step(actions_t.cpu().numpy())
        total_steps += NUM_FISH
        ep_counter += int(np.sum(dones))
        
        step_buffer.append({"obs":obs_t, "actions":actions_t, "logp":logp_t, "reward":torch.tensor(rewards,dtype=torch.float32), "done":torch.tensor(dones,dtype=torch.bool), "value":value.flatten()})
        obs_t = torch.tensor(next_obs, dtype=torch.float32)

        if env.step_count % 8 == 0: # Reduced frequency of state updates to speed up training
            state = env.get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})

        if total_steps >= BATCH_SIZE:
            with torch.no_grad(): _, next_value = model(obs_t)
            values, rewards, dones = torch.stack([b["value"] for b in step_buffer]), torch.stack([b["reward"] for b in step_buffer]), torch.stack([b["done"] for b in step_buffer])
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(step_buffer))):
                delta = rewards[t] + GAMMA * next_value.squeeze() * (~dones[t]) - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (~dones[t]) * gae
                advantages[t] = gae
                next_value = values[t]
            
            b_obs, b_actions, b_logp = torch.cat([b["obs"] for b in step_buffer]), torch.cat([b["actions"] for b in step_buffer]), torch.cat([b["logp"] for b in step_buffer])
            b_adv = (advantages.flatten() - advantages.flatten().mean()) / (advantages.flatten().std() + 1e-8)
            b_returns = (advantages + values).flatten()

            for _ in range(EPOCHS):
                idxs = torch.randperm(b_obs.shape[0])
                for start in range(0, b_obs.shape[0], MINI_BATCH):
                    mb_idxs = idxs[start:start+MINI_BATCH]
                    dist, value = model(b_obs[mb_idxs])
                    ratio = (dist.log_prob(b_actions[mb_idxs]) - b_logp[mb_idxs]).exp()
                    pg_loss = -torch.min(ratio * b_adv[mb_idxs], torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * b_adv[mb_idxs]).mean()
                    loss = pg_loss + 0.5 * ((value.flatten() - b_returns[mb_idxs]).pow(2)).mean() - ENT_COEF * dist.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": torch.stack([b["reward"].mean() for b in step_buffer]).mean().item(), "loss": loss.item()})
            step_buffer, total_steps = [], 0
            
            # Reset environment after each training batch to start fresh
            obs = env.reset()
            obs_t = torch.tensor(obs, dtype=torch.float32)

    await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item() if 'loss' in locals() else 0}})

# Run & Inference
POLICIES_DIR = "policies"
os.makedirs(POLICIES_DIR, exist_ok=True)
_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}

def infer_action_fish(obs: List[float], policy: ActorCritic) -> int:
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist, _ = policy(obs_t)
        action = dist.sample()
    return action.item()

async def run_fish(websocket: WebSocket, env: FishEnv):
    if not env.trained_policy:
        await websocket.send_json({"type": "error", "message": "No trained policy available."})
        return

    from starlette.websockets import WebSocketState
    food_tree = None # Initialize food_tree
    while websocket.application_state == WebSocketState.CONNECTED and env.fish:
        
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        food_locations = np.argwhere(env.grid == food_idx)
        if len(food_locations) > 0:
            food_tree = cKDTree(food_locations)

        actions = [infer_action_fish(get_fish_state_vector(f, env.grid, food_tree), env.trained_policy) for f in env.fish]
        env.step(np.array(actions))
        state = env.get_state_for_viz()
        
        try:
            await websocket.send_json({"type": "run_step", "state": state})
            await websocket.send_json({"type": "progress", "episode": env.step_count, "reward": sum(f.energy for f in env.fish), "loss": None})
            await asyncio.sleep(0.1)
        except Exception:
            break
    
    if not env.fish:
        await websocket.send_json({"type": "info", "message": "All fish have died. Simulation over."}) 