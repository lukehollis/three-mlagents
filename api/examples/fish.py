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
NUM_FISH = 32
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 100, "color": [0.8, 0.8, 0.2]}, # Prioritize eating, increased reward
    "coral_a": {"value": 0, "color": [0.9, 0.3, 0.3]},
    "coral_b": {"value": 0, "color": [0.3, 0.9, 0.3]},
    "rock": {"value": 0, "color": [0.5, 0.5, 0.5]},
    "sand": {"value": 0, "color": [0.8, 0.7, 0.5]},
    "shark": {"value": -100, "color": [0.2, 0.2, 0.3]},
}
DISCRETE_ACTIONS = [
    "move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "move_z-",
    "eat", "wait"
]
ACTION_MAP_MOVE = {
    "move_x+": np.array([1, 0, 0]), "move_x-": np.array([-1, 0, 0]),
    "move_y+": np.array([0, 1, 0]), "move_y-": np.array([0, -1, 0]),
    "move_z+": np.array([0, 0, 1]), "move_z-": np.array([0, 0, -1]),
}

# --- Shark Class ---
class Shark:
    def __init__(self, pos: np.ndarray):
        self.id = "shark"
        self.pos = pos.astype(np.float32)
        self.color = [0.3, 0.4, 0.5]
        self.velocity = (np.random.rand(3) - 0.5) * 2
        self.speed = 1.5

    def move(self, fish_list: List['Fish']):
        if not fish_list:
            self.velocity += (np.random.rand(3) - 0.5) * 0.5
        else:
            swarm_center = np.mean([f.pos for f in fish_list], axis=0)
            direction_to_swarm = swarm_center - self.pos
            dist = np.linalg.norm(direction_to_swarm)
            if dist > 1: direction_to_swarm /= dist
            self.velocity = self.velocity * 0.9 + direction_to_swarm * 0.1 + (np.random.rand(3) - 0.5) * 0.2
        
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
        self.pos += self.velocity
        self.pos[0] = np.clip(self.pos[0], 0, GRID_SIZE_X - 1)
        self.pos[1] = np.clip(self.pos[1], 2, GRID_SIZE_Y - 1)
        self.pos[2] = np.clip(self.pos[2], 0, GRID_SIZE_Z - 1)

# --- Fish Class ---
class Fish:
    def __init__(self, fish_id: int, pos: np.ndarray):
        self.id = fish_id
        self.pos = pos.astype(np.float32)
        self.energy = 100.0
        self.color = [random.random(), 0.5, 1.0 - random.random()]
        self.velocity = np.zeros(3, dtype=np.float32)

# --- Environment Class ---
class FishEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.step_count = 0
        self.fish: List[Fish] = []
        self.shark: Shark = None
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
        self.shark = Shark(np.array([
            random.randint(0, GRID_SIZE_X - 1),
            random.randint(20, GRID_SIZE_Y - 20),
            random.randint(0, GRID_SIZE_Z - 1)
        ]))
        
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        food_locations = np.argwhere(self.grid == food_idx)
        food_tree = cKDTree(food_locations) if len(food_locations) > 0 else None
        return np.array([get_fish_state_vector(f, self.grid, self.fish, self.shark, food_tree) for f in self.fish])

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
        for _ in range(150): self.grid[random.randint(0, GRID_SIZE_X - 1), random.randint(2, GRID_SIZE_Y - 1), random.randint(0, GRID_SIZE_Z - 1)] = food_idx

    def _get_reward(self, fish: Fish, action: str, data: Any, food_tree: cKDTree) -> float:
        reward = -0.01 # Minimal base penalty to encourage action

        # Strong reward for moving towards the nearest food, which is in the observation space
        if action in ACTION_MAP_MOVE and food_tree is not None:
            dist_before, idx = food_tree.query(fish.pos)
            if np.isfinite(dist_before):
                nearest_food_pos = food_tree.data[idx]
                new_pos = data
                dist_after = np.linalg.norm(new_pos - nearest_food_pos)
                
                # Reward based on getting closer to the single nearest food
                if dist_after < dist_before:
                    reward += (dist_before - dist_after) * 2.0 # Strong reward signal

        if action in ACTION_MAP_MOVE or action == "eat":
            target_pos = data
            if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                target_pos_int = tuple(np.round(target_pos).astype(int))
                # Clip again to be safe
                target_pos_int = (
                    np.clip(target_pos_int[0], 0, GRID_SIZE_X - 1),
                    np.clip(target_pos_int[1], 0, GRID_SIZE_Y - 1),
                    np.clip(target_pos_int[2], 0, GRID_SIZE_Z - 1),
                )
                if self.grid[target_pos_int] == list(ENTITY_TYPES.keys()).index("food") + 1:
                    reward += ENTITY_TYPES["food"]["value"] # Big reward for eating
        
        nearby_fish = [f for f in self.fish if f.id != fish.id and np.linalg.norm(f.pos - fish.pos) < 20]
        if len(nearby_fish) > 2:
            centroid = np.mean([f.pos for f in nearby_fish], axis=0)
            # Make schooling a weaker incentive compared to food
            reward += max(0, 1.0 - np.linalg.norm(fish.pos - centroid) / 20.0) * 0.05 
        else:
            reward -= 0.05 # Smaller penalty for being alone

        dist_to_shark = np.linalg.norm(fish.pos - self.shark.pos)
        # Keep shark penalty high, it's a terminal condition
        if dist_to_shark < 15: reward -= max(0, 1.0 - dist_to_shark / 15.0) * 20 # Increased shark penalty
        return reward

    def _execute_actions(self, fish_actions: List[Tuple[str, Any]]):
        randomized_order = list(zip(self.fish, fish_actions))
        random.shuffle(randomized_order)
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1

        for fish, (action, data) in randomized_order:
            fish.energy -= 0.2
            
            moved = False
            old_pos = fish.pos.copy()

            if action in ACTION_MAP_MOVE: 
                fish.energy -= 0.3
                
                if data is not None:
                    target_pos_int = np.round(data).astype(int)
                    target_pos_int[0] = np.clip(target_pos_int[0], 0, GRID_SIZE_X - 1)
                    target_pos_int[1] = np.clip(target_pos_int[1], 2, GRID_SIZE_Y - 1)
                    target_pos_int[2] = np.clip(target_pos_int[2], 0, GRID_SIZE_Z - 1)
                    
                    target_cell_val = self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]]

                    if target_cell_val == 0: # water
                        fish.pos = data
                        moved = True
                    elif target_cell_val == food_idx:
                        fish.pos = data
                        fish.energy += ENTITY_TYPES["food"]["value"]
                        self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] = 0
                        moved = True
            
            elif action == "eat":
                target_pos_int = np.round(fish.pos).astype(int)
                if 0 <= target_pos_int[0] < GRID_SIZE_X and 0 <= target_pos_int[1] < GRID_SIZE_Y and 0 <= target_pos_int[2] < GRID_SIZE_Z:
                    if self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] == food_idx:
                        fish.energy += ENTITY_TYPES["food"]["value"]
                        self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] = 0

            if moved:
                new_velocity = fish.pos - old_pos
                fish.velocity = fish.velocity * 0.7 + new_velocity * 0.3
            else:
                fish.velocity *= 0.9

        self.shark.move(self.fish)
        dones = []
        for f in self.fish:
            if np.linalg.norm(f.pos - self.shark.pos) < 2: 
                f.energy = 0
            is_done = f.energy <= 0
            dones.append(is_done)
            if is_done:
                f.pos = np.array([random.randint(0, GRID_SIZE_X-1), random.randint(10, GRID_SIZE_Y-10), random.randint(0, GRID_SIZE_Z-1)], dtype=np.float32)
                f.energy = 100.0
                
        if self.step_count % 20 == 0:
            for _ in range(10): self.grid[random.randint(0,GRID_SIZE_X-1), random.randint(2,GRID_SIZE_Y-1), random.randint(0,GRID_SIZE_Z-1)] = food_idx
        
        return dones

    def step(self, actions_np: np.ndarray):
        self.step_count += 1
        fish_actions, rewards = [], []
        
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        food_locations = np.argwhere(self.grid == food_idx)
        food_tree = cKDTree(food_locations) if len(food_locations) > 0 else None

        for i, fish in enumerate(self.fish):
            action_name = DISCRETE_ACTIONS[actions_np[i]]
            action_data = fish.pos + ACTION_MAP_MOVE[action_name] if action_name in ACTION_MAP_MOVE else fish.pos
            fish_actions.append((action_name, action_data))
            rewards.append(self._get_reward(fish, action_name, action_data, food_tree))

        dones = self._execute_actions(fish_actions)
        
        final_rewards = np.array(rewards, dtype=np.float32)
        final_rewards[dones] = -20.0 # Large penalty for dying

        next_obs = np.array([get_fish_state_vector(f, self.grid, self.fish, self.shark, food_tree) for f in self.fish])
        return next_obs, final_rewards, np.array(dones)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {"grid": self.grid.tolist(), "agents": [{"id": f.id, "pos": f.pos.tolist(), "energy": f.energy, "color": f.color, "velocity": f.velocity.tolist()} for f in self.fish], "shark": {"id": self.shark.id, "pos": self.shark.pos.tolist(), "color": self.shark.color}, "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z], "resource_types": ENTITY_TYPES}

def get_fish_state_vector(fish: Fish, grid: np.ndarray, all_fish: List['Fish'], shark: 'Shark', food_tree: cKDTree) -> np.ndarray:
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
    nearby_fish = [f for f in all_fish if f.id != fish.id and np.linalg.norm(f.pos - fish.pos) < 20]
    if len(nearby_fish) > 0:
        centroid = np.mean([f.pos for f in nearby_fish], axis=0)
        dist_to_centroid = np.linalg.norm(fish.pos - centroid)
        swarm_info = np.concatenate([centroid/np.array([GRID_SIZE_X,GRID_SIZE_Y,GRID_SIZE_Z]), [dist_to_centroid/30.0], [len(nearby_fish)/NUM_FISH]])
    else: swarm_info = np.zeros(5)
    
    shark_vec = (shark.pos - fish.pos) / np.array([GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z])

    food_vec = np.zeros(3)
    if food_tree is not None:
        dist, idx = food_tree.query(fish.pos)
        if np.isfinite(dist):
            nearest_food_pos = food_tree.data[idx]
            food_vec = (nearest_food_pos - fish.pos) / np.array([GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z])

    return np.concatenate([view_flat, energy_vec, swarm_info, shark_vec, food_vec])

# PPO Hyperparameters & Training Loop
BATCH_SIZE, MINI_BATCH, EPOCHS, GAMMA, GAE_LAMBDA, CLIP_EPS, ENT_COEF, LR = 8192, 512, 4, 0.99, 0.95, 0.2, 0.01, 3e-4

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

        if env.step_count % 8 == 0:
            state = env.get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
            await asyncio.sleep(0.01)

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

        actions = [infer_action_fish(get_fish_state_vector(f, env.grid, env.fish, env.shark, food_tree), env.trained_policy) for f in env.fish]
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