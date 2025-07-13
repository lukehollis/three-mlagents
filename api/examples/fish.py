import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.spatial import cKDTree

# --- Environment Constants ---
GRID_SIZE = 20
NUM_FISH = 10
NUM_FOOD = 20
REWARD_FOOD = 20.0
REWARD_STEP = -0.1
# For frontend compatibility, we define ENTITY_TYPES but only use food and water
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 1, "color": [0.8, 0.8, 0.2]},
}

# --- Action Definitions ---
DISCRETE_ACTIONS = ["up", "down", "left", "right", "forward", "backward"]
ACTION_MAP = {
    "up": np.array([0, 1, 0]),
    "down": np.array([0, -1, 0]),
    "left": np.array([-1, 0, 0]),
    "right": np.array([1, 0, 0]),
    "forward": np.array([0, 0, 1]),
    "backward": np.array([0, 0, -1]),
}

# --- Q-Learning Network ---
class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Environment Class ---
class MultiFishEnv:
    def __init__(self):
        self.fish_pos = np.zeros((NUM_FISH, 3), dtype=np.int32)
        self.food_pos = np.zeros((NUM_FOOD, 3), dtype=np.int32)
        self.food_tree = None
        self.step_count = 0
        self.trained_policy: QNet = None
        self.reset()

    def reset(self):
        self.step_count = 0
        self.fish_pos = np.random.randint(0, GRID_SIZE, size=(NUM_FISH, 3))
        self.food_pos = np.random.randint(0, GRID_SIZE, size=(NUM_FOOD, 3))
        self._update_food_tree()
        return self._get_states()

    def _update_food_tree(self):
        if self.food_pos.shape[0] > 0:
            self.food_tree = cKDTree(self.food_pos)
        else:
            self.food_tree = None

    def _get_states(self):
        states = np.zeros((NUM_FISH, 3), dtype=np.float32)
        if self.food_tree is None:
            return states
        
        # Find the closest food for all fish at once
        dist, idx = self.food_tree.query(self.fish_pos)
        
        for i in range(NUM_FISH):
            if np.isfinite(dist[i]):
                closest_food_pos = self.food_tree.data[idx[i]]
                state_vec = (closest_food_pos - self.fish_pos[i]).astype(np.float32)
                norm = np.linalg.norm(state_vec)
                if norm > 0:
                    state_vec /= norm
                states[i] = state_vec
        return states

    def step(self, actions):
        rewards = np.full(NUM_FISH, REWARD_STEP)
        
        for i in range(NUM_FISH):
            action = DISCRETE_ACTIONS[actions[i]]
            delta = ACTION_MAP[action]
            self.fish_pos[i] += delta
        
        self.fish_pos = np.clip(self.fish_pos, 0, GRID_SIZE - 1)
        self.step_count += 1

        done = False
        
        # Check for food eaten
        eaten_food_indices = set()
        if self.food_tree:
            # Find fish that are very close to any food
            collisions = self.food_tree.query_ball_point(self.fish_pos, r=0.5, p=2)
            for fish_idx, food_indices in enumerate(collisions):
                if food_indices: # This fish is near at least one food
                    rewards[fish_idx] = REWARD_FOOD
                    done = True # End episode if any fish eats
                    for food_idx in food_indices:
                        eaten_food_indices.add(food_idx)

        if eaten_food_indices:
            # Remove eaten food and respawn
            self.food_pos = np.delete(self.food_pos, list(eaten_food_indices), axis=0)
            new_food = np.random.randint(0, GRID_SIZE, size=(len(eaten_food_indices), 3))
            self.food_pos = np.vstack([self.food_pos, new_food])
            self._update_food_tree()

        # Episode also ends if it takes too long
        if self.step_count >= 100:
            done = True
        
        return self._get_states(), rewards, done

    def get_state_for_viz(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=int)
        food_val = list(ENTITY_TYPES.keys()).index("food") + 1
        for fx, fy, fz in self.food_pos:
            grid[fx][fy][fz] = food_val
        
        agents = []
        for i in range(NUM_FISH):
            agents.append({
                "id": i,
                "pos": [int(self.fish_pos[i, 0]), int(self.fish_pos[i, 1]), int(self.fish_pos[i, 2])],
                "energy": 100,
                "color": [0.2 + i*0.05, 0.5, 1.0 - i*0.05],
                "velocity": [0,0,0]
            })

        return {
            "grid": grid.tolist(),
            "agents": agents,
            "grid_size": [GRID_SIZE, GRID_SIZE, GRID_SIZE],
            "resource_types": ENTITY_TYPES,
        }

# --- Training Loop ---
EPISODES = 4000
GAMMA = 0.99
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 3000
LR = 1e-4

async def train_fish(websocket: WebSocket, env: MultiFishEnv):
    input_size = 3
    output_size = len(DISCRETE_ACTIONS)
    
    model = QNet(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    epsilon = EPSILON_START
    total_steps = 0
    
    for ep in range(EPISODES):
        states = env.reset()
        ep_reward_sum = 0
        
        for step in range(100):
            total_steps += 1
            
            actions = []
            states_t = torch.tensor(states, dtype=torch.float32)
            
            # Get actions for all fish
            if random.random() < epsilon:
                actions = np.random.randint(0, output_size, size=NUM_FISH).tolist()
            else:
                with torch.no_grad():
                    q_values = model(states_t)
                    actions = torch.argmax(q_values, dim=1).tolist()

            next_states, rewards, done = env.step(actions)
            ep_reward_sum += np.sum(rewards)

            # Batch update Q-learning
            # We treat each fish's experience as an independent sample
            q_preds = model(states_t).gather(1, torch.tensor(actions).unsqueeze(-1)).squeeze()
            
            with torch.no_grad():
                next_states_t = torch.tensor(next_states, dtype=torch.float32)
                q_next = model(next_states_t).max(dim=1).values
            
            q_targets = torch.tensor(rewards, dtype=torch.float32) + GAMMA * q_next * (1 - int(done))

            loss = nn.functional.mse_loss(q_preds, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 5 == 0:
                await websocket.send_json({
                    "type": "train_step", "state": env.get_state_for_viz(), "episode": ep
                })
            
            states = next_states
            if done:
                break
        
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * ep / EPSILON_DECAY)
        
        if (ep + 1) % 10 == 0:
            await websocket.send_json({
                "type": "progress", "episode": ep + 1, "reward": float(ep_reward_sum), "loss": loss.item()
            })

    await websocket.send_json({"type": "trained", "model_info": {"episodes": EPISODES, "loss": loss.item()}})

# --- Inference / Run Loop ---
async def run_fish(websocket: WebSocket, env: MultiFishEnv):
    if not env.trained_policy:
        await websocket.send_json({"type": "error", "message": "No trained policy available."})
        return

    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        states = env.reset()
        
        # Run for more steps to allow for more observation
        for step in range(500):
            states_t = torch.tensor(states, dtype=torch.float32)
            
            # --- FIX: Add epsilon-greedy exploration to break deterministic loops ---
            if random.random() < 0.1: # 10% chance of random action
                actions = np.random.randint(0, len(DISCRETE_ACTIONS), size=NUM_FISH).tolist()
            else:
                with torch.no_grad():
                    q_values = env.trained_policy(states_t)
                    actions = torch.argmax(q_values, dim=1).tolist()
            
            # We ignore the 'done' flag during inference to let the simulation run continuously.
            next_states, _, done = env.step(actions)
            states = next_states
            
            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.1)

            # No longer breaking when an episode would be 'done' in training.
            # This allows observing the fish over a longer period.
            if done:
                await asyncio.sleep(0.5) # Still pause briefly when food is eaten
                # but we don't break the loop anymore 