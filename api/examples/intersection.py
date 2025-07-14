import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.spatial import cKDTree
from torch.distributions import Categorical

# --- Environment Constants ---
GRID_SIZE = 64
NUM_VEHICLES = 32
NUM_OBSTACLES = 16
REWARD_PROGRESS = 1.0
REWARD_COLLISION = -50.0
REWARD_STEP = -0.1
VEHICLE_SPEED = 1.0
PREDATOR_SPEED = 0.8  # Slower than vehicles

# For frontend compatibility, define ENTITY_TYPES
ENTITY_TYPES = {
    "road": {"value": 0, "color": [0.2, 0.2, 0.2]},
    "obstacle": {"value": 1, "color": [0.8, 0.2, 0.2]},
    "predator": {"value": 2, "color": [1, 1, 1]},
}

# --- Action Definitions ---
DISCRETE_ACTIONS = ["up", "down", "left", "right", "forward", "backward"]
ACTION_MAP = {
    "up": np.array([0, 3, 0]),
    "down": np.array([0, -3, 0]),
    "left": np.array([-3, 0, 0]),
    "right": np.array([3, 0, 0]),
    "forward": np.array([0, 0, 3]),
    "backward": np.array([0, 0, -3]),
}

# --- PPO Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, state, action=None):
        probs = self.actor(state)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

# --- Environment Class ---
class MultiVehicleEnv:
    def __init__(self):
        self.vehicle_pos = np.zeros((NUM_VEHICLES, 3), dtype=np.int32)
        self.obstacle_pos = np.zeros((NUM_OBSTACLES, 3), dtype=np.int32)
        self.predator_pos = np.zeros(3, dtype=np.float32)
        self.obstacle_tree = None
        self.vehicle_tree = None
        self.step_count = 0
        self.trained_policy: ActorCritic = None
        self.reset()

    def reset(self):
        self.step_count = 0
        self.vehicle_pos = np.random.randint(0, GRID_SIZE, size=(NUM_VEHICLES, 3))
        self.obstacle_pos = np.random.randint(0, GRID_SIZE, size=(NUM_OBSTACLES, 3))
        self.predator_pos = np.random.randint(0, GRID_SIZE, size=3).astype(np.float32)
        self._update_obstacle_tree()
        self._update_vehicle_tree()
        return self._get_states()

    def _update_obstacle_tree(self):
        if self.obstacle_pos.shape[0] > 0:
            self.obstacle_tree = cKDTree(self.obstacle_pos)
        else:
            self.obstacle_tree = None

    def _update_vehicle_tree(self):
        if self.vehicle_pos.shape[0] > 0:
            self.vehicle_tree = cKDTree(self.vehicle_pos)
        else:
            self.vehicle_tree = None

    def _get_states(self):
        states = np.zeros((NUM_VEHICLES, 6), dtype=np.float32)
        
        # Vector to closest obstacle
        if self.obstacle_tree:
            dist, idx = self.obstacle_tree.query(self.vehicle_pos)
            for i in range(NUM_VEHICLES):
                if np.isfinite(dist[i]):
                    closest_obstacle_pos = self.obstacle_tree.data[idx[i]]
                    states[i, :3] = (closest_obstacle_pos - self.vehicle_pos[i]).astype(np.float32)
        
        # Vector from predator
        for i in range(NUM_VEHICLES):
            states[i, 3:] = (self.vehicle_pos[i] - self.predator_pos).astype(np.float32)

        return states

    def step(self, actions):
        rewards = np.full(NUM_VEHICLES, REWARD_STEP)

        # Get distance to obstacles before moving
        old_dist_to_obstacle = np.full(NUM_VEHICLES, np.inf, dtype=np.float32)
        if self.obstacle_tree:
            dist, _ = self.obstacle_tree.query(self.vehicle_pos)
            if dist is not None:
                old_dist_to_obstacle = dist
        
        # 1. Move vehicles
        for i in range(NUM_VEHICLES):
            action = DISCRETE_ACTIONS[actions[i]]
            delta = ACTION_MAP[action]
            self.vehicle_pos[i] += delta
        self.vehicle_pos = np.clip(self.vehicle_pos, 0, GRID_SIZE - 1)
        self._update_vehicle_tree()

        # Add reward for avoiding obstacles or progressing
        if self.obstacle_tree:
            new_dist, _ = self.obstacle_tree.query(self.vehicle_pos)
            if new_dist is not None:
                dist_diff = old_dist_to_obstacle - new_dist
                proximity_penalty = np.minimum(0, dist_diff) * -1  # Penalize getting closer
                rewards[np.isfinite(proximity_penalty)] += proximity_penalty[np.isfinite(proximity_penalty)]

        # 2. Move predator towards nearest vehicle
        if self.vehicle_tree:
            dist, idx = self.vehicle_tree.query(self.predator_pos)
            if np.isfinite(dist):
                closest_vehicle_pos = self.vehicle_tree.data[idx]
                move_dir = closest_vehicle_pos - self.predator_pos
                norm = np.linalg.norm(move_dir)
                if norm > 0:
                    move_dir /= norm
                self.predator_pos += move_dir * PREDATOR_SPEED
                self.predator_pos = np.clip(self.predator_pos, 0, GRID_SIZE - 1)

        self.step_count += 1
        done = False
        
        # 3. Check for predator collisions
        collided_vehicle_indices = set()
        if self.vehicle_tree:
            predator_collisions = self.vehicle_tree.query_ball_point(self.predator_pos, r=2.0)
            for vehicle_idx in predator_collisions:
                rewards[vehicle_idx] += REWARD_COLLISION
                collided_vehicle_indices.add(vehicle_idx)
        
        # Respawn collided vehicles
        if collided_vehicle_indices:
            for i in collided_vehicle_indices:
                self.vehicle_pos[i] = np.random.randint(0, GRID_SIZE, size=3)
            self._update_vehicle_tree()

        # 4. Check for obstacle collisions
        collided_obstacle_indices = set()
        if self.obstacle_tree:
            collisions = self.obstacle_tree.query_ball_point(self.vehicle_pos, r=0.5, p=2)
            for vehicle_idx, obstacle_indices in enumerate(collisions):
                if vehicle_idx in collided_vehicle_indices:
                    continue
                if obstacle_indices:
                    rewards[vehicle_idx] += REWARD_COLLISION
                    done = True 
                    for obstacle_idx in obstacle_indices:
                        collided_obstacle_indices.add(obstacle_idx)

        if collided_obstacle_indices:
            self.obstacle_pos = np.delete(self.obstacle_pos, list(collided_obstacle_indices), axis=0)
            new_obstacles = np.random.randint(0, GRID_SIZE, size=(len(collided_obstacle_indices), 3))
            self.obstacle_pos = np.vstack([self.obstacle_pos, new_obstacles])
            self._update_obstacle_tree()

        if self.step_count >= 200:
            done = True
        
        return self._get_states(), rewards, done

    def get_state_for_viz(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=int)
        obstacle_val = list(ENTITY_TYPES.keys()).index("obstacle") + 1
        for ox, oy, oz in self.obstacle_pos:
            grid[ox][oy][oz] = obstacle_val
        
        agents = []
        for i in range(NUM_VEHICLES):
            agents.append({
                "id": i,
                "pos": [int(self.vehicle_pos[i, 0]), int(self.vehicle_pos[i, 1]), int(self.vehicle_pos[i, 2])],
                "energy": 100,  # Placeholder, can adapt for vehicle health or something
                "color": [0.2 + i*0.05, 1.0 - i*0.05, 0.5],
                "velocity": [0,0,0]
            })

        return {
            "grid": grid.tolist(),
            "agents": agents,
            "predator": {
                "pos": self.predator_pos.astype(int).tolist(),
                "color": ENTITY_TYPES["predator"]["color"],
            },
            "grid_size": [GRID_SIZE, GRID_SIZE, GRID_SIZE],
            "resource_types": ENTITY_TYPES,
        }

# --- PPO Training Loop ---
EPISODES = 1000
GAMMA = 0.99
LR = 3e-4
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
BATCH_SIZE = 128

async def train_intersection(websocket: WebSocket, env: MultiVehicleEnv):
    input_size = 6  # obstacle_vec (3) + predator_vec (3)
    output_size = len(DISCRETE_ACTIONS)
    
    model = ActorCritic(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    for ep in range(EPISODES):
        states = env.reset()
        ep_reward_sum = 0
        
        # --- Collect Trajectories ---
        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_dones = [], [], [], [], []
        
        for step in range(BATCH_SIZE):
            states_t = torch.tensor(states, dtype=torch.float32)
            
            with torch.no_grad():
                actions_t, log_probs_t, _ = model.get_action(states_t)
            
            actions = actions_t.tolist()
            log_probs = log_probs_t.tolist()

            next_states, rewards, done = env.step(actions)
            ep_reward_sum += np.sum(rewards)

            batch_states.append(states)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)
            batch_dones.append([done] * NUM_VEHICLES)

            states = next_states
            if done:
                states = env.reset()
        
        # --- Compute Advantages and Returns ---
        states_t = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, last_values = model(states_t)
        
        returns = last_values.squeeze().numpy()
        advantages = np.zeros(NUM_VEHICLES)
        
        all_advantages = torch.zeros((BATCH_SIZE, NUM_VEHICLES), dtype=torch.float32)
        all_returns = torch.zeros((BATCH_SIZE, NUM_VEHICLES), dtype=torch.float32)

        for t in reversed(range(BATCH_SIZE)):
            rewards_t = np.array(batch_rewards[t])
            dones_t = np.array(batch_dones[t])
            
            td_error = rewards_t + GAMMA * returns * (1 - dones_t) - model(torch.tensor(batch_states[t], dtype=torch.float32))[1].detach().squeeze().numpy()
            advantages = td_error + GAMMA * GAE_LAMBDA * advantages * (1 - dones_t)
            returns = rewards_t + GAMMA * returns * (1 - dones_t)
            
            all_advantages[t] = torch.tensor(advantages, dtype=torch.float32)
            all_returns[t] = torch.tensor(returns, dtype=torch.float32)

        b_states = torch.tensor(np.array(batch_states), dtype=torch.float32).view(-1, input_size)
        b_actions = torch.tensor(np.array(batch_actions), dtype=torch.int64).view(-1)
        b_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32).view(-1)
        b_advantages = all_advantages.view(-1)
        b_returns = all_returns.view(-1)

        for _ in range(UPDATE_EPOCHS):
            _, log_probs, entropy = model.get_action(b_states, b_actions)
            values = model(b_states)[1].squeeze()

            ratio = torch.exp(log_probs - b_log_probs)
            
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.functional.mse_loss(values, b_returns)
            
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        await websocket.send_json({
            "type": "train_step", "state": env.get_state_for_viz(), "episode": ep
        })
        
        await websocket.send_json({
            "type": "progress", "episode": ep + 1, "reward": float(ep_reward_sum / BATCH_SIZE), "loss": loss.item()
        })

    await websocket.send_json({"type": "trained", "model_info": {"episodes": EPISODES, "loss": loss.item()}})


# --- Inference / Run Loop ---
async def run_intersection(websocket: WebSocket, env: MultiVehicleEnv):
    if not env.trained_policy:
        await websocket.send_json({"type": "error", "message": "No trained policy available."})
        return

    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        states = env.reset()
        
        for step in range(500):
            states_t = torch.tensor(states, dtype=torch.float32)
            
            with torch.no_grad():
                actions_t, _, _ = env.trained_policy.get_action(states_t)
                actions = actions_t.tolist()
            
            next_states, _, done = env.step(actions)
            states = next_states
            
            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.1)

            if done:
                await asyncio.sleep(0.5) 