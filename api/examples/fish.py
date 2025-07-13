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
NUM_FISH = 32
NUM_FOOD = 32
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
class MultiFishEnv:
    def __init__(self):
        self.fish_pos = np.zeros((NUM_FISH, 3), dtype=np.int32)
        self.food_pos = np.zeros((NUM_FOOD, 3), dtype=np.int32)
        self.food_tree = None
        self.step_count = 0
        self.trained_policy: ActorCritic = None
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
                # No normalization for PPO, to provide distance information
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
        if self.step_count >= 200: # Increased step count for PPO
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

# --- PPO Training Loop ---
EPISODES = 4000
GAMMA = 0.99
LR = 3e-4
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
BATCH_SIZE = 128

async def train_fish(websocket: WebSocket, env: MultiFishEnv):
    input_size = 3
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
            batch_dones.append([done] * NUM_FISH)

            states = next_states
            if done:
                states = env.reset() # Reset if an episode ends mid-batch
        
        # --- Compute Advantages and Returns ---
        states_t = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, last_values = model(states_t)
        
        returns = last_values.squeeze().numpy()
        advantages = np.zeros(NUM_FISH)
        
        # These need to be tensors for the update loop
        all_advantages = torch.zeros((BATCH_SIZE, NUM_FISH), dtype=torch.float32)
        all_returns = torch.zeros((BATCH_SIZE, NUM_FISH), dtype=torch.float32)

        for t in reversed(range(BATCH_SIZE)):
            # rewards and dones are for all fish at timestep t
            rewards_t = np.array(batch_rewards[t])
            dones_t = np.array(batch_dones[t])
            
            # GAE
            td_error = rewards_t + GAMMA * returns * (1 - dones_t) - model(torch.tensor(batch_states[t], dtype=torch.float32))[1].detach().squeeze().numpy()
            advantages = td_error + GAMMA * GAE_LAMBDA * advantages * (1 - dones_t)
            returns = rewards_t + GAMMA * returns * (1 - dones_t)
            
            all_advantages[t] = torch.tensor(advantages, dtype=torch.float32)
            all_returns[t] = torch.tensor(returns, dtype=torch.float32)

        # Flatten batches for update
        b_states = torch.tensor(np.array(batch_states), dtype=torch.float32).view(-1, input_size)
        b_actions = torch.tensor(np.array(batch_actions), dtype=torch.int64).view(-1)
        b_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32).view(-1)
        b_advantages = all_advantages.view(-1)
        b_returns = all_returns.view(-1)

        # --- Update Policy ---
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

        if (ep + 1) % 5 == 0:
            await websocket.send_json({
                "type": "train_step", "state": env.get_state_for_viz(), "episode": ep
            })
        
        if (ep + 1) % 10 == 0:
            await websocket.send_json({
                "type": "progress", "episode": ep + 1, "reward": float(ep_reward_sum / BATCH_SIZE), "loss": loss.item()
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
            
            with torch.no_grad():
                # During inference, we take the most likely action (no sampling)
                probs, _ = env.trained_policy(states_t)
                actions = torch.argmax(probs, dim=1).tolist()
            
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