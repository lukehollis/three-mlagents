import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.spatial import cKDTree
from torch.distributions import Categorical, Normal
import logging

# Configure logging to write to a file, overwriting it on each run
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='fish.log',
    filemode='w'
)


# --- Environment Constants ---
GRID_SIZE = 24 # Reduced grid size
NUM_FISH = 1
NUM_FOOD = 1 # Start with one food for curriculum
REWARD_FOOD = 50.0 # Increased reward
REWARD_DEATH = -10.0 # Reduced penalty
MAX_ENERGY = 200 # Increased energy
ENERGY_FROM_FOOD = 50
ENERGY_DECAY_STEP = 1
REWARD_STEP = -1
SHARK_SPEED = 0.8 # Slower than fish
FISH_SPEED = 1.5 # Fish can move faster

# For frontend compatibility, we define ENTITY_TYPES but only use food and water
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 1, "color": [0.8, 0.8, 0.2]},
    "shark": {"value": 2, "color": [1, 1, 1]},
}

# --- PPO Actor-Critic Network for Continuous Actions ---
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()  # Output a direction vector between -1 and 1
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Use a learnable parameter for log standard deviation
        self.log_std = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, x):
        # The actor outputs the mean of the policy distribution
        mu = self.actor(x)
        value = self.critic(x)
        return mu, value

    def get_action(self, state, action=None):
        mu, _ = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# --- Environment Class ---
class MultiFishEnv:
    def __init__(self):
        self.fish_pos = np.zeros((NUM_FISH, 3), dtype=np.float32) # Use float for position
        self.fish_energy = np.full(NUM_FISH, MAX_ENERGY, dtype=np.int32)
        self.food_pos = np.zeros((NUM_FOOD, 3), dtype=np.int32)
        # --- Disable Shark ---
        self.shark_pos = np.array([-1000, -1000, -1000], dtype=np.float32)
        self.food_tree = None
        self.fish_tree = None
        self.step_count = 0
        self.trained_policy: ActorCritic = None
        self.reset()

    def reset(self):
        self.step_count = 0
        
        # --- Curriculum: Start fish very close to a single food pellet ---
        self.food_pos = np.array([[GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2]])
        
        # Place fish randomly in a 3x3x3 box around the food
        offset = np.random.randint(-2, 3, size=3)
        self.fish_pos = (self.food_pos + offset).astype(np.float32)

        self.fish_energy.fill(MAX_ENERGY)
        # --- Disable Shark ---
        self.shark_pos = np.array([-1000, -1000, -1000], dtype=np.float32)
        self._update_food_tree()
        self._update_fish_tree()
        
        return self._get_state_dict()

    def _get_state_dict(self):
        state_dicts = []
        food_dists, food_indices = self.food_tree.query(self.fish_pos) if self.food_tree and self.food_pos.shape[0] > 0 else (np.full(NUM_FISH, np.inf), None)

        for i in range(NUM_FISH):
            state = {}

            # Food info
            if self.food_tree and food_indices is not None and np.isfinite(food_dists[i]):
                closest_food_pos = self.food_tree.data[food_indices[i]]
                vec_to_food = (closest_food_pos - self.fish_pos[i]).astype(np.float32)
                dist_to_food = food_dists[i]
                if dist_to_food > 0:
                    state['vec_to_food'] = vec_to_food / dist_to_food
                else:
                    state['vec_to_food'] = np.zeros(3, dtype=np.float32)
                state['dist_to_food'] = dist_to_food / GRID_SIZE
            else:
                state['vec_to_food'] = np.zeros(3, dtype=np.float32)
                state['dist_to_food'] = 1.0
            
            state_dicts.append(state)
            
        return state_dicts

    def _update_food_tree(self):
        if self.food_pos.shape[0] > 0:
            self.food_tree = cKDTree(self.food_pos)
        else:
            self.food_tree = None

    def _update_fish_tree(self):
        if self.fish_pos.shape[0] > 0:
            self.fish_tree = cKDTree(self.fish_pos)
        else:
            self.fish_tree = None

    def step(self, actions):
        # Get distances before moving
        old_dist_to_food = np.linalg.norm(self.fish_pos - self.food_tree.data[self.food_tree.query(self.fish_pos)[1]], axis=1) if self.food_tree else np.full(NUM_FISH, np.inf)

        rewards = np.zeros(NUM_FISH, dtype=np.float32)
        self.fish_energy -= ENERGY_DECAY_STEP

        # 1. Move fish
        for i in range(NUM_FISH):
            action = actions[i] # Continuous action
            delta = action * FISH_SPEED # Scale action by speed
            self.fish_pos[i] += delta
        self.fish_pos = np.clip(self.fish_pos, 0, GRID_SIZE - 1)
        self._update_fish_tree()

        # --- Reward Shaping ---
        # 1. Food Proximity Reward
        if self.food_tree:
            new_dist_to_food = np.linalg.norm(self.fish_pos - self.food_tree.data[self.food_tree.query(self.fish_pos)[1]], axis=1)
            food_dist_diff = old_dist_to_food - new_dist_to_food
            rewards += food_dist_diff * 5.0 # Increased reward for getting closer

        # 2. Move shark towards nearest fish
        # Shark is disabled for now to simplify learning

        self.step_count += 1
        done = False
        
        # 3. Check for deaths (shark or starvation)
        dead_fish_indices = set()
        
        # Starvation
        starved_indices = np.where(self.fish_energy <= 0)[0]
        for fish_idx in starved_indices:
            logging.info(f"  OUCH! Fish {fish_idx} starved to death.")
            rewards[fish_idx] += REWARD_DEATH
            dead_fish_indices.add(fish_idx)
        
        # Respawn dead fish
        if dead_fish_indices:
            done = True # Episode ends on death
            for i in dead_fish_indices:
                self.fish_pos[i] = np.random.randint(0, GRID_SIZE, size=3)
                self.fish_energy[i] = MAX_ENERGY # Reset energy
            self._update_fish_tree()

        # 4. Check for food eaten by surviving fish
        eaten_food_indices = set()
        if self.food_tree:
            # Create a new tree for only the surviving fish
            surviving_fish_mask = np.ones(NUM_FISH, dtype=bool)
            surviving_fish_mask[list(dead_fish_indices)] = False
            
            surviving_fish_pos = self.fish_pos[surviving_fish_mask]
            
            if surviving_fish_pos.shape[0] > 0:
                surviving_fish_tree = cKDTree(surviving_fish_pos)
                collisions = self.food_tree.query_ball_point(surviving_fish_pos, r=1.5, p=2)
                
                original_indices = np.where(surviving_fish_mask)[0]

                for i, food_indices in enumerate(collisions):
                    if food_indices:
                        fish_idx = original_indices[i]
                        logging.info(f"  YUM! Fish {fish_idx} ate food.")
                        rewards[fish_idx] += REWARD_FOOD
                        self.fish_energy[fish_idx] = min(MAX_ENERGY, self.fish_energy[fish_idx] + ENERGY_FROM_FOOD)
                        done = True 
                        for food_idx in food_indices:
                            eaten_food_indices.add(food_idx)

        if eaten_food_indices:
            logging.info(f"  Food eaten: {len(eaten_food_indices)}. Food left: {len(self.food_pos) - len(eaten_food_indices)}")
            self.food_pos = np.delete(self.food_pos, list(eaten_food_indices), axis=0)
            # Respawn the eaten food
            new_food = np.random.randint(0, GRID_SIZE, size=(len(eaten_food_indices), 3))
            if self.food_pos.shape[0] == 0:
                self.food_pos = new_food
            else:
                self.food_pos = np.vstack([self.food_pos, new_food])
            
            logging.info(f"  Food respawned. New food count: {len(self.food_pos)}")
            self._update_food_tree()

        state_dicts = self._get_state_dict()
        logging.info(f"Step {self.step_count} | Action: {np.array2string(actions[0], formatter={'float_kind':lambda x: f'{x:4.1f}'})} | Energy: {self.fish_energy[0]:<3} | Reward: {rewards[0]:<5.1f} | State: { {k: f'{v:.2f}' if isinstance(v, float) else v for k, v in state_dicts[0].items()} }")
        
        return state_dicts, rewards, done

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
                "energy": int(self.fish_energy[i]),
                "color": [0.2 + i*0.05, 0.5, 1.0 - i*0.05],
                "velocity": [0,0,0]
            })

        return {
            "grid": grid.tolist(),
            "agents": agents,
            "shark": {
                "pos": self.shark_pos.astype(int).tolist(),
                "color": ENTITY_TYPES["shark"]["color"],
            },
            "grid_size": [GRID_SIZE, GRID_SIZE, GRID_SIZE],
            "resource_types": ENTITY_TYPES,
        }

def _state_dict_to_tensor(state_dicts):
    """Converts a list of state dicts to a tensor."""
    batch_size = len(state_dicts)
    tensor = torch.zeros((batch_size, 4), dtype=torch.float32)
    for i, state in enumerate(state_dicts):
        tensor[i, 0:3] = torch.tensor(state.get('vec_to_food', np.zeros(3)), dtype=torch.float32)
        tensor[i, 3] = state.get('dist_to_food', 1.0)
    return tensor

# --- PPO Training Loop ---
EPISODES = 1000
GAMMA = 0.99
LR = 3e-5 # Reduced learning rate
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 10 # Increased update epochs
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.0001 # Drastically reduced entropy
BATCH_SIZE = 128
TARGET_UPDATE_TAU = 0.005 # Tau for soft target update

async def train_fish(websocket: WebSocket, env: MultiFishEnv):
    input_size = 4 # vec_food(3), dist_food(1)
    output_size = 3 # Continuous action space (x, y, z)
    
    model = ActorCritic(input_size, output_size)
    target_model = ActorCritic(input_size, output_size)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    for ep in range(EPISODES):
        states_dict = env.reset()
        ep_reward_sum = 0
        
        # --- Collect Trajectories ---
        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_dones = [], [], [], [], []
        
        for step in range(BATCH_SIZE):
            states_t = _state_dict_to_tensor(states_dict)
            
            with torch.no_grad():
                actions_t, log_probs_t, _ = model.get_action(states_t)
            
            actions = actions_t.cpu().numpy() # Convert to numpy array
            log_probs = log_probs_t.tolist()

            next_states_dict, rewards, done = env.step(actions)
            ep_reward_sum += np.sum(rewards)

            batch_states.append(states_dict)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)
            batch_dones.append([done] * NUM_FISH)

            states_dict = next_states_dict
            if done:
                states_dict = env.reset() # Reset if an episode ends mid-batch
        
        # --- Compute Advantages and Returns ---
        with torch.no_grad():
            next_states_t = _state_dict_to_tensor(states_dict)
            _, last_values = target_model(next_states_t)
        
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
            states_t_at_t = _state_dict_to_tensor(batch_states[t])
            with torch.no_grad():
                _, values_t = target_model(states_t_at_t)
            
            td_error = rewards_t + GAMMA * returns * (1 - dones_t) - values_t.detach().squeeze().numpy()
            advantages = td_error + GAMMA * GAE_LAMBDA * advantages * (1 - dones_t)
            returns = rewards_t + GAMMA * returns * (1 - dones_t)
            
            all_advantages[t] = torch.tensor(advantages, dtype=torch.float32)
            all_returns[t] = torch.tensor(returns, dtype=torch.float32)

        # Flatten batches for update
        flat_states_list = [item for sublist in batch_states for item in sublist]
        b_states = _state_dict_to_tensor(flat_states_list)
        b_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32).view(-1, 3) # Continuous actions
        b_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32).view(-1)
        b_advantages = all_advantages.view(-1)
        b_returns = all_returns.view(-1)

        # --- Normalize Advantages ---
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # --- Update Policy ---
        for _ in range(UPDATE_EPOCHS):
            mu, log_probs, entropy = model.get_action(b_states, b_actions)
            values = model(b_states)[1].squeeze()

            ratio = torch.exp(log_probs - b_log_probs)
            
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.functional.mse_loss(values, b_returns)
            
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            # --- Clip Gradients ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # --- Soft update of the target network ---
            for target_param, local_param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(TARGET_UPDATE_TAU * local_param.data + (1.0 - TARGET_UPDATE_TAU) * target_param.data)


        logging.info(f"\n--- Episode {ep+1} Summary ---")
        logging.info(f"  Total Reward: {ep_reward_sum:.2f}")
        logging.info(f"  Avg Reward:   {ep_reward_sum / BATCH_SIZE:.2f}")
        logging.info(f"  Final Loss:   {loss.item():.4f}")
        logging.info(f"    - Policy Loss: {policy_loss.item():.4f}")
        logging.info(f"    - Value Loss:  {value_loss.item():.4f}")
        logging.info(f"  Entropy:      {entropy.mean().item():.4f}")
        logging.info(f"  Std Dev:      {torch.exp(model.log_std).mean().item():.4f}\n")

        await websocket.send_json({
            "type": "train_step", "state": env.get_state_for_viz(), "episode": ep
        })
        
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
        states_dict = env.reset()
        
        # Run for more steps to allow for more observation
        for step in range(500):
            states_t = _state_dict_to_tensor(states_dict)
            
            with torch.no_grad():
                # During inference, we sample from the policy's distribution
                actions_t, _, _ = env.trained_policy.get_action(states_t)
                actions = actions_t.cpu().numpy() # Convert to numpy
            
            # We ignore the 'done' flag during inference to let the simulation run continuously.
            next_states_dict, _, done = env.step(actions)
            states_dict = next_states_dict
            
            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.1)

            # No longer breaking when an episode would be 'done' in training.
            # This allows observing the fish over a longer period.
            if done:
                await asyncio.sleep(0.5) # Still pause briefly when food is eaten
                # but we don't break the loop anymore 