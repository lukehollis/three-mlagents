import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fish.log'),
    filemode='w'
)

# --- Environment Constants ---
GRID_SIZE = 64
NUM_FISH = 32
NUM_FOOD = 32
REWARD_FOOD = 100.0
MAX_ENERGY = 1000  # Very high so fish doesn't die
ENERGY_FROM_FOOD = 200
ENERGY_DECAY_STEP = 0.1  # Very slow decay
FISH_SPEED = 2.0  # Fast movement

ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 1, "color": [0.8, 0.8, 0.2]},
    "shark": {"value": 2, "color": [1, 1, 1]},
}

# --- Super Simple Network ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: just distance to food (1 value)
        # Output: speed multiplier (1 value)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1 for speed
        )

    def forward(self, x):
        return self.net(x)

# --- Dead Simple Environment ---
class FishEnv:
    def __init__(self):
        self.fish_pos = np.zeros((NUM_FISH, 3), dtype=np.float32)
        self.fish_energy = np.full(NUM_FISH, MAX_ENERGY, dtype=np.float32)
        self.food_pos = np.zeros((NUM_FOOD, 3), dtype=np.float32)
        self.shark_pos = np.array([-1000, -1000, -1000], dtype=np.float32)
        self.step_count = 0
        self.trained_policy = None
        self.total_food_eaten = 0
        self.reset()

    def reset(self):
        self.step_count = 0
        self.fish_energy.fill(MAX_ENERGY)
        
        # Place food randomly
        self.food_pos[0] = np.random.uniform(10, GRID_SIZE-10, size=3)
        
        # Place fish randomly but away from food
        while True:
            self.fish_pos[0] = np.random.uniform(10, GRID_SIZE-10, size=3)
            distance = np.linalg.norm(self.fish_pos[0] - self.food_pos[0])
            if distance > 15:  # Start far from food
                break
        
        logging.info(f"Reset: Fish at {self.fish_pos[0]}, Food at {self.food_pos[0]}")
        return self._get_state()

    def _get_state(self):
        """Super simple state: just normalized distance to nearest food for each fish"""
        states = []
        for i in range(NUM_FISH):
            # Find nearest food for this fish
            distances_to_all_food = [np.linalg.norm(self.fish_pos[i] - food_pos) for food_pos in self.food_pos]
            min_distance = min(distances_to_all_food)
            normalized_distance = min_distance / (GRID_SIZE * np.sqrt(3))  # Normalize to 0-1
            states.append(np.array([normalized_distance], dtype=np.float32))
        return states

    def step(self, actions):
        self.step_count += 1
        old_distance = np.linalg.norm(self.fish_pos[0] - self.food_pos[0])
        
        # ALWAYS move toward food - this is the key insight
        food_direction = self.food_pos[0] - self.fish_pos[0]
        food_direction = food_direction / (np.linalg.norm(food_direction) + 1e-8)  # Normalize
        
        # Use network output as speed multiplier (0-1)
        speed_multiplier = actions[0]
        if isinstance(speed_multiplier, np.ndarray):
            speed_multiplier = speed_multiplier.item()  # Extract scalar from numpy array
        elif hasattr(speed_multiplier, 'item'):
            speed_multiplier = speed_multiplier.item()  # Handle torch tensors too
        
        # Fish ALWAYS moves toward food, network just controls speed
        movement = food_direction * FISH_SPEED * speed_multiplier
        
        old_pos = self.fish_pos[0].copy()
        self.fish_pos[0] += movement
        self.fish_pos[0] = np.clip(self.fish_pos[0], 0, GRID_SIZE - 1)
        
        logging.info(f"Step {self.step_count}: Old pos {old_pos}, Food at {self.food_pos[0]}")
        logging.info(f"  Food direction: {food_direction}, Speed mult: {speed_multiplier:.3f}")
        logging.info(f"  Movement: {movement}, New pos: {self.fish_pos[0]}")
        
        # Decay energy very slowly
        self.fish_energy -= ENERGY_DECAY_STEP
        
        new_distance = np.linalg.norm(self.fish_pos[0] - self.food_pos[0])
        reward = 0.0
        done = False
        
        # Check if food is eaten
        if new_distance <= 4.0:
            reward = REWARD_FOOD
            self.fish_energy[0] = min(MAX_ENERGY, self.fish_energy[0] + ENERGY_FROM_FOOD)
            self.total_food_eaten += 1
            
            # Respawn food far away
            self.food_pos[0] = np.random.uniform(5, GRID_SIZE-5, size=3)
            new_distance = np.linalg.norm(self.fish_pos[0] - self.food_pos[0])
            
            logging.info(f"*** FOOD EATEN! Total: {self.total_food_eaten} ***")
            logging.info(f"  Reward: {REWARD_FOOD}, New food at: {self.food_pos[0]}")
            done = True
        else:
            # Big reward for moving toward food
            distance_improvement = old_distance - new_distance
            reward = distance_improvement * 10.0  # Big multiplier
            
            # Extra reward for being close
            if new_distance < 10.0:
                reward += 5.0
            elif new_distance < 20.0:
                reward += 2.0
            
            # Small penalty for being slow
            if speed_multiplier < 0.5:
                reward -= 1.0
        
        # Death check (very unlikely with high energy)
        if self.fish_energy[0] <= 0:
            reward = -50.0
            self.fish_pos[0] = np.random.uniform(10, GRID_SIZE-10, size=3)
            self.fish_energy[0] = MAX_ENERGY
            done = True
            logging.info(f"Fish died! Respawned at: {self.fish_pos[0]}")
        
        logging.info(f"  Distance: {old_distance:.2f} -> {new_distance:.2f}, Reward: {reward:.2f}")
        
        return self._get_state(), [reward], done

    def get_state_for_viz(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=int)
        food_val = list(ENTITY_TYPES.keys()).index("food") + 1
        fx, fy, fz = self.food_pos[0].astype(int)
        if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE and 0 <= fz < GRID_SIZE:
            grid[fx][fy][fz] = food_val
        
        agents = []
        for i in range(NUM_FISH):
            agents.append({
                "id": i,
                "pos": [int(self.fish_pos[i, 0]), int(self.fish_pos[i, 1]), int(self.fish_pos[i, 2])],
                "energy": int(self.fish_energy[i]),
                "color": [0.2, 0.8, 1.0],
                "velocity": [0, 0, 0]
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

# --- Training Parameters ---
EPISODES = 200
LEARNING_RATE = 1e-3
STEPS_PER_EPISODE = 50

async def train_fish(websocket: WebSocket, env: FishEnv):
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    env.trained_policy = model
    
    logging.info("Starting SIMPLE fish training - fish will ALWAYS move toward food!")
    
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        rewards = []
        states = []
        actions = []
        
        for step in range(STEPS_PER_EPISODE):
            state_tensor = torch.FloatTensor(state[0]).unsqueeze(0)
            
            # Get action from network (speed multiplier)
            with torch.no_grad():
                speed_mult = model(state_tensor)
            
            speed_mult_np = speed_mult.squeeze().numpy()
            next_state, reward, done = env.step([speed_mult_np])
            
            states.append(state[0])
            actions.append(speed_mult_np)
            rewards.append(reward[0])
            
            episode_reward += reward[0]
            state = next_state
            
            # Send update to frontend frequently
            if step % 5 == 0:
                await websocket.send_json({"type": "train_step", "state": env.get_state_for_viz(), "episode": episode})
            
            if done:
                break
        
        # Simple learning: reward high speeds when we got good rewards
        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
            
            # Only update if we had decent performance
            if avg_reward > 0:
                # Convert to tensors
                states_tensor = torch.FloatTensor(states)
                target_speeds = torch.FloatTensor([min(1.0, max(0.5, 0.5 + avg_reward/50.0)) for _ in actions])
                
                # Forward pass
                predicted_speeds = model(states_tensor).squeeze()
                
                # Loss: encourage speeds that led to good rewards
                loss = nn.MSELoss()(predicted_speeds, target_speeds)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                logging.info(f"Episode {episode+1}: Reward={episode_reward:.1f}, AvgReward={avg_reward:.2f}, Loss={loss.item():.4f}, FoodEaten={env.total_food_eaten}")
            else:
                logging.info(f"Episode {episode+1}: Reward={episode_reward:.1f}, FoodEaten={env.total_food_eaten} (no learning)")
        
        await websocket.send_json({
            "type": "progress", 
            "episode": episode + 1, 
            "reward": float(episode_reward), 
            "loss": 0.0
        })
    
    await websocket.send_json({
        "type": "trained", 
        "model_info": {"episodes": EPISODES, "total_food_eaten": env.total_food_eaten}
    })

async def run_fish(websocket: WebSocket, env: FishEnv):
    if not env.trained_policy:
        await websocket.send_json({"type": "error", "message": "No trained policy available."})
        return

    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        state = env.reset()
        
        for step in range(1000):
            state_tensor = torch.FloatTensor(state[0]).unsqueeze(0)
            
            with torch.no_grad():
                speed_mult = env.trained_policy(state_tensor)
            
            speed_mult_np = speed_mult.squeeze().numpy()
            next_state, _, done = env.step([speed_mult_np])
            state = next_state
            
            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.1)
            
            if done:
                await asyncio.sleep(0.5) 