import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging

# Configure logging (disabled for cleaner operation)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fish.log'),
#     filemode='w'
# )

# --- Environment Constants ---
GRID_SIZE = 128
NUM_FISH = 128
NUM_FOOD = 128
REWARD_FOOD = 100.0
MAX_ENERGY = 100  # Very high so fish doesn't die
ENERGY_FROM_FOOD = 20
ENERGY_DECAY_STEP = 1  # Very slow decay
FISH_SPEED = 2.0  # Fast movement
FISH_RADIUS = 1.5  # Fish radius for collision detection
FISH_SPACING = FISH_RADIUS * 2 + 1.0  # Minimum distance between fish centers

# Shark constants
SHARK_SPEED = 1.5  # Slightly slower than fish so they can escape
SHARK_CATCH_DISTANCE = 3.0  # Distance at which shark catches fish
SHARK_RADIUS = 2.0  # Larger than fish
PENALTY_SHARK_CLOSE = -10.0  # Penalty for being close to shark
PENALTY_SHARK_DEATH = -200.0  # Heavy penalty for being caught

ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]},
    "food": {"value": 1, "color": [0.8, 0.8, 0.2]},
    "shark": {"value": 2, "color": [1, 1, 1]},
}

# --- Simple Network with Shark Awareness ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: distance to food + distance to shark (2 values)
        # Output: speed multiplier (1 value)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
        
        # Place all food randomly
        for i in range(NUM_FOOD):
            self.food_pos[i] = np.random.uniform(10, GRID_SIZE-10, size=3)
        
        # Place shark randomly (away from edges)
        self.shark_pos = np.random.uniform(SHARK_RADIUS + 5, GRID_SIZE - SHARK_RADIUS - 5, size=3)
        
        # Place all fish randomly with proper spacing (and away from shark)
        for i in range(NUM_FISH):
            attempts = 0
            while attempts < 100:  # Max attempts to find valid position
                candidate_pos = np.random.uniform(FISH_RADIUS + 2, GRID_SIZE - FISH_RADIUS - 2, size=3)
                
                # Check if this position conflicts with existing fish
                valid_position = True
                for j in range(i):
                    distance = np.linalg.norm(candidate_pos - self.fish_pos[j])
                    if distance < FISH_SPACING:
                        valid_position = False
                        break
                
                # Also check distance from shark (start fish far from shark)
                shark_distance = np.linalg.norm(candidate_pos - self.shark_pos)
                if shark_distance < 20.0:  # Start fish far from shark
                    valid_position = False
                
                if valid_position:
                    self.fish_pos[i] = candidate_pos
                    break
                
                attempts += 1
            
            # If we couldn't find a valid position after many attempts, place randomly
            if attempts >= 100:
                self.fish_pos[i] = np.random.uniform(FISH_RADIUS + 2, GRID_SIZE - FISH_RADIUS - 2, size=3)
        
        return self._get_state()

    def _resolve_fish_collisions(self, intended_positions):
        """Resolve collisions between fish by adjusting their positions"""
        adjusted_positions = intended_positions.copy()
        
        # Multiple passes to resolve all collisions
        for _ in range(5):  # Max 5 iterations to resolve conflicts
            collision_found = False
            
            for i in range(NUM_FISH):
                for j in range(i + 1, NUM_FISH):
                    # Calculate distance between fish i and j
                    distance = np.linalg.norm(adjusted_positions[i] - adjusted_positions[j])
                    
                    if distance < FISH_SPACING:
                        collision_found = True
                        
                        # Calculate separation vector
                        if distance > 1e-6:  # Avoid division by zero
                            separation_vector = (adjusted_positions[i] - adjusted_positions[j]) / distance
                        else:
                            # If fish are exactly on top of each other, separate randomly
                            separation_vector = np.random.uniform(-1, 1, 3)
                            separation_vector = separation_vector / np.linalg.norm(separation_vector)
                        
                        # Move fish apart to maintain minimum spacing
                        overlap = FISH_SPACING - distance
                        move_distance = overlap / 2.0
                        
                        adjusted_positions[i] += separation_vector * move_distance
                        adjusted_positions[j] -= separation_vector * move_distance
                        
                        # Keep fish within bounds
                        adjusted_positions[i] = np.clip(adjusted_positions[i], FISH_RADIUS, GRID_SIZE - FISH_RADIUS)
                        adjusted_positions[j] = np.clip(adjusted_positions[j], FISH_RADIUS, GRID_SIZE - FISH_RADIUS)
            
            if not collision_found:
                break
        
        return adjusted_positions

    def _get_state(self):
        """State: normalized distance to nearest food + distance to shark for each fish"""
        states = []
        for i in range(NUM_FISH):
            # Find nearest food for this fish
            distances_to_all_food = [np.linalg.norm(self.fish_pos[i] - food_pos) for food_pos in self.food_pos]
            min_food_distance = min(distances_to_all_food)
            normalized_food_distance = min_food_distance / (GRID_SIZE * np.sqrt(3))  # Normalize to 0-1
            
            # Distance to shark
            shark_distance = np.linalg.norm(self.fish_pos[i] - self.shark_pos)
            normalized_shark_distance = shark_distance / (GRID_SIZE * np.sqrt(3))  # Normalize to 0-1
            
            states.append(np.array([normalized_food_distance, normalized_shark_distance], dtype=np.float32))
        return states

    def step(self, actions):
        self.step_count += 1
        
        # Calculate old distances for each fish to their nearest food and shark
        old_food_distances = []
        old_shark_distances = []
        nearest_food_indices = []
        for i in range(NUM_FISH):
            distances_to_all_food = [np.linalg.norm(self.fish_pos[i] - food_pos) for food_pos in self.food_pos]
            min_distance = min(distances_to_all_food)
            nearest_food_idx = distances_to_all_food.index(min_distance)
            old_food_distances.append(min_distance)
            nearest_food_indices.append(nearest_food_idx)
            
            # Track old shark distance
            shark_distance = np.linalg.norm(self.fish_pos[i] - self.shark_pos)
            old_shark_distances.append(shark_distance)
        
        # Calculate intended positions for each fish (before collision resolution)
        intended_positions = self.fish_pos.copy()
        
        for i in range(NUM_FISH):
            # ALWAYS move toward nearest food
            nearest_food_pos = self.food_pos[nearest_food_indices[i]]
            food_direction = nearest_food_pos - self.fish_pos[i]
            food_direction = food_direction / (np.linalg.norm(food_direction) + 1e-8)  # Normalize
            
            # Use network output as speed multiplier (0-1)
            speed_multiplier = actions[i]
            if isinstance(speed_multiplier, np.ndarray):
                speed_multiplier = speed_multiplier.item()  # Extract scalar from numpy array
            elif hasattr(speed_multiplier, 'item'):
                speed_multiplier = speed_multiplier.item()  # Handle torch tensors too
            
            # Fish ALWAYS moves toward food, network just controls speed
            movement = food_direction * FISH_SPEED * speed_multiplier
            
            # Calculate intended position (before collision resolution)
            intended_positions[i] += movement
            intended_positions[i] = np.clip(intended_positions[i], FISH_RADIUS, GRID_SIZE - FISH_RADIUS)
        
        # Resolve fish collisions
        final_positions = self._resolve_fish_collisions(intended_positions)
        
        # Apply final positions
        old_positions = self.fish_pos.copy()
        self.fish_pos = final_positions
        
        # SHARK MOVEMENT: Move shark toward nearest fish
        old_shark_pos = self.shark_pos.copy()
        if NUM_FISH > 0:
            # Find nearest fish to shark
            distances_to_fish = [np.linalg.norm(self.shark_pos - fish_pos) for fish_pos in self.fish_pos]
            nearest_fish_idx = distances_to_fish.index(min(distances_to_fish))
            nearest_fish_pos = self.fish_pos[nearest_fish_idx]
            
            # Move shark toward nearest fish
            shark_direction = nearest_fish_pos - self.shark_pos
            shark_direction = shark_direction / (np.linalg.norm(shark_direction) + 1e-8)  # Normalize
            shark_movement = shark_direction * SHARK_SPEED
            
            self.shark_pos += shark_movement
            self.shark_pos = np.clip(self.shark_pos, SHARK_RADIUS, GRID_SIZE - SHARK_RADIUS)
        

        
        # Decay energy very slowly for all fish
        self.fish_energy -= ENERGY_DECAY_STEP
        
        # Calculate rewards for each fish
        rewards = []
        done = False
        food_eaten_indices = set()
        
        for i in range(NUM_FISH):
            # Find new nearest food and distance (after collision resolution)
            distances_to_all_food = [np.linalg.norm(self.fish_pos[i] - food_pos) for food_pos in self.food_pos]
            new_food_distance = min(distances_to_all_food)
            nearest_food_idx = distances_to_all_food.index(new_food_distance)
            
            # Check distance to shark (CRITICAL for survival!)
            new_shark_distance = np.linalg.norm(self.fish_pos[i] - self.shark_pos)
            
            reward = 0.0
            fish_caught_by_shark = False
            
            # CHECK IF SHARK CAUGHT FISH (most important check!)
            if new_shark_distance <= SHARK_CATCH_DISTANCE:
                reward = PENALTY_SHARK_DEATH
                fish_caught_by_shark = True
                done = True
            # Check if food is eaten (only if not caught by shark)
            elif new_food_distance <= 4.0:
                reward = REWARD_FOOD
                self.fish_energy[i] = min(MAX_ENERGY, self.fish_energy[i] + ENERGY_FROM_FOOD)
                self.total_food_eaten += 1
                food_eaten_indices.add(nearest_food_idx)
                done = True
            else:
                # Reward based on actual movement toward food (after collision resolution)
                actual_movement_distance = np.linalg.norm(self.fish_pos[i] - old_positions[i])
                food_distance_improvement = old_food_distances[i] - new_food_distance
                shark_distance_improvement = new_shark_distance - old_shark_distances[i]  # Positive when moving away from shark
                
                # Primary reward for moving toward food
                reward = food_distance_improvement * 10.0
                
                # MAJOR reward for staying away from shark
                reward += shark_distance_improvement * 15.0  # Even more important than food!
                
                # Bonus for actually moving (even if collision-adjusted)
                reward += actual_movement_distance * 2.0
                
                # Extra reward for being close to food
                if new_food_distance < 10.0:
                    reward += 5.0
                elif new_food_distance < 20.0:
                    reward += 2.0
                
                # MAJOR penalty for being close to shark
                if new_shark_distance < 10.0:
                    reward += PENALTY_SHARK_CLOSE  # -10
                elif new_shark_distance < 15.0:
                    reward -= 5.0  # Medium penalty
                
                # Small penalty for being slow (based on intended speed)
                speed_multiplier = actions[i]
                if isinstance(speed_multiplier, np.ndarray):
                    speed_multiplier = speed_multiplier.item()
                if speed_multiplier < 0.5:
                    reward -= 1.0
                
                # Small penalty for being too close to other fish (encourages spreading out)
                min_fish_distance = float('inf')
                for j in range(NUM_FISH):
                    if i != j:
                        fish_distance = np.linalg.norm(self.fish_pos[i] - self.fish_pos[j])
                        min_fish_distance = min(min_fish_distance, fish_distance)
                
                if min_fish_distance < FISH_SPACING * 1.5:
                    reward -= 0.5  # Small penalty for crowding
            
            # Death/respawn checks
            if fish_caught_by_shark or self.fish_energy[i] <= 0:
                if self.fish_energy[i] <= 0:
                    reward = -50.0  # Energy death penalty
                
                # Respawn fish with proper spacing (away from shark and other fish)
                for attempt in range(100):
                    candidate_pos = np.random.uniform(FISH_RADIUS + 2, GRID_SIZE - FISH_RADIUS - 2, size=3)
                    valid_pos = True
                    
                    # Check distance from other fish
                    for j in range(NUM_FISH):
                        if i != j and np.linalg.norm(candidate_pos - self.fish_pos[j]) < FISH_SPACING:
                            valid_pos = False
                            break
                    
                    # Check distance from shark (spawn far away!)
                    shark_distance = np.linalg.norm(candidate_pos - self.shark_pos)
                    if shark_distance < 25.0:  # Spawn far from shark
                        valid_pos = False
                    
                    if valid_pos:
                        self.fish_pos[i] = candidate_pos
                        break
                else:
                    # Fallback if no valid position found
                    self.fish_pos[i] = np.random.uniform(FISH_RADIUS + 2, GRID_SIZE - FISH_RADIUS - 2, size=3)
                
                self.fish_energy[i] = MAX_ENERGY
                done = True
            
            rewards.append(reward)
        
        # Respawn eaten food
        if food_eaten_indices:
            for food_idx in food_eaten_indices:
                self.food_pos[food_idx] = np.random.uniform(5, GRID_SIZE-5, size=3)
        
        return self._get_state(), rewards, done

    def get_state_for_viz(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=int)
        food_val = list(ENTITY_TYPES.keys()).index("food") + 1
        
        # Place all food items in the grid
        for i in range(NUM_FOOD):
            fx, fy, fz = self.food_pos[i].astype(int)
            if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE and 0 <= fz < GRID_SIZE:
                grid[fx][fy][fz] = food_val
        
        agents = []
        for i in range(NUM_FISH):
            # Calculate fish velocity direction (toward nearest food)
            distances_to_all_food = [np.linalg.norm(self.fish_pos[i] - food_pos) for food_pos in self.food_pos]
            min_distance = min(distances_to_all_food)
            nearest_food_idx = distances_to_all_food.index(min_distance)
            nearest_food_pos = self.food_pos[nearest_food_idx]
            
            food_direction = nearest_food_pos - self.fish_pos[i]
            food_direction = food_direction / (np.linalg.norm(food_direction) + 1e-8)  # Normalize
            velocity = (food_direction * FISH_SPEED).tolist()
            
            # Give different fish slightly different colors
            color_variation = i / NUM_FISH * 0.3  # Vary color slightly
            agents.append({
                "id": i,
                "pos": [float(self.fish_pos[i, 0]), float(self.fish_pos[i, 1]), float(self.fish_pos[i, 2])],
                "energy": int(self.fish_energy[i]),
                "color": [0.2 + color_variation, 0.8, 1.0 - color_variation],
                "velocity": velocity
            })

        # Calculate shark velocity (toward nearest fish)
        shark_velocity = [0, 0, 0]
        if NUM_FISH > 0:
            distances_to_fish = [np.linalg.norm(self.shark_pos - fish_pos) for fish_pos in self.fish_pos]
            nearest_fish_idx = distances_to_fish.index(min(distances_to_fish))
            nearest_fish_pos = self.fish_pos[nearest_fish_idx]
            
            shark_direction = nearest_fish_pos - self.shark_pos
            shark_direction = shark_direction / (np.linalg.norm(shark_direction) + 1e-8)  # Normalize
            shark_velocity = (shark_direction * SHARK_SPEED).tolist()

        return {
            "grid": grid.tolist(),
            "agents": agents,
            "shark": {
                "pos": self.shark_pos.astype(float).tolist(),
                "color": ENTITY_TYPES["shark"]["color"],
                "velocity": shark_velocity
            },
            "grid_size": [GRID_SIZE, GRID_SIZE, GRID_SIZE],
            "resource_types": ENTITY_TYPES,
        }

# --- Training Parameters ---
EPISODES = 800  # More episodes for complex shark-avoidance learning
LEARNING_RATE = 1e-3
STEPS_PER_EPISODE = 50

async def train_fish(websocket: WebSocket, env: FishEnv):
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    env.trained_policy = model
    
    for episode in range(EPISODES):
        states = env.reset()
        episode_reward = 0
        all_rewards = []
        all_states = []
        all_actions = []
        
        for step in range(STEPS_PER_EPISODE):
            # Batch process all fish states for faster inference
            states_batch = torch.FloatTensor(states)  # Shape: (NUM_FISH, 2)
            
            with torch.no_grad():
                speed_mults = model(states_batch)  # Shape: (NUM_FISH, 1)
            
            fish_actions = speed_mults.squeeze().numpy().tolist()
            if isinstance(fish_actions, float):  # Handle single fish case
                fish_actions = [fish_actions]
            
            # Step environment with all fish actions
            next_states, step_rewards, done = env.step(fish_actions)
            
            # Store data for all fish
            for i in range(NUM_FISH):
                all_states.append(states[i])
                all_actions.append(fish_actions[i])
                all_rewards.append(step_rewards[i])
                episode_reward += step_rewards[i]
            
            states = next_states
            
            # Send update to frontend less frequently during training
            if step % 10 == 0:
                await websocket.send_json({"type": "train_step", "state": env.get_state_for_viz(), "episode": episode})
            
            if done:
                break
        
        # Simple learning: reward high speeds when we got good rewards
        episode_loss = 0.0  # Track loss for this episode
        if len(all_rewards) > 0:
            avg_reward = np.mean(all_rewards)
            
            # Only update if we had decent performance
            if avg_reward > 0:
                # Convert to tensors
                states_tensor = torch.FloatTensor(all_states)
                # Target speed based on reward: higher reward = encourage faster movement
                target_speeds = torch.FloatTensor([min(1.0, max(0.5, 0.5 + reward/50.0)) for reward in all_rewards])
                
                # Forward pass
                predicted_speeds = model(states_tensor).squeeze()
                
                # Loss: encourage speeds that led to good rewards
                loss = nn.MSELoss()(predicted_speeds, target_speeds)
                episode_loss = loss.item()  # Capture the actual loss value
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        await websocket.send_json({
            "type": "progress", 
            "episode": episode + 1, 
            "reward": float(episode_reward), 
            "loss": float(episode_loss)  # Send actual loss instead of 0.0
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
        states = env.reset()
        
        for step in range(1000):
            # Batch process all fish states for faster inference
            states_batch = torch.FloatTensor(states)  # Shape: (NUM_FISH, 2)
            
            with torch.no_grad():
                speed_mults = env.trained_policy(states_batch)  # Shape: (NUM_FISH, 1)
            
            fish_actions = speed_mults.squeeze().numpy().tolist()
            if isinstance(fish_actions, float):  # Handle single fish case
                fish_actions = [fish_actions]
            
            next_states, _, done = env.step(fish_actions)
            states = next_states
            
            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.0001)  # 200 FPS for very smooth movement
            
            if done:
                await asyncio.sleep(0.5) 