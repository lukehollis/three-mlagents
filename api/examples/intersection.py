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
NUM_VEHICLES = 16
REWARD_PROGRESS = 1.0
REWARD_COLLISION = -50.0
REWARD_STEP = -0.1
VEHICLE_MIN_SPEED = 0.5
VEHICLE_MAX_SPEED = 2.0
ACCELERATION = 0.2
REWARD_RED_LIGHT = -25.0

# --- Traffic Light Definitions ---
class TrafficLightController:
    def __init__(self, cycle_time=200):
        # 0: NS Green, EW Red | 1: NS Red, EW Green
        self.state = 0
        self.cycle_time = cycle_time
        self.timer = 0

    def update(self):
        self.timer += 1
        if self.timer >= self.cycle_time:
            self.state = 1 - self.state
            self.timer = 0

INTERSECTIONS = {
    'main': {'pos': np.array([0, 0, 0]), 'radius': 10.0, 'controlled_by': 'NS'},
    'secondary': {'pos': np.array([-25, 0, 0]), 'radius': 8.0, 'controlled_by': 'EW'},
    'third': {'pos': np.array([20, 0, 0]), 'radius': 8.0, 'controlled_by': 'EW'}
}

# --- Waypoint and Path Definitions ---
WAYPOINTS = {
    # Primary Roads
    'H_E': np.array([40.0, 0.0, 0.0]), 'H_W': np.array([-40.0, 0.0, 0.0]),
    'V_N': np.array([0.0, 0.0, 20.0]),
    'I_CENTER': np.array([0.0, 0.0, 0.0]),

    # Feeder roads
    'TR_N': np.array([20.0, 0.0, 20.0]), 'TR_I': np.array([20.0, 0.0, 0.0]),
    'BL_S': np.array([-25.0, 0.0, -20.0]), 'BL_I': np.array([-25.0, 0.0, 0.0]),

    # Curved road from South-East
    'CR_S': np.array([25.0, 0.0, -20.0]),
    'CR_M': np.array([10.0, 0.0, -10.0]),
}


PATHS = {
    # E-W Traffic
    'E_to_W': (['H_E', 'TR_I', 'I_CENTER', 'BL_I', 'H_W'], 'EW'),
    'W_to_E': (['H_W', 'BL_I', 'I_CENTER', 'TR_I', 'H_E'], 'EW'),

    # From South (Curve)
    'S_Curve_to_N': (['CR_S', 'CR_M', 'I_CENTER', 'V_N'], 'NS'),
    'S_Curve_to_W': (['CR_S', 'CR_M', 'I_CENTER', 'BL_I', 'H_W'], 'NS'),
    'S_Curve_to_E': (['CR_S', 'CR_M', 'I_CENTER', 'TR_I', 'H_E'], 'NS'),

    # From North
    'N_to_S_Curve': (['V_N', 'I_CENTER', 'CR_M', 'CR_S'], 'NS'),
    'N_to_W': (['V_N', 'I_CENTER', 'BL_I', 'H_W'], 'NS'),
    'N_to_E': (['V_N', 'I_CENTER', 'TR_I', 'H_E'], 'NS'),

    # From feeder roads
    'TR_N_to_W': (['TR_N', 'TR_I', 'I_CENTER', 'BL_I', 'H_W'], 'EW'),
    'TR_N_to_S': (['TR_N', 'TR_I', 'I_CENTER', 'CR_M', 'CR_S'], 'EW'),
    'BL_S_to_E': (['BL_S', 'BL_I', 'I_CENTER', 'TR_I', 'H_E'], 'EW'),
    'BL_S_to_N': (['BL_S', 'BL_I', 'I_CENTER', 'V_N'], 'EW'),
}


# --- Action Definitions ---
# 0: decelerate, 1: maintain speed, 2: accelerate
DISCRETE_ACTIONS = ["decelerate", "maintain", "accelerate"]

# --- PPO Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
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
        self.vehicles = []
        self.step_count = 0
        self.trained_policy: ActorCritic = None
        self.light_controller = TrafficLightController()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.vehicles = []
        for i in range(NUM_VEHICLES):
            self.spawn_vehicle(i)
        return self._get_states()

    def spawn_vehicle(self, vehicle_id):
        path_name = random.choice(list(PATHS.keys()))
        path_wps, path_group = PATHS[path_name]
        
        # Remove existing vehicle with same ID
        self.vehicles = [v for v in self.vehicles if v['id'] != vehicle_id]

        self.vehicles.append({
            'id': vehicle_id,
            'pos': np.copy(WAYPOINTS[path_wps[0]]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'speed': VEHICLE_MIN_SPEED,
            'path_name': path_name,
            'path_wps': path_wps,
            'path_group': path_group,
            'wp_idx': 1
        })

    def _get_states(self):
        if not self.vehicles:
            return np.zeros((0, 7)) # Return empty if no vehicles
            
        states = np.zeros((len(self.vehicles), 7), dtype=np.float32)
        vehicle_positions = np.array([v['pos'] for v in self.vehicles])
        vehicle_tree = cKDTree(vehicle_positions)

        for i, v in enumerate(self.vehicles):
            # Vector to next waypoint
            target_wp = WAYPOINTS[v['path_wps'][v['wp_idx']]]
            vec_to_wp = target_wp - v['pos']
            
            # Distance to nearest vehicle
            dist, _ = vehicle_tree.query(v['pos'], k=2) # k=2 to get nearest other
            nearest_dist = dist[1] if len(dist) > 1 and np.isfinite(dist[1]) else 30.0

            # Find next intersection and light state
            dist_to_light = 100.0
            light_state = 0.0 # 0: no light, 1: green, -1: red
            for name, intersection in INTERSECTIONS.items():
                dist_to_isect = np.linalg.norm(v['pos'] - intersection['pos'])
                if dist_to_isect < 40.0: # Only consider nearby lights
                    if dist_to_isect < dist_to_light:
                        dist_to_light = dist_to_isect
                        # Is our path group allowed to go?
                        # state 0 = NS green, state 1 = EW green
                        is_ns_path = v['path_group'] == 'NS'
                        is_ns_green = self.light_controller.state == 0
                        
                        if (is_ns_path and is_ns_green) or (not is_ns_path and not is_ns_green):
                            light_state = 1.0 # Green
                        else:
                            light_state = -1.0 # Red


            states[i, 0] = v['speed']
            states[i, 1:4] = vec_to_wp / np.linalg.norm(vec_to_wp) if np.linalg.norm(vec_to_wp) > 0 else vec_to_wp
            states[i, 4] = nearest_dist
            states[i, 5] = light_state
            states[i, 6] = dist_to_light / 40.0 # Normalized
        
        return states

    def step(self, actions):
        rewards = np.full(len(self.vehicles), REWARD_STEP)
        self.light_controller.update()
        
        vehicle_positions_before = np.array([v['pos'] for v in self.vehicles])

        for i, v in enumerate(self.vehicles):
            # Check for red light violation
            for name, intersection in INTERSECTIONS.items():
                dist_to_isect = np.linalg.norm(v['pos'] - intersection['pos'])
                if dist_to_isect < intersection['radius']:
                    is_ns_path = v['path_group'] == 'NS'
                    is_ew_path = v['path_group'] == 'EW'
                    is_ns_green = self.light_controller.state == 0
                    is_ew_green = self.light_controller.state == 1

                    if (is_ns_path and not is_ns_green) or (is_ew_path and not is_ew_green):
                        rewards[i] += REWARD_RED_LIGHT

            # 1. Update speed based on action
            action = actions[i]
            if action == 0: # decelerate
                v['speed'] = max(VEHICLE_MIN_SPEED, v['speed'] - ACCELERATION)
            elif action == 2: # accelerate
                v['speed'] = min(VEHICLE_MAX_SPEED, v['speed'] + ACCELERATION)
            
            # 2. Move vehicle towards next waypoint
            target_wp_pos = WAYPOINTS[v['path_wps'][v['wp_idx']]]
            direction = target_wp_pos - v['pos']
            dist_to_wp = np.linalg.norm(direction)
            
            if dist_to_wp > 0:
                v['velocity'] = (direction / dist_to_wp) * v['speed']
                
            v['pos'] += v['velocity']

            # 3. Check if waypoint is reached
            if np.linalg.norm(target_wp_pos - v['pos']) < v['speed']: # If close enough
                if v['wp_idx'] < len(v['path_wps']) - 1:
                    v['wp_idx'] += 1
                else: # End of path
                    rewards[i] += REWARD_PROGRESS * 20 # Bonus for finishing
                    self.spawn_vehicle(v['id'])
                    continue # skip collision check for respawned car
        
        # 4. Collision detection
        if len(self.vehicles) > 1:
            vehicle_positions_after = np.array([v['pos'] for v in self.vehicles])
            vehicle_tree = cKDTree(vehicle_positions_after)
            collisions = vehicle_tree.query_pairs(r=1.5) # Collision radius
            
            collided_indices = set()
            for c1, c2 in collisions:
                rewards[c1] += REWARD_COLLISION
                rewards[c2] += REWARD_COLLISION
                collided_indices.add(c1)
                collided_indices.add(c2)
            
            for idx in collided_indices:
                self.spawn_vehicle(self.vehicles[idx]['id'])
        
        self.step_count += 1
        done = self.step_count >= 1000 # Episode ends after N steps
        
        return self._get_states(), rewards, done

    def get_state_for_viz(self):
        agents = []
        for v in self.vehicles:
            agents.append({
                "id": v['id'],
                "pos": [float(v['pos'][0]), float(v['pos'][1]), float(v['pos'][2])],
                "energy": (v['speed'] / VEHICLE_MAX_SPEED) * 100,
                "velocity": [float(v['velocity'][0]), float(v['velocity'][1]), float(v['velocity'][2])]
            })

        return {"agents": agents, "lights": self.light_controller.state}

# --- PPO Training Loop ---
EPISODES = 1000
GAMMA = 0.99
LR = 3e-4
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
BATCH_SIZE = 256 # Num steps to collect before update

async def train_intersection(websocket: WebSocket, env: MultiVehicleEnv):
    input_size = 7  # speed, vec_to_wp (3), nearest_dist, light_state, dist_to_light
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
            if states.shape[0] == 0: # Handle case where all vehicles crashed
                states = env.reset()
                continue
                
            states_t = torch.tensor(states, dtype=torch.float32)
            
            with torch.no_grad():
                actions_t, log_probs_t, _ = model.get_action(states_t)
            
            actions = actions_t.tolist()
            log_probs = log_probs_t.tolist()

            next_states, rewards, done = env.step(actions)
            ep_reward_sum += np.sum(rewards)

            if step % 16 == 0: # Send updates periodically to frontend
                await websocket.send_json({
                    "type": "train_step", "state": env.get_state_for_viz(), "episode": ep
                })
                await asyncio.sleep(0.01)

            batch_states.append(states)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)
            batch_dones.append([done] * len(env.vehicles))

            states = next_states
            if done:
                states = env.reset()
        
        # --- Compute Advantages and Returns ---
        if states.shape[0] > 0:
            states_t = torch.tensor(states, dtype=torch.float32)
            with torch.no_grad():
                _, last_values = model(states_t)
            last_values = last_values.squeeze().numpy()
        else:
            last_values = np.zeros(NUM_VEHICLES)


        # This logic needs to be robust to changing number of agents
        # For simplicity, we'll align based on NUM_VEHICLES and pad if necessary
        all_advantages = torch.zeros((BATCH_SIZE, NUM_VEHICLES), dtype=torch.float32)
        all_returns = torch.zeros((BATCH_SIZE, NUM_VEHICLES), dtype=torch.float32)
        
        # GAE calculation
        # This is complex with variable agents, so we will use a simplified return calculation
        # for this example, focusing on getting the simulation working.
        
        flat_rewards = [r for batch_r in batch_rewards for r in batch_r]
        if not flat_rewards: flat_rewards = [0]
        
        b_states = torch.tensor(np.concatenate(batch_states), dtype=torch.float32)
        b_actions = torch.tensor(np.concatenate(batch_actions), dtype=torch.int64)
        b_log_probs = torch.tensor(np.concatenate(batch_log_probs), dtype=torch.float32)

        # Simplified returns (reward-to-go)
        returns = []
        discounted_reward = 0
        for rewards in reversed(batch_rewards):
            for r in reversed(rewards):
                discounted_reward = r + GAMMA * discounted_reward
                returns.insert(0, discounted_reward)
        
        b_returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(b_returns) > len(b_states):
            b_returns = b_returns[:len(b_states)]
        
        b_advantages = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)


        # --- Update Policy ---
        for _ in range(UPDATE_EPOCHS):
            if len(b_states) == 0: continue
            
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
            "type": "progress", "episode": ep + 1, "reward": float(ep_reward_sum / BATCH_SIZE), "loss": loss.item() if 'loss' in locals() else 0
        })

    await websocket.send_json({"type": "trained", "model_info": {"episodes": EPISODES, "loss": loss.item() if 'loss' in locals() else 0}})


# --- Inference / Run Loop ---
async def run_intersection(websocket: WebSocket, env: MultiVehicleEnv):
    if not env.trained_policy:
        await websocket.send_json({"type": "error", "message": "No trained policy available."})
        return

    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        states = env.reset()
        
        for step in range(500):
            if states.shape[0] > 0:
                states_t = torch.tensor(states, dtype=torch.float32)
                
                with torch.no_grad():
                    actions_t, _, _ = env.trained_policy.get_action(states_t)
                    actions = actions_t.tolist()
                
                next_states, _, done = env.step(actions)
                states = next_states
            else: # No vehicles
                done = True

            await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
            await asyncio.sleep(0.1)

            if done:
                states = env.reset()
                await asyncio.sleep(0.5) 