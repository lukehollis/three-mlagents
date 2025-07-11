import asyncio
import random
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging

from services.llm import get_json, get_embedding
import torch
import torch.nn as nn
import torch.optim as optim
from policies.minefarm_policy import ActorCritic # Note: Reusing policy for now

logger = logging.getLogger(__name__)

# --- Constants ---
GRID_SIZE_X = 64 
GRID_SIZE_Y = 64
GRID_SIZE_Z = 64
NUM_FISH = 32
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]}, # Not actually placed, it's the default
    "food": {"value": 10, "color": [0.8, 0.8, 0.2]}, # Yellow particles
    "coral_a": {"value": 0, "color": [0.9, 0.3, 0.3]}, # Red coral
    "coral_b": {"value": 0, "color": [0.3, 0.9, 0.3]}, # Green coral
    "rock": {"value": 0, "color": [0.5, 0.5, 0.5]},
    "sand": {"value": 0, "color": [0.8, 0.7, 0.5]},
    "shark": {"value": -100, "color": [0.2, 0.2, 0.3]},
}
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30

DISCRETE_ACTIONS = [
    "move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "move_z-",
    "eat", "talk", "wait"
]
ACTION_MAP_MOVE = {
    "move_x+": np.array([1, 0, 0]),
    "move_x-": np.array([-1, 0, 0]),
    "move_y+": np.array([0, 1, 0]),
    "move_y-": np.array([0, -1, 0]),
    "move_z+": np.array([0, 0, 1]),
    "move_z-": np.array([0, 0, -1]),
}

# --- Shark Class ---
class Shark:
    def __init__(self, pos: np.ndarray):
        self.id = "shark"
        self.pos = pos
        self.color = [0.3, 0.4, 0.5]
        self.velocity = (np.random.rand(3) - 0.5) * 2
        self.speed = 1.5

    def move(self, fish_list: List['Fish']):
        if not fish_list:
            # Wander aimlessly
            self.velocity += (np.random.rand(3) - 0.5) * 0.5
        else:
            # Target the center of the swarm
            swarm_center = np.mean([f.pos for f in fish_list], axis=0)
            direction_to_swarm = swarm_center - self.pos
            dist = np.linalg.norm(direction_to_swarm)
            if dist > 1:
                direction_to_swarm /= dist

            # Add some inertia and randomness
            self.velocity = self.velocity * 0.9 + direction_to_swarm * 0.1 + (np.random.rand(3) - 0.5) * 0.2

        # Normalize velocity to maintain speed
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

        self.pos = self.pos + self.velocity

        # Keep shark within bounds
        self.pos[0] = np.clip(self.pos[0], 0, GRID_SIZE_X - 1)
        self.pos[1] = np.clip(self.pos[1], 2, GRID_SIZE_Y - 1)
        self.pos[2] = np.clip(self.pos[2], 0, GRID_SIZE_Z - 1)


# --- Fish Class ---
class Fish:
    def __init__(self, fish_id: int, pos: np.ndarray):
        self.id = fish_id
        self.pos = pos
        self.energy = 100.0
        self.goal = None # e.g., "food" or "school"
        self.color = [random.random(), 0.5, 1.0 - random.random()] # Bluish-purplish fish
        self.llm_intent = None
        self.is_thinking = False
        self.memory_vector = np.zeros(384)

    def update_memory(self, text: str):
        new_embedding = get_embedding(text)
        self.memory_vector = (self.memory_vector * 0.9) + (new_embedding * 0.1)

    async def decide_action_llm(self, grid: np.ndarray, all_fish: List['Fish'], messages: List[Dict], step_count: int, shark: 'Shark'):
        self.is_thinking = True
        try:
            recent_messages = messages[-5:]
            entity_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(ENTITY_TYPES.keys()) if name != 'water'])
            
            x_start, x_end = max(0, self.pos[0]-3), min(GRID_SIZE_X, self.pos[0]+4)
            y_start, y_end = max(0, self.pos[1]-3), min(GRID_SIZE_Y, self.pos[1]+4)
            z_start, z_end = max(0, self.pos[2]-3), min(GRID_SIZE_Z, self.pos[2]+4)
            view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

            other_fish_positions = {f.id: f.pos.tolist() for f in all_fish if f.id != self.id and np.linalg.norm(f.pos - self.pos) < 10}
            shark_pos_str = "not visible"
            if np.linalg.norm(self.pos - shark.pos) < 30:
                shark_pos_str = f"{shark.pos.tolist()}"

            prompt = f"""You are a fish in a swarm in a 3D underwater world. Your ID is {self.id}.
A deadly shark is hunting nearby. Your absolute priority is to avoid it. The shark is at {shark_pos_str}.
Your current position is [x, y, z]: {self.pos.tolist()}.
Your current energy is {self.energy}. You lose energy by moving, and gain it by eating food. If it reaches 0, you are removed.
Your current goal is to '{self.goal if self.goal else "survive"}'.
Your current memory of recent events is summarized as: {np.round(self.memory_vector[:5], 2).tolist()}...
Recent messages from other fish: {json.dumps(recent_messages)}
Nearby fish: {json.dumps(other_fish_positions)}

The world is {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}. Entities are encoded as numbers. 0 is empty water. The bottom of the world is sand.
Entity map: {entity_map_str}.
You can see a 7x7x7 area around you. Your view:
{json.dumps(view)}

Your available actions are "move", "eat", "talk", or "wait".
- move: requires integer [x, y, z] coordinates for the next step. You can only move to an adjacent square.
- eat: requires target coordinates [x, y, z]. If food is there, you eat it.
- talk: requires an object with {{ "message": string, "recipient_id": integer (optional) }}.
- wait: does nothing.

Based on your state, what is your next action? Your primary goals are to find food to keep your energy up and to stay with the swarm for safety. If you see food (entity 2), move towards it and eat it. If you see other fish, try to move towards the center of their group (schooling behavior). If you are alone, explore to find the swarm or food. Above all, if you see the shark, flee in the opposite direction.
"""

            action_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["move", "eat", "talk", "wait"]},
                    "data": {"oneOf": [
                        {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                        {"type": "string"},
                        {"type": "object"}, 
                    ]}
                },
                "required": ["action", "data"]
            }
            
            response = await asyncio.wait_for(
                get_json(
                    prompt=prompt, model="gemma2:latest", response_schema=action_schema,
                    schema_name="fish_action_decision", should_use_ollama=True
                ),
                timeout=10.0
            )
            
            self.llm_intent = (response.get("action", "wait"), response.get("data"))
            
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": "...", "response": response} # Truncate prompt for logging
            return log_entry

        except asyncio.TimeoutError:
            self.llm_intent = ("wait", None)
            return {"agent_id": self.id, "step": step_count, "error": "Timeout"}
        except Exception as e:
            self.llm_intent = ("wait", None)
            return {"agent_id": self.id, "step": step_count, "error": str(e)}
        finally:
            self.is_thinking = False

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray, all_fish: List['Fish'], shark: 'Shark') -> Tuple[str, Any]:
        # --- Policy Decision ---
        # 1. Use the trained actor-critic policy if available
        if trained_policy:
            state_vector = get_fish_state_vector(self, grid, all_fish, shark)
            action_index, _, _ = trained_policy.get_action(state_vector) 
            action_name = DISCRETE_ACTIONS[action_index]

            if action_name in ACTION_MAP_MOVE:
                return ("move", self.pos + ACTION_MAP_MOVE[action_name])
            elif action_name == "eat":
                 return ("eat", self.pos) # Try to eat at current location
            elif action_name == "talk":
                 return ("talk", {"message": f"Energy: {self.energy}"})
            else: # wait
                return ("wait", None)

        # 2. If no trained policy, use the LLM's most recent intent
        if self.llm_intent:
            action, data = self.llm_intent
            self.llm_intent = None

            if action == "move" and data:
                target_pos = np.array(data)
                direction = np.sign(target_pos - self.pos)
                if np.sum(np.abs(direction)) <= 1 and np.sum(np.abs(direction)) > 0:
                     return ("move", self.pos + direction)
            elif action in ["eat", "talk"]:
                return (action, data)
        
        # 3. Default behavior: random walk
        move = random.choice(list(ACTION_MAP_MOVE.values()))
        next_pos = self.pos + np.array(move)
        return ("move", next_pos)

# --- Environment Class ---
class FishSwarmEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.llm_logs: List[Dict] = []
        self.messages: List[Dict] = []
        self._spawn_scene()
        self.step_count = 0
        self.fish = []
        self.trained_policy: "ActorCritic" = None
        self.shark = Shark(np.array([
            random.randint(0, GRID_SIZE_X - 1),
            random.randint(20, GRID_SIZE_Y - 20),
            random.randint(0, GRID_SIZE_Z - 1)
        ]))
        for i in range(NUM_FISH):
            pos = np.array([
                random.randint(0, GRID_SIZE_X - 1),
                random.randint(10, GRID_SIZE_Y - 10), # Spawn away from floor/surface
                random.randint(0, GRID_SIZE_Z - 1)
            ])
            self.fish.append(Fish(i, pos))

    def _spawn_scene(self):
        # 1. Create a sandy bottom
        sand_idx = list(ENTITY_TYPES.keys()).index("sand") + 1
        self.grid[:, 0:2, :] = sand_idx # Two layers of sand
        
        # 2. Add some rocks and corals
        rock_idx = list(ENTITY_TYPES.keys()).index("rock") + 1
        coral_a_idx = list(ENTITY_TYPES.keys()).index("coral_a") + 1
        coral_b_idx = list(ENTITY_TYPES.keys()).index("coral_b") + 1

        for _ in range(30): # Add 30 rock/coral features
            px = random.randint(0, GRID_SIZE_X - 1)
            pz = random.randint(0, GRID_SIZE_Z - 1)
            base_y = 2
            
            feature_type = random.choice(['rock', 'coral'])
            if feature_type == 'rock':
                rock_height = random.randint(1, 4)
                for y_off in range(rock_height):
                    self.grid[px, base_y + y_off, pz] = rock_idx
            else:
                coral_height = random.randint(2, 6)
                coral_type = random.choice([coral_a_idx, coral_b_idx])
                for y_off in range(coral_height):
                    self.grid[px, base_y + y_off, pz] = coral_type

        # 3. Scatter food particles
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
        for _ in range(150): # Add 150 food particles
            self.grid[
                random.randint(0, GRID_SIZE_X - 1),
                random.randint(2, GRID_SIZE_Y - 1), # Food floats
                random.randint(0, GRID_SIZE_Z - 1)
            ] = food_idx

    def _calculate_total_energy(self):
        return sum(f.energy for f in self.fish)

    def _get_reward(self, fish: Fish, action: str, data: Any, all_fish: List['Fish'], shark: 'Shark') -> float:
        """Calculates reward for a single fish's action."""
        reward = 0.0
        
        # Energy cost for existing/moving
        if action == "move":
            reward -= 0.5
            # Add reward for moving to food
            target_pos = data
            if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
                if self.grid[int(target_pos[0]), int(target_pos[1]), int(target_pos[2])] == food_idx:
                    reward += ENTITY_TYPES["food"]["value"]
        else:
            reward -= 0.2

        if action == "eat":
            # This is now mostly handled by movement, but we can leave a small reward for LLM-based decisions
            target_pos = data
            if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                entity_idx = self.grid[int(target_pos[0]), int(target_pos[1]), int(target_pos[2])]
                food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
                if entity_idx == food_idx:
                    reward += ENTITY_TYPES["food"]["value"]
        
        # Schooling reward
        nearby_fish = [f for f in all_fish if f.id != fish.id and np.linalg.norm(f.pos - fish.pos) < 20]
        if len(nearby_fish) > 2: # Need at least a few fish to be a school
            centroid = np.mean([f.pos for f in nearby_fish], axis=0)
            dist_to_centroid = np.linalg.norm(fish.pos - centroid)
            # Reward for being close, capped at a max distance.
            reward += max(0, 1.0 - dist_to_centroid / 20.0) * 0.5
        else:
            # Penalty for being alone
            reward -= 0.2

        # Shark avoidance penalty
        dist_to_shark = np.linalg.norm(fish.pos - shark.pos)
        if dist_to_shark < 15:
            reward -= max(0, 1.0 - dist_to_shark / 15.0) * 10
            
        return reward

    def _execute_actions(self, fish_actions: List[Tuple[str, Any]]):
        randomized_order = list(zip(self.fish, fish_actions))
        random.shuffle(randomized_order)
        food_idx = list(ENTITY_TYPES.keys()).index("food") + 1

        for fish, (action, data) in randomized_order:
            fish.energy -= 0.2 # Base energy cost per step
            if action == "move" and data is not None:
                fish.energy -= 0.3 # Extra cost for moving
                
                target_pos_int = np.round(data).astype(int)
                target_pos_int[0] = np.clip(target_pos_int[0], 0, GRID_SIZE_X - 1)
                target_pos_int[1] = np.clip(target_pos_int[1], 2, GRID_SIZE_Y - 1) # Don't go into sand
                target_pos_int[2] = np.clip(target_pos_int[2], 0, GRID_SIZE_Z - 1)
                
                target_cell_val = self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]]

                # Fish can move into water or onto food
                if target_cell_val == 0: # water
                    fish.pos = data
                elif target_cell_val == food_idx:
                    fish.pos = data
                    fish.energy += ENTITY_TYPES["food"]["value"]
                    self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] = 0 # Food consumed
                    fish.update_memory("I ate food.")
                # else: it's a solid object, do not move

            elif action == "eat" and data is not None:
                # This action is now mostly for the LLM. The policy agent eats by moving.
                # We'll keep the logic but it will be rarely triggered by the policy.
                target_pos_int = np.round(data).astype(int)
                if 0 <= target_pos_int[0] < GRID_SIZE_X and 0 <= target_pos_int[1] < GRID_SIZE_Y and 0 <= target_pos_int[2] < GRID_SIZE_Z:
                    entity_idx = self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]]
                    if entity_idx == food_idx:
                        fish.energy += ENTITY_TYPES["food"]["value"]
                        self.grid[target_pos_int[0], target_pos_int[1], target_pos_int[2]] = 0 # Food consumed
                        fish.update_memory("I ate food by explicitly eating.")

            elif action == "talk" and data is not None:
                # Same as minefarm
                pass

        # Move shark and check for eaten fish
        self.shark.move(self.fish)
        for f in self.fish:
            if np.linalg.norm(f.pos - self.shark.pos) < 4: # Shark's "mouth" size
                f.energy = 0
                f.update_memory("I was eaten by a shark.")

        # Remove fish that ran out of energy
        self.fish = [f for f in self.fish if f.energy > 0]
        # Occasionally add new food
        if self.step_count % 20 == 0:
            food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
            for _ in range(10): # Add 10 new food particles
                self.grid[
                    random.randint(0, GRID_SIZE_X - 1),
                    random.randint(2, GRID_SIZE_Y - 1),
                    random.randint(0, GRID_SIZE_Z - 1)
                ] = food_idx


    async def step(self):
        self.step_count += 1
        
        llm_tasks = []
        for fish in self.fish:
            if not fish.is_thinking:
                task = asyncio.create_task(
                    fish.decide_action_llm(self.grid, self.fish, self.messages, self.step_count, self.shark)
                )
                llm_tasks.append(task)

        fish_actions = []
        for fish in self.fish:
            fish_actions.append(fish.get_fast_action(self.trained_policy, self.grid, self.fish, self.shark))

        if llm_tasks:
            done, _ = await asyncio.wait(llm_tasks, timeout=0.1)
            for task in done:
                log_entry = task.result()
                if log_entry:
                    self.llm_logs.append(log_entry)
                    if len(self.llm_logs) > MAX_LLM_LOGS:
                        self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]
        
        self._execute_actions(fish_actions)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": f.id, "pos": f.pos.tolist(), "energy": f.energy, "color": f.color} for f in self.fish],
            "shark": {"id": self.shark.id, "pos": self.shark.pos.tolist(), "color": self.shark.color},
            "llm_logs": self.llm_logs,
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "resource_types": ENTITY_TYPES,
            "messages": self.messages,
        }

def get_fish_state_vector(fish: Fish, grid: np.ndarray, all_fish: List['Fish'], shark: 'Shark') -> np.ndarray:
    # 1. Local View (5x5x5 = 125 values)
    pos = np.round(fish.pos).astype(int)
    view = np.zeros((5, 5, 5))
    x_start, x_end = max(0, pos[0]-2), min(GRID_SIZE_X, pos[0]+3)
    y_start, y_end = max(0, pos[1]-2), min(GRID_SIZE_Y, pos[1]+3)
    z_start, z_end = max(0, pos[2]-2), min(GRID_SIZE_Z, pos[2]+3)
    
    # Ensure indices are integers
    x_start, x_end = int(x_start), int(x_end)
    y_start, y_end = int(y_start), int(y_end)
    z_start, z_end = int(z_start), int(z_end)

    grid_slice = grid[x_start:x_end, y_start:y_end, z_start:z_end]
    
    pad_x_before = 2 - (pos[0] - x_start)
    pad_y_before = 2 - (pos[1] - y_start)
    pad_z_before = 2 - (pos[2] - z_start)
    
    view[pad_x_before:pad_x_before+grid_slice.shape[0], 
         pad_y_before:pad_y_before+grid_slice.shape[1], 
         pad_z_before:pad_z_before+grid_slice.shape[2]] = grid_slice
    view_flat = view.flatten()

    # 2. Energy level (1 value)
    energy_vec = np.array([fish.energy / 100.0]) # Normalize

    # 3. Memory Vector (384 values)
    memory_vec = fish.memory_vector
    
    # 4. Swarm dynamics (5 values)
    nearby_fish = [f for f in all_fish if f.id != fish.id and np.linalg.norm(f.pos - fish.pos) < 20]
    if len(nearby_fish) > 0:
        centroid = np.mean([f.pos for f in nearby_fish], axis=0)
        dist_to_centroid = np.linalg.norm(fish.pos - centroid)
        normalized_centroid = centroid / np.array([GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z])
        normalized_dist = np.array([dist_to_centroid / 30.0]) # Cap distance at 30
        normalized_count = np.array([len(nearby_fish) / NUM_FISH])
        swarm_info = np.concatenate([normalized_centroid, normalized_dist, normalized_count])
    else:
        swarm_info = np.zeros(5)

    # 5. Shark location (3 values)
    shark_vec = (shark.pos - fish.pos) / np.array([GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z])

    state_vector = np.concatenate([view_flat, energy_vec, memory_vec, swarm_info, shark_vec])
    return state_vector

# --- PPO Hyperparameters ---
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

async def train_fish_swarm(websocket: WebSocket, env: FishSwarmEnv):
    try:
        await websocket.send_json({"type": "debug", "message": "Entered train_fish_swarm()"})
        
        dummy_fish = env.fish[0]
        dummy_obs = get_fish_state_vector(dummy_fish, env.grid, env.fish, env.shark)
        obs_size = dummy_obs.shape[0]
        action_size = len(DISCRETE_ACTIONS)

        model = ActorCritic(obs_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        env.trained_policy = model

        step_buffer: list[dict] = []
        ep_counter = 0
        total_steps = 0

        while ep_counter < 5000:
            if not env.fish:
                await websocket.send_json({"type": "error", "message": "All fish have died. Stopping training."})
                break
            
            num_alive = len(env.fish)
            agent_states = [get_fish_state_vector(fish, env.grid, env.fish, env.shark) for fish in env.fish]
            obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)

            with torch.no_grad():
                dist, value = model(obs_t)
                actions_t = dist.sample()
                logp_t = dist.log_prob(actions_t)
                
            actions_np = actions_t.cpu().numpy()
            
            rewards = []
            dones = []
            
            agent_actions_for_env = []
            for i, fish in enumerate(env.fish):
                action_name = DISCRETE_ACTIONS[actions_np[i]]
                action_data = None
                if action_name in ACTION_MAP_MOVE:
                    action_data = fish.pos + ACTION_MAP_MOVE[action_name]
                    agent_actions_for_env.append(("move", action_data))
                elif action_name == "eat":
                    action_data = fish.pos
                    agent_actions_for_env.append(("eat", action_data))
                else: # talk, wait
                    agent_actions_for_env.append((action_name, None))

                reward = env._get_reward(fish, agent_actions_for_env[-1][0], agent_actions_for_env[-1][1], env.fish, env.shark)
                rewards.append(reward)
                dones.append(fish.energy <= 0)

            pad_size = NUM_FISH - num_alive
            mask_t = torch.ones(num_alive, dtype=torch.bool)
            mask_t = torch.nn.functional.pad(mask_t, (0, pad_size), value=False)
            
            obs_t = torch.nn.functional.pad(obs_t, (0, 0, 0, pad_size), value=0)
            actions_t = torch.nn.functional.pad(actions_t, (0, pad_size), value=0)
            logp_t = torch.nn.functional.pad(logp_t, (0, pad_size), value=0)
            rewards_t = torch.nn.functional.pad(torch.tensor(rewards, dtype=torch.float32), (0, pad_size), value=0)
            dones_t = torch.nn.functional.pad(torch.tensor(dones, dtype=torch.float32), (0, pad_size), value=1) # Pad with 1 (True) for terminal
            value_t = torch.nn.functional.pad(value.flatten(), (0, pad_size), value=0)

            step_buffer.append({
                "obs": obs_t, "actions": actions_t, "logp": logp_t, 
                "reward": rewards_t, 
                "done": dones_t, 
                "value": value_t,
                "mask": mask_t,
            })
            
            env._execute_actions(agent_actions_for_env)
            env.step_count += 1
            total_steps += len(env.fish)
            ep_counter += len(env.fish)

            if env.step_count % 8 == 0:
                state = env.get_state_for_viz()
                state['agents'] = state.pop('agents')
                await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
                await asyncio.sleep(0.01)

            if total_steps >= BATCH_SIZE:
                with torch.no_grad():
                    next_value = torch.zeros(len(env.fish))
                    if env.fish:
                        next_agent_states = [get_fish_state_vector(fish, env.grid, env.fish, env.shark) for fish in env.fish]
                        next_obs_t = torch.tensor(np.array(next_agent_states), dtype=torch.float32)
                        _, next_value = model(next_obs_t)
                        next_value = next_value.squeeze()

                num_steps = len(step_buffer)
                
                values = torch.stack([b["value"] for b in step_buffer])
                rewards = torch.stack([b["reward"] for b in step_buffer])
                dones = torch.stack([b["done"] for b in step_buffer])
                
                advantages = torch.zeros_like(rewards)
                gae = 0.0
                for t in reversed(range(num_steps)):
                    # This logic needs adjustment for variable number of agents
                    # For now, we proceed with a simplification
                    num_agents_t = step_buffer[t]['reward'].shape[0]
                    next_val_t = next_value if t == num_steps - 1 else values[t+1]
                    
                    if next_val_t.shape[0] != num_agents_t: # If agents died
                        # Align shapes by padding/truncating; this is a simplification
                        if next_val_t.shape[0] > num_agents_t:
                            next_val_t = next_val_t[:num_agents_t]
                        else:
                            next_val_t = torch.nn.functional.pad(next_val_t, (0, num_agents_t - next_val_t.shape[0]))
                    
                    delta = rewards[t] + GAMMA * next_val_t * (1 - dones[t]) - values[t]
                    gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
                    advantages[t] = gae
                
                returns = advantages + values
                
                b_obs = torch.cat([b["obs"] for b in step_buffer])
                b_actions = torch.cat([b["actions"] for b in step_buffer])
                b_logp = torch.cat([b["logp"] for b in step_buffer])
                b_mask = torch.stack([b["mask"] for b in step_buffer]).flatten()

                b_adv = advantages.flatten()
                b_returns = returns.flatten()
                
                b_adv = b_adv[b_mask] # Select only valid entries
                b_returns = b_returns[b_mask]
                b_obs = b_obs[b_mask]
                b_actions = b_actions[b_mask]
                b_logp = b_logp[b_mask]

                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                for _ in range(EPOCHS):
                    idxs = torch.randperm(b_obs.shape[0])
                    for start in range(0, b_obs.shape[0], MINI_BATCH):
                        mb_idxs = idxs[start : start + MINI_BATCH]
                        dist, value = model(b_obs[mb_idxs])
                        logp_new = dist.log_prob(b_actions[mb_idxs])
                        entropy_bonus = dist.entropy().mean()
                        ratio = (logp_new - b_logp[mb_idxs]).exp()
                        pg_loss1 = -b_adv[mb_idxs] * ratio
                        pg_loss2 = -b_adv[mb_idxs] * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        v_loss = 0.5 * ((value.flatten() - b_returns[mb_idxs]).pow(2)).mean()
                        loss = pg_loss - ENT_COEF * entropy_bonus + v_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                avg_reward = float(torch.stack([b["reward"].mean() for b in step_buffer]).mean().cpu().item())
                step_buffer = []
                total_steps = 0
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": loss.item()})

        await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item() if 'loss' in locals() else 0}})
    except Exception as e:
        logger.error(f"Error during FishSwarm training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"})


# --- Websocket runner ---
async def run_fish_swarm(websocket: WebSocket, env: FishSwarmEnv):
    """
    Runs the FishSwarm simulation loop.
    Assumes the environment is already created and the initial state has been sent.
    """
    running = True
    while running and env.fish:
        await env.step()
        state = env.get_state_for_viz()
        state['agents'] = state.pop('agents') # Rename for frontend
        try:
            await websocket.send_json({"type": "run_step", "state": state})
            reward = env._calculate_total_energy()
            await websocket.send_json({"type": "progress", "episode": env.step_count, "reward": reward, "loss": None})
            await asyncio.sleep(0.1) # Faster simulation speed
        except Exception:
            running = False
    
    if not env.fish:
        await websocket.send_json({"type": "info", "message": "All fish have died. Simulation over."}) 