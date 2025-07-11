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
NUM_FISH = 30
ENTITY_TYPES = {
    "water": {"value": 0, "color": [0.1, 0.3, 0.8]}, # Not actually placed, it's the default
    "food": {"value": 10, "color": [0.8, 0.8, 0.2]}, # Yellow particles
    "coral_a": {"value": 0, "color": [0.9, 0.3, 0.3]}, # Red coral
    "coral_b": {"value": 0, "color": [0.3, 0.9, 0.3]}, # Green coral
    "rock": {"value": 0, "color": [0.5, 0.5, 0.5]},
    "sand": {"value": 0, "color": [0.8, 0.7, 0.5]},
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

    async def decide_action_llm(self, grid: np.ndarray, all_fish: List['Fish'], messages: List[Dict], step_count: int):
        self.is_thinking = True
        try:
            recent_messages = messages[-5:]
            entity_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(ENTITY_TYPES.keys()) if name != 'water'])
            
            x_start, x_end = max(0, self.pos[0]-3), min(GRID_SIZE_X, self.pos[0]+4)
            y_start, y_end = max(0, self.pos[1]-3), min(GRID_SIZE_Y, self.pos[1]+4)
            z_start, z_end = max(0, self.pos[2]-3), min(GRID_SIZE_Z, self.pos[2]+4)
            view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

            other_fish_positions = {f.id: f.pos.tolist() for f in all_fish if f.id != self.id and np.linalg.norm(f.pos - self.pos) < 10}

            prompt = f"""You are a fish in a swarm in a 3D underwater world. Your ID is {self.id}.
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

Based on your state, what is your next action? Your primary goals are to find food to keep your energy up and to stay with the swarm for safety. If you see food (entity 2), move towards it and eat it. If you see other fish, try to move towards the center of their group (schooling behavior). If you are alone, explore to find the swarm or food.
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

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray) -> Tuple[str, Any]:
        # --- Policy Decision ---
        # 1. Use the trained actor-critic policy if available
        if trained_policy:
            state_vector = get_fish_state_vector(self, grid)
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

    def _get_reward(self, fish: Fish, action: str, data: Any) -> float:
        """Calculates reward for a single fish's action."""
        reward = 0.0
        
        # Energy cost for existing/moving
        if action == "move":
            reward -= 0.5
        else:
            reward -= 0.2

        if action == "eat":
            target_pos = data
            if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                entity_idx = self.grid[target_pos[0], target_pos[1], target_pos[2]]
                food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
                if entity_idx == food_idx:
                    reward += ENTITY_TYPES["food"]["value"]
        
        return reward

    def _execute_actions(self, fish_actions: List[Tuple[str, Any]]):
        randomized_order = list(zip(self.fish, fish_actions))
        random.shuffle(randomized_order)

        for fish, (action, data) in randomized_order:
            fish.energy -= 0.2 # Base energy cost per step
            if action == "move" and data is not None:
                fish.energy -= 0.3 # Extra cost for moving
                data[0] = np.clip(data[0], 0, GRID_SIZE_X - 1)
                data[1] = np.clip(data[1], 2, GRID_SIZE_Y - 1) # Don't go into sand
                data[2] = np.clip(data[2], 0, GRID_SIZE_Z - 1)
                
                # Fish can't move into solid objects
                if self.grid[data[0], data[1], data[2]] == 0:
                    fish.pos = data

            elif action == "eat" and data is not None:
                target_pos = data
                if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                    entity_idx = self.grid[target_pos[0], target_pos[1], target_pos[2]]
                    food_idx = list(ENTITY_TYPES.keys()).index("food") + 1
                    if entity_idx == food_idx:
                        fish.energy += ENTITY_TYPES["food"]["value"]
                        self.grid[target_pos[0], target_pos[1], target_pos[2]] = 0 # Food consumed
                        fish.update_memory("I ate food.")

            elif action == "talk" and data is not None:
                # Same as minefarm
                pass

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
                    fish.decide_action_llm(self.grid, self.fish, self.messages, self.step_count)
                )
                llm_tasks.append(task)

        fish_actions = []
        for fish in self.fish:
            fish_actions.append(fish.get_fast_action(self.trained_policy, self.grid))

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
            "llm_logs": self.llm_logs,
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "resource_types": ENTITY_TYPES,
            "messages": self.messages,
        }

def get_fish_state_vector(fish: Fish, grid: np.ndarray) -> np.ndarray:
    # 1. Local View (5x5x5 = 125 values)
    pos = fish.pos
    view = np.zeros((5, 5, 5))
    x_start, x_end = max(0, pos[0]-2), min(GRID_SIZE_X, pos[0]+3)
    y_start, y_end = max(0, pos[1]-2), min(GRID_SIZE_Y, pos[1]+3)
    z_start, z_end = max(0, pos[2]-2), min(GRID_SIZE_Z, pos[2]+3)
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
    
    state_vector = np.concatenate([view_flat, energy_vec, memory_vec])
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
        dummy_obs = get_fish_state_vector(dummy_fish, env.grid)
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

            agent_states = [get_fish_state_vector(fish, env.grid) for fish in env.fish]
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

                reward = env._get_reward(fish, agent_actions_for_env[-1][0], agent_actions_for_env[-1][1])
                rewards.append(reward)
                dones.append(fish.energy <= 0)

            step_buffer.append({
                "obs": obs_t, "actions": actions_t, "logp": logp_t, 
                "reward": torch.tensor(rewards, dtype=torch.float32), 
                "done": torch.tensor(dones, dtype=torch.float32), 
                "value": value.flatten()
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
                        next_agent_states = [get_fish_state_vector(fish, env.grid) for fish in env.fish]
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
                b_adv = advantages.flatten()
                b_returns = returns.flatten()
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