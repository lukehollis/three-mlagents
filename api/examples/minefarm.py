import asyncio
import random
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
from services.llm import get_json

# --- Constants ---
GRID_SIZE_X = 64
GRID_SIZE_Y = 16
GRID_SIZE_Z = 64
NUM_AGENTS = 20
RESOURCE_TYPES = {
    "grass": {"value": 0, "color": [0.2, 0.6, 0.2]},
    "dirt": {"value": 0, "color": [0.6, 0.4, 0.2]},
    "wood": {"value": 1, "color": [0.5, 0.3, 0.1]}, # Brown for tree trunks
    "stone": {"value": 2, "color": [0.5, 0.5, 0.5]},
    "iron": {"value": 5, "color": [0.8, 0.8, 0.9]},
    "gold": {"value": 10, "color": [0.9, 0.8, 0.2]},
    "diamond": {"value": 20, "color": [0.7, 0.9, 1.0]},
    "crystal": {"value": 50, "color": [0.5, 0.2, 0.8]},
    "obsidian": {"value": 30, "color": [0.1, 0.1, 0.2]},
}
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {res: 0 for res in RESOURCE_TYPES}
        self.goal = None # e.g., "wood" or "stone"
        self.path = []
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]

    async def decide_action(self, grid: np.ndarray, agents: List['Agent'], messages: List[Dict], step_count: int) -> Tuple[str, Any, Dict]:
        # LLM-based decision making.
        log_entry = None
        recent_messages = messages[-5:] # Only use last 5 messages for brevity
        resource_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(RESOURCE_TYPES.keys())])
        
        # Safe slicing for agent's 3D view (5x5x5)
        x_start, x_end = max(0, self.pos[0]-2), min(GRID_SIZE_X, self.pos[0]+3)
        y_start, y_end = max(0, self.pos[1]-2), min(GRID_SIZE_Y, self.pos[1]+3)
        z_start, z_end = max(0, self.pos[2]-2), min(GRID_SIZE_Z, self.pos[2]+3)
        view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

        prompt = f"""You are a mining agent in a 3D grid world. Your ID is {self.id}.
Your current position is [x, y, z]: {self.pos.tolist()}. Y is the vertical axis.
Your inventory is {self.inventory}.
Your current goal is to collect '{self.goal if self.goal else "anything"}'.
Recent messages from other agents: {json.dumps(recent_messages)}

The world is {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}. Resources are encoded as numbers. 0 is empty (air). The world is mostly solid stone underground; you must mine to find resources.
Resource map: {resource_map_str}.
You can see a 5x5x5 area around you. Your view:
{json.dumps(view)}

Your available actions are "move", "mine", "talk", or "wait".
- move: requires integer [x, y, z] coordinates for the next step. You can only move to an adjacent square (including up/down).
- mine: requires integer [x, y, z] coordinates of an adjacent resource to mine.
- talk: requires a short string message for other agents.
- wait: does nothing.

Based on your state, what is your next action? If you have no goal, set one by talking about a valuable resource. If you see your goal, move towards it or mine it. If you don't see your goal, explore randomly (especially downwards for valuable ores) or ask for help.
"""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "mine", "talk", "wait"]},
                "data": {"oneOf": [
                    {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    {"type": "string"},
                    {"type": "object"}, # Allow empty dict for wait
                ]}
            },
            "required": ["action", "data"]
        }

        try:
            response = await get_json(
                prompt=prompt,
                model="gemma3n:latest",
                response_schema=action_schema,
                schema_name="agent_action_decision_3d",
                should_use_ollama=True
            )
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": prompt, "response": response}
            action = response.get("action", "wait")
            data = response.get("data")

            # Basic validation of LLM output for 3D
            if action == "move":
                target_pos = np.array(data)
                # Ensure agent only moves to adjacent squares in 3D
                if np.sum(np.abs(target_pos - self.pos)) == 1:
                    return ("move", target_pos, log_entry)
            elif action == "mine":
                 target_pos = np.array(data)
                 # Can mine current or adjacent square in 3D
                 if np.sum(np.abs(target_pos - self.pos)) <= 1:
                    return ("mine", target_pos, log_entry)
            elif action == "talk":
                return ("talk", str(data), log_entry)
            
            return ("wait", None, log_entry)

        except Exception as e:
            print(f"Agent {self.id} LLM call failed: {e}")
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": prompt, "error": str(e)}
            # Fallback to simple random walk on error (3D)
            move = random.choice([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)])
            next_pos = self.pos + np.array(move)
            return ("move", next_pos, log_entry)


# --- Environment Class ---
class MineFarmEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.llm_logs: List[Dict] = []
        self._spawn_resources()
        self.step_count = 0
        self.agents = []
        for i in range(NUM_AGENTS):
            # Find a valid spawn point on the surface
            while True:
                start_x = random.randint(0, GRID_SIZE_X - 1)
                start_z = random.randint(0, GRID_SIZE_Z - 1)

                # Find the highest solid block y-coordinate by scanning down from the top
                y_surface = -1
                for y in range(GRID_SIZE_Y - 1, -1, -1):
                    if self.grid[start_x, y, start_z] != 0:
                        y_surface = y
                        break

                # If we found a surface and there's space above it, spawn the agent
                if y_surface != -1 and y_surface + 1 < GRID_SIZE_Y:
                    spawn_y = y_surface + 1
                    self.agents.append(Agent(i, np.array([start_x, spawn_y, start_z])))
                    break
                # If the column is all air or the top block is solid, retry


    def _spawn_resources(self):
        # 1. Generate a height map for the terrain using sine waves for smooth hills
        height_map = np.zeros((GRID_SIZE_X, GRID_SIZE_Z))
        
        # Parameters for terrain generation - randomized for variety
        num_octaves = random.randint(2, 4)
        base_height = GRID_SIZE_Y / 2.5

        octaves = []
        for _ in range(num_octaves):
            octaves.append({
                "freq_x": random.uniform(0.05, 0.2),
                "freq_z": random.uniform(0.05, 0.2),
                "amp": random.uniform(GRID_SIZE_Y * 0.1, GRID_SIZE_Y * 0.25),
            })

        for x in range(GRID_SIZE_X):
            for z in range(GRID_SIZE_Z):
                h = base_height
                for i, o in enumerate(octaves):
                    # Use a mix of sin and cos for more variation
                    if i % 2 == 0:
                        h += o["amp"] * np.sin(o["freq_x"] * x) * np.cos(o["freq_z"] * z)
                    else:
                        h += o["amp"] * np.cos(o["freq_x"] * x) * np.sin(o["freq_z"] * z)
                
                # Add some more random noise for less uniformity
                h += random.uniform(-0.5, 0.5)

                height_map[x, z] = h

        # 2. Generate a surface map for block types using more noise
        surface_map = np.zeros((GRID_SIZE_X, GRID_SIZE_Z))
        surface_octaves = []
        for _ in range(random.randint(2, 3)):
            surface_octaves.append({
                "freq_x": random.uniform(0.15, 0.4), # Higher frequency for smaller patches
                "freq_z": random.uniform(0.15, 0.4),
                "amp": random.uniform(0.8, 1.2),
            })
        for x in range(GRID_SIZE_X):
            for z in range(GRID_SIZE_Z):
                for o in surface_octaves:
                    surface_map[x, z] += o["amp"] * np.sin(o["freq_x"] * x) * np.cos(o["freq_z"] * z)
        
        # Normalize surface map to a 0-1 range to create resource patches
        s_min, s_max = np.min(surface_map), np.max(surface_map)
        if s_max > s_min:
            surface_map = (surface_map - s_min) / (s_max - s_min)
        else:
            surface_map.fill(0.5)

        # 3. Create the world based on height and surface maps
        res_keys = list(RESOURCE_TYPES.keys())
        stone_idx = res_keys.index("stone") + 1
        grass_idx = res_keys.index("grass") + 1
        dirt_idx = res_keys.index("dirt") + 1
        wood_idx = res_keys.index("wood") + 1

        for x in range(GRID_SIZE_X):
            for z in range(GRID_SIZE_Z):
                height = int(np.clip(height_map[x, z], 1, GRID_SIZE_Y - 2))
                
                # Fill everything below the surface with stone
                self.grid[x, :height, z] = stone_idx
                
                # Use noise map to create patches of grass, dirt, and stone
                noise_val = surface_map[x, z]
                if noise_val < 0.65:    # 65% chance of grass
                    surface_block = grass_idx
                elif noise_val < 0.90:  # 25% chance of dirt
                    surface_block = dirt_idx
                else:                   # 10% chance of stone
                    surface_block = stone_idx
                self.grid[x, height, z] = surface_block

                # Place wood (as trees) on grass blocks
                if surface_block == grass_idx and random.random() < 0.01:
                    tree_height = random.randint(3, 6)
                    for i in range(1, tree_height):
                        if height + i < GRID_SIZE_Y:
                            self.grid[x, height + i, z] = wood_idx
        
        # 4. Embed veins of other resources into the stone
        for i, res_type in enumerate(res_keys[4:]): # Skip grass, dirt, wood, stone
            res_idx = res_keys.index(res_type) + 1
            
            # More valuable resources are deeper and in smaller veins
            max_depth_ratio = 0.8 - (i / len(RESOURCE_TYPES)) * 0.7
            min_depth_ratio = max(0.1, max_depth_ratio - 0.3)
            
            num_veins = random.randint(15, 25)
            
            for _ in range(num_veins):
                # Pick a random column to start the vein
                px = random.randint(0, GRID_SIZE_X - 1)
                pz = random.randint(0, GRID_SIZE_Z - 1)
                
                surface_height = int(np.clip(height_map[px, pz], 1, GRID_SIZE_Y - 2))
                
                # Define vein depth based on ratios of the available stone height
                min_y = int(surface_height * min_depth_ratio)
                max_y = int(surface_height * max_depth_ratio)
                if min_y >= max_y: continue

                py = random.randint(min_y, max_y)
                
                vein_size = random.randint(50, 150)

                for _ in range(vein_size):
                    # Use gaussian distribution to make veins more "blobby"
                    x_offset = int(random.gauss(0, 2.0))
                    y_offset = int(random.gauss(0, 1.0))
                    z_offset = int(random.gauss(0, 2.0))
                    
                    x, y, z = px + x_offset, py + y_offset, pz + z_offset
                    
                    # Ensure vein is within bounds and only replaces stone below the surface
                    if 0 <= x < GRID_SIZE_X and 0 <= z < GRID_SIZE_Z:
                        current_surface_height = int(np.clip(height_map[x, z], 1, GRID_SIZE_Y - 2))
                        if 0 <= y < current_surface_height:
                            if self.grid[x, y, z] == stone_idx:
                                self.grid[x, y, z] = res_idx

    def _calculate_reward(self):
        total_value = 0
        for agent in self.agents:
            for res, count in agent.inventory.items():
                total_value += RESOURCE_TYPES[res]["value"] * count
        return total_value

    async def step(self):
        self.step_count += 1
        
        messages_for_llm = []
        # Iterate in reverse to find recent talk messages for the prompt context
        for log in reversed(self.llm_logs):
            if not log.get("error") and log.get("response", {}).get("action") == "talk":
                messages_for_llm.append({
                    "agent_id": log["agent_id"],
                    "message": log["response"]["data"],
                    "step": log["step"]
                })
            if len(messages_for_llm) >= MAX_MESSAGES:
                break
        messages_for_llm.reverse() # Restore chronological order

        results = await asyncio.gather(
            *[agent.decide_action(self.grid, self.agents, messages_for_llm, self.step_count) for agent in self.agents]
        )

        agent_actions = []
        new_llm_logs = []
        for res in results:
            agent_actions.append((res[0], res[1]))
            if res[2]:
                new_llm_logs.append(res[2])
        
        self.llm_logs.extend(new_llm_logs)
        if len(self.llm_logs) > MAX_LLM_LOGS:
            self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]

        randomized_order = list(zip(self.agents, agent_actions))
        random.shuffle(randomized_order)

        for agent, (action, data) in randomized_order:
            if action == "move" and data is not None:
                # Ensure agent stays within bounds
                data[0] = np.clip(data[0], 0, GRID_SIZE_X - 1)
                data[1] = np.clip(data[1], 0, GRID_SIZE_Y - 1)
                data[2] = np.clip(data[2], 0, GRID_SIZE_Z - 1)
                agent.pos = data
            elif action == "mine" and data is not None:
                res_pos = data
                if 0 <= res_pos[0] < GRID_SIZE_X and 0 <= res_pos[1] < GRID_SIZE_Y and 0 <= res_pos[2] < GRID_SIZE_Z:
                    res_idx = self.grid[res_pos[0], res_pos[1], res_pos[2]]
                    if res_idx > 0:
                        res_name = list(RESOURCE_TYPES.keys())[int(res_idx) - 1]
                        agent.inventory[res_name] += 1
                        self.grid[res_pos[0], res_pos[1], res_pos[2]] = 0 # Resource depleted
                        agent.goal = None # Find a new goal
            elif action == "talk" and data is not None:
                # "talk" actions are now implicitly logged via llm_logs, no separate handling needed.
                pass

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color} for a in self.agents],
            "llm_logs": self.llm_logs,
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "resource_types": RESOURCE_TYPES,
        }

# --- Websocket runner ---
async def run_minefarm(websocket: WebSocket):
    env = MineFarmEnv()
    await websocket.send_json({"type": "init", "state": env.get_state_for_viz()})
    running = True

    async def receive_commands():
        nonlocal running
        try:
            while running:
                data = await websocket.receive_json()
                if data.get("cmd") == "run":
                    pass # Command is now implicit after first message
                if data.get("cmd") == "stop":
                    running = False
                    break
        except Exception:
            running = False
            
    cmd_task = asyncio.create_task(receive_commands())

    while running:
        await env.step()
        state = env.get_state_for_viz()
        try:
            await websocket.send_json({"type": "run_step", "state": state})

            # Send progress update for chart
            reward = env._calculate_reward()
            progress_update = {
                "type": "progress",
                "episode": env.step_count,
                "reward": reward,
                "loss": None # Placeholder for loss
            }
            await websocket.send_json(progress_update)

            await asyncio.sleep(0.5) # Simulation speed
        except Exception:
            running = False
            break # Exit while loop

    if not cmd_task.done():
        cmd_task.cancel() 