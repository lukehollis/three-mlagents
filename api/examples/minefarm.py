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
from policies.minefarm_policy import ActorCritic

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

CRAFTING_RECIPES = {
    "stone_pickaxe": {"craft_time": 2, "value": 25, "recipe": {"stone": 3, "wood": 2}},
    "iron_pickaxe": {"craft_time": 4, "value": 100, "recipe": {"iron": 3, "wood": 2}},
    "gold_pickaxe": {"craft_time": 6, "value": 250, "recipe": {"gold": 3, "wood": 2}},
    "diamond_pickaxe": {"craft_time": 10, "value": 1000, "recipe": {"diamond": 3, "obsidian": 2}},
    "crystal_wand": {"craft_time": 15, "value": 3000, "recipe": {"crystal": 5, "gold": 2, "wood": 1}},
}

DISCRETE_ACTIONS = [
    "move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "move_z-",
    "mine_self", "mine_up", "mine_down", "mine_x+", "mine_x-", "mine_z+", "mine_z-",
    "talk", "wait"
]
ACTION_MAP_MOVE = {
    "move_x+": np.array([1, 0, 0]),
    "move_x-": np.array([-1, 0, 0]),
    "move_y+": np.array([0, 1, 0]),
    "move_y-": np.array([0, -1, 0]),
    "move_z+": np.array([0, 0, 1]),
    "move_z-": np.array([0, 0, -1]),
}
ACTION_MAP_MINE = {
    "mine_self": np.array([0, 0, 0]),
    "mine_up":   np.array([0, 1, 0]),
    "mine_down": np.array([0, -1, 0]),
    "mine_x+":   np.array([1, 0, 0]),
    "mine_x-":   np.array([-1, 0, 0]),
    "mine_z+":   np.array([0, 0, 1]),
    "mine_z-":   np.array([0, 0, -1]),
}


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {res: 0 for res in list(RESOURCE_TYPES.keys()) + list(CRAFTING_RECIPES.keys())}
        self.goal = None # e.g., "wood" or "stone"
        self.path = []
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]
        self.llm_intent = None # Stores the action suggested by the LLM
        self.is_thinking = False # Flag to prevent concurrent LLM calls
        self.memory_vector = np.zeros(384) # all-MiniLM-L6-v2 produces embeddings of size 384

    def update_memory(self, text: str):
        # Update agent's memory with a new text embedding using a moving average
        new_embedding = get_embedding(text)
        self.memory_vector = (self.memory_vector * 0.9) + (new_embedding * 0.1)

    async def decide_action_llm(self, grid: np.ndarray, agents: List['Agent'], messages: List[Dict], step_count: int, offers: List[Dict]):
        # This function runs the LLM call in the background.
        self.is_thinking = True
        try:
            # The existing LLM decision logic is moved here.
            recent_messages = messages[-5:]
            resource_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(RESOURCE_TYPES.keys())])
            
            x_start, x_end = max(0, self.pos[0]-2), min(GRID_SIZE_X, self.pos[0]+3)
            y_start, y_end = max(0, self.pos[1]-2), min(GRID_SIZE_Y, self.pos[1]+3)
            z_start, z_end = max(0, self.pos[2]-2), min(GRID_SIZE_Z, self.pos[2]+3)
            view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

            prompt = f"""You are a mining agent in a 3D grid world. Your ID is {self.id}.
Your current position is [x, y, z]: {self.pos.tolist()}. Y is the vertical axis.
Your inventory is {self.inventory}.
Your current goal is to collect '{self.goal if self.goal else "anything"}'.
Your current memory of recent events is summarized as: {np.round(self.memory_vector[:5], 2).tolist()}...
Recent messages from other agents: {json.dumps(recent_messages)}
Crafting recipes available: {json.dumps(CRAFTING_RECIPES)}
Open trade offers: {json.dumps(offers)}

The world is {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}. Resources are encoded as numbers. 0 is empty (air). The world is mostly solid stone underground; you must mine to find resources.
Resource map: {resource_map_str}.
You can see a 5x5x5 area around you. Your view:
{json.dumps(view)}

Your available actions are "move", "mine", "talk", "craft", "offer", "accept_offer", or "wait".
- move: requires integer [x, y, z] coordinates for the next step. You can only move to an adjacent square (including up/down).
- mine: requires integer [x, y, z] coordinates of an adjacent resource to mine.
- talk: requires an object with {{ "message": string, "recipient_id": integer (optional) }}. If no recipient, it's a broadcast.
- craft: requires the string name of the item to craft from the recipe list.
- offer: requires an object with {{ "item_to_give": string, "amount_to_give": int, "item_to_receive": string, "amount_to_receive": int }}.
- accept_offer: requires the integer `offer_id` of the offer to accept.
- wait: does nothing.

Based on your state, what is your next action? If you need resources for crafting, check for trade offers or create your own. If you have surplus resources, offer them for trade. If you have no goal, set one by talking about a valuable resource or a craftable item. If you see your goal, move towards it or mine it. If you have the resources, craft a valuable item. If you don't see your goal, explore randomly (especially downwards for valuable ores) or ask for help.
"""

            action_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["move", "mine", "talk", "craft", "offer", "accept_offer", "wait"]},
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
                    prompt=prompt,
                    model="gemma2:latest",
                    response_schema=action_schema,
                    schema_name="agent_action_decision_3d",
                    should_use_ollama=True
                ),
                timeout=10.0 # Increased timeout as it's non-blocking now
            )
            
            # Instead of returning, store the result as the agent's intent
            self.llm_intent = (response.get("action", "wait"), response.get("data"))
            
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": prompt, "response": response}
            # This log will be collected by the environment later
            return log_entry

        except asyncio.TimeoutError:
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": "LLM Timeout", "error": "Timeout"}
            self.llm_intent = ("wait", None) # On timeout, just wait
            return log_entry
        except Exception as e:
            log_entry = {"agent_id": self.id, "step": step_count, "prompt": "LLM Error", "error": str(e)}
            self.llm_intent = ("wait", None) # On error, just wait
            return log_entry
        finally:
            self.is_thinking = False

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray) -> Tuple[str, Any]:
        # --- Policy Decision ---
        # 1. Use the trained actor-critic policy if available
        if trained_policy:
            state_vector = get_agent_state_vector(self, grid)
            # The get_action method on the policy will return action_index, log_prob, value
            action_index, _, _ = trained_policy.get_action(state_vector) 
            action_name = DISCRETE_ACTIONS[action_index]

            if action_name in ACTION_MAP_MOVE:
                return ("move", self.pos + ACTION_MAP_MOVE[action_name])
            elif action_name in ACTION_MAP_MINE:
                return ("mine", self.pos + ACTION_MAP_MINE[action_name])
            elif action_name == "talk":
                 return ("talk", {"message": f"My goal is {self.goal}"})
            else: # wait
                return ("wait", None)

        # 2. If no trained policy, use the LLM's most recent intent
        if self.llm_intent:
            action, data = self.llm_intent
            self.llm_intent = None  # Consume the intent

            if action == "move" and data:
                target_pos = np.array(data)
                direction = np.sign(target_pos - self.pos)
                if np.sum(np.abs(direction)) == 1:
                     return ("move", self.pos + direction)
            elif action in ["mine", "talk", "craft", "offer", "accept_offer"]:
                return (action, data)
        
        # 3. Default behavior: random walk
        move = random.choice(list(ACTION_MAP_MOVE.values()))
        next_pos = self.pos + np.array(move)
        return ("move", next_pos)


# --- Environment Class ---
class MineFarmEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.llm_logs: List[Dict] = []
        self.trade_offers: List[Dict] = []
        self.messages: List[Dict] = []
        self._spawn_resources()
        self.step_count = 0
        self.agents = []
        self.trained_policy: "ActorCritic" = None
        for i in range(NUM_AGENTS):
            # Find a valid spawn point on the surface
            tries = 0
            spawned = False
            while not spawned:
                tries += 1
                if tries > 1000:
                    logger.warning(f"Could not find a random spawn point for agent {i}. Using fallback.")
                    # Fallback: find the surface height at the center of the map
                    center_x, center_z = GRID_SIZE_X // 2, GRID_SIZE_Z // 2
                    y_surface = -1
                    for y in range(GRID_SIZE_Y - 1, -1, -1):
                        if self.grid[center_x, y, center_z] != 0:
                            y_surface = y
                            break
                    if y_surface != -1 and y_surface + 1 < GRID_SIZE_Y:
                        spawn_y = y_surface + 1
                        self.agents.append(Agent(i, np.array([center_x, spawn_y, center_z])))
                    else:
                        # If center is also invalid (e.g. a pillar to the sky), spawn at a default height
                        self.agents.append(Agent(i, np.array([center_x, GRID_SIZE_Y // 2, center_z])))
                    spawned = True
                    continue

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
                    spawned = True
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
                res_value = 0
                if res in RESOURCE_TYPES:
                    res_value = RESOURCE_TYPES[res]["value"]
                elif res in CRAFTING_RECIPES:
                    res_value = CRAFTING_RECIPES[res]["value"]
                total_value += res_value * count
        return total_value

    def _get_reward(self, agent: Agent, action: str, data: Any) -> float:
        """Calculates reward for a single agent's action."""
        reward = 0.0
        
        # Small cost for existing
        reward -= 0.01

        if action == "move":
            reward -= 0.02 # Cost of moving

        elif action == "mine":
            res_pos = data
            if 0 <= res_pos[0] < GRID_SIZE_X and 0 <= res_pos[1] < GRID_SIZE_Y and 0 <= res_pos[2] < GRID_SIZE_Z:
                res_idx = self.grid[res_pos[0], res_pos[1], res_pos[2]]
                if res_idx > 0:
                    res_name = list(RESOURCE_TYPES.keys())[int(res_idx) - 1]
                    res_value = RESOURCE_TYPES[res_name]["value"]
                    reward += res_value if res_value > 0 else 0.1 # Reward for mining something
                else:
                    reward -= 0.1 # Penalty for mining empty space
            else:
                reward -= 0.2 # Penalty for mining out of bounds

        elif action == "craft":
             recipe_info = CRAFTING_RECIPES.get(data)
             if recipe_info:
                reward += recipe_info.get("value", 0) * 0.5 # Big reward for crafting

        # Add other reward conditions (e.g., for trading) here...

        return reward

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        randomized_order = list(zip(self.agents, agent_actions))
        random.shuffle(randomized_order)

        for agent, (action, data) in randomized_order:
            # We'll need to get reward here before executing the action
            reward = self._get_reward(agent, action, data)
            # In a full PPO impl, we'd store this reward with the state & action.
            # For now, we're just setting up the function.

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
                        agent.update_memory(f"I mined {res_name}.")
            elif action == "talk" and data is not None:
                message_data = {
                    "sender_id": agent.id,
                    "message": data.get("message"),
                    "recipient_id": data.get("recipient_id"), # Will be None for broadcast
                    "step": self.step_count,
                }
                self.messages.append(message_data)
                if len(self.messages) > MAX_MESSAGES:
                    self.messages = self.messages[-MAX_MESSAGES:]
                # All agents that hear the message update their memory
                if "recipient_id" in data and data["recipient_id"] is not None:
                    recipient = next((a for a in self.agents if a.id == data["recipient_id"]), None)
                    if recipient:
                        recipient.update_memory(f"Agent {agent.id} said: {data['message']}")
                else: # Broadcast
                    for a in self.agents:
                        if a.id != agent.id:
                            a.update_memory(f"Agent {agent.id} said: {data['message']}")

            elif action == "craft" and data is not None:
                recipe_info = CRAFTING_RECIPES.get(data)
                if recipe_info:
                    can_craft = True
                    for resource, amount in recipe_info["recipe"].items():
                        if agent.inventory.get(resource, 0) < amount:
                            can_craft = False
                            break
                    
                    if can_craft:
                        for resource, amount in recipe_info["recipe"].items():
                            agent.inventory[resource] -= amount
                        agent.inventory[data] = agent.inventory.get(data, 0) + 1
                        agent.goal = None # Crafted, find a new goal
                        agent.update_memory(f"I crafted a {data}.")
            elif action == "offer" and data is not None:
                item_to_give = data.get("item_to_give")
                amount_to_give = data.get("amount_to_give")
                if agent.inventory.get(item_to_give, 0) >= amount_to_give:
                    offer_id = len(self.trade_offers)
                    self.trade_offers.append({
                        "offer_id": offer_id,
                        "agent_id": agent.id,
                        "gives": {"item": item_to_give, "amount": amount_to_give},
                        "receives": {"item": data.get("item_to_receive"), "amount": data.get("amount_to_receive")},
                        "status": "open",
                    })
                    agent.update_memory(f"I offered {amount_to_give} {item_to_give} for {data.get('amount_to_receive')} {data.get('item_to_receive')}.")
            elif action == "accept_offer" and data is not None:
                offer_id = data
                if 0 <= offer_id < len(self.trade_offers):
                    offer = self.trade_offers[offer_id]
                    if offer["status"] == "open" and offer["agent_id"] != agent.id:
                        offering_agent = next((a for a in self.agents if a.id == offer["agent_id"]), None)
                        if offering_agent:
                            # Check if accepting agent has the required resources
                            receives_item = offer["receives"]["item"]
                            receives_amount = offer["receives"]["amount"]
                            if agent.inventory.get(receives_item, 0) >= receives_amount:
                                # Execute trade
                                # Payer gives items
                                agent.inventory[receives_item] -= receives_amount
                                offering_agent.inventory[receives_item] = offering_agent.inventory.get(receives_item, 0) + receives_amount
                                # Offerer gives items
                                gives_item = offer["gives"]["item"]
                                gives_amount = offer["gives"]["amount"]
                                offering_agent.inventory[gives_item] -= gives_amount
                                agent.inventory[gives_item] = agent.inventory.get(gives_item, 0) + gives_amount
                                
                                offer["status"] = "completed"
                                # Both agents update their memory
                                offer_text = f"{offer['gives']['amount']} {offer['gives']['item']} for {offer['receives']['amount']} {offer['receives']['item']}"
                                agent.update_memory(f"I accepted an offer for {offer_text}.")
                                offering_agent.update_memory(f"My offer for {offer_text} was accepted.")


    async def step(self):
        self.step_count += 1
        
        # Prune completed/invalid trade offers
        self.trade_offers = [offer for offer in self.trade_offers if offer.get("status") == "open"]

        # --- LLM Thinking (Asynchronous) ---
        llm_tasks = []
        for agent in self.agents:
            if not agent.is_thinking:
                task = asyncio.create_task(
                    agent.decide_action_llm(self.grid, self.agents, self.messages, self.step_count, self.trade_offers)
                )
                llm_tasks.append(task)

        # --- Fast Action Execution (Synchronous) ---
        agent_actions = []
        for agent in self.agents:
            agent_actions.append(agent.get_fast_action(self.trained_policy, self.grid))

        # Collect logs from completed LLM tasks without blocking
        if llm_tasks:
            done, _ = await asyncio.wait(llm_tasks, timeout=0.1) # Short timeout to not block the loop
            for task in done:
                log_entry = task.result()
                if log_entry:
                    self.llm_logs.append(log_entry)
                    if len(self.llm_logs) > MAX_LLM_LOGS:
                        self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]
        
        self._execute_actions(agent_actions)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color} for a in self.agents],
            "llm_logs": self.llm_logs,
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "resource_types": RESOURCE_TYPES,
            "crafting_recipes": CRAFTING_RECIPES,
            "trade_offers": self.trade_offers,
            "messages": self.messages,
        }

    def get_dynamic_state_for_viz(self) -> Dict[str, Any]:
        """A lightweight version of get_state_for_viz for frequent updates."""
        # Only send the parts of the state that change frequently
        return {
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color} for a in self.agents],
            "trade_offers": self.trade_offers,
            "messages": self.messages,
            # The grid is sent once on init and we can send incremental updates if needed
            # For now, we assume mining actions update the frontend's grid copy
        }


def get_agent_state_vector(agent: Agent, grid: np.ndarray) -> np.ndarray:
    # 1. Local View (5x5x5 = 125 values)
    pos = agent.pos
    x_start, x_end = max(0, pos[0]-2), min(GRID_SIZE_X, pos[0]+3)
    y_start, y_end = max(0, pos[1]-2), min(GRID_SIZE_Y, pos[1]+3)
    z_start, z_end = max(0, pos[2]-2), min(GRID_SIZE_Z, pos[2]+3)
    
    view = np.zeros((5, 5, 5))
    
    # Get the slice from the grid
    grid_slice = grid[x_start:x_end, y_start:y_end, z_start:z_end]
    
    # Calculate padding dimensions
    pad_x_before = 2 - (pos[0] - x_start)
    pad_y_before = 2 - (pos[1] - y_start)
    pad_z_before = 2 - (pos[2] - z_start)
    
    # Place the slice into the view array
    view[pad_x_before:pad_x_before+grid_slice.shape[0], 
         pad_y_before:pad_y_before+grid_slice.shape[1], 
         pad_z_before:pad_z_before+grid_slice.shape[2]] = grid_slice

    view_flat = view.flatten()

    # 2. Inventory (as a vector)
    inventory_vec = np.array(list(agent.inventory.values()))

    # 3. Memory Vector (already a vector)
    memory_vec = agent.memory_vector
    
    # Concatenate all parts into a single state vector
    # Ensure all parts have a consistent size
    state_vector = np.concatenate([view_flat, inventory_vec, memory_vec])
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

async def train_minefarm(websocket: WebSocket, env: MineFarmEnv):
    """
    Train the MineFarm agents using Proximal Policy Optimization (PPO).
    """
    try:
        await websocket.send_json({"type": "debug", "message": "Entered train_minefarm()"})
        # env = MineFarmEnv() # DO NOT CREATE A NEW ENV
        
        # Determine obs and action sizes from the environment
        # Note: This is a multi-agent environment, but we use a shared policy.
        # The observation size is for a single agent.
        
        # Calculate obs_size dynamically
        dummy_agent = env.agents[0]
        dummy_obs = get_agent_state_vector(dummy_agent, env.grid)
        obs_size = dummy_obs.shape[0]
        action_size = len(DISCRETE_ACTIONS)

        # Initialize model and optimizer
        model = ActorCritic(obs_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        env.trained_policy = model # Agents will use this model for actions

        step_buffer: list[dict] = []
        ep_counter = 0
        total_steps = 0

        while ep_counter < 5000: # Train for a fixed number of agent "episodes" or interactions
            if env.step_count % 25 == 0:
                await websocket.send_json({"type": "debug", "message": f"Training loop step {env.step_count}"})
            
            # --- Collect a batch of experience from all agents ---
            # Unlike the glider, we have one environment with N agents.
            # We'll gather one step of experience from each agent.
            
            agent_states = [get_agent_state_vector(agent, env.grid) for agent in env.agents]
            obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)

            with torch.no_grad():
                dist, value = model(obs_t)
                actions_t = dist.sample()
                logp_t = dist.log_prob(actions_t)
                
            actions_np = actions_t.cpu().numpy()
            
            # --- Execute actions for all agents and collect results ---
            rewards = []
            dones = [] # In this env, 'done' isn't really a concept, so it will always be False.
                      # We can simulate episodes by resetting agents or tracking individual stats.
            
            agent_actions_for_env = []
            for i, agent in enumerate(env.agents):
                action_name = DISCRETE_ACTIONS[actions_np[i]]
                if action_name in ACTION_MAP_MOVE:
                    action_data = agent.pos + ACTION_MAP_MOVE[action_name]
                    agent_actions_for_env.append(("move", action_data))
                elif action_name in ACTION_MAP_MINE:
                    action_data = agent.pos + ACTION_MAP_MINE[action_name]
                    agent_actions_for_env.append(("mine", action_data))
                else: # talk, wait
                    agent_actions_for_env.append((action_name, None))

                # Get reward for the action
                # Note: The _get_reward function needs the *result* of the action.
                # We are getting it before, which is a slight simplification.
                reward = env._get_reward(agent, agent_actions_for_env[-1][0], agent_actions_for_env[-1][1])
                rewards.append(reward)
                dones.append(False) # No terminal states in this persistent world.

            # Store the experience
            step_buffer.append({
                "obs": obs_t, 
                "actions": actions_t, 
                "logp": logp_t, 
                "reward": torch.tensor(rewards, dtype=torch.float32), 
                "done": torch.tensor(dones, dtype=torch.float32), 
                "value": value.flatten()
            })
            
            # --- Update environment state ---
            env._execute_actions(agent_actions_for_env)
            env.step_count += 1
            total_steps += len(env.agents)
            ep_counter += len(env.agents) # Count each agent step as an "episode" for progress

            # --- Visualize state on the frontend periodically ---
            if env.step_count % 8 == 0: # Update frontend every 8 steps
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
                await asyncio.sleep(0.01) # give websocket time to send

            # --- PPO Update phase ---
            if total_steps >= BATCH_SIZE:
                with torch.no_grad():
                    next_agent_states = [get_agent_state_vector(agent, env.grid) for agent in env.agents]
                    next_obs_t = torch.tensor(np.array(next_agent_states), dtype=torch.float32)
                    _, next_value = model(next_obs_t)
                    next_value = next_value.squeeze()

                # Process buffer to calculate advantages
                num_steps = len(step_buffer)
                num_agents = len(env.agents)
                
                values = torch.stack([b["value"] for b in step_buffer])
                rewards = torch.stack([b["reward"] for b in step_buffer])
                dones = torch.stack([b["done"] for b in step_buffer])
                
                # GAE Calculation
                advantages = torch.zeros_like(rewards)
                gae = 0.0
                for t in reversed(range(num_steps)):
                    # If we were using dones, next_value would be masked where dones[t] is true.
                    # Since done is always false, this simplifies.
                    delta = rewards[t] + GAMMA * next_value - values[t]
                    gae = delta + GAMMA * GAE_LAMBDA * gae
                    advantages[t] = gae
                    next_value = values[t] # The value of the state at time t is the "next_value" for t-1
                
                returns = advantages + values
                
                # Flatten the batch for training
                b_obs = torch.cat([b["obs"] for b in step_buffer])
                b_actions = torch.cat([b["actions"] for b in step_buffer])
                b_logp = torch.cat([b["logp"] for b in step_buffer])
                b_adv = advantages.flatten()
                b_returns = returns.flatten()
                
                # Normalize advantages
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                # --- Training Epochs ---
                for _ in range(EPOCHS):
                    idxs = torch.randperm(b_obs.shape[0])
                    for start in range(0, b_obs.shape[0], MINI_BATCH):
                        mb_idxs = idxs[start : start + MINI_BATCH]
                        
                        mb_obs = b_obs[mb_idxs]
                        mb_actions = b_actions[mb_idxs]
                        mb_logp_old = b_logp[mb_idxs]
                        mb_adv = b_adv[mb_idxs]
                        mb_returns = b_returns[mb_idxs]

                        dist, value = model(mb_obs)
                        logp_new = dist.log_prob(mb_actions)
                        entropy_bonus = dist.entropy().mean()
                        
                        ratio = (logp_new - mb_logp_old).exp()
                        
                        # Policy loss
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        v_loss = 0.5 * ((value.flatten() - mb_returns).pow(2)).mean()
                        
                        loss = pg_loss - ENT_COEF * entropy_bonus + v_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # --- Post-update ---
                avg_reward = float(torch.stack([b["reward"] for b in step_buffer]).mean().cpu().item())
                
                step_buffer = []
                total_steps = 0
                
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": loss.item()})
                await websocket.send_json({"type": "debug", "message": "Completed PPO update"})

        await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item()}})
        await websocket.send_json({"type": "debug", "message": "Training complete"})
    except Exception as e:
        logger.error(f"Error during MineFarm training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"})


# --- Websocket runner ---
async def run_minefarm(websocket: WebSocket, env: MineFarmEnv):
    """
    Runs the MineFarm simulation loop.
    Assumes the environment is already created and the initial state has been sent.
    """
    running = True

    async def receive_commands():
        nonlocal running
        try:
            # After 'run' is received, we only need to listen for 'stop'.
            while running:
                data = await websocket.receive_json()
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