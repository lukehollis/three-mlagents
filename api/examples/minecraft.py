import asyncio
import random
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from services.llm import get_json, get_embedding

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Global websocket reference for logging to frontend
_current_websocket = None

def log_to_frontend(message: str):
    """Send log message to frontend InfoPanel if websocket is available"""
    if _current_websocket:
        # Get the current event loop and create a task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_current_websocket.send_json({
                "type": "debug", 
                "message": message
            }))

# --- Constants ---
GRID_SIZE_X = 64
GRID_SIZE_Y = 16
GRID_SIZE_Z = 64
NUM_AGENTS = 8
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
LLM_CALL_FREQUENCY = 10
USE_LOCAL_OLLAMA = True 

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
    "talk"
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


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_logits = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, valid_actions_mask: torch.Tensor = None):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        
        if valid_actions_mask is not None:
            # Where the mask is False, we set logits to a very low number to make their probability ~0
            logits[~valid_actions_mask] = -1e9

        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

    def get_action(self, obs: np.ndarray, action: torch.Tensor = None, valid_actions_mask: torch.Tensor = None):
        """
        Get an action from the policy, either by sampling or using a provided action.
        Handles both single observations and batches.
        Applies an action mask to prevent selection of invalid actions.
        """
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs

        if valid_actions_mask is not None and not isinstance(valid_actions_mask, torch.Tensor):
            valid_actions_mask = torch.from_numpy(valid_actions_mask).bool()
        
        # Add a batch dimension if it's a single observation
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            if valid_actions_mask is not None and valid_actions_mask.dim() == 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)

        dist, value = self.forward(obs_t, valid_actions_mask=valid_actions_mask)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)

        # If it was a single observation, return single items
        if obs_t.shape[0] == 1:
            return action.item(), log_prob.item(), value.item()
        
        return action, log_prob, value 


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {res: 0 for res in list(RESOURCE_TYPES.keys()) + list(CRAFTING_RECIPES.keys())}
        # Give agents initial goals to encourage exploration and communication
        valuable_resources = ["iron", "gold", "diamond", "crystal"]
        self.goal = random.choice(valuable_resources)
        self.path = []
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]
        self.llm_intent = None # Stores the action suggested by the LLM
        self.is_thinking = False # Flag to prevent concurrent LLM calls
        self.last_llm_step = -10 # Last step when LLM was called (start with -10 for immediate first call)
        self.memory_vector = np.zeros(384, dtype=np.float32) # all-MiniLM-L6-v2 produces embeddings of size 384
        self.memory_stream = [] # Store last 10 events for context

    def update_memory(self, text: str):
        # Update agent's memory with a new text embedding using a moving average
        new_embedding = get_embedding(text)
        self.memory_vector = (self.memory_vector * 0.9) + (new_embedding.astype(np.float32) * 0.1)
    
    def add_to_memory_stream(self, event: str, step: int = None):
        """Add an event to the agent's memory stream, keeping only the last 10 events"""
        if step is not None:
            event_entry = f"Step {step}: {event}"
        else:
            event_entry = event
        
        self.memory_stream.append(event_entry)
        
        # Keep only the last 10 events
        if len(self.memory_stream) > 10:
            self.memory_stream = self.memory_stream[-10:]

    async def decide_action_llm(self, grid: np.ndarray, agents: List['Agent'], messages: List[Dict], step_count: int, offers: List[Dict]):
        # This function runs the LLM call in the background.
        print(f"DEBUG: Agent {self.id}: Starting LLM call at step {step_count}")
        log_to_frontend(f"🤖 Agent {self.id}: Starting LLM call at step {step_count}")
        self.is_thinking = True

        # The existing LLM decision logic is moved here.
        recent_messages = messages[-5:]
        resource_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(RESOURCE_TYPES.keys())])
        
        x_start, x_end = max(0, self.pos[0]-2), min(GRID_SIZE_X, self.pos[0]+3)
        y_start, y_end = max(0, self.pos[1]-2), min(GRID_SIZE_Y, self.pos[1]+3)
        z_start, z_end = max(0, self.pos[2]-2), min(GRID_SIZE_Z, self.pos[2]+3)
        view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

        template = f"""
        Examples of valid JSON responses:
        
        For move action:
        {{
            "action": "move",
            "data": [32, 8, 32]
        }}
        
        For mine action:
        {{
            "action": "mine", 
            "data": [32, 7, 32]
        }}
        
        For talk action:
        {{
            "action": "talk",
            "data": {{"message": "Hello everyone!", "recipient_id": null}}
        }}
        
        For craft action:
        {{
            "action": "craft",
            "data": "stone_pickaxe"
        }}
        
        For offer action:
        {{
            "action": "offer",
            "data": {{"item_to_give": "stone", "amount_to_give": 2, "item_to_receive": "wood", "amount_to_receive": 1}}
        }}
        
        For accept_offer action:
        {{
            "action": "accept_offer",
            "data": 0
        }}
        """

        system_prompt = f"""
        You are a mining agent in a 3D grid world. Your ID is {self.id}.
        You must respond in valid JSON format only. Here are the valid response formats:
        
        {template}
        
        Choose the appropriate action and provide the data in the exact format shown above.
        """

        prompt = f"""You are a mining agent in a 3D grid world. Your ID is {self.id}.
Your current position is [x, y, z]: {self.pos.tolist()}. Y is the vertical axis.
Your inventory is {self.inventory}.
Your current goal is to collect '{self.goal if self.goal else "anything"}'.
Your current memory of recent events is summarized as: {np.round(self.memory_vector[:5], 2).tolist()}...

Your recent action history (last 10 events):
{chr(10).join(self.memory_stream) if self.memory_stream else "No recent events recorded."}

Recent messages from other agents: {json.dumps(recent_messages)}
Crafting recipes available: {json.dumps(CRAFTING_RECIPES)}
Open trade offers: {json.dumps(offers)}

The world is {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}. Resources are encoded as numbers. 0 is empty (air). The world is mostly solid stone underground; you must mine to find resources.
Resource map: {resource_map_str}.
You can see a 5x5x5 area around you. Your view:
{json.dumps(view)}

Your available actions are "move", "mine", "talk", "craft", "offer", or "accept_offer".
- move: requires integer [x, y, z] coordinates for the next step. You can only move to an adjacent square (including up/down).
- mine: requires integer [x, y, z] coordinates of an adjacent resource to mine.
- talk: requires an object with {{ "message": string, "recipient_id": integer (optional) }}. If no recipient, it's a broadcast.
- craft: requires the string name of the item to craft from the recipe list.
- offer: requires an object with {{ "item_to_give": string, "amount_to_give": int, "item_to_receive": string, "amount_to_receive": int }}.
- accept_offer: requires the integer `offer_id` of the offer to accept.

Based on your state, what is your next action? 

PRIORITIZE COMMUNICATION: Talk frequently to coordinate with other agents, ask for help, offer trades, or share information about resources you've found. Communication is key to success in this world.

If you need resources for crafting, check for trade offers or create your own. If you have surplus resources, offer them for trade. If you have no goal, set one by talking about a valuable resource or a craftable item. If you see your goal, move towards it or mine it. If you have the resources, craft a valuable item. If you don't see your goal, explore randomly (especially downwards for valuable ores) or ask for help.

Consider talking if: you just found something valuable, you need help finding resources, you want to make a trade, you want to share your location, or you want to coordinate with other agents.

CRITICAL: You MUST respond with valid JSON only. No explanations, no text outside JSON.

{template}

Pick ONE action and respond in the exact JSON format shown above.
"""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "mine", "talk", "craft", "offer", "accept_offer"]},
                "data": {"oneOf": [
                    {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    {"type": "string"},
                    {"type": "object"}, 
                ]}
            },
            "required": ["action", "data"]
        }
        
        print(f"DEBUG: Agent {self.id}: About to call get_json with prompt length: {len(prompt)}")
        
        if USE_LOCAL_OLLAMA:
            response = await asyncio.wait_for(
                get_json(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    model="gemma3n:latest",
                    response_schema=action_schema,
                    schema_name="agent_action_decision_3d",
                    should_use_ollama=True
                ),
                timeout=20.0
            )
        else:
            response = await asyncio.wait_for(
                get_json(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    model="anthropic/claude-sonnet-4",
                    response_schema=action_schema,
                    schema_name="agent_action_decision_3d",
                    # should_use_ollama=True
                ),
                timeout=20.0
            )

        
        print(f"DEBUG: Agent {self.id}: LLM response received: {response}")
        
        # Instead of returning, store the result as the agent's intent
        self.llm_intent = (response.get("action", "move"), response.get("data"))
        
        print(f"DEBUG: Agent {self.id}: Intent set to: {self.llm_intent}")
        log_to_frontend(f"🧠 Agent {self.id}: LLM response received - action: {response.get('action')}, data: {response.get('data')}")
        log_to_frontend(f"💭 Agent {self.id}: Intent set to: {self.llm_intent}")
        log_to_frontend(f"Agent {self.id} LLM set intent: {self.llm_intent}")
        
        # Generate conversational text for the MessagePanel
        conversation_prompt = f"""You are Agent {self.id} in a mining world. Based on your current situation and recent action decision, provide a brief conversational response about what you're thinking or doing. Keep it short and natural, like you're talking to other agents.

Current situation:
- Position: {self.pos.tolist()}
- Goal: {self.goal}
- Inventory highlights: {[(k,v) for k,v in self.inventory.items() if v > 0][:3]}
- Recent action decision: {response.get('action')} with data {response.get('data')}
- Recent messages: {[msg.get('message', '') for msg in recent_messages[-2:]]}

Respond as Agent {self.id} in first person, conversationally:"""

        print(f"DEBUG: Agent {self.id}: About to generate conversation text")
        
        # Generate conversational response
        if USE_LOCAL_OLLAMA:
            conversation_response = await asyncio.wait_for(
                get_json(
                    prompt=conversation_prompt,
                    model="gemma3n:latest", 
                    response_schema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    },
                    schema_name="agent_conversation",
                    should_use_ollama=True
                ),
                timeout=3.0 
            )
        else:
            conversation_response = await asyncio.wait_for(
                get_json(
                    prompt=conversation_prompt,
                    model="anthropic/claude-sonnet-4",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    },
                    schema_name="agent_conversation",
                ),
                timeout=10.0
            )
            
        conversation_text = conversation_response.get("message", f"Agent {self.id}: Thinking about my next move...")
        print(f"DEBUG: Agent {self.id}: Generated conversation: {conversation_text}")
        
        # Send conversation text to frontend MessagePanel
        log_to_frontend(f"💬 {conversation_text}")
        
        # Also send directly as a message to ensure it reaches MessagePanel
        if _current_websocket:
            try:
                await _current_websocket.send_json({
                    "type": "agent_message",
                    "agent_id": self.id,
                    "message": conversation_text,
                    "step": step_count
                })
            except Exception as e:
                print(f"DEBUG: Failed to send conversation message to frontend: {e}")
        
        await asyncio.sleep(0.01)  # Give time for message to send
        
        log_entry = {
            "agent_id": self.id, 
            "step": step_count, 
            "prompt": prompt, 
            "response": response,
            "conversation": conversation_text
        }
        
        print(f"DEBUG: Agent {self.id}: LLM call finished, is_thinking = False")
        log_to_frontend(f"✅ Agent {self.id}: LLM call finished, is_thinking = False")
        self.is_thinking = False
        return log_entry

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray) -> Tuple[str, Any]:
        log_to_frontend(f"🎯 Agent {self.id}: get_fast_action called, llm_intent = {self.llm_intent}")
        
        # --- LLM Intent Check First (for communication) ---
        # Always prioritize LLM intent for talk actions, even with trained policy
        if self.llm_intent:
            action, data = self.llm_intent
            log_to_frontend(f"💬 Agent {self.id}: Using LLM intent: {action}, {data}")
            log_to_frontend(f"Agent {self.id} using LLM intent: {action}, {data}")
            # If it's a communication action, always use LLM
            if action in ["talk", "craft", "offer", "accept_offer"]:
                self.llm_intent = None  # Consume the intent
                log_to_frontend(f"🗣️ Agent {self.id}: Returning LLM communication action: {action}")
                return (action, data)
            # For move actions, validate and use if reasonable
            elif action == "move" and data:
                target_pos = np.array(data)
                direction = np.sign(target_pos - self.pos)
                if np.sum(np.abs(direction)) == 1:
                    self.llm_intent = None  # Consume the intent
                    return ("move", self.pos + direction)
            # For mine actions, validate before using
            elif action == "mine" and data:
                self.llm_intent = None  # Consume the intent
                mine_pos_int = [int(p) for p in data]
                # Check if target is within bounds and not empty
                if 0 <= mine_pos_int[0] < GRID_SIZE_X and \
                   0 <= mine_pos_int[1] < GRID_SIZE_Y and \
                   0 <= mine_pos_int[2] < GRID_SIZE_Z and \
                   grid[mine_pos_int[0], mine_pos_int[1], mine_pos_int[2]] != 0:
                    return (action, mine_pos_int)
                else:
                    log_to_frontend(f"🤫 Agent {self.id}: LLM chose to mine invalid block at {mine_pos_int}, moving randomly instead.")
                    # Override with random movement instead of wait
                    move = random.choice(list(ACTION_MAP_MOVE.values()))
                    return ("move", self.pos + move)

        # --- Policy Decision (for basic movement/mining) ---
        # Use the trained actor-critic policy for basic actions if available
        if trained_policy:
            log_to_frontend(f"🧮 Agent {self.id}: Using trained policy")
            state_vector = get_agent_state_vector(self, grid)
            action_index, _, _ = trained_policy.get_action(state_vector) 
            action_name = DISCRETE_ACTIONS[action_index]
            log_to_frontend(f"🎲 Agent {self.id}: Policy chose action: {action_name}")

            if action_name in ACTION_MAP_MOVE:
                log_to_frontend(f"🚶 Agent {self.id}: Policy move action")
                return ("move", self.pos + ACTION_MAP_MOVE[action_name])
            elif action_name in ACTION_MAP_MINE:
                log_to_frontend(f"⛏️ Agent {self.id}: Policy mine action")
                mine_pos = self.pos + ACTION_MAP_MINE[action_name]
                mine_pos_int = [int(p) for p in mine_pos]
                # Action Masking: Check if the target is valid before returning the action
                if 0 <= mine_pos_int[0] < GRID_SIZE_X and \
                   0 <= mine_pos_int[1] < GRID_SIZE_Y and \
                   0 <= mine_pos_int[2] < GRID_SIZE_Z and \
                   grid[mine_pos_int[0], mine_pos_int[1], mine_pos_int[2]] != 0:
                    return ("mine", mine_pos_int)
                else:
                    # If the action is invalid, the agent moves randomly instead.
                    # This implicitly punishes the policy for choosing bad actions.
                    log_to_frontend(f"🤫 Agent {self.id}: Policy chose to mine invalid block at {mine_pos_int}, moving randomly instead.")
                    move = random.choice(list(ACTION_MAP_MOVE.values()))
                    return ("move", self.pos + move)
            elif action_name == "talk":
                # NEVER use basic messages - talk actions should ONLY come from LLM
                # If policy wants to talk but no LLM intent, move randomly instead
                log_to_frontend(f"🤫 Agent {self.id}: Policy wanted to talk but no LLM intent - moving randomly instead")
                move = random.choice(list(ACTION_MAP_MOVE.values()))
                return ("move", self.pos + move)
            else: # fallback case
                log_to_frontend(f"🎲 Agent {self.id}: Policy chose unknown action, moving randomly")
                move = random.choice(list(ACTION_MAP_MOVE.values()))
                return ("move", self.pos + move)
        
        # --- Default behavior: random walk ---
        log_to_frontend(f"🎲 Agent {self.id}: No policy, using random walk")
        move = random.choice(list(ACTION_MAP_MOVE.values()))
        next_pos = self.pos + np.array(move)
        return ("move", next_pos)


# --- Environment Class ---
class MineCraftEnv:
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
                    y_surface = self._find_surface_height(center_x, center_z)
                    if y_surface != -1 and y_surface + 1 < GRID_SIZE_Y:
                        spawn_y = y_surface + 1
                        agent = Agent(i, np.array([center_x, spawn_y, center_z]))
                        agent.add_to_memory_stream(f"Spawned at position {agent.pos.tolist()} with goal: {agent.goal}", 0)
                        self.agents.append(agent)
                    else:
                        # If center is also invalid (e.g. a pillar to the sky), spawn at a default height
                        agent = Agent(i, np.array([center_x, GRID_SIZE_Y // 2, center_z]))
                        agent.add_to_memory_stream(f"Spawned at position {agent.pos.tolist()} with goal: {agent.goal}", 0)
                        self.agents.append(agent)
                    spawned = True
                    continue

                start_x = random.randint(0, GRID_SIZE_X - 1)
                start_z = random.randint(0, GRID_SIZE_Z - 1)

                # Find the highest solid block y-coordinate by scanning down from the top
                y_surface = self._find_surface_height(start_x, start_z)

                # If we found a surface and there's space above it, spawn the agent
                if y_surface != -1 and y_surface + 1 < GRID_SIZE_Y:
                    spawn_y = y_surface + 1
                    agent = Agent(i, np.array([start_x, spawn_y, start_z]))
                    agent.add_to_memory_stream(f"Spawned at position {agent.pos.tolist()} with goal: {agent.goal}", 0)
                    self.agents.append(agent)
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

    def _find_surface_height(self, x: int, z: int) -> int:
        """Find the highest solid block at the given x,z coordinate. Returns -1 if no solid block found."""
        if not (0 <= x < GRID_SIZE_X and 0 <= z < GRID_SIZE_Z):
            return -1
        
        for y in range(GRID_SIZE_Y - 1, -1, -1):
            if self.grid[x, y, z] != 0:
                return y
        return -1

    def _apply_gravity(self, agent: Agent) -> None:
        """Apply gravity to an agent, making them fall to the nearest solid surface."""
        x, y, z = int(agent.pos[0]), int(agent.pos[1]), int(agent.pos[2])
        
        # Find the surface height at this x,z position
        surface_y = self._find_surface_height(x, z)
        
        if surface_y != -1:
            # Agent should be standing on the surface (one block above the highest solid block)
            target_y = surface_y + 1
            # Make sure target_y is within bounds
            target_y = max(0, min(GRID_SIZE_Y - 1, target_y))
            agent.pos[1] = target_y
        else:
            # No solid ground found, keep agent at a safe height
            agent.pos[1] = max(0, min(GRID_SIZE_Y - 1, agent.pos[1]))

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
            reward -= 0.05 # Increased cost of moving to encourage deliberate action

        elif action == "mine":
            # Ensure coords are integers for grid access
            res_pos = [int(p) for p in data]
            if 0 <= res_pos[0] < GRID_SIZE_X and 0 <= res_pos[1] < GRID_SIZE_Y and 0 <= res_pos[2] < GRID_SIZE_Z:
                res_idx = self.grid[res_pos[0], res_pos[1], res_pos[2]]
                if res_idx > 0:
                    res_name = list(RESOURCE_TYPES.keys())[int(res_idx) - 1]
                    res_value = RESOURCE_TYPES[res_name]["value"]
                    reward += res_value if res_value > 0 else 0.1 # Reward for mining something
                else:
                    reward -= 0.5 # Sharper penalty for mining empty space
            else:
                reward -= 1.0 # Even sharper penalty for mining out of bounds

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
                new_x = np.clip(data[0], 0, GRID_SIZE_X - 1)
                new_z = np.clip(data[2], 0, GRID_SIZE_Z - 1)
                old_pos = agent.pos.copy()
                
                # Update position horizontally first
                agent.pos[0] = new_x
                agent.pos[2] = new_z
                
                # Apply gravity to ensure agent lands on solid ground
                self._apply_gravity(agent)
                
                # Record movement event
                agent.add_to_memory_stream(f"Moved from {old_pos.tolist()} to {agent.pos.tolist()}", self.step_count)
            elif action == "mine" and data is not None:
                # Convert mining position to integers to ensure proper grid indexing
                res_pos = [int(data[0]), int(data[1]), int(data[2])]
                if 0 <= res_pos[0] < GRID_SIZE_X and 0 <= res_pos[1] < GRID_SIZE_Y and 0 <= res_pos[2] < GRID_SIZE_Z:
                    res_idx = self.grid[res_pos[0], res_pos[1], res_pos[2]]
                    if res_idx > 0:
                        res_name = list(RESOURCE_TYPES.keys())[int(res_idx) - 1]
                        agent.inventory[res_name] += 1
                        self.grid[res_pos[0], res_pos[1], res_pos[2]] = 0 # Resource depleted
                        log_to_frontend(f"⛏️ Agent {agent.id} mined {res_name} at {res_pos}")  # Debug logging
                        
                        # Record successful mining event
                        agent.add_to_memory_stream(f"Successfully mined {res_name} at {res_pos}. Now have {agent.inventory[res_name]} {res_name}.", self.step_count)
                        
                        # Assign a new goal instead of setting to None
                        valuable_resources = ["iron", "gold", "diamond", "crystal"]
                        craftable_items = list(CRAFTING_RECIPES.keys())
                        all_goals = valuable_resources + craftable_items
                        old_goal = agent.goal
                        agent.goal = random.choice(all_goals)
                        agent.update_memory(f"I mined {res_name}. My new goal is {agent.goal}.")
                        
                        # Record goal change
                        if old_goal != agent.goal:
                            agent.add_to_memory_stream(f"Changed goal from {old_goal} to {agent.goal}", self.step_count)
                        
                        # Apply gravity in case the agent mined the block they were standing on
                        self._apply_gravity(agent)
                    else:
                        log_to_frontend(f"❌ Agent {agent.id} tried to mine empty space at {res_pos}")  # Debug logging
                        agent.add_to_memory_stream(f"Tried to mine empty space at {res_pos}", self.step_count)
                else:
                    log_to_frontend(f"❌ Agent {agent.id} tried to mine out of bounds at {res_pos}")  # Debug logging
                    agent.add_to_memory_stream(f"Tried to mine out of bounds at {res_pos}", self.step_count)
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
                
                # Record talk event
                if data.get("recipient_id") is not None:
                    agent.add_to_memory_stream(f"Sent message to Agent {data.get('recipient_id')}: '{data.get('message')}'", self.step_count)
                else:
                    agent.add_to_memory_stream(f"Broadcast message: '{data.get('message')}'", self.step_count)
                
                # Send talk message immediately to frontend MessagePanel
                if _current_websocket:
                    asyncio.create_task(_current_websocket.send_json({
                        "type": "agent_message",
                        "agent_id": agent.id,
                        "message": data.get("message"),
                        "step": self.step_count,
                        "recipient_id": data.get("recipient_id")
                    }))
                # All agents that hear the message update their memory
                if "recipient_id" in data and data["recipient_id"] is not None:
                    recipient = next((a for a in self.agents if a.id == data["recipient_id"]), None)
                    if recipient:
                        recipient.update_memory(f"Agent {agent.id} said: {data['message']}")
                        recipient.add_to_memory_stream(f"Received private message from Agent {agent.id}: '{data['message']}'", self.step_count)
                else: # Broadcast
                    for a in self.agents:
                        if a.id != agent.id:
                            a.update_memory(f"Agent {agent.id} said: {data['message']}")
                            a.add_to_memory_stream(f"Heard Agent {agent.id} say: '{data['message']}'", self.step_count)

            elif action == "craft" and data is not None:
                recipe_info = CRAFTING_RECIPES.get(data)
                if recipe_info:
                    can_craft = True
                    for resource, amount in recipe_info["recipe"].items():
                        if agent.inventory.get(resource, 0) < amount:
                            can_craft = False
                            break
                    
                    if can_craft:
                        # Record resources used
                        used_resources = ", ".join([f"{amount} {resource}" for resource, amount in recipe_info["recipe"].items()])
                        
                        for resource, amount in recipe_info["recipe"].items():
                            agent.inventory[resource] -= amount
                        agent.inventory[data] = agent.inventory.get(data, 0) + 1
                        
                        # Record successful crafting event
                        agent.add_to_memory_stream(f"Successfully crafted {data} using {used_resources}. Now have {agent.inventory[data]} {data}.", self.step_count)
                        
                        # Assign a new goal instead of setting to None  
                        valuable_resources = ["iron", "gold", "diamond", "crystal"]
                        craftable_items = list(CRAFTING_RECIPES.keys())
                        all_goals = valuable_resources + craftable_items
                        old_goal = agent.goal
                        agent.goal = random.choice(all_goals)
                        agent.update_memory(f"I crafted a {data}. My new goal is {agent.goal}.")
                        
                        # Record goal change
                        if old_goal != agent.goal:
                            agent.add_to_memory_stream(f"Changed goal from {old_goal} to {agent.goal}", self.step_count)
                    else:
                        # Record failed crafting attempt
                        missing_resources = []
                        for resource, amount in recipe_info["recipe"].items():
                            if agent.inventory.get(resource, 0) < amount:
                                missing_resources.append(f"{amount - agent.inventory.get(resource, 0)} {resource}")
                        agent.add_to_memory_stream(f"Failed to craft {data} - missing: {', '.join(missing_resources)}", self.step_count)
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
                    
                    # Record trade offer event
                    agent.add_to_memory_stream(f"Created trade offer #{offer_id}: give {amount_to_give} {item_to_give} for {data.get('amount_to_receive')} {data.get('item_to_receive')}", self.step_count)
                else:
                    # Record failed offer attempt
                    agent.add_to_memory_stream(f"Failed to create trade offer - not enough {item_to_give} (have {agent.inventory.get(item_to_give, 0)}, need {amount_to_give})", self.step_count)
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
                                
                                # Record trade completion events
                                agent.add_to_memory_stream(f"Successfully accepted trade offer #{offer_id}: gave {receives_amount} {receives_item}, received {gives_amount} {gives_item}", self.step_count)
                                offering_agent.add_to_memory_stream(f"My trade offer #{offer_id} was accepted: gave {gives_amount} {gives_item}, received {receives_amount} {receives_item}", self.step_count)
                            else:
                                # Record failed trade attempt
                                agent.add_to_memory_stream(f"Failed to accept trade offer #{offer_id} - not enough {receives_item} (have {agent.inventory.get(receives_item, 0)}, need {receives_amount})", self.step_count)
                        else:
                            # Record failed trade attempt - offering agent not found
                            agent.add_to_memory_stream(f"Failed to accept trade offer #{offer_id} - offering agent not found", self.step_count)
                    else:
                        # Record failed trade attempt - invalid offer
                        agent.add_to_memory_stream(f"Failed to accept trade offer #{offer_id} - offer closed or invalid", self.step_count)
                else:
                    # Record failed trade attempt - offer doesn't exist
                    agent.add_to_memory_stream(f"Failed to accept trade offer #{offer_id} - offer doesn't exist", self.step_count)



    async def step(self):
        self.step_count += 1
        
        # Prune completed/invalid trade offers
        self.trade_offers = [offer for offer in self.trade_offers if offer.get("status") == "open"]

        # --- LLM Thinking (Asynchronous) ---
        llm_tasks = []
        task_to_agent = {}  # Map tasks to their corresponding agents
        # Limit concurrent LLM calls to prevent overload - only allow 5 agents to think at once
        max_concurrent_llm = 5
        current_thinking_count = sum(1 for agent in self.agents if agent.is_thinking)
        
        print(f"DEBUG: Step {self.step_count}: Current thinking count: {current_thinking_count}")
        
        for agent in self.agents:
            if current_thinking_count >= max_concurrent_llm:
                print(f"DEBUG: Max concurrent LLM calls reached ({max_concurrent_llm})")
                break
                
            # Reduce LLM call frequency - every 8 steps instead of 5
            steps_since_last_llm = self.step_count - agent.last_llm_step
            print(f"DEBUG: Agent {agent.id}: steps_since_last_llm={steps_since_last_llm}, is_thinking={agent.is_thinking}")
            
            if not agent.is_thinking and steps_since_last_llm >= LLM_CALL_FREQUENCY:
                print(f"DEBUG: Creating LLM task for Agent {agent.id}")
                log_to_frontend(f"🚀 Creating LLM task for Agent {agent.id}")
                agent.last_llm_step = self.step_count
                
                # Wrap in try-catch to handle any errors
                async def safe_llm_call(agent_ref):
                    print(f"DEBUG: Starting safe_llm_call for Agent {agent_ref.id}")
                    log_entry = await agent_ref.decide_action_llm(
                        self.grid, self.agents, self.messages, self.step_count, self.trade_offers
                    )
                    print(f"DEBUG: safe_llm_call completed for Agent {agent_ref.id}")
                    return log_entry
                
                task = asyncio.create_task(safe_llm_call(agent))
                llm_tasks.append(task)
                task_to_agent[task] = agent  # Track which agent this task belongs to
                current_thinking_count += 1
        
        if len(llm_tasks) > 0:
            print(f"DEBUG: Step {self.step_count}: Created {len(llm_tasks)} LLM tasks")
            log_to_frontend(f"📋 Step {self.step_count}: Created {len(llm_tasks)} LLM tasks")
        else:
            print(f"DEBUG: Step {self.step_count}: No LLM tasks created")

        # --- Fast Action Execution (Synchronous) ---
        agent_actions = []
        for agent in self.agents:
            agent_actions.append(agent.get_fast_action(self.trained_policy, self.grid))

        # Collect logs from completed LLM tasks without blocking
        if llm_tasks:
            print(f"DEBUG: Step {self.step_count}: Waiting for {len(llm_tasks)} LLM tasks to complete")
            log_to_frontend(f"⏳ Step {self.step_count}: Waiting for {len(llm_tasks)} LLM tasks to complete")
            
            done, pending = await asyncio.wait(llm_tasks, timeout=30.0) # Reasonable timeout - calls typically complete in 1.5-2s
            
            print(f"DEBUG: Step {self.step_count}: {len(done)} LLM tasks completed, {len(pending)} still pending")
            log_to_frontend(f"✅ Step {self.step_count}: {len(done)} LLM tasks completed, {len(pending)} still pending")
            
            # Process completed tasks
            for task in done:
                agent = task_to_agent.get(task)
                try:
                    log_entry = task.result()
                    print(f"DEBUG: LLM task result: {log_entry}")
                    if log_entry:
                        print(f"DEBUG: Collected log entry from Agent {log_entry.get('agent_id')}")
                        log_to_frontend(f"📝 Collected log entry from Agent {log_entry.get('agent_id')}")
                        self.llm_logs.append(log_entry)
                        if len(self.llm_logs) > MAX_LLM_LOGS:
                            self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]
                        log_to_frontend(f"Agent {log_entry.get('agent_id')} LLM call completed")
                        
                        # Also send the conversation text to messages if it exists
                        if log_entry.get('conversation'):
                            print(f"DEBUG: Adding conversation to messages: {log_entry.get('conversation')}")
                            message_data = {
                                "sender_id": log_entry.get('agent_id'),
                                "message": log_entry.get('conversation'),
                                "recipient_id": None,  # Broadcast
                                "step": self.step_count,
                                # Remove "type" field so it appears as regular message in MessagePanel
                            }
                            self.messages.append(message_data)
                            if len(self.messages) > MAX_MESSAGES:
                                self.messages = self.messages[-MAX_MESSAGES:]
                except Exception as e:
                    # Handle failed LLM tasks
                    if agent:
                        print(f"DEBUG: LLM task failed for Agent {agent.id}: {e}")
                        log_to_frontend(f"❌ LLM task failed for Agent {agent.id}: {e}")
                        agent.is_thinking = False  # Reset the thinking flag
                    else:
                        print(f"DEBUG: Unknown LLM task failed: {e}")
                        log_to_frontend(f"❌ Unknown LLM task failed: {e}")
            
            # Cancel pending tasks and reset their agents' is_thinking flag
            for task in pending:
                agent = task_to_agent.get(task)
                if agent:
                    print(f"DEBUG: Cancelling pending LLM task for Agent {agent.id}")
                    log_to_frontend(f"❌ Cancelling pending LLM task for Agent {agent.id}")
                    agent.is_thinking = False  # Reset the thinking flag
                else:
                    print(f"DEBUG: Cancelling unknown pending LLM task")
                    log_to_frontend(f"❌ Cancelling unknown pending LLM task")
                task.cancel()
        
        self._execute_actions(agent_actions)
        
        # Apply gravity to all agents to ensure they stay grounded
        for agent in self.agents:
            self._apply_gravity(agent)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color, "memory_stream": a.memory_stream, "goal": a.goal} for a in self.agents],
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
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color, "memory_stream": a.memory_stream, "goal": a.goal} for a in self.agents],
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


def get_valid_actions_mask(agent: Agent, grid: np.ndarray) -> np.ndarray:
    """
    Generates a boolean mask for all discrete actions, where True means the action is valid.
    """
    mask = np.zeros(len(DISCRETE_ACTIONS), dtype=bool)
    
    # Check move actions
    for i, (action_name, move_dir) in enumerate(ACTION_MAP_MOVE.items()):
        target_pos = agent.pos + move_dir
        # For simplicity, we assume moves are generally valid unless they go into a solid block.
        # A more complex check would look at clipping and gravity.
        # But for masking, we just check if the target block is not solid (value > 0).
        if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
            # Allow moving into air
            if grid[int(target_pos[0]), int(target_pos[1]), int(target_pos[2])] == 0:
                 action_idx = DISCRETE_ACTIONS.index(action_name)
                 mask[action_idx] = True

    # Check mine actions
    for i, (action_name, mine_dir) in enumerate(ACTION_MAP_MINE.items()):
        target_pos = agent.pos + mine_dir
        if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
            # Allow mining non-air blocks
            if grid[int(target_pos[0]), int(target_pos[1]), int(target_pos[2])] != 0:
                action_idx = DISCRETE_ACTIONS.index(action_name)
                mask[action_idx] = True

    # Talk is always valid
    mask[DISCRETE_ACTIONS.index("talk")] = True

    # If no actions are valid for some reason (e.g., agent is trapped), enable all movement actions as fallback
    if not np.any(mask):
        for action_name in ACTION_MAP_MOVE.keys():
            action_idx = DISCRETE_ACTIONS.index(action_name)
            mask[action_idx] = True
        
    return mask


# --- PPO Hyperparameters ---
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

async def train_minecraft(websocket: WebSocket, env: MineCraftEnv):
    """
    Train the MineCraft agents using Proximal Policy Optimization (PPO).
    """
    global _current_websocket
    _current_websocket = websocket
    
    try:
        await websocket.send_json({"type": "debug", "message": "Entered train_minecraft()"})
        # env = MineCraftEnv() # DO NOT CREATE A NEW ENV
        
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
        current_loss = None  # Track the most recent loss value

        while ep_counter < 5000: # Train for a fixed number of agent "episodes" or interactions
            if env.step_count % 25 == 0:
                await websocket.send_json({"type": "debug", "message": f"Training loop step {env.step_count}"})
            
            # --- Collect a batch of experience from all agents ---
            # Unlike the glider, we have one environment with N agents.
            # We'll gather one step of experience from each agent.
            
            agent_states = [get_agent_state_vector(agent, env.grid) for agent in env.agents]
            obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)
            
            # Generate valid action masks for each agent
            valid_masks = np.array([get_valid_actions_mask(agent, env.grid) for agent in env.agents])
            masks_t = torch.from_numpy(valid_masks).bool()

            with torch.no_grad():
                dist, value = model(obs_t, valid_actions_mask=masks_t)
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
                elif action_name == "talk":
                    # Generate varied talk messages for better communication
                    talk_options = [
                        f"My goal is {agent.goal}",
                        f"I have {sum(agent.inventory.values())} items total",
                        f"Looking for {agent.goal}, anyone seen it?",
                        f"I'm at position {agent.pos.tolist()}",
                        f"Anyone want to trade?",
                        "Hello everyone!",
                        f"I just mined something valuable!",
                        f"Has anyone found {agent.goal}?"
                    ]
                    message = random.choice(talk_options)
                    agent_actions_for_env.append(("talk", {"message": message}))
                else: # fallback - move randomly
                    move = random.choice(list(ACTION_MAP_MOVE.values()))
                    agent_actions_for_env.append(("move", agent.pos + move))

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
                "value": value.flatten(),
                "mask": masks_t
            })
            
            # --- Update environment state ---
            env._execute_actions(agent_actions_for_env)
            env.step_count += 1
            total_steps += len(env.agents)
            ep_counter += len(env.agents) # Count each agent step as an "episode" for progress

            # --- Send progress updates more frequently for chart ---
            if env.step_count % 16 == 0: # Send progress every 16 steps
                current_reward = float(torch.tensor(rewards, dtype=torch.float32).mean().cpu().item())
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": current_reward, "loss": current_loss})

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
                b_masks = torch.cat([b["mask"] for b in step_buffer])
                
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
                        mb_masks = b_masks[mb_idxs]

                        dist, value = model(mb_obs, valid_actions_mask=mb_masks)
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
                current_loss = loss.item()  # Update the tracked loss value
                
                step_buffer = []
                total_steps = 0
                
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": current_loss})
                await websocket.send_json({"type": "debug", "message": "Completed PPO update"})

        await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item()}})
        await websocket.send_json({"type": "debug", "message": "Training complete"})
    except Exception as e:
        logger.error(f"Error during MineCraft training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"})


# --- Websocket runner ---
async def run_minecraft(websocket: WebSocket, env: MineCraftEnv):
    """
    Runs the MineCraft simulation loop.
    Assumes the environment is already created and the initial state has been sent.
    """
    global _current_websocket
    _current_websocket = websocket
    
    running = True

    async def receive_commands():
        nonlocal running

        # After 'run' is received, we only need to listen for 'stop'.
        while running:
            data = await websocket.receive_json()
            if data.get("cmd") == "stop":
                running = False
                break
            
    cmd_task = asyncio.create_task(receive_commands())

    while running:
        await env.step()
        state = env.get_state_for_viz()

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

    if not cmd_task.done():
        cmd_task.cancel() 


