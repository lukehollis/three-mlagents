import asyncio
import random
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from starlette.websockets import WebSocketState, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from services.llm import get_json, get_embedding

logger = logging.getLogger(__name__)

# Global websocket reference for logging to frontend
_current_websocket = None

RETRO_SCIFI_COLORS = [
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.6, 0.0],  # Bright Orange
    [0.7, 1.0, 0.0],  # Lime Green
    [0.1, 0.5, 1.0],  # Electric Blue
    [1.0, 1.0, 0.2],  # Bright Yellow
    [1.0, 0.2, 0.8],  # Pink
    [0.8, 0.0, 1.0],  # Purple
    [0.2, 1.0, 0.2],  # Green
]

def log_to_frontend(message: str):
    """Send log message to frontend InfoPanel if websocket is available"""
    if _current_websocket:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_current_websocket.send_json({
                "type": "debug", 
                "message": message
            }))

# --- Constants ---
NUM_PEDESTRIANS = 12
NUM_BUSINESSES = 8
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 15
USE_LOCAL_OLLAMA = True 
MAX_STEPS_PER_EPISODE = 2000

# SB3/PPO Training constants
TOTAL_TIMESTEPS = 200000
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_ENT_COEF = 0.01
PPO_LR = 3e-4

# Resource System
RESOURCES = {
    "money": {"value": 1, "color": [1.0, 0.8, 0.2]},  # Gold
    "wood": {"value": 2, "color": [0.5, 0.3, 0.1]},   # Brown
    "stone": {"value": 3, "color": [0.5, 0.5, 0.5]},  # Gray
    "steel": {"value": 5, "color": [0.7, 0.7, 0.8]},  # Silver
    "glass": {"value": 4, "color": [0.8, 0.9, 1.0]},  # Light blue
    "concrete": {"value": 3, "color": [0.6, 0.6, 0.6]}, # Dark gray
    "tools": {"value": 8, "color": [0.9, 0.5, 0.1]},   # Orange
    "blueprints": {"value": 15, "color": [0.2, 0.4, 1.0]} # Blue
}

# Building Recipes (like crafting in Minecraft)
BUILDING_RECIPES = {
    "house": {
        "height": 2, "base_value": 100, "build_time": 5,
        "recipe": {"wood": 10, "stone": 5, "tools": 2}
    },
    "apartment": {
        "height": 4, "base_value": 300, "build_time": 12,
        "recipe": {"wood": 20, "concrete": 15, "steel": 8, "tools": 4}
    },
    "office": {
        "height": 6, "base_value": 500, "build_time": 20,
        "recipe": {"steel": 15, "glass": 20, "concrete": 25, "tools": 6, "blueprints": 2}
    },
    "skyscraper": {
        "height": 12, "base_value": 2000, "build_time": 50,
        "recipe": {"steel": 50, "glass": 40, "concrete": 60, "tools": 15, "blueprints": 8}
    },
    "factory": {
        "height": 3, "base_value": 400, "build_time": 15,
        "recipe": {"steel": 25, "concrete": 20, "tools": 10, "blueprints": 3}
    }
}

BUSINESS_TYPES = ["restaurant", "shop", "office", "factory", "market", "bank"]

# Discrete Actions for RL
DISCRETE_ACTIONS = [
    "move_north", "move_south", "move_east", "move_west",
    "gather_resources", "work_at_business", "start_building", "contribute_building",
    "communicate"
]

ACTION_MAP_MOVE = {
    "move_north": np.array([0.0001, 0]),
    "move_south": np.array([-0.0001, 0]),
    "move_east": np.array([0, 0.0001]),
    "move_west": np.array([0, -0.0001]),
}

class SimCityPPOPolicy(BaseFeaturesExtractor):
    """
    Custom features extractor for the SimCity environment.
    It processes the structured observation data into a flat vector.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Define the structure of the observation space
        agent_pos_size = observation_space["agent_pos"].shape[0]
        resources_size = observation_space["resources"].shape[0]
        agent_stats_size = observation_space["agent_stats"].shape[0]
        env_state_size = observation_space["env_state"].shape[0]
        memory_size = observation_space["memory"].shape[0]

        total_input_size = agent_pos_size + resources_size + agent_stats_size + env_state_size + memory_size
        
        self.extractor = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all parts of the observation into a single tensor
        combined_obs = torch.cat([
            observations["agent_pos"],
            observations["resources"],
            observations["agent_stats"],
            observations["env_state"],
            observations["memory"],
        ], dim=1)
        
        return self.extractor(combined_obs)

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
            logits[~valid_actions_mask] = -1e9

        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

    def get_action(self, obs: np.ndarray, action: torch.Tensor = None, valid_actions_mask: torch.Tensor = None):
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs

        if valid_actions_mask is not None and not isinstance(valid_actions_mask, torch.Tensor):
            valid_actions_mask = torch.from_numpy(valid_actions_mask).bool()
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            if valid_actions_mask is not None and valid_actions_mask.dim() == 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)

        dist, value = self.forward(obs_t, valid_actions_mask=valid_actions_mask)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)

        if obs_t.shape[0] == 1:
            return action.item(), log_prob.item(), value.item()
        
        return action, log_prob, value

# --- Entity Classes ---
class Building:
    def __init__(self, building_id: int, pos: np.ndarray, building_type: str, initiator_id: int):
        self.id = building_id
        self.pos = pos
        self.type = building_type
        self.recipe = BUILDING_RECIPES[building_type]
        self.height = self.recipe["height"]
        self.base_value = self.recipe["base_value"]
        self.build_time = self.recipe["build_time"]
        
        # Construction progress
        self.progress = 0  # 0 to build_time
        self.status = "planning"  # planning, under_construction, completed
        self.contributors = [initiator_id]  # List of agent IDs who contributed
        self.resources_contributed = {resource: 0 for resource in self.recipe["recipe"]}
        self.resources_needed = self.recipe["recipe"].copy()
        
        # Economic attributes
        self.daily_income = 0.0
        self.employees = []
        
    def can_start_construction(self) -> bool:
        """Check if all required resources have been contributed"""
        for resource, needed in self.resources_needed.items():
            if self.resources_contributed[resource] < needed:
                return False
        return True
    
    def contribute_resource(self, resource: str, amount: int, contributor_id: int) -> int:
        """Contribute resources to the building. Returns amount actually contributed."""
        needed = self.resources_needed[resource] - self.resources_contributed[resource]
        actual_contribution = min(amount, needed)
        
        if actual_contribution > 0:
            self.resources_contributed[resource] += actual_contribution
            if contributor_id not in self.contributors:
                self.contributors.append(contributor_id)
        
        return actual_contribution
    
    def advance_construction(self) -> bool:
        """Advance construction by one step. Returns True if building is completed."""
        if self.status == "planning" and self.can_start_construction():
            self.status = "under_construction"
        
        if self.status == "under_construction":
            self.progress += 1
            if self.progress >= self.build_time:
                self.status = "completed"
                self.daily_income = self.base_value * 0.1  # 10% of value as daily income
                return True
        
        return False

class TrafficLight:
    def __init__(self, light_id: int, pos: np.ndarray, initial_state: str = 'green', cycle_time: int = 200):
        self.id = light_id
        self.pos = pos
        self.state = initial_state 
        self.cycle_time = cycle_time
        self.timer = random.randint(0, cycle_time)

    def step(self):
        self.timer += 1
        if self.timer >= self.cycle_time:
            self.timer = 0
            self.state = 'green' if self.state == 'red' else 'red'

class Business:
    def __init__(self, business_id: int, pos: np.ndarray, business_type: str):
        self.id = business_id
        self.pos = pos
        self.type = business_type
        self.inventory = {resource: random.randint(10, 50) for resource in RESOURCES.keys()}
        self.prices = {resource: RESOURCES[resource]["value"] * random.uniform(0.8, 1.2) for resource in RESOURCES.keys()}
        self.customers_served = 0
        self.revenue = 0.0
        self.resource_generation_rate = {resource: random.uniform(0.1, 0.5) for resource in RESOURCES.keys()}
        
    def generate_resources(self):
        """Generate resources based on business type"""
        for resource, rate in self.resource_generation_rate.items():
            if random.random() < rate:
                self.inventory[resource] += random.randint(1, 3)
        
    def serve_customer(self, customer_needs: Dict[str, int]) -> Tuple[Dict[str, int], float]:
        """Serve a customer and return what was provided"""
        provided = {}
        cost = 0.0
        
        for resource, needed in customer_needs.items():
            if resource in self.inventory and self.inventory[resource] >= needed:
                self.inventory[resource] -= needed
                provided[resource] = needed
                cost += needed * self.prices[resource]
        
        if provided:
            self.customers_served += 1
            self.revenue += cost
            
        return provided, cost
    
    def hire_worker(self, worker_id: int) -> float:
        """Hire a worker and return their daily wage"""
        wage = random.uniform(50, 100)
        return wage

class Pedestrian:
    def __init__(self, ped_id: int, start_pos: np.ndarray, graph: 'networkx.MultiDiGraph'):
        self.id = ped_id
        self.graph = graph
        self.pos = start_pos.copy()
        self.speed = random.uniform(0.5, 1.5)
        self.color = random.choice(RETRO_SCIFI_COLORS)
        
        # Resource inventory (new system)
        self.resources = {resource: random.randint(5, 15) for resource in RESOURCES.keys()}
        self.resources["money"] = random.randint(100, 300)
        
        # Basic needs and satisfaction
        basic_needs = ["money", "wood", "stone"]
        self.needs = {resource: random.randint(1, 3) for resource in basic_needs}
        self.satisfaction = 50.0
        
        # Building and collaboration
        self.building_goal = random.choice(list(BUILDING_RECIPES.keys()))
        self.current_building_project = None  # Building ID they're working on
        self.building_contributions = {}  # Track contributions to buildings
        
        # Movement and pathfinding
        self.target_pos = None
        self.target_business = None
        self.target_building = None
        self.state = random.choice(['wandering', 'gathering', 'working', 'building', 'communicating'])
        self.state_timer = 0
        self.path_progress = 0.0
        
        # RL and LLM integration
        self.llm_intent = None
        self.is_thinking = False
        self.last_llm_step = -20
        self.memory_vector = np.zeros(384, dtype=np.float32)
        self.memory_stream = []
        
        # Find nearest nodes for pathfinding
        self._find_nearest_nodes()
        
    def _find_nearest_nodes(self):
        """Find the nearest graph nodes to current position"""
        min_dist = float('inf')
        self.current_node = None
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_pos = np.array([node_data['y'], node_data['x']])
            dist = np.linalg.norm(self.pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                self.current_node = node_id
    
    def set_destination(self, target_pos: np.ndarray):
        """Set a destination and find path"""
        min_dist = float('inf')
        target_node = None
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_pos = np.array([node_data['y'], node_data['x']])
            dist = np.linalg.norm(target_pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                target_node = node_id
        
        if target_node and self.current_node:
            path = ox.shortest_path(self.graph, self.current_node, target_node, weight='length')
            if path and len(path) > 1:
                self.path = path
                self.path_index = 0
                self.target_pos = target_pos
                self.path_progress = 0.0
                return True
        
        return False
    
    async def decide_action_llm(self, businesses: List[Business], buildings: List[Building], 
                               other_agents: List['Pedestrian'], messages: List[Dict], 
                               step_count: int):
        """LLM-powered decision making for complex coordination tasks"""
        log_to_frontend(f"🤖 Agent {self.id}: Starting LLM call at step {step_count}")
        self.is_thinking = True

        # Gather context about current situation
        nearby_buildings = [b for b in buildings if np.linalg.norm(self.pos - b.pos) < 0.001]
        available_building_types = list(BUILDING_RECIPES.keys())
        current_projects = [b for b in buildings if b.status in ["planning", "under_construction"]]
        
        recent_messages = messages[-5:]
        
        system_prompt = f"""
        You are a city planning agent with ID {self.id}. You can collaborate with other agents to build structures.
        
        Valid JSON responses:
        {{
            "action": "move",
            "data": {{"target": "business|building|explore", "specific_id": 0}}
        }}
        
        {{
            "action": "gather_resources", 
            "data": {{"business_id": 0}}
        }}
        
        {{
            "action": "start_building",
            "data": {{"building_type": "house", "reason": "We need more housing"}}
        }}
        
        {{
            "action": "contribute_building",
            "data": {{"building_id": 0, "resources": {{"wood": 5, "stone": 3}}}}
        }}
        
        {{
            "action": "communicate",
            "data": {{"message": "Let's build a skyscraper together!", "recipient_id": null}}
        }}
        """

        prompt = f"""You are Agent {self.id} in a collaborative city building simulation.

Current Status:
- Position: {self.pos.tolist()}
- Resources: {self.resources}
- Current goal: Build a {self.building_goal}
- Current project: {self.current_building_project if self.current_building_project is not None else 'None'}
- State: {self.state}
- Satisfaction: {self.satisfaction:.1f}

Recent memories: {' | '.join(self.memory_stream[-3:]) if self.memory_stream else 'None'}

Environment:
- Available building types: {available_building_types}
- Current building projects: {len(current_projects)} active
- Nearby businesses: {len([b for b in businesses if np.linalg.norm(self.pos - b.pos) < 0.001])}

Building Recipes: {json.dumps(BUILDING_RECIPES)}

Recent messages from other agents: {json.dumps(recent_messages)}

Current building projects needing help:
{json.dumps([{
    "id": b.id, "type": b.type, "status": b.status, 
    "progress": b.progress, "resources_needed": {{k: v - b.resources_contributed[k] for k, v in b.resources_needed.items() if v > b.resources_contributed[k]}},
    "contributors": len(b.contributors)
} for b in current_projects[:3]])}

IMPORTANT PRIORITIES (follow these in order):
1. **FINISH EXISTING PROJECTS FIRST** - If there are building projects in "planning" status that need resources you have, CONTRIBUTE to them instead of starting new ones
2. **FOCUS ON YOUR CURRENT PROJECT** - If you have a current_building_project, prioritize contributing to it
3. **HELP OTHERS COMPLETE** - Look for buildings that are close to starting construction or completion
4. **COMMUNICATE** frequently to coordinate with other agents
5. **GATHER RESOURCES** only if you need them to contribute to existing projects
6. **START NEW BUILDINGS** only if there are very few active projects (less than {len(self.pedestrians) // 3}) and you have no current project

FORBIDDEN: Do NOT start new buildings if you already have a current_building_project or if there are many unfinished projects!

Choose your action focusing on COMPLETION over starting new projects:

Respond with valid JSON only, no explanations."""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "gather_resources", "start_building", "contribute_building", "communicate"]},
                "data": {"type": "object"}
            },
            "required": ["action", "data"]
        }
        
        if USE_LOCAL_OLLAMA:
            response = await asyncio.wait_for(
                get_json(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    model="gemma3n:latest",
                    response_schema=action_schema,
                    schema_name="simcity_agent_decision",
                    should_use_ollama=True
                ),
                timeout=25.0
            )
        else:
            response = await asyncio.wait_for(
                get_json(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    model="anthropic/claude-sonnet-4",
                    response_schema=action_schema,
                    schema_name="simcity_agent_decision"
                ),
                timeout=15.0
            )

        self.llm_intent = (response.get("action", "move"), response.get("data", {}))
        
        log_to_frontend(f"🧠 Agent {self.id}: LLM response - action: {response.get('action')}, data: {response.get('data')}")
        
        # Generate conversational text
        conversation_prompt = f"""Agent {self.id} just decided to {response.get('action')} with data {response.get('data')}. 
        Current situation: Building goal is {self.building_goal}, have {sum(self.resources.values())} total resources.
        Provide a brief conversational response about what you're thinking or planning. Keep it natural and collaborative."""

        if USE_LOCAL_OLLAMA:
            conversation_response = await asyncio.wait_for(
                get_json(
                    prompt=conversation_prompt,
                    model="gemma3n:latest", 
                    response_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                    schema_name="agent_conversation",
                    should_use_ollama=True
                ),
                timeout=10.0 
            )
        else:
            conversation_response = await asyncio.wait_for(
                get_json(
                    prompt=conversation_prompt,
                    model="anthropic/claude-sonnet-4",
                    response_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                    schema_name="agent_conversation"
                ),
                timeout=10.0
            )
            
        conversation_text = conversation_response.get("message", f"Agent {self.id}: Planning my next move...")
        
        # Send conversation text to frontend
        log_to_frontend(f"💬 {conversation_text}")
        
        if _current_websocket:
            await _current_websocket.send_json({
                "type": "agent_message",
                "agent_id": self.id,
                "message": conversation_text,
                "step": step_count
            })
        
        self.is_thinking = False
        
        return {
            "agent_id": self.id, 
            "step": step_count, 
            "prompt": prompt, 
            "response": response,
            "conversation": conversation_text
        }

    def get_fast_action(self, trained_policy: "ActorCritic", businesses: List[Business], 
                       buildings: List[Building]) -> Tuple[str, Any]:
        """Get action using trained RL policy or LLM intent"""
        log_to_frontend(f"🎯 Agent {self.id}: get_fast_action called, llm_intent = {self.llm_intent}")
        
        # Prioritize LLM intent for complex actions
        if self.llm_intent:
            action, data = self.llm_intent
            log_to_frontend(f"💬 Agent {self.id}: Using LLM intent: {action}, {data}")
            
            if action in ["communicate", "start_building", "contribute_building"]:
                self.llm_intent = None
                return (action, data)
            elif action == "gather_resources":
                self.llm_intent = None
                return (action, data)
            elif action == "move":
                self.llm_intent = None
                return (action, data)

        # Use RL policy for basic movement and resource gathering
        if trained_policy:
            log_to_frontend(f"🧮 Agent {self.id}: Using trained policy")
            state_vector = self._get_state_vector(businesses, buildings)
            action_index, _, _ = trained_policy.get_action(state_vector) 
            action_name = DISCRETE_ACTIONS[action_index]
            log_to_frontend(f"🎲 Agent {self.id}: Policy chose action: {action_name}")

            if action_name in ACTION_MAP_MOVE:
                move_vector = ACTION_MAP_MOVE[action_name]
                return ("move", {"target": "explore", "move_vector": move_vector})
            elif action_name == "gather_resources":
                nearest_business = min(businesses, key=lambda b: np.linalg.norm(self.pos - b.pos))
                return ("gather_resources", {"business_id": nearest_business.id})
            elif action_name == "work_at_business":
                nearest_business = min(businesses, key=lambda b: np.linalg.norm(self.pos - b.pos))
                return ("work_at_business", {"business_id": nearest_business.id})
            elif action_name == "start_building":
                return ("start_building", {"building_type": self.building_goal, "reason": "AI policy decision"})
            elif action_name == "contribute_building":
                active_projects = [b for b in buildings if b.status in ["planning", "under_construction"]]
                if active_projects:
                    target_building = min(active_projects, key=lambda b: np.linalg.norm(self.pos - b.pos))
                    contribution = {}
                    for resource, needed in target_building.resources_needed.items():
                        available = self.resources.get(resource, 0)
                        if available > 0 and target_building.resources_contributed[resource] < needed:
                            contribution[resource] = min(available, needed - target_building.resources_contributed[resource])
                    return ("contribute_building", {"building_id": target_building.id, "resources": contribution})
            elif action_name == "communicate":
                messages = [
                    f"I'm looking to build a {self.building_goal}",
                    f"Anyone need help with construction?",
                    f"I have {sum(self.resources.values())} resources to share",
                    f"Let's work together on a big project!"
                ]
                return ("communicate", {"message": random.choice(messages), "recipient_id": None})
        
        # Default: random movement
        log_to_frontend(f"🎲 Agent {self.id}: No policy, using random movement")
        move_vector = random.choice(list(ACTION_MAP_MOVE.values()))
        return ("move", {"target": "explore", "move_vector": move_vector})

    def _get_state_vector(self, businesses: List[Business], buildings: List[Building]) -> np.ndarray:
        """Create state vector for RL policy"""
        # Position (normalized)
        pos_vector = self.pos / 100.0  # Normalize to roughly 0-1 range
        
        # Resources (normalized by typical max values)
        resource_vector = np.array([self.resources.get(r, 0) / 100.0 for r in RESOURCES.keys()])
        
        # Satisfaction and state
        satisfaction_vector = np.array([self.satisfaction / 100.0])
        
        # Distance to nearest business
        if businesses:
            nearest_business_dist = min([np.linalg.norm(self.pos - b.pos) for b in businesses])
            business_vector = np.array([nearest_business_dist])
        else:
            business_vector = np.array([1.0])
        
        # Number of active building projects
        active_projects = len([b for b in buildings if b.status in ["planning", "under_construction"]])
        building_vector = np.array([active_projects / 10.0])  # Normalize
        
        # Memory vector (truncated to manageable size)
        memory_vector = self.memory_vector[:50]  # Use first 50 components
        
        # Concatenate all vectors
        state_vector = np.concatenate([
            pos_vector, resource_vector, satisfaction_vector, 
            business_vector, building_vector, memory_vector
        ])
        
        return state_vector.astype(np.float32)

    def step(self, businesses: List[Business], buildings: List[Building], 
             traffic_lights: List[TrafficLight]):
        """Update pedestrian state and position"""
        self.state_timer += 1
        
        # State transitions based on current situation
        if self.state == 'wandering' and self.state_timer > 30:
            if any(b.status == "planning" for b in buildings):
                self.state = 'building'
            elif random.random() < 0.4:
                self.state = 'gathering'
            self.state_timer = 0
        elif self.state == 'gathering' and self.state_timer > 50:
            if random.random() < 0.5:
                self.state = 'building' if buildings else 'working'
            else:
                self.state = 'wandering'
            self.state_timer = 0
        elif self.state == 'building' and self.state_timer > 40:
            self.state = 'communicating' if random.random() < 0.6 else 'wandering'
            self.state_timer = 0
        elif self.state == 'communicating' and self.state_timer > 20:
            self.state = 'building' if buildings else 'gathering'
            self.state_timer = 0
        elif self.state == 'working' and self.state_timer > 60:
            self.state = 'wandering'
            self.state_timer = 0

        # Movement
        if hasattr(self, 'path') and hasattr(self, 'path_index'):
            self._move_along_path()
            
        # Update satisfaction based on progress towards goals
        self._update_satisfaction(buildings)
    
    def _move_along_path(self):
        """Move along the current path"""
        if not hasattr(self, 'path') or self.path_index >= len(self.path) - 1:
            return
            
        p1 = self.graph.nodes[self.path[self.path_index]]
        p2 = self.graph.nodes[self.path[self.path_index + 1]]
        
        start_pos = np.array([p1['y'], p1['x']])
        end_pos = np.array([p2['y'], p2['x']])
        
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction_norm = direction / distance
            movement = direction_norm * self.speed * 0.00001
            self.pos += movement
            
            if np.linalg.norm(self.pos - end_pos) < 0.00001:
                self.path_index += 1
                self.current_node = self.path[self.path_index]
                
                if self.path_index >= len(self.path) - 1:
                    self.target_pos = None
    
    def _update_satisfaction(self, buildings: List[Building]):
        """Update satisfaction based on building progress and personal needs"""
        # Base satisfaction from personal resource fulfillment
        resource_satisfaction = min(100, sum(self.resources.values()) / 10)
        
        # Bonus from building contributions
        building_bonus = len(self.building_contributions) * 10
        
        # Community building progress bonus
        completed_buildings = len([b for b in buildings if b.status == "completed"])
        community_bonus = completed_buildings * 5
        
        self.satisfaction = max(0, min(100, resource_satisfaction + building_bonus + community_bonus))
    
    def add_to_memory_stream(self, event: str, step: int = None):
        """Add event to memory stream"""
        if step is not None:
            event_entry = f"Step {step}: {event}"
        else:
            event_entry = event
        
        self.memory_stream.append(event_entry)
        
        if len(self.memory_stream) > 10:
            self.memory_stream = self.memory_stream[-10:]
        
        # Update memory vector
        self.update_memory(event_entry)
    
    def update_memory(self, text: str):
        """Update agent's memory with new text embedding"""
        new_embedding = get_embedding(text)
        self.memory_vector = (self.memory_vector * 0.9) + (new_embedding.astype(np.float32) * 0.1)

# --- Environment Class ---
class SimCityEnv(gym.Env):
    def __init__(self, training_mode: bool = True):
        super().__init__()
        self.training_mode = training_mode
        self.llm_logs: List[Dict] = []
        self.messages: List[Dict] = []
        self.step_count = 0
        self.pedestrians: List[Pedestrian] = []
        self.businesses: List[Business] = []
        self.buildings: List[Building] = []
        self.traffic_lights: List[TrafficLight] = []
        self.running = False
        self.trained_policy: "ActorCritic" = None
        
        # Define action and observation spaces for SB3
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        
        # Define observation space using a dictionary
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
            "resources": spaces.Box(low=0, high=np.inf, shape=(len(RESOURCES),), dtype=np.float32),
            "agent_stats": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32), # satisfaction
            "env_state": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32), # nearest business dist, active projects
            "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        })

        # Load real city data
        self.location_point = (37.7749, -122.4194)  # Downtown San Francisco
        self.graph = ox.graph_from_point(self.location_point, dist=900, network_type='drive')
        
        # Add elevation data if possible
        google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if google_api_key:
            ox.add_node_elevations_google(self.graph, api_key=google_api_key)
        
        self.road_network_for_viz = self._get_road_network_for_viz()
        self._create_businesses()
        self._create_traffic_lights()
        self._create_pedestrians()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.llm_logs.clear()
        self.messages.clear()
        self.buildings.clear()
        
        self._create_businesses()
        self._create_traffic_lights()
        self._create_pedestrians()
        
        # In training mode, we need to return an observation for a single agent
        if self.training_mode and self.pedestrians:
            return self._get_obs_for_agent(self.pedestrians[0]), {}
        
        return self._get_dummy_obs(), {} # Return a dummy observation if no pedestrians

    def _get_dummy_obs(self):
        return {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "resources": np.zeros(len(RESOURCES), dtype=np.float32),
            "agent_stats": np.zeros(1, dtype=np.float32),
            "env_state": np.zeros(2, dtype=np.float32),
            "memory": np.zeros(50, dtype=np.float32),
        }

    def _get_obs_for_agent(self, agent: Pedestrian) -> Dict[str, np.ndarray]:
        """Create observation dictionary for a single agent"""
        # Position
        pos_vector = agent.pos.astype(np.float32)
        
        # Resources
        resource_vector = np.array([agent.resources.get(r, 0) for r in RESOURCES.keys()], dtype=np.float32)
        
        # Satisfaction
        satisfaction_vector = np.array([agent.satisfaction], dtype=np.float32)
        
        # Distance to nearest business
        if self.businesses:
            nearest_business_dist = min([np.linalg.norm(agent.pos - b.pos) for b in self.businesses])
        else:
            nearest_business_dist = 1.0
            
        # Number of active building projects
        active_projects = len([b for b in self.buildings if b.status in ["planning", "under_construction"]])
        env_state_vector = np.array([nearest_business_dist, active_projects], dtype=np.float32)
        
        # Memory vector
        memory_vector = agent.memory_vector[:50].astype(np.float32)
        
        return {
            "agent_pos": pos_vector,
            "resources": resource_vector,
            "agent_stats": satisfaction_vector,
            "env_state": env_state_vector,
            "memory": memory_vector,
        }

    def step(self, action: int):
        """Execute one simulation step for a single agent during training"""
        self.step_count += 1
        agent = self.pedestrians[0]
        action_name = DISCRETE_ACTIONS[action]

        data = {}
        if action_name == "start_building":
            data = {"building_type": agent.building_goal, "reason": "AI policy decision"}
        elif action_name == "contribute_building":
            active_projects = [b for b in self.buildings if b.status in ["planning", "under_construction"]]
            if active_projects:
                target_building = min(active_projects, key=lambda b: np.linalg.norm(agent.pos - b.pos))
                contribution = {res: min(agent.resources.get(res, 0), needed - target_building.resources_contributed[res])
                                for res, needed in target_building.resources_needed.items() if agent.resources.get(res, 0) > 0 and target_building.resources_contributed[res] < needed}
                data = {"building_id": target_building.id, "resources": contribution}
        elif action_name in ACTION_MAP_MOVE:
            data = {"target": "explore", "move_vector": ACTION_MAP_MOVE[action_name]}
        elif action_name == "gather_resources" and self.businesses:
            nearest_business = min(self.businesses, key=lambda b: np.linalg.norm(agent.pos - b.pos))
            data = {"business_id": nearest_business.id}
        elif action_name == "work_at_business" and self.businesses:
            nearest_business = min(self.businesses, key=lambda b: np.linalg.norm(agent.pos - b.pos))
            data = {"business_id": nearest_business.id}
        elif action_name == "communicate":
            data = {"message": f"I'm planning to build a {agent.building_goal}", "recipient_id": None}
            
        reward = self._get_reward(agent, action_name, data)
        
        agent_actions = [(action_name, data)]
        for other_agent in self.pedestrians[1:]:
            random_action, random_data = other_agent.get_fast_action(None, self.businesses, self.buildings)
            agent_actions.append((random_action, random_data))

        self._execute_actions(agent_actions)

        for light in self.traffic_lights:
            light.step()
        for business in self.businesses:
            business.generate_resources()
        for building in self.buildings:
            if building.advance_construction():
                pass
        for p in self.pedestrians:
            p.step(self.businesses, self.buildings, self.traffic_lights)
            
        terminated = self.step_count >= MAX_STEPS_PER_EPISODE
        truncated = False
        obs = self._get_obs_for_agent(agent)
        info = {"is_success": terminated} if terminated or truncated else {}
        if "episode" not in info:
             info["episode"] = {
                "r": reward,
                "l": self.step_count,
            }

        return obs, reward, terminated, truncated, info

    def get_state_for_viz(self) -> Dict[str, Any]:
        """Get current state for visualization"""
        return {
            "pedestrians": [{
                "id": p.id,
                "pos": p.pos.tolist(),
                "color": p.color,
                "state": p.state,
                "satisfaction": p.satisfaction,
                "resources": p.resources,
                "memory_stream": p.memory_stream,
                "building_goal": p.building_goal,
                "current_building_project": p.current_building_project
            } for p in self.pedestrians],
            "businesses": [{
                "id": b.id,
                "pos": b.pos.tolist(),
                "type": b.type,
                "inventory": b.inventory,
                "customers_served": b.customers_served,
                "revenue": b.revenue
            } for b in self.businesses],
            "buildings": [{
                "id": b.id,
                "pos": b.pos.tolist(),
                "type": b.type,
                "height": b.height,
                "status": b.status,
                "progress": b.progress,
                "build_time": b.build_time,
                "resources_needed": b.resources_needed,
                "resources_contributed": b.resources_contributed,
                "contributors": b.contributors,
                "base_value": b.base_value,
                "daily_income": b.daily_income
            } for b in self.buildings],
            "traffic_lights": [{
                "id": l.id,
                "pos": l.pos.tolist(),
                "state": l.state
            } for l in self.traffic_lights],
            "road_network": self.road_network_for_viz,
            "messages": self.messages,
            "llm_logs": self.llm_logs,
            "resources": RESOURCES,
            "building_recipes": BUILDING_RECIPES
        }

    def _get_road_network_for_viz(self):
        """Convert road network to visualization format"""
        lines = []
        for u, v, data in self.graph.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                lines.append([[ys[i], xs[i]] for i in range(len(xs))])
        return lines

    def _create_businesses(self):
        """Create businesses at strategic locations"""
        high_degree_nodes = [node for node, degree in self.graph.degree() if degree >= 3]
        business_nodes = random.sample(high_degree_nodes, min(NUM_BUSINESSES, len(high_degree_nodes)))
        
        for i, node_id in enumerate(business_nodes):
            node_data = self.graph.nodes[node_id]
            business_pos = np.array([node_data['y'], node_data['x']])
            business_type = random.choice(BUSINESS_TYPES)
            
            offset = np.array([random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)])
            business_pos += offset
            
            self.businesses.append(Business(i, business_pos, business_type))

    def _create_traffic_lights(self):
        """Create traffic lights at major intersections"""
        intersections = [node for node, degree in self.graph.degree() if degree > 3]
        selected_intersections = random.sample(intersections, min(8, len(intersections)))
        
        for i, node_id in enumerate(selected_intersections):
            node_data = self.graph.nodes[node_id]
            light_pos = np.array([node_data['y'], node_data['x']])
            self.traffic_lights.append(TrafficLight(i, light_pos))

    def _create_pedestrians(self):
        """Create pedestrians at random locations"""
        all_nodes = list(self.graph.nodes())
        
        for i in range(NUM_PEDESTRIANS):
            start_node = random.choice(all_nodes)
            node_data = self.graph.nodes[start_node]
            start_pos = np.array([node_data['y'], node_data['x']])
            
            offset = np.array([random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)])
            start_pos += offset
            
            pedestrian = Pedestrian(i, start_pos, self.graph)
            self.pedestrians.append(pedestrian)

    def add_message(self, ped_id: int, message: str):
        """Add a message to the environment's message list"""
        if len(self.messages) > MAX_MESSAGES:
            self.messages.pop(0)
        self.messages.append({
            "sender_id": ped_id,
            "recipient_id": None,
            "message": message,
            "step": self.step_count
        })

    def _find_valid_building_spot(self, agent_pos: np.ndarray, max_tries=50) -> np.ndarray | None:
        """Find a valid spot for a new building, not on a road and not too close to others."""
        for _ in range(max_tries):
            # Propose a candidate position near the agent, but with a minimum distance
            angle = random.uniform(0, 2 * np.pi)
            # Corresponds to ~40-80 meters, to ensure we're not right on top of the agent
            distance = random.uniform(0.0004, 0.0008) 
            offset = np.array([distance * np.cos(angle), distance * np.sin(angle)])
            candidate_pos = agent_pos + offset
            
            # Check 1: Not too close to a road
            is_on_road = False
            min_road_dist = 0.002 # ~200 meters, to account for building size
            
            try:
                # Use osmnx to find the distance to the nearest road edge
                _, _, _, dist_to_road = ox.distance.nearest_edges(self.graph, X=candidate_pos[1], Y=candidate_pos[0], return_dist=True)
                if dist_to_road < min_road_dist:
                    is_on_road = True
            except Exception:
                # Fallback if nearest_edges fails, check manually
                candidate_point = Point(candidate_pos[1], candidate_pos[0])
                for u, v, data in self.graph.edges(data=True):
                    if 'geometry' in data and candidate_point.distance(data['geometry']) < min_road_dist:
                        is_on_road = True
                        break
            
            if is_on_road:
                continue

            # Check 2: Not too close to another building or business
            is_too_close = False
            min_building_dist = 0.0006 # ~60 meters, to ensure better spacing
            
            for building in self.buildings:
                dist = np.linalg.norm(candidate_pos - building.pos)
                if dist < min_building_dist:
                    is_too_close = True
                    break
            if is_too_close:
                continue
            
            for business in self.businesses:
                 dist = np.linalg.norm(candidate_pos - business.pos)
                 if dist < min_building_dist:
                    is_too_close = True
                    break
            if is_too_close:
                continue

            # All checks passed, this is a valid spot
            return candidate_pos

        log_to_frontend("Could not find a valid building spot after many tries.")
        return None

    def _calculate_reward(self) -> float:
        """Calculate total environment reward based on collaborative progress"""
        # Community building progress
        building_value = sum([b.base_value for b in self.buildings if b.status == "completed"])
        
        # Resource distribution efficiency
        total_resources = sum([sum(p.resources.values()) for p in self.pedestrians])
        resource_efficiency = min(100, total_resources / (len(self.pedestrians) * 10))
        
        # Average satisfaction
        avg_satisfaction = np.mean([p.satisfaction for p in self.pedestrians])
        
        # Collaboration bonus (number of buildings with multiple contributors)
        collaboration_bonus = len([b for b in self.buildings if len(b.contributors) > 1]) * 50
        
        return building_value + resource_efficiency + avg_satisfaction + collaboration_bonus

    def _get_reward(self, agent: Pedestrian, action: str, data: Any) -> float:
        """Calculate reward for individual agent action"""
        reward = 0.0
        
        # Small cost for existing
        reward -= 0.01

        if action == "move":
            reward -= 0.02  # Small movement cost

        elif action == "gather_resources":
            reward += 5.0  # Reward resource gathering

        elif action == "work_at_business":
            reward += 8.0  # Reward productive work

        elif action == "start_building":
            building_type = data.get("building_type", "house")
            base_value = BUILDING_RECIPES.get(building_type, {}).get("base_value", 100)
            
            # Count unfinished buildings
            unfinished_buildings = len([b for b in self.buildings if b.status in ["planning", "under_construction"]])
            
            # Heavy penalty for starting new buildings when there are many unfinished ones
            if unfinished_buildings > len(self.pedestrians) // 3:  # More than 1/3 of agents worth of buildings
                reward -= base_value * 0.5  # Large penalty
                reward -= unfinished_buildings * 10  # Additional penalty per unfinished building
            else:
                reward += base_value * 0.1  # Reward for initiating valuable projects only when not too many active

        elif action == "contribute_building":
            resources_contributed = data.get("resources", {})
            if not resources_contributed:  # No actual contribution
                reward -= 5.0  # Penalty for trying to contribute nothing
                return reward
                
            contribution_value = sum([RESOURCES[r]["value"] * amount for r, amount in resources_contributed.items()])
            base_contribution_reward = contribution_value * 0.5
            
            # Find the building being contributed to
            building_id = data.get("building_id")
            building = next((b for b in self.buildings if b.id == building_id), None)
            
            if building:
                # Extra rewards for strategic contributions
                if building.status == "planning":
                    # Check if this contribution could help the building start construction
                    remaining_resources = {}
                    for resource, needed in building.resources_needed.items():
                        remaining = needed - building.resources_contributed[resource]
                        if remaining > 0:
                            remaining_resources[resource] = remaining
                    
                    # Check how much of the remaining resources this contribution covers
                    total_remaining_value = sum([RESOURCES[r]["value"] * amount for r, amount in remaining_resources.items()])
                    covered_value = sum([RESOURCES[r]["value"] * min(amount, remaining_resources.get(r, 0)) 
                                       for r, amount in resources_contributed.items()])
                    
                    if total_remaining_value > 0:
                        completion_percentage = covered_value / total_remaining_value
                        
                        # Massive bonus if this contribution allows building to start construction
                        if all(building.resources_contributed[r] + resources_contributed.get(r, 0) >= needed 
                               for r, needed in building.resources_needed.items()):
                            reward += 50.0  # Huge bonus for enabling construction start
                        
                        # Additional bonus based on how much progress this makes
                        reward += completion_percentage * 30.0
                
                elif building.status == "under_construction":
                    # Bonus for contributing to buildings already under construction
                    progress_ratio = building.progress / building.build_time
                    reward += base_contribution_reward * (1.0 + progress_ratio)  # More reward for buildings closer to completion
                
                # Bonus if this is the agent's current project
                if agent.current_building_project == building_id:
                    reward += 10.0  # Loyalty bonus for sticking to current project
                
                reward += base_contribution_reward
            else:
                reward += base_contribution_reward

        elif action == "communicate":
            reward += 3.0  # Reward communication for coordination

        return reward

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        """Execute actions for all agents"""
        randomized_order = list(zip(self.pedestrians, agent_actions))
        random.shuffle(randomized_order)

        for agent, (action, data) in randomized_order:
            if action == "move" and data:
                self._execute_move_action(agent, data)
            elif action == "gather_resources" and data:
                self._execute_gather_action(agent, data)
            elif action == "work_at_business" and data:
                self._execute_work_action(agent, data)
            elif action == "start_building" and data:
                self._execute_start_building_action(agent, data)
            elif action == "contribute_building" and data:
                self._execute_contribute_building_action(agent, data)
            elif action == "communicate" and data:
                self._execute_communicate_action(agent, data)

    def _execute_move_action(self, agent: Pedestrian, data: Dict):
        """Execute movement action"""
        if "move_vector" in data:
            # Direct movement from RL policy
            move_vector = np.array(data["move_vector"])
            agent.pos += move_vector
            agent.add_to_memory_stream(f"Moved by vector {move_vector.tolist()}", self.step_count)
        elif data.get("target") == "business" and "specific_id" in data:
            # Move towards specific business
            business_id = data["specific_id"]
            business = next((b for b in self.businesses if b.id == business_id), None)
            if business:
                agent.set_destination(business.pos)
                agent.add_to_memory_stream(f"Moving towards business {business_id}", self.step_count)
        elif data.get("target") == "building" and "specific_id" in data:
            # Move towards specific building
            building_id = data["specific_id"]
            building = next((b for b in self.buildings if b.id == building_id), None)
            if building:
                agent.set_destination(building.pos)
                agent.add_to_memory_stream(f"Moving towards building {building_id}", self.step_count)
        else:
            # Random exploration
            all_nodes = list(self.graph.nodes())
            if all_nodes:
                random_node = random.choice(all_nodes)
                node_data = self.graph.nodes[random_node]
                random_pos = np.array([node_data['y'], node_data['x']])
                agent.set_destination(random_pos)
                agent.add_to_memory_stream("Exploring randomly", self.step_count)

    def _execute_gather_action(self, agent: Pedestrian, data: Dict):
        """Execute resource gathering action"""
        business_id = data.get("business_id")
        business = next((b for b in self.businesses if b.id == business_id), None)
        
        if business and np.linalg.norm(agent.pos - business.pos) < 0.0005:
            # Agent is close enough to business
            resources_needed = {r: random.randint(1, 5) for r in random.sample(list(RESOURCES.keys()), 2)}
            provided, cost = business.serve_customer(resources_needed)
            
            if provided and agent.resources["money"] >= cost:
                agent.resources["money"] -= cost
                for resource, amount in provided.items():
                    agent.resources[resource] += amount
                
                log_to_frontend(f"💰 Agent {agent.id} gathered {provided} for ${cost:.1f}")
                agent.add_to_memory_stream(f"Gathered {provided} from business {business_id} for ${cost:.1f}", self.step_count)
            else:
                agent.add_to_memory_stream(f"Failed to gather resources from business {business_id} - insufficient funds", self.step_count)
        else:
            # Move towards business first
            if business:
                agent.set_destination(business.pos)
                agent.add_to_memory_stream(f"Moving towards business {business_id} to gather resources", self.step_count)

    def _execute_work_action(self, agent: Pedestrian, data: Dict):
        """Execute work action"""
        business_id = data.get("business_id")
        business = next((b for b in self.businesses if b.id == business_id), None)
        
        if business and np.linalg.norm(agent.pos - business.pos) < 0.0005:
            wage = business.hire_worker(agent.id)
            agent.resources["money"] += wage
            business.revenue -= wage
            
            log_to_frontend(f"💼 Agent {agent.id} worked and earned ${wage:.1f}")
            agent.add_to_memory_stream(f"Worked at business {business_id} and earned ${wage:.1f}", self.step_count)
        else:
            if business:
                agent.set_destination(business.pos)
                agent.add_to_memory_stream(f"Moving towards business {business_id} to work", self.step_count)

    def _execute_start_building_action(self, agent: Pedestrian, data: Dict):
        """Execute start building action"""
        building_type = data.get("building_type", "house")
        reason = data.get("reason", "Community need")
        
        if building_type not in BUILDING_RECIPES:
            agent.add_to_memory_stream(f"Failed to start building - unknown type: {building_type}", self.step_count)
            return
        
        # IMPORTANT: Prevent agents from starting new buildings if they already have an active project
        if agent.current_building_project is not None:
            # Check if their current project still exists and isn't completed
            current_building = next((b for b in self.buildings if b.id == agent.current_building_project), None)
            if current_building and current_building.status != "completed":
                agent.add_to_memory_stream(f"Cannot start new building - already working on building {agent.current_building_project}", self.step_count)
                log_to_frontend(f"🚫 Agent {agent.id} tried to start {building_type} but already working on building {agent.current_building_project}")
                return
            else:
                # Clear the project if it's completed or doesn't exist
                agent.current_building_project = None
        
        # Limit total number of planning/under_construction buildings to prevent too many unfinished projects
        active_buildings = [b for b in self.buildings if b.status in ["planning", "under_construction"]]
        max_concurrent_buildings = max(3, len(self.pedestrians) // 2)  # At least 3, or half the number of agents
        
        if len(active_buildings) >= max_concurrent_buildings:
            agent.add_to_memory_stream(f"Cannot start new building - too many active projects ({len(active_buildings)}/{max_concurrent_buildings})", self.step_count)
            log_to_frontend(f"🚫 Agent {agent.id} tried to start {building_type} but too many active projects ({len(active_buildings)}/{max_concurrent_buildings})")
            return
        
        # Find a valid location for the new building
        building_pos = self._find_valid_building_spot(agent.pos)
        
        if building_pos is None:
            agent.add_to_memory_stream(f"Failed to find a valid spot to build a {building_type}", self.step_count)
            return
        
        # Create new building project
        building_id = len(self.buildings)
        new_building = Building(building_id, building_pos, building_type, agent.id)
        self.buildings.append(new_building)
        
        agent.current_building_project = building_id
        agent.building_contributions[building_id] = {}
        
        log_to_frontend(f"🏗️ Agent {agent.id} started building project: {building_type} (#{building_id})")
        agent.add_to_memory_stream(f"Started {building_type} building project (ID: {building_id}). Reason: {reason}", self.step_count)
        
        # Send to all agents
        self.add_message(agent.id, f"I've started a {building_type} project! Who wants to help? Reason: {reason}")

    def _execute_contribute_building_action(self, agent: Pedestrian, data: Dict):
        """Execute contribute to building action"""
        building_id = data.get("building_id")
        resources_to_contribute = data.get("resources", {})
        
        building = next((b for b in self.buildings if b.id == building_id), None)
        if not building:
            agent.add_to_memory_stream(f"Failed to contribute - building {building_id} not found", self.step_count)
            return
        
        if building.status == "completed":
            agent.add_to_memory_stream(f"Building {building_id} already completed", self.step_count)
            # Clear current project if it's this building
            if agent.current_building_project == building_id:
                agent.current_building_project = None
            return
        
        total_contributed = {}
        contribution_value = 0
        
        for resource, amount in resources_to_contribute.items():
            if agent.resources.get(resource, 0) >= amount:
                actual_contribution = building.contribute_resource(resource, amount, agent.id)
                if actual_contribution > 0:
                    agent.resources[resource] -= actual_contribution
                    total_contributed[resource] = actual_contribution
                    contribution_value += RESOURCES[resource]["value"] * actual_contribution
        
        if total_contributed:
            # Update current project if not set or if contributing to a different building
            if agent.current_building_project is None or agent.current_building_project != building_id:
                # Check if their old project is completed
                if agent.current_building_project is not None:
                    old_building = next((b for b in self.buildings if b.id == agent.current_building_project), None)
                    if old_building and old_building.status == "completed":
                        log_to_frontend(f"✅ Agent {agent.id}'s project {agent.current_building_project} completed, switching to {building_id}")
                
                agent.current_building_project = building_id
                agent.add_to_memory_stream(f"Switched focus to building project {building_id}", self.step_count)
            
            # Track personal contributions
            if building_id not in agent.building_contributions:
                agent.building_contributions[building_id] = {}
            
            for resource, amount in total_contributed.items():
                agent.building_contributions[building_id][resource] = \
                    agent.building_contributions[building_id].get(resource, 0) + amount
            
            log_to_frontend(f"🔨 Agent {agent.id} contributed {total_contributed} to building {building_id}")
            agent.add_to_memory_stream(f"Contributed {total_contributed} to {building.type} project (value: ${contribution_value})", self.step_count)
            
            # Check if building can now start construction
            if building.status == "planning" and building.can_start_construction():
                log_to_frontend(f"🚀 Building {building_id} ({building.type}) can now start construction!")
                agent.add_to_memory_stream(f"Building {building_id} ready for construction thanks to my contribution!", self.step_count)
            
            # Notify others
            self.add_message(agent.id, f"I contributed {total_contributed} to the {building.type} project! Progress continues.")
        else:
            agent.add_to_memory_stream(f"No resources available to contribute to building {building_id}", self.step_count)

    def _execute_communicate_action(self, agent: Pedestrian, data: Dict):
        """Execute communication action"""
        message = data.get("message", "Hello!")
        recipient_id = data.get("recipient_id")
        
        message_data = {
            "sender_id": agent.id,
            "message": message,
            "recipient_id": recipient_id,
            "step": self.step_count,
        }
        self.messages.append(message_data)
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]
        
        # Update memory for relevant agents
        if recipient_id is not None:
            recipient = next((a for a in self.pedestrians if a.id == recipient_id), None)
            if recipient:
                recipient.update_memory(f"Agent {agent.id} said: {message}")
                recipient.add_to_memory_stream(f"Received message from Agent {agent.id}: '{message}'", self.step_count)
                agent.add_to_memory_stream(f"Sent message to Agent {recipient_id}: '{message}'", self.step_count)
        else:
            # Broadcast
            for other_agent in self.pedestrians:
                if other_agent.id != agent.id:
                    other_agent.update_memory(f"Agent {agent.id} said: {message}")
                    other_agent.add_to_memory_stream(f"Heard Agent {agent.id} say: '{message}'", self.step_count)
            agent.add_to_memory_stream(f"Broadcast message: '{message}'", self.step_count)
        
        # Send to frontend
        if _current_websocket:
            asyncio.create_task(_current_websocket.send_json({
                "type": "agent_message",
                "agent_id": agent.id,
                "message": message,
                "step": self.step_count,
                "recipient_id": recipient_id
            }))

    async def step(self):
        """Execute one simulation step"""
        self.step_count += 1
        
        # --- LLM Thinking (Asynchronous) ---
        llm_tasks = []
        task_to_agent = {}
        max_concurrent_llm = 4
        current_thinking_count = sum(1 for agent in self.pedestrians if agent.is_thinking)
        
        for agent in self.pedestrians:
            if current_thinking_count >= max_concurrent_llm:
                break
                
            steps_since_last_llm = self.step_count - agent.last_llm_step
            
            if not agent.is_thinking and steps_since_last_llm >= LLM_CALL_FREQUENCY:
                agent.last_llm_step = self.step_count
                
                async def safe_llm_call(agent_ref):
                    return await agent_ref.decide_action_llm(
                        self.businesses, self.buildings, self.pedestrians, self.messages, self.step_count
                    )
                
                task = asyncio.create_task(safe_llm_call(agent))
                llm_tasks.append(task)
                task_to_agent[task] = agent
                current_thinking_count += 1
        
        # --- Fast Action Execution (Synchronous) ---
        agent_actions = []
        for agent in self.pedestrians:
            action = agent.get_fast_action(self.trained_policy, self.businesses, self.buildings)
            agent_actions.append(action)

        # Collect LLM logs
        if llm_tasks:
            done, pending = await asyncio.wait(llm_tasks, timeout=30.0)
            
            for task in done:
                try:
                    log_entry = task.result()
                    if log_entry:
                        self.llm_logs.append(log_entry)
                        if len(self.llm_logs) > MAX_LLM_LOGS:
                            self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]
                        
                        # Add conversation to messages
                        if log_entry.get('conversation'):
                            message_data = {
                                "sender_id": log_entry.get('agent_id'),
                                "message": log_entry.get('conversation'),
                                "recipient_id": None,
                                "step": self.step_count,
                            }
                            self.messages.append(message_data)
                            if len(self.messages) > MAX_MESSAGES:
                                self.messages = self.messages[-MAX_MESSAGES:]
                except Exception as e:
                    agent = task_to_agent.get(task)
                    if agent:
                        agent.is_thinking = False
            
            for task in pending:
                agent = task_to_agent.get(task)
                if agent:
                    agent.is_thinking = False
                task.cancel()
        
        # Execute all actions
        self._execute_actions(agent_actions)
        
        # Update traffic lights
        for light in self.traffic_lights:
            light.step()
        
        # Update businesses (generate resources)
        for business in self.businesses:
            business.generate_resources()
        
        # Update building construction progress
        for building in self.buildings:
            if building.advance_construction():
                log_to_frontend(f"🏗️ Building {building.id} ({building.type}) completed!")
                # Notify all contributors and clear their current_building_project if it was this building
                for contributor_id in building.contributors:
                    contributor = next((p for p in self.pedestrians if p.id == contributor_id), None)
                    if contributor:
                        contributor.add_to_memory_stream(f"Building {building.id} ({building.type}) completed! I was a contributor.", self.step_count)
                        # Clear current project if it was this building
                        if contributor.current_building_project == building.id:
                            contributor.current_building_project = None
                            log_to_frontend(f"✅ Agent {contributor.id} completed their building project {building.id}")
        
        # Update pedestrians
        for pedestrian in self.pedestrians:
            pedestrian.step(self.businesses, self.buildings, self.traffic_lights)

def get_agent_state_vector(agent: Pedestrian, businesses: List[Business], buildings: List[Building]) -> np.ndarray:
    """Create state vector for RL policy"""
    return agent._get_state_vector(businesses, buildings)

def get_valid_actions_mask(agent: Pedestrian, businesses: List[Business], buildings: List[Building]) -> np.ndarray:
    """Generate valid actions mask for agent"""
    mask = np.ones(len(DISCRETE_ACTIONS), dtype=bool)
    
    # Movement actions are generally always valid
    # Gather resources only valid if there are businesses with resources
    if not any(sum(b.inventory.values()) > 0 for b in businesses):
        gather_idx = DISCRETE_ACTIONS.index("gather_resources")
        mask[gather_idx] = False
    
    # Work only valid if businesses exist
    if not businesses:
        work_idx = DISCRETE_ACTIONS.index("work_at_business")
        mask[work_idx] = False
    
    # Building actions depend on resources and projects
    start_building_idx = DISCRETE_ACTIONS.index("start_building")
    contribute_idx = DISCRETE_ACTIONS.index("contribute_building")
    
    # Can start building only if:
    # 1. Agent has some resources
    # 2. Agent doesn't already have a current project
    # 3. There aren't too many active buildings
    active_buildings = [b for b in buildings if b.status in ["planning", "under_construction"]]
    max_concurrent_buildings = 6  # Reasonable default limit
    
    if (sum(agent.resources.values()) < 10 or 
        agent.current_building_project is not None or 
        len(active_buildings) >= max_concurrent_buildings):
        mask[start_building_idx] = False
    
    # Can contribute if there are active projects and agent has resources
    if not active_buildings or sum(agent.resources.values()) < 5:
        mask[contribute_idx] = False
    
    return mask

class WebSocketCallback(BaseCallback):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop, verbose=0):
        super(WebSocketCallback, self).__init__(verbose)
        self.websocket = websocket
        self.loop = loop
        self.should_stop = False
        self.episode_count = 0

    def _send_message_safely(self, payload: dict) -> bool:
        if self.should_stop or self.websocket.application_state != WebSocketState.CONNECTED:
            if not self.should_stop:
                print("WebSocket is not connected. Stopping training.")
                self.should_stop = True
            return False

        future = asyncio.run_coroutine_threadsafe(
            self.websocket.send_json(payload), self.loop
        )
        try:
            return future.result(timeout=2.0)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.should_stop = True
            return False

    def _on_step(self) -> bool:
        if self.should_stop:
            return False

        # Log episode info when an episode ends
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                info = self.locals["infos"][i]
                if "episode" in info:
                    ep_info = info["episode"]
                    payload = {
                        "type": "progress",
                        "episode": self.episode_count,
                        "reward": ep_info["r"],
                        "loss": self.logger.name_to_value.get("train/loss"),
                        "buildings_total": len(self.training_env.get_attr("buildings")[i]),
                        "buildings_completed": len([b for b in self.training_env.get_attr("buildings")[i] if b.status == "completed"]),
                    }
                    if not self._send_message_safely(payload):
                        return False

        # Send visualization state periodically
        if self.num_timesteps % 100 == 0:
            state = self.training_env.env_method("get_state_for_viz")[0]
            total_buildings = len(state.get("buildings", []))
            completed_buildings = len([b for b in state.get("buildings", []) if b.get("status") == "completed"])

            payload = {
                "type": "train_step", 
                "state": state, 
                "episode": self.episode_count,
                "buildings_total": total_buildings,
                "buildings_completed": completed_buildings
            }
            if not self._send_message_safely(payload):
                return False

        return not self.should_stop

    def stop_training(self):
        self.should_stop = True

async def train_simcity(websocket: WebSocket, env: SimCityEnv):
    """Train the SimCity agents using PPO for collaborative building"""
    global _current_websocket
    _current_websocket = websocket
    
    loop = asyncio.get_running_loop()

    def train_in_thread():
        vec_env = make_vec_env(lambda: SimCityEnv(training_mode=True), n_envs=4)

        policy_kwargs = dict(
            features_extractor_class=SimCityPPOPolicy,
            features_extractor_kwargs=dict(features_dim=128),
        )

        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_EPS,
            ent_coef=PPO_ENT_COEF,
            learning_rate=PPO_LR,
            n_epochs=PPO_N_EPOCHS,
            batch_size=PPO_BATCH_SIZE,
            n_steps=PPO_N_STEPS,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./simcity_tensorboard/"
        )

        callback = WebSocketCallback(websocket, loop)
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        
        # Save the trained model
        model_path = "policies/simcity_ppo_model.zip"
        model.save(model_path)
        
        # Send completion message
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "trained",
                "model_info": {"path": model_path}
            }),
            loop
        )

    training_task = asyncio.to_thread(train_in_thread)
    await training_task

# --- Websocket handlers ---
async def run_simcity(websocket: WebSocket, env: SimCityEnv):
    """Run the simulation"""
    global _current_websocket
    _current_websocket = websocket
    
    # Load the trained model
    model_path = "policies/simcity_ppo_model.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        env.trained_policy = model
    else:
        logger.warning("No trained model found for SimCity. Actions will be random.")
        env.trained_geo_policy = None

    env.running = True
    step_rewards = []

    while env.running:
        # For run mode, we step the entire environment at once.
        # We can get actions for all agents from the policy.
        
        if env.trained_policy:
            # In a multi-agent env, you'd typically get an action for each agent
            # Here, for simplicity, we'll get an action for the first agent
            # and have others act randomly or with a simpler logic.
            obs = env._get_obs_for_agent(env.pedestrians[0])
            action, _ = env.trained_policy.predict(obs, deterministic=True)
            action_name = DISCRETE_ACTIONS[action]
            
            # Create dummy data for the action
            data = {}
            if action_name == "start_building":
                data = {"building_type": env.pedestrians[0].building_goal}

            # We need to manually call the execution logic for the agents
            env._execute_move_action(env.pedestrians[0], {"move_vector": ACTION_MAP_MOVE.get(action_name)})

        # The full step logic is now more complex than the gym step
        # We'll use the original async step method from the class
        await env_step_full(env)

        # Send state update
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state})
        
        # Calculate metrics for charts
        total_reward = env._calculate_reward()
        avg_satisfaction = np.mean([p.satisfaction for p in env.pedestrians])
        total_buildings = len(env.buildings)
        completed_buildings = len([b for b in env.buildings if b.status == "completed"])
        
        step_rewards.append(total_reward)
        
        await websocket.send_json({
            "type": "progress",
            "episode": env.step_count,
            "reward": total_reward,
            "satisfaction": avg_satisfaction,
            "buildings_total": total_buildings,
            "buildings_completed": completed_buildings,
            "loss": None
        })
        
        await asyncio.sleep(0.15)  # Control simulation speed
            

async def env_step_full(env: SimCityEnv):
    """The original, comprehensive step function for the environment."""
    env.step_count += 1
    # This function would contain the logic from the original `step` method
    # before it was adapted for the gym interface.
    # For now, we'll just call the individual components.

    # 1. Get actions for all agents (simplified for this example)
    agent_actions = []
    for agent in env.pedestrians:
        if env.trained_policy and agent.id == 0:
             obs = env._get_obs_for_agent(agent)
             action, _ = env.trained_policy.predict(obs, deterministic=True)
             action_name = DISCRETE_ACTIONS[action]
             data = {} # dummy data
        else:
            action_name = random.choice(DISCRETE_ACTIONS)
            data = {} # dummy data
        agent_actions.append((action_name, data))
    
    # 2. Execute actions
    env._execute_actions(agent_actions)

    # 3. Update world state
    for light in env.traffic_lights:
        light.step()
    for business in env.businesses:
        business.generate_resources()
    for building in env.buildings:
        if building.advance_construction():
            log_to_frontend(f"🏗️ Building {building.id} ({building.type}) completed!")
    
    # 4. Update pedestrians
    for pedestrian in env.pedestrians:
        pedestrian.step(env.businesses, env.buildings, env.traffic_lights)
