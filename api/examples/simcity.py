import asyncio
import random
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging
import osmnx as ox
from shapely.geometry import Point, LineString

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

RETRO_SCIFI_COLORS = [
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.6, 0.0],  # Bright Orange
    [0.7, 1.0, 0.0],  # Lime Green
    [0.1, 0.5, 1.0],  # Electric Blue
    [1.0, 1.0, 0.2],  # Bright Yellow
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
GRID_SIZE_X = 64
GRID_SIZE_Y = 20
GRID_SIZE_Z = 64
NUM_AGENTS = 5
NUM_CARS = 10
INITIAL_BUDGET = 100000
INITIAL_POPULATION = 0

BUILDING_TYPES = {
    # Non-buildable ground types
    "grass": {"cost": 0, "income": 0, "population": 0, "color": [0.2, 0.6, 0.2]},
    "road": {"cost": 0, "income": 0, "population": 0, "color": [0.2, 0.2, 0.2]},
    # Buildable types
    "residential": {"cost": 100, "income": 10, "population": 5, "color": [0.2, 0.8, 0.2]},
    "commercial": {"cost": 200, "income": 50, "population": 0, "color": [0.2, 0.2, 0.8]},
    "industrial": {"cost": 300, "income": 100, "population": 0, "color": [0.8, 0.8, 0.2]},
    "park": {"cost": 50, "income": 0, "population": 0, "color": [0.1, 0.5, 0.1]},
}

MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 10
USE_LOCAL_OLLAMA = True 

# Define discrete actions for the PPO policy
build_actions = [f"build_{btype}" for btype in ["residential", "commercial", "industrial", "park"]]
move_actions = ["move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "move_z-"]
destroy_actions = ["destroy_x+", "destroy_x-", "destroy_y+", "destroy_y-", "destroy_z+", "destroy_z-"]
DISCRETE_ACTIONS = move_actions + build_actions + destroy_actions + ["talk", "wait"]

ACTION_MAP_MOVE = {
    "move_x+": np.array([1, 0, 0]), "move_x-": np.array([-1, 0, 0]),
    "move_y+": np.array([0, 1, 0]), "move_y-": np.array([0, -1, 0]),
    "move_z+": np.array([0, 0, 1]), "move_z-": np.array([0, 0, -1]),
}
ACTION_MAP_DESTROY = {
    "destroy_x+": np.array([1, 0, 0]), "destroy_x-": np.array([-1, 0, 0]),
    "destroy_y+": np.array([0, 1, 0]), "destroy_y-": np.array([0, -1, 0]),
    "destroy_z+": np.array([0, 0, 1]), "destroy_z-": np.array([0, 0, -1]),
}

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh())
        self.actor_logits = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, valid_actions_mask: torch.Tensor = None):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        
        # Apply mask if provided
        if valid_actions_mask is not None:
            logits = logits + (valid_actions_mask - 1) * 1e8
        
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

    def get_action(self, obs: np.ndarray, action: torch.Tensor = None, valid_actions_mask: torch.Tensor = None):
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs
        if obs_t.dim() == 1: 
            obs_t = obs_t.unsqueeze(0)
            
        dist, value = self.forward(obs_t, valid_actions_mask)
        if action is None: 
            action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if obs_t.shape[0] == 1: 
            return action.item(), log_prob.item(), value.item()
        return action, log_prob, value

class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {btype: 20 for btype in ["residential", "commercial", "industrial", "park"]}
        self.money = 10000  # Individual agent money
        self.goal = random.choice(["residential", "commercial", "industrial", "park"])
        self.color = [random.random(), random.random(), random.random()]
        self.llm_intent = None
        self.is_thinking = False
        self.last_llm_step = -LLM_CALL_FREQUENCY
        self.memory_vector = np.zeros(384, dtype=np.float32)
        self.memory_stream = []

    def update_memory(self, text: str):
        """Update agent's memory with a new text embedding using a moving average"""
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

    async def decide_action_llm(self, grid: np.ndarray, city_stats: Dict, messages: List[Dict], step_count: int, offers: List[Dict]):
        self.is_thinking = True
        log_to_frontend(f"🤖 Agent {self.id}: Starting LLM call at step {step_count}")
        
        view_radius = 2
        x_start, x_end = max(0, self.pos[0]-view_radius), min(GRID_SIZE_X, self.pos[0]+view_radius+1)
        y_start, y_end = max(0, self.pos[1]-view_radius), min(GRID_SIZE_Y, self.pos[1]+view_radius+1)
        z_start, z_end = max(0, self.pos[2]-view_radius), min(GRID_SIZE_Z, self.pos[2]+view_radius+1)
        view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

        template = """
Examples of valid JSON responses:

For move action:
{
    "action": "move",
    "data": [32, 8, 32]
}

For build action:
{
    "action": "build", 
    "data": {"block_type": "residential", "position": [32, 7, 32]}
}

For destroy action:
{
    "action": "destroy",
    "data": [32, 7, 32]
}

For talk action:
{
    "action": "talk",
    "data": {"message": "Hello everyone!", "recipient_id": null}
}

For offer action:
{
    "action": "offer",
    "data": {"item_to_give": "residential", "amount_to_give": 2, "item_to_receive": "commercial", "amount_to_receive": 1, "money_offered": 500}
}

For accept_offer action:
{
    "action": "accept_offer",
    "data": 0
}
"""

        system_prompt = f"""
You are a city construction agent in a 3D grid world. Your ID is {self.id}.
You must respond in valid JSON format only. Here are the valid response formats:

{template}

Choose the appropriate action and provide the data in the exact format shown above.
"""

        building_map_str = ", ".join([f"{i+1}: {name}" for i, name in enumerate(BUILDING_TYPES.keys())])
        recent_messages = messages[-5:]

        prompt = f"""You are a city construction agent in a 3D grid world. Your ID is {self.id}.
Your current position is [x, y, z]: {self.pos.tolist()}. Y is the vertical axis.
Your inventory is {self.inventory}.
Your money: ${self.money}
Your current goal is to build '{self.goal}' buildings.
City stats: {city_stats}

Your recent action history (last 10 events):
{chr(10).join(self.memory_stream) if self.memory_stream else "No recent events recorded."}

Recent messages from other agents: {json.dumps(recent_messages)}
Open trade offers: {json.dumps(offers)}

The world is {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}. Buildings are encoded as numbers. 0 is empty (air), 2 is road (DO NOT BUILD ON ROADS).
Building map: {building_map_str}.
You can see a 5x5x5 area around you. Your view:
{json.dumps(view)}

Your available actions are "move", "build", "destroy", "talk", "offer", or "accept_offer".
- move: requires integer [x, y, z] coordinates for the next step. You can only move to an adjacent square.
- build: requires an object with {{"block_type": string, "position": [x, y, z]}} of an adjacent location. NEVER build on roads (value 2)!
- destroy: requires integer [x, y, z] coordinates of an adjacent building to demolish.
- talk: requires an object with {{"message": string, "recipient_id": integer (optional)}}. If no recipient, it's a broadcast.
- offer: requires an object with {{"item_to_give": string, "amount_to_give": int, "item_to_receive": string, "amount_to_receive": int, "money_offered": int}}.
- accept_offer: requires the integer offer_id of the offer to accept.

PRIORITIZE COMMUNICATION: Talk frequently to coordinate with other agents, negotiate trades, or share city planning ideas.

Build strategically near roads but NOT ON roads. Consider the city's RCI (Residential/Commercial/Industrial) demand. Trade with other agents to get needed building types.

CRITICAL: You MUST respond with valid JSON only. No explanations, no text outside JSON.

{template}

Pick ONE action and respond in the exact JSON format shown above.
"""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "build", "destroy", "talk", "offer", "accept_offer"]},
                "data": {"oneOf": [
                    {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    {"type": "object"},
                ]}
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
                    schema_name="agent_action_decision_simcity",
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
                    schema_name="agent_action_decision_simcity",
                ),
                timeout=20.0
            )

        self.llm_intent = (response.get("action", "move"), response.get("data"))
        log_to_frontend(f"🧠 Agent {self.id}: LLM response - action: {response.get('action')}, data: {response.get('data')}")
        
        # Generate conversational text for the MessagePanel
        conversation_prompt = f"""You are Agent {self.id} in a city building world. Based on your current situation and recent action decision, provide a brief conversational response about what you're thinking or doing. Keep it short and natural, like you're talking to other agents.

Current situation:
- Position: {self.pos.tolist()}
- Goal: Build {self.goal} buildings
- Money: ${self.money}
- Inventory highlights: {[(k,v) for k,v in self.inventory.items() if v > 0][:3]}
- Recent action decision: {response.get('action')} with data {response.get('data')}
- Recent messages: {[msg.get('message', '') for msg in recent_messages[-2:]]}

Respond as Agent {self.id} in first person, conversationally:"""

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
            
        conversation_text = conversation_response.get("message", f"Agent {self.id}: Planning my next building...")
        log_to_frontend(f"💬 {conversation_text}")
        
        # Send conversation text to frontend MessagePanel
        if _current_websocket:
            await _current_websocket.send_json({
                "type": "agent_message",
                "agent_id": self.id,
                "message": conversation_text,
                "step": step_count
            })
        
        self.is_thinking = False
        return {"agent_id": self.id, "step": step_count, "response": response}

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray, env: "SimCityEnv") -> Tuple[str, Any]:
        """Get a fast action using the trained policy when LLM is not available"""
        state_vec = get_agent_state_vector(self, grid, env)
        valid_actions_mask = get_valid_actions_mask(self, grid)
        action_idx, _, _ = trained_policy.get_action(state_vec, valid_actions_mask=valid_actions_mask)
        action_name = DISCRETE_ACTIONS[action_idx]
        
        if "build_" in action_name:
            building_type = action_name.replace("build_", "")
            # Try to build at feet level
            target_pos = self.pos + np.array([0, -1, 0])
            return "build", {"block_type": building_type, "position": target_pos.tolist()}
        elif "destroy_" in action_name:
            direction = action_name.replace("destroy_", "")
            target_pos = self.pos + ACTION_MAP_DESTROY[f"destroy_{direction}"]
            return "destroy", target_pos.tolist()
        elif "move_" in action_name:
            direction = action_name.replace("move_", "")
            target_pos = self.pos + ACTION_MAP_MOVE[f"move_{direction}"]
            return "move", target_pos.tolist()
        elif action_name == "talk":
            return "talk", {"message": f"Agent {self.id} working on the city", "recipient_id": None}
        
        return "wait", None

class TrafficLight:
    def __init__(self, light_id: int, pos: np.ndarray, initial_state: str = 'green', cycle_time: int = 200):
        self.id = light_id
        self.pos = pos
        self.state = initial_state 
        self.cycle_time = cycle_time
        self.timer = random.randint(0, cycle_time)

    def update(self):
        self.timer += 1
        if self.timer >= self.cycle_time:
            self.timer = 0
            self.state = 'green' if self.state == 'red' else 'red'

class Pedestrian:
    def __init__(self, ped_id: int, start_pos: np.ndarray, end_pos: np.ndarray, speed: float = 1.0, initial_state: str = 'waiting'):
        self.id = ped_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.pos = start_pos.copy()
        self.speed = speed
        self.state = initial_state
        self.path_progress = 0.0
        self.wait_timer = 0
    
    def step(self, traffic_light_state: str):
        if self.state == 'waiting':
            if random.random() < 0.005:
                self.state = 'jaywalking'
                self.wait_timer = 0
                return

            if traffic_light_state == 'green':
                self.wait_timer = 0
                self.state = 'crossing'
            else:
                self.wait_timer += 1
        
        elif self.state == 'crossing' or self.state == 'jaywalking':
            total_dist = np.linalg.norm(self.end_pos - self.start_pos)
            if total_dist > 0:
                self.path_progress += self.speed / total_dist
                self.pos = self.start_pos + (self.end_pos - self.start_pos) * self.path_progress
            
            if self.path_progress >= 1.0:
                self.state = 'waiting'
                self.path_progress = 0
                self.start_pos, self.end_pos = self.end_pos, self.start_pos
                self.pos = self.start_pos.copy()

class Car:
    def __init__(self, car_id: int, graph, start_node, path, color):
        self.id = car_id
        self.graph = graph
        self.path = path
        self.path_index = 0
        self.distance_on_segment = 0.0
        self.pos = np.array(graph.nodes[start_node]['pos'])
        self.speed = random.uniform(0.1, 0.3)
        self.color = color

    def move(self, traffic_lights):
        if self.path_index >= len(self.path) - 1:
            return

        current_node_id = self.path[self.path_index]
        next_node_id = self.path[self.path_index + 1]
        light = traffic_lights.get(next_node_id)

        current_segment_vec = np.array(self.graph.nodes[next_node_id]['pos']) - np.array(self.graph.nodes[current_node_id]['pos'])
        segment_len = np.linalg.norm(current_segment_vec)
        dist_to_light = segment_len - self.distance_on_segment

        if light and light.state == 'red' and dist_to_light < 5:
            return

        self.distance_on_segment += self.speed

        if self.distance_on_segment >= segment_len:
            self.distance_on_segment -= segment_len
            self.path_index += 1
            if self.path_index >= len(self.path) - 1:
                return

        current_node_id = self.path[self.path_index]
        next_node_id = self.path[self.path_index + 1]
        start_pos = np.array(self.graph.nodes[current_node_id]['pos'])
        end_pos = np.array(self.graph.nodes[next_node_id]['pos'])
        
        current_segment_vec = end_pos - start_pos
        segment_len = np.linalg.norm(current_segment_vec)
        
        if segment_len > 0:
            ratio = self.distance_on_segment / segment_len
            self.pos = start_pos + ratio * current_segment_vec

class SimCityEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.agents: List[Agent] = []
        self.cars: List[Car] = []
        self.pedestrians: List[Pedestrian] = []
        self.traffic_lights: Dict[int, TrafficLight] = {}
        self.messages = []
        self.offers = []
        self.step_count = 0
        self.budget = INITIAL_BUDGET
        self.population = INITIAL_POPULATION
        self.trained_policy: "ActorCritic" = None

        log_to_frontend("Fetching street data from OpenStreetMap...")
        self._setup_map()
        self._spawn_entities()
        log_to_frontend("Street data processed and entities spawned.")

    def _setup_map(self):
        location_point = (40.7128, -74.0060)  # New York City
        self.graph = ox.graph_from_point(location_point, dist=500, network_type='drive')
        
        nodes_df, edges_df = ox.graph_to_gdfs(self.graph)
        min_x, min_y = nodes_df.geometry.x.min(), nodes_df.geometry.y.min()
        max_x, max_y = nodes_df.geometry.x.max(), nodes_df.geometry.y.max()
        
        # Store the actual map bounds for the frontend
        self.map_center = [(min_y + max_y) / 2, (min_x + max_x) / 2]  # [lat, lon] for Google Maps
        self.map_bounds = {
            "north": max_y,
            "south": min_y,
            "east": max_x, 
            "west": min_x
        }
        
        # Calculate appropriate zoom level based on bounds
        lat_diff = max_y - min_y
        lon_diff = max_x - min_x
        max_diff = max(lat_diff, lon_diff)
        if max_diff > 0.02:
            self.map_zoom = 14
        elif max_diff > 0.01:
            self.map_zoom = 15
        else:
            self.map_zoom = 16

        def scale_coords(lon, lat):
            x = int(np.interp(lon, [min_x, max_x], [0, GRID_SIZE_X - 1]))
            z = int(np.interp(lat, [min_y, max_y], [0, GRID_SIZE_Z - 1]))
            return x, z

        road_idx = list(BUILDING_TYPES.keys()).index("road") + 1
        for u, v, data in self.graph.edges(data=True):
            p1_lon, p1_lat = self.graph.nodes[u]['x'], self.graph.nodes[u]['y']
            p2_lon, p2_lat = self.graph.nodes[v]['x'], self.graph.nodes[v]['y']
            x1, z1 = scale_coords(p1_lon, p1_lat)
            x2, z2 = scale_coords(p2_lon, p2_lat)
            self.graph.nodes[u]['pos'] = (x1, 0, z1)
            self.graph.nodes[v]['pos'] = (x2, 0, z2)
            
            # Draw line on grid
            num_points = max(abs(x2 - x1), abs(z2 - z1)) + 1
            for i in range(num_points):
                t = i / (num_points - 1)
                x, z = int(x1 * (1 - t) + x2 * t), int(z1 * (1 - t) + z2 * t)
                if 0 <= x < GRID_SIZE_X and 0 <= z < GRID_SIZE_Z:
                    self.grid[x, 0, z] = road_idx
        
        self.road_network_for_viz = [[self.graph.nodes[u]['pos'], self.graph.nodes[v]['pos']] for u, v, _ in self.graph.edges(data=True)]

    def _spawn_entities(self):
        road_idx = list(BUILDING_TYPES.keys()).index("road") + 1
        road_coords = np.argwhere(self.grid[:, 0, :] == road_idx)
        
        for i in range(NUM_AGENTS):
            idx = random.choice(range(len(road_coords)))
            x, z = road_coords[idx]
            self.agents.append(Agent(i, np.array([x, 1, z])))

        graph_nodes = list(self.graph.nodes)
        if not graph_nodes: 
            return

        # Spawn cars with paths
        for i in range(NUM_CARS):
            while True:
                start_node, end_node = random.sample(graph_nodes, 2)
                path = ox.shortest_path(self.graph, start_node, end_node, weight='length')
                if path and len(path) > 1:
                    self.cars.append(Car(i, self.graph, start_node, path, random.choice(RETRO_SCIFI_COLORS)))
                    break

        # Spawn traffic lights and pedestrians
        light_id_counter = 0
        ped_id_counter = 0
        for i, (node_id, data) in enumerate(self.graph.nodes(data=True)):
            if self.graph.degree(node_id) > 2:
                self.traffic_lights[node_id] = TrafficLight(light_id_counter, data['pos'])
                light_id_counter += 1

                crosswalk_start = np.array(data['pos']) + np.array([1, 0, 1])
                crosswalk_end = np.array(data['pos']) - np.array([1, 0, 1])
                self.pedestrians.append(Pedestrian(ped_id_counter, crosswalk_start, crosswalk_end))
                ped_id_counter += 1

    def _update_simulation(self):
        for light in self.traffic_lights.values(): 
            light.update()
        for car in self.cars: 
            car.move(self.traffic_lights)
        for ped in self.pedestrians:
            closest_light_node = min(self.traffic_lights.keys(), key=lambda node_id: np.linalg.norm(np.array(self.graph.nodes[node_id]['pos']) - ped.pos)) if self.traffic_lights else None
            light_state = self.traffic_lights[closest_light_node].state if closest_light_node else 'red'
            ped.step(light_state)

    def _calculate_reward(self):
        # Calculate city-wide metrics
        income = np.sum([BUILDING_TYPES[list(BUILDING_TYPES.keys())[idx-1]]['income'] for idx in self.grid.flatten() if idx > 2])
        population = np.sum([BUILDING_TYPES[list(BUILDING_TYPES.keys())[idx-1]]['population'] for idx in self.grid.flatten() if idx > 2])
        self.budget += income
        self.population = population
        return population + (income / 10)

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        """Execute a list of actions for all agents"""
        for i, (action, data) in enumerate(agent_actions):
            agent = self.agents[i]
            
            if action == "move" and data:
                target_pos = np.array(data)
                if (0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 
                    0 <= target_pos[2] < GRID_SIZE_Z and np.linalg.norm(target_pos - agent.pos) <= 1.5):
                    if self.grid[target_pos[0], target_pos[1], target_pos[2]] == 0:
                        agent.pos = target_pos
                        agent.add_to_memory_stream(f"Moved to {target_pos.tolist()}", self.step_count)
            
            elif action == "build" and data:
                block_type = data.get("block_type")
                position = np.array(data.get("position", agent.pos))
                
                if (block_type in agent.inventory and agent.inventory[block_type] > 0 and
                    0 <= position[0] < GRID_SIZE_X and 0 <= position[1] < GRID_SIZE_Y and 
                    0 <= position[2] < GRID_SIZE_Z and np.linalg.norm(position - agent.pos) <= 1.5):
                    
                    # Check if building on road (road_idx = 2)
                    if self.grid[position[0], 0, position[2]] == 2:
                        agent.add_to_memory_stream(f"Cannot build on road at {position.tolist()}", self.step_count)
                        continue
                    
                    if self.grid[position[0], position[1], position[2]] == 0:
                        building_cost = BUILDING_TYPES[block_type]["cost"]
                        if agent.money >= building_cost:
                            agent.inventory[block_type] -= 1
                            agent.money -= building_cost
                            self.grid[position[0], position[1], position[2]] = list(BUILDING_TYPES.keys()).index(block_type) + 1
                            agent.add_to_memory_stream(f"Built {block_type} at {position.tolist()}", self.step_count)
            
            elif action == "destroy" and data:
                position = np.array(data)
                if (0 <= position[0] < GRID_SIZE_X and 0 <= position[1] < GRID_SIZE_Y and 
                    0 <= position[2] < GRID_SIZE_Z and np.linalg.norm(position - agent.pos) <= 1.5):
                    
                    block_idx = self.grid[position[0], position[1], position[2]]
                    if block_idx > 2:  # Can't destroy road or grass
                        block_type = list(BUILDING_TYPES.keys())[block_idx - 1]
                        agent.inventory[block_type] += 1
                        agent.money += BUILDING_TYPES[block_type]["cost"] // 2  # Half refund
                        self.grid[position[0], position[1], position[2]] = 0
                        agent.add_to_memory_stream(f"Destroyed {block_type} at {position.tolist()}", self.step_count)
            
            elif action == "talk" and data:
                message = data.get("message", "")
                recipient_id = data.get("recipient_id")
                
                message_obj = {
                    "sender_id": agent.id,
                    "recipient_id": recipient_id,
                    "message": message,
                    "step": self.step_count
                }
                self.messages.append(message_obj)
                
                if len(self.messages) > MAX_MESSAGES:
                    self.messages = self.messages[-MAX_MESSAGES:]
                
                agent.add_to_memory_stream(f"Said: {message}", self.step_count)
                
                # Send to frontend
                if _current_websocket:
                    asyncio.create_task(_current_websocket.send_json({
                        "type": "agent_message",
                        "agent_id": agent.id,
                        "message": message,
                        "step": self.step_count
                    }))
            
            elif action == "offer" and data:
                offer = {
                    "id": len(self.offers),
                    "agent_id": agent.id,
                    "item_to_give": data.get("item_to_give"),
                    "amount_to_give": data.get("amount_to_give", 1),
                    "item_to_receive": data.get("item_to_receive"),
                    "amount_to_receive": data.get("amount_to_receive", 1),
                    "money_offered": data.get("money_offered", 0),
                    "step": self.step_count
                }
                self.offers.append(offer)
                agent.add_to_memory_stream(f"Made trade offer: {offer}", self.step_count)
            
            elif action == "accept_offer" and data is not None:
                offer_id = data
                if 0 <= offer_id < len(self.offers):
                    offer = self.offers[offer_id]
                    offerer = self.agents[offer["agent_id"]]
                    
                    # Check if trade is possible
                    if (offerer.inventory.get(offer["item_to_give"], 0) >= offer["amount_to_give"] and
                        agent.inventory.get(offer["item_to_receive"], 0) >= offer["amount_to_receive"] and
                        offerer.money >= offer["money_offered"]):
                        
                        # Execute trade
                        offerer.inventory[offer["item_to_give"]] -= offer["amount_to_give"]
                        offerer.inventory[offer["item_to_receive"]] += offer["amount_to_receive"]
                        offerer.money -= offer["money_offered"]
                        
                        agent.inventory[offer["item_to_give"]] += offer["amount_to_give"]
                        agent.inventory[offer["item_to_receive"]] -= offer["amount_to_receive"]
                        agent.money += offer["money_offered"]
                        
                        agent.add_to_memory_stream(f"Completed trade with agent {offer['agent_id']}", self.step_count)
                        offerer.add_to_memory_stream(f"Trade completed with agent {agent.id}", self.step_count)
                        
                        # Remove the offer
                        self.offers.pop(offer_id)

    async def step(self):
        self.step_count += 1
        self._update_simulation()
        
        # LLM decisions (can run in parallel)
        llm_tasks = []
        for agent in self.agents:
            if not agent.is_thinking and (self.step_count - agent.last_llm_step) >= LLM_CALL_FREQUENCY:
                agent.last_llm_step = self.step_count
                stats = {"budget": self.budget, "population": self.population, "step": self.step_count}
                llm_tasks.append(asyncio.create_task(agent.decide_action_llm(self.grid, stats, self.messages, self.step_count, self.offers)))

        # Collect actions from agents
        agent_actions = []
        for agent in self.agents:
            if agent.llm_intent:
                action, data = agent.llm_intent
                agent.llm_intent = None
                agent_actions.append((action, data))
            elif self.trained_policy:
                action, data = agent.get_fast_action(self.trained_policy, self.grid, self)
                agent_actions.append((action, data))
            else:
                agent_actions.append(("wait", None))

        # Execute all actions
        self._execute_actions(agent_actions)
        
        # Finish awaiting LLM tasks
        if llm_tasks: 
            await asyncio.gather(*llm_tasks, return_exceptions=True)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "color": a.color, "type": "pedestrian", "inventory": a.inventory, "money": a.money} for a in self.agents],
            "cars": [{"id": c.id, "pos": c.pos.tolist(), "color": c.color, "type": "car"} for c in self.cars],
            "pedestrians": [{"id": p.id, "pos": p.pos.tolist(), "state": p.state} for p in self.pedestrians],
            "traffic_lights": [{"id": l.id, "pos": l.pos, "state": l.state} for l in self.traffic_lights.values()],
            "city_stats": {"budget": self.budget, "population": self.population, "rci_demand": self.get_rci_demand()},
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "building_types": BUILDING_TYPES,
            "road_network": self.road_network_for_viz,
            "messages": self.messages,
            "offers": self.offers,
            # Map overlay information for Google Maps (using actual OSM bounds)
            "map_center": self.map_center,
            "map_zoom": self.map_zoom,
            "map_rotation_degrees": 0,  # No rotation
            "map_bounds": self.map_bounds
        }

    def get_rci_demand(self):
        pop = self.population
        
        res_zones = np.count_nonzero(self.grid == list(BUILDING_TYPES.keys()).index("residential") + 1)
        com_zones = np.count_nonzero(self.grid == list(BUILDING_TYPES.keys()).index("commercial") + 1)
        ind_zones = np.count_nonzero(self.grid == list(BUILDING_TYPES.keys()).index("industrial") + 1)

        res_demand = max(0, (pop / 5) - res_zones) 
        com_demand = max(0, (pop / 20) - com_zones)
        ind_demand = max(0, (pop / 15) - ind_zones)

        total_demand = res_demand + com_demand + ind_demand + 1e-6
        return {
            "residential": (res_demand / total_demand) * 10,
            "commercial": (com_demand / total_demand) * 10,
            "industrial": (ind_demand / total_demand) * 10,
        }

def get_agent_state_vector(agent: Agent, grid: np.ndarray, env: SimCityEnv) -> np.ndarray:
    """Create a fixed-size state vector for the policy."""
    pos = agent.pos.astype(int)
    view = np.zeros((3, 3, 3))
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x, y, z = pos[0]+dx, pos[1]+dy, pos[2]+dz
                if 0 <= x < GRID_SIZE_X and 0 <= y < GRID_SIZE_Y and 0 <= z < GRID_SIZE_Z:
                    view[dx+1, dy+1, dz+1] = grid[x, y, z]
    
    inventory_vec = np.array(list(agent.inventory.values()))
    rci_demand = env.get_rci_demand()
    stats_vec = np.array([env.budget, env.population, rci_demand['residential'], rci_demand['commercial'], rci_demand['industrial'], agent.money])
    pos_vec = agent.pos
    
    return np.concatenate([view.flatten(), inventory_vec, stats_vec, pos_vec]).astype(np.float32)

def get_valid_actions_mask(agent: Agent, grid: np.ndarray) -> torch.Tensor:
    """Generate a mask for valid actions"""
    mask = torch.ones(len(DISCRETE_ACTIONS))
    
    for i, action in enumerate(DISCRETE_ACTIONS):
        if "build_" in action:
            building_type = action.replace("build_", "")
            if agent.inventory.get(building_type, 0) <= 0 or agent.money < BUILDING_TYPES[building_type]["cost"]:
                mask[i] = 0
        elif "move_" in action:
            direction = action.replace("move_", "")
            target_pos = agent.pos + ACTION_MAP_MOVE[f"move_{direction}"]
            if not (0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z):
                mask[i] = 0
    
    return mask

# PPO Hyperparameters
BATCH_SIZE = 512
MINI_BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

async def train_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket
    _current_websocket = websocket
    await websocket.send_json({"type": "debug", "message": "Starting SimCity training..."})
    
    dummy_obs = get_agent_state_vector(env.agents[0], env.grid, env)
    model = ActorCritic(dummy_obs.shape[0], len(DISCRETE_ACTIONS))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    step_buffer = []
    episode = 0
    total_steps = 0
    current_loss = None

    while episode < 10000:
        obs_list = [get_agent_state_vector(agent, env.grid, env) for agent in env.agents]
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)
        
        valid_masks = torch.stack([get_valid_actions_mask(agent, env.grid) for agent in env.agents])

        with torch.no_grad():
            actions_t, logp_t, value_t = model.get_action(obs_t, valid_actions_mask=valid_masks)
        
        # Execute actions and get rewards
        agent_actions = []
        rewards = []
        for i, agent in enumerate(env.agents):
            action_name = DISCRETE_ACTIONS[actions_t[i].item()]
            if "build_" in action_name:
                building_type = action_name.replace("build_", "")
                target_pos = agent.pos + np.array([0, -1, 0])
                agent_actions.append(("build", {"block_type": building_type, "position": target_pos.tolist()}))
            elif "destroy_" in action_name:
                direction = action_name.replace("destroy_", "")
                target_pos = agent.pos + ACTION_MAP_DESTROY[f"destroy_{direction}"]
                agent_actions.append(("destroy", target_pos.tolist()))
            elif "move_" in action_name:
                direction = action_name.replace("move_", "")
                target_pos = agent.pos + ACTION_MAP_MOVE[f"move_{direction}"]
                agent_actions.append(("move", target_pos.tolist()))
            else:
                agent_actions.append(("wait", None))
        
        prev_reward = env._calculate_reward()
        env._execute_actions(agent_actions)
        env._update_simulation()
        new_reward = env._calculate_reward()
        
        for i in range(len(env.agents)):
            rewards.append(new_reward - prev_reward + 0.1)  # Small positive reward for action
        
        done = env.budget < 0 or episode > 10000
        
        step_buffer.append({
            "obs": obs_t, 
            "actions": actions_t, 
            "logp": logp_t, 
            "reward": torch.tensor(rewards), 
            "done": torch.tensor([done] * len(env.agents)), 
            "value": value_t.flatten()
        })
        
        episode += 1
        total_steps += len(env.agents)

        if done:
            await websocket.send_json({"type": "debug", "message": "Episode finished. Resetting..."})
            env = SimCityEnv()
            env.trained_policy = model

        if episode % 100 == 0:
            avg_reward = float(torch.tensor(rewards).mean().item())
            await websocket.send_json({"type": "progress", "episode": episode, "reward": avg_reward, "loss": current_loss})
            await websocket.send_json({"type": "train_step", "state": env.get_state_for_viz(), "episode": episode})
            await asyncio.sleep(0.01)

        if total_steps >= BATCH_SIZE:
            # PPO Update
            with torch.no_grad():
                next_obs = [get_agent_state_vector(a, env.grid, env) for a in env.agents]
                next_masks = torch.stack([get_valid_actions_mask(a, env.grid) for a in env.agents])
                _, _, next_value = model.get_action(np.array(next_obs), valid_actions_mask=next_masks)
            
            advantages = torch.zeros(len(step_buffer), len(env.agents))
            gae = torch.zeros(len(env.agents))
            
            for t in reversed(range(len(step_buffer))):
                delta = step_buffer[t]["reward"] + GAMMA * next_value * (1 - step_buffer[t]["done"]) - step_buffer[t]["value"]
                gae = delta + GAMMA * GAE_LAMBDA * (1 - step_buffer[t]["done"]) * gae
                advantages[t] = gae
                next_value = step_buffer[t]["value"]
            
            returns = advantages + torch.stack([b["value"] for b in step_buffer])
            
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
                    pg_loss = torch.max(-b_adv[mb_idxs] * ratio, -b_adv[mb_idxs] * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)).mean()
                    v_loss = 0.5 * ((value.flatten() - b_returns[mb_idxs]).pow(2)).mean()
                    loss = pg_loss - ENT_COEF * entropy_bonus + v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            current_loss = loss.item()
            step_buffer = []
            total_steps = 0
            await websocket.send_json({"type": "debug", "message": "Completed PPO update"})

    await websocket.send_json({"type": "trained", "model_info": {"episodes": episode, "loss": current_loss}})

async def run_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket
    _current_websocket = websocket
    running = True
    
    async def receive_commands():
        nonlocal running
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
        
        reward = env._calculate_reward()
        progress_update = {
            "type": "progress",
            "episode": env.step_count,
            "reward": reward,
            "loss": None
        }
        await websocket.send_json(progress_update)
        await asyncio.sleep(0.5)
    
    if not cmd_task.done():
        cmd_task.cancel() 