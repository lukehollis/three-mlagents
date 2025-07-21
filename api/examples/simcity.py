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

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

    def get_action(self, obs: np.ndarray, action: torch.Tensor = None):
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs
        if obs_t.dim() == 1: obs_t = obs_t.unsqueeze(0)
        dist, value = self.forward(obs_t)
        if action is None: action = dist.sample()
        log_prob = dist.log_prob(action)
        if obs_t.shape[0] == 1: return action.item(), log_prob.item(), value.item()
        return action, log_prob, value

# --- Entity Classes ---
class Pedestrian: # This is our main agent
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {btype: 20 for btype in ["residential", "commercial", "industrial", "park"]}
        self.goal = "Contribute to building a vibrant and efficient city."
        self.color = [random.random(), random.random(), random.random()]
        self.llm_intent = None
        self.is_thinking = False
        self.last_llm_step = -LLM_CALL_FREQUENCY
        self.memory_stream = []

    def add_to_memory_stream(self, event: str, step: int):
        self.memory_stream.append(f"Step {step}: {event}")
        if len(self.memory_stream) > 10: self.memory_stream.pop(0)

    async def decide_action_llm(self, grid: np.ndarray, city_stats: Dict, messages: List[Dict], step_count: int):
        self.is_thinking = True
        log_to_frontend(f"🤖 Agent {self.id}: Thinking...")
        
        view_radius = 2
        x_start, x_end = max(0, self.pos[0]-view_radius), min(GRID_SIZE_X, self.pos[0]+view_radius+1)
        y_start, y_end = max(0, self.pos[1]-view_radius), min(GRID_SIZE_Y, self.pos[1]+view_radius+1)
        z_start, z_end = max(0, self.pos[2]-view_radius), min(GRID_SIZE_Z, self.pos[2]+view_radius+1)
        view = grid[x_start:x_end, y_start:y_end, z_start:z_end].tolist()

        prompt = f"""You are a construction agent (ID: {self.id}) in a 3D city.
Your goal: {self.goal}. Your position: {self.pos.tolist()}. Inventory: {self.inventory}.
City stats: {city_stats}. Recent messages: {messages[-3:]}
Your 5x5x5 view (0=air): {view}
Your available actions are "build", "destroy", "move", "talk", "wait".
- build: {{ "block_type": "...", "position": [x, y, z] }} (must be adjacent to you)
- destroy: {{ "position": [x, y, z] }} (must be adjacent)
- move: {{ "position": [x, y, z] }} (must be adjacent)
- talk: {{ "message": "..." }}
Based on your state, what is your next high-level action? Respond with valid JSON only.
Example: {{ "action": "build", "data": {{ "block_type": "residential", "position": [{self.pos[0]+1}, 1, {self.pos[2]}] }} }}"""

        response = await get_json(prompt=prompt, model="gemma3n:latest" if USE_LOCAL_OLLAMA else "anthropic/claude-sonnet-4")
        self.llm_intent = (response.get("action"), response.get("data"))
        self.is_thinking = False
        return {"agent_id": self.id, "step": step_count, "response": response}

class Car:
    def __init__(self, car_id: int, graph, start_node):
        self.id = car_id
        self.graph = graph
        self.pos = np.array(graph.nodes[start_node]['pos'])
        self.path = [start_node]
        self.edge_progress = 0.0
        self.speed = random.uniform(0.1, 0.3)
        self.color = [random.random(), 0.1, 0.1]
        self._find_new_path()

    def _find_new_path(self):
        current_node = self.path[-1]
        neighbors = list(self.graph.neighbors(current_node))
        if not neighbors: 
            self.path = [current_node] # Stuck
            return
        next_node = random.choice(neighbors)
        self.path.append(next_node)

    def move(self, traffic_lights):
        if len(self.path) < 2:
            self._find_new_path()
            return
        
        u, v = self.path[0], self.path[1]
        
        # Check traffic light at the end of the current edge (node v)
        light = traffic_lights.get(v)
        if light and light.state == 'red' and self.edge_progress > 0.9:
            return # Stop for red light
        
        self.edge_progress += self.speed
        if self.edge_progress >= 1.0:
            self.edge_progress = 0.0
            self.path.pop(0)
            if not self.path: self._find_new_path()
            u, v = self.path[0], self.path[1]

        start_pos = np.array(self.graph.nodes[u]['pos'])
        end_pos = np.array(self.graph.nodes[v]['pos'])
        self.pos = start_pos + (end_pos - start_pos) * self.edge_progress

class TrafficLight:
    def __init__(self, light_id, pos):
        self.id = light_id
        self.pos = pos
        self.state = 'red'
        self.timer = random.randint(5, 20)

    def update(self):
        self.timer -= 1
        if self.timer <= 0:
            self.state = 'green' if self.state == 'red' else 'red'
            self.timer = random.randint(10, 30) if self.state == 'green' else random.randint(40, 60)

# --- Environment Class ---
class SimCityEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z), dtype=int)
        self.agents = []
        self.cars = []
        self.traffic_lights = {}
        self.llm_logs, self.messages = [], []
        self.step_count, self.budget, self.population = 0, INITIAL_BUDGET, INITIAL_POPULATION
        self.trained_policy: "ActorCritic" = None

        log_to_frontend("Fetching street data from OpenStreetMap...")
        self._setup_map()
        self._spawn_entities()
        log_to_frontend("Street data processed and entities spawned.")

    def _setup_map(self):
        location_point = (40.7128, -74.0060) # New York City
        self.graph = ox.graph_from_point(location_point, dist=500, network_type='drive')
        
        nodes_df, edges_df = ox.graph_to_gdfs(self.graph)
        min_x, min_y = nodes_df.geometry.x.min(), nodes_df.geometry.y.min()
        max_x, max_y = nodes_df.geometry.x.max(), nodes_df.geometry.y.max()

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
            self.agents.append(Pedestrian(i, np.array([x, 1, z])))

        graph_nodes = list(self.graph.nodes)
        for i in range(NUM_CARS):
            self.cars.append(Car(i, self.graph, random.choice(graph_nodes)))

        for i, (node, data) in enumerate(self.graph.nodes(data=True)):
            if self.graph.degree(node) > 2: # Is an intersection
                self.traffic_lights[node] = TrafficLight(i, data['pos'])

    def _update_simulation(self):
        for light in self.traffic_lights.values(): light.update()
        for car in self.cars: car.move(self.traffic_lights)
        
        income = np.sum([BUILDING_TYPES[list(BUILDING_TYPES.keys())[idx-1]]['income'] for idx in self.grid.flatten() if idx > 2])
        population = np.sum([BUILDING_TYPES[list(BUILDING_TYPES.keys())[idx-1]]['population'] for idx in self.grid.flatten() if idx > 2])
        self.budget += income
        self.population = population

    def _get_reward(self, agent: Pedestrian, action: str, data: Any) -> float:
        # Simple reward: budget change + population change
        prev_budget, prev_pop = self.budget, self.population
        self._update_simulation()
        reward = (self.budget - prev_budget) + (self.population - prev_pop) * 10
        if "build" in action and data: reward += 10 # Bonus for building
        if "destroy" in action: reward -= 5
        if "wait" in action: reward -= 1
        return reward

    def _execute_agent_action(self, agent: Pedestrian, action_name: str):
        # ... (Execution logic for move, build, destroy) ...
        if action_name in ACTION_MAP_MOVE:
            target_pos = agent.pos + ACTION_MAP_MOVE[action_name]
            # Check if target is air
            if 0 <= target_pos[0] < GRID_SIZE_X and 0 <= target_pos[1] < GRID_SIZE_Y and 0 <= target_pos[2] < GRID_SIZE_Z:
                if self.grid[target_pos[0], target_pos[1], target_pos[2]] == 0:
                    agent.pos = target_pos

        elif "build_" in action_name:
            b_type = action_name.replace("build_", "")
            build_pos = agent.pos + np.array([0, -1, 0]) # Build at feet
            if agent.inventory.get(b_type, 0) > 0 and 0 <= build_pos[0] < GRID_SIZE_X and 0 <= build_pos[1] < GRID_SIZE_Y and 0 <= build_pos[2] < GRID_SIZE_Z:
                # Check if building on non-road
                if self.grid[build_pos[0], 0, build_pos[2]] != list(BUILDING_TYPES.keys()).index("road") + 1:
                    agent.inventory[b_type] -= 1
                    self.grid[build_pos[0], build_pos[1], build_pos[2]] = list(BUILDING_TYPES.keys()).index(b_type) + 1

        elif action_name in ACTION_MAP_DESTROY:
            destroy_pos = agent.pos + ACTION_MAP_DESTROY[action_name]
            if 0 <= destroy_pos[0] < GRID_SIZE_X and 0 <= destroy_pos[1] < GRID_SIZE_Y and 0 <= destroy_pos[2] < GRID_SIZE_Z:
                block_idx = self.grid[destroy_pos[0], destroy_pos[1], destroy_pos[2]]
                if block_idx > 2: # Can't destroy road or grass
                    b_type = list(BUILDING_TYPES.keys())[block_idx - 1]
                    agent.inventory[b_type] += 1
                    self.grid[destroy_pos[0], destroy_pos[1], destroy_pos[2]] = 0

    async def step(self):
        self.step_count += 1
        self._update_simulation()
        
        # LLM decisions (can run in parallel)
        llm_tasks = []
        for agent in self.agents:
            if not agent.is_thinking and (self.step_count - agent.last_llm_step) >= LLM_CALL_FREQUENCY:
                agent.last_llm_step = self.step_count
                stats = {"budget": self.budget, "population": self.population, "step": self.step_count}
                llm_tasks.append(asyncio.create_task(agent.decide_action_llm(self.grid, stats, self.messages, self.step_count)))

        # PPO policy actions (synchronous)
        agent_actions_for_env = []
        for agent in self.agents:
            if agent.llm_intent: # Prioritize LLM
                action, data = agent.llm_intent
                # ... (translate high-level LLM action to low-level if needed) ...
                agent.llm_intent = None
            
            # Fallback to policy
            state_vec = get_agent_state_vector(agent, self.grid, self)
            action_idx, _, _ = self.trained_policy.get_action(state_vec)
            action_name = DISCRETE_ACTIONS[action_idx]
            self._execute_agent_action(agent, action_name)
        
        # Finish awaiting LLM tasks
        if llm_tasks: await asyncio.gather(*llm_tasks)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "color": a.color, "type": "pedestrian"} for a in self.agents],
            "cars": [{"id": c.id, "pos": c.pos.tolist(), "color": c.color, "type": "car"} for c in self.cars],
            "traffic_lights": [{"id": l.id, "pos": l.pos, "state": l.state} for l in self.traffic_lights.values()],
            "city_stats": {"budget": self.budget, "population": self.population},
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z],
            "building_types": BUILDING_TYPES,
            "road_network": self.road_network_for_viz,
        }

def get_agent_state_vector(agent: Pedestrian, grid: np.ndarray, env: SimCityEnv) -> np.ndarray:
    """Create a fixed-size state vector for the policy."""
    # Local 3x3x3 view
    pos = agent.pos.astype(int)
    view = np.zeros((3, 3, 3))
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x, y, z = pos[0]+dx, pos[1]+dy, pos[2]+dz
                if 0 <= x < GRID_SIZE_X and 0 <= y < GRID_SIZE_Y and 0 <= z < GRID_SIZE_Z:
                    view[dx+1, dy+1, dz+1] = grid[x, y, z]
    
    inventory_vec = np.array(list(agent.inventory.values()))
    stats_vec = np.array([env.budget, env.population])
    pos_vec = agent.pos
    
    return np.concatenate([view.flatten(), inventory_vec, stats_vec, pos_vec]).astype(np.float32)

# --- PPO Hyperparameters ---
BATCH_SIZE = 512; MINI_BATCH = 64; EPOCHS = 4; GAMMA = 0.99
GAE_LAMBDA = 0.95; CLIP_EPS = 0.2; ENT_COEF = 0.01; LR = 3e-4

async def train_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket; _current_websocket = websocket
    await websocket.send_json({"type": "debug", "message": "Starting SimCity training..."})
    
    dummy_obs = get_agent_state_vector(env.agents[0], env.grid, env)
    model = ActorCritic(dummy_obs.shape[0], len(DISCRETE_ACTIONS))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env.trained_policy = model

    step_buffer: list[dict] = []
    ep_counter = 0; total_steps = 0; current_loss = None

    while ep_counter < 10000:
        env._update_simulation()
        obs_list = [get_agent_state_vector(agent, env.grid, env) for agent in env.agents]
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)

        with torch.no_grad():
            actions_t, logp_t, value_t = model.get_action(obs_t)
        
        rewards, dones = [], []
        for i, agent in enumerate(env.agents):
            action_name = DISCRETE_ACTIONS[actions_t[i].item()]
            reward = env._get_reward(agent, action_name, None)
            env._execute_agent_action(agent, action_name)
            rewards.append(reward)
            dones.append(env.budget < 0)

        step_buffer.append({"obs": obs_t, "actions": actions_t, "logp": logp_t, "reward": torch.tensor(rewards), "done": torch.tensor(dones), "value": value_t.flatten()})
        
        ep_counter += len(env.agents); total_steps += len(env.agents)

        if any(dones):
            await websocket.send_json({"type": "debug", "message": "Episode finished (Bankrupt). Resetting..."})
            env = SimCityEnv(); env.trained_policy = model

        if ep_counter % 100 == 0:
            avg_reward = float(torch.tensor(rewards).mean().item())
            await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": current_loss})
            await websocket.send_json({"type": "train_step", "state": env.get_state_for_viz(), "episode": ep_counter})
            await asyncio.sleep(0.01)

        if total_steps >= BATCH_SIZE:
            # ... (Standard PPO Update Logic) ...
            with torch.no_grad():
                next_obs = [get_agent_state_vector(a, env.grid, env) for a in env.agents]
                _, _, next_value = model.get_action(np.array(next_obs))
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
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            current_loss = loss.item(); step_buffer = []; total_steps = 0
            await websocket.send_json({"type": "debug", "message": "Completed PPO update"})

    await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": current_loss}})

async def run_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket; _current_websocket = websocket
    running = True
    async def receive_commands():
        nonlocal running
        while running:
            try:
                data = await websocket.receive_json()
                if data.get("cmd") == "stop": running = False; break
            except Exception: running = False; break
    cmd_task = asyncio.create_task(receive_commands())

    while running:
        await env.step()
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state})
        progress = {"type": "progress", "episode": env.step_count, "reward": env.population, "loss": None}
        await websocket.send_json(progress)
        await asyncio.sleep(0.2)
    
    if not cmd_task.done(): cmd_task.cancel() 