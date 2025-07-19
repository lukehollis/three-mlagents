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
NUM_AGENTS = 1
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 10
USE_LOCAL_OLLAMA = True 

DISCRETE_ACTIONS = [
    "accelerate", "decelerate", "maintain_speed", "slight_left", "slight_right"
]

FEATURE_LABELS = {
    0: "Path Direction X",
    1: "Path Direction Y",
    2: "Current Speed",
    3: "Distance to Next Turn",
    4: "Total Distance Remaining",
    5: "Is On Final Segment",
    6: "Current Heading",
    7: "Heading Error",
    8: "Nearest Pedestrian Distance",
    9: "Nearest Traffic Light State"
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

    def get_local_feature_importance(self, obs: torch.Tensor, action: torch.Tensor, top_k=3):
        """Calculates local, instance-based feature importance using gradients."""
        # Ensure the observation tensor can have gradients calculated for it
        obs.requires_grad_(True)

        # Re-compute the forward pass to build the computation graph
        dist, _ = self.forward(obs)
        log_prob = dist.log_prob(action)

        # Calculate the gradients of the log-probability of the action with respect to the inputs
        # using torch.autograd.grad to avoid interfering with the main training gradients.
        grads = torch.autograd.grad(log_prob, obs)[0]
        
        feature_importance = grads.abs().squeeze()

        # We can now disable grad on obs as we are done with it.
        obs.requires_grad_(False)

        # Get the indices of the top_k most important features
        top_indices = torch.topk(feature_importance, k=top_k).indices.tolist()

        # Map indices to labels
        top_features = [FEATURE_LABELS.get(i, f"Unknown Feature {i}") for i in top_indices]
        
        return top_features

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

# --- New Entity Classes ---
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

class Pedestrian:
    def __init__(self, ped_id: int, start_pos: np.ndarray, end_pos: np.ndarray, speed: float = 1.0, initial_state: str = 'waiting'):
        self.id = ped_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.pos = start_pos.copy()
        self.speed = speed
        self.state = initial_state # 'waiting', 'crossing', 'jaywalking'
        self.path_progress = 0.0
        self.wait_timer = 0
    
    def step(self, traffic_light_state: str):
        if self.state == 'waiting':
            # Jaywalking logic
            if random.random() < 0.005: # 0.5% chance to jaywalk each step
                self.state = 'jaywalking'
                self.wait_timer = 0
                return # Start jaywalking on the next step

            if traffic_light_state == 'green': # Assuming green for cars means red for peds
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
                # Swap start and end to walk back and forth
                self.start_pos, self.end_pos = self.end_pos, self.start_pos
                self.pos = self.start_pos.copy()


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, start_node: int, goal_node: int, path: list, graph: 'networkx.MultiDiGraph', graph_proj: 'networkx.MultiDiGraph'):
        self.id = agent_id
        
        self.graph = graph
        self.graph_proj = graph_proj
        self.start_node = start_node
        self.goal_node = goal_node
        self.path = path
        self.path_index = 0
        self.distance_on_segment = 0.0
        
        self.pos = np.array([self.graph.nodes[self.path[0]]['y'], self.graph.nodes[self.path[0]]['x']])
        self.heading = 0
        self._update_heading()
        self.pitch = 0

        self.speed = 0.0
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]
        self.memory_stream = []

    def reset(self, start_node: int, goal_node: int, path: list):
        self.start_node = start_node
        self.goal_node = goal_node
        self.path = path
        self.path_index = 0
        self.distance_on_segment = 0.0
        self.pos = np.array([self.graph.nodes[self.path[0]]['y'], self.graph.nodes[self.path[0]]['x']])
        self.heading = 0
        self._update_heading()
        self.pitch = 0
        self.speed = 0.0
        self.memory_stream = []


    def _update_heading(self):
        if self.path_index < len(self.path) - 1:
            p1 = self.graph.nodes[self.path[self.path_index]]
            p2 = self.graph.nodes[self.path[self.path_index + 1]]
            vec = np.array([p2['y'] - p1['y'], p2['x'] - p1['x']])
            self.heading = np.degrees(np.arctan2(vec[1], vec[0]))

    def get_goal(self):
        return np.array([self.graph.nodes[self.goal_node]['y'], self.graph.nodes[self.goal_node]['x']])
    
    def add_to_memory_stream(self, event: str, step: int = None):
        event_entry = f"Step {step}: {event}" if step is not None else event
        self.memory_stream.append(event_entry)
        if len(self.memory_stream) > 10:
            self.memory_stream.pop(0)


# --- Environment Class ---
class SelfDrivingCarEnv:
    def __init__(self):
        self.llm_logs: List[Dict] = []
        self.messages: List[Dict] = []
        self.step_count = 0
        self.agents = []
        self.pedestrians: List[Pedestrian] = []
        self.traffic_lights: List[TrafficLight] = []
        self.trained_policy: "ActorCritic" = None
        
        self.location_point = (40.758896, -73.985130) # Times Square
        self.graph = ox.graph_from_point(self.location_point, dist=500, network_type='drive')
        try:
            google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if google_api_key:
                ox.add_node_elevations_google(self.graph, api_key=google_api_key)
            else:
                logger.warning("GOOGLE_MAPS_API_KEY not found, proceeding without elevation data.")
        except Exception as e:
            logger.error(f"Failed to get elevation data: {e}")

        self.graph_proj = ox.project_graph(self.graph)
        
        self.road_network_for_viz = self._get_road_network_for_viz()
        self._create_traffic_lights_and_pedestrians()
        self._create_agents(NUM_AGENTS)

    def _create_traffic_lights_and_pedestrians(self):
        # Find intersections (nodes with more than 2 edges)
        intersections = [node for node, degree in self.graph.degree() if degree > 2]
        
        # Place traffic lights and pedestrian crossings at some intersections
        num_to_place = min(len(intersections), 5) # Limit to 5 for performance
        selected_nodes = random.sample(intersections, num_to_place)

        for i, node_id in enumerate(selected_nodes):
            node_data = self.graph.nodes[node_id]
            light_pos = np.array([node_data['y'], node_data['x']])
            self.traffic_lights.append(TrafficLight(light_id=i, pos=light_pos))

            # Add a pedestrian crossing at this light
            # Define a simple crossing path across the intersection
            crosswalk_start = light_pos + np.array([0.0001, 0.0001]) # small offset
            crosswalk_end = light_pos - np.array([0.0001, 0.0001])
            self.pedestrians.append(Pedestrian(ped_id=i, start_pos=crosswalk_start, end_pos=crosswalk_end))

        # Add more random pedestrians along sidewalks
        ped_id_counter = len(self.pedestrians)
        all_edges = list(self.graph.edges())
        num_peds_to_add = 30
        
        if not all_edges: return

        sampled_edges = random.sample(all_edges, min(num_peds_to_add, len(all_edges)))
        
        for u, v in sampled_edges:
            p_start = np.array([self.graph.nodes[u]['y'], self.graph.nodes[u]['x']])
            p_end = np.array([self.graph.nodes[v]['y'], self.graph.nodes[v]['x']])
            
            vec = p_end - p_start
            if np.linalg.norm(vec) < 1e-6: continue
                
            perp_vec = np.array([-vec[1], vec[0]])
            perp_vec_norm = perp_vec / np.linalg.norm(perp_vec)
            
            offset_dist_degrees = 0.00004 
            
            side = random.choice([-1, 1])
            offset = side * offset_dist_degrees * perp_vec_norm
            
            sidewalk_start = p_start + offset
            
            if random.random() < 0.3: # 30% chance to be a jaywalker
                jaywalk_end = p_end - offset # Target the other side of the street
                new_ped = Pedestrian(
                    ped_id=ped_id_counter, 
                    start_pos=sidewalk_start, 
                    end_pos=jaywalk_end,
                    initial_state='jaywalking'
                )
            else: # Normal sidewalk pedestrian
                sidewalk_end = p_end + offset
                new_ped = Pedestrian(
                    ped_id=ped_id_counter,
                    start_pos=sidewalk_start,
                    end_pos=sidewalk_end
                )
                
            self.pedestrians.append(new_ped)
            ped_id_counter += 1

    def add_message(self, agent_id: int, message: str):
        """Adds a message to the environment's message list."""
        if len(self.messages) > MAX_MESSAGES:
            self.messages.pop(0)
        self.messages.append({
            "sender_id": agent_id,
            "recipient_id": None,
            "message": message,
            "step": self.step_count
        })

    def _create_agents(self, num_agents):
        self.agents = []
        for i in range(num_agents):
            agent = self._create_single_agent(i)
            self.agents.append(agent)

    def _create_single_agent(self, agent_id):
        while True:
            start_node, goal_node = random.sample(list(self.graph.nodes), 2)
            try:
                path = ox.shortest_path(self.graph, start_node, goal_node, weight='length')
                if path and len(path) > 1:
                    return Agent(agent_id, start_node, goal_node, path, self.graph, self.graph_proj)
            except Exception:
                pass

    def reset_agent(self, agent_id: int):
        while True:
            start_node, goal_node = random.sample(list(self.graph.nodes), 2)
            try:
                path = ox.shortest_path(self.graph, start_node, goal_node, weight='length')
                if path and len(path) > 1:
                    self.agents[agent_id].reset(start_node, goal_node, path)
                    return
            except Exception:
                pass

    def _get_road_network_for_viz(self):
        lines = []
        for u, v, data in self.graph.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                lines.append([[ys[i], xs[i]] for i in range(len(xs))])
        return lines

    def _get_reward(self, agent: Agent, action: str, data: Any) -> float:
        reward = 0
        
        # Big penalty for collisions or red lights
        for ped in self.pedestrians:
            if np.linalg.norm(agent.pos - ped.pos) < 0.0002: # Collision threshold
                reward -= 200
        
        # Check for red light violation
        for light in self.traffic_lights:
            # Is agent near this light?
            if np.linalg.norm(agent.pos - light.pos) < 0.0003:
                if light.state == 'red':
                    # Penalize based on speed, more for going faster through a red
                    reward -= 50 * (agent.speed + 1)

        if agent.path_index >= len(agent.path) - 1:
            return 100.0
        
        # Reward for speed, penalize for being too slow or stopped
        reward += agent.speed * 0.1 - 0.05
        
        # Penalize for turning when not necessary
        if "left" in action or "right" in action:
            reward -= 0.2
            
        return reward

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        rewards = []
        dones = []

        for agent, (action, data) in zip(self.agents, agent_actions):
            if action == "accelerate": agent.speed += 0.5
            elif action == "decelerate": agent.speed -= 0.5
            elif action == "slight_left": agent.heading -= 5
            elif action == "slight_right": agent.heading += 5
            
            agent.speed = np.clip(agent.speed, 0, 15)
            agent.heading %= 360

            if agent.path_index >= len(agent.path) - 1:
                agent.speed = 0
                rewards.append(self._get_reward(agent, action, data))
                dones.append(True)
                continue

            dist_to_move = agent.speed
            agent.distance_on_segment += dist_to_move
            
            while agent.path_index < len(agent.path) - 1:
                p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
                p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 1]]
                segment_len = np.linalg.norm(np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']]))

                if agent.distance_on_segment >= segment_len:
                    agent.distance_on_segment -= segment_len
                    agent.path_index += 1
                    agent._update_heading()
                else:
                    break
            
            if agent.path_index >= len(agent.path) - 1:
                agent.path_index = len(agent.path) - 1
                agent.pos = np.array([self.graph.nodes[agent.path[-1]]['y'], self.graph.nodes[agent.path[-1]]['x']])
                agent.speed = 0
                agent.add_to_memory_stream("Goal reached!", self.step_count)
                rewards.append(self._get_reward(agent, action, data))
                dones.append(True)
                continue

            p1_geo = self.graph.nodes[agent.path[agent.path_index]]
            p2_geo = self.graph.nodes[agent.path[agent.path_index+1]]
            vec_geo = np.array([p2_geo['y'] - p1_geo['y'], p2_geo['x'] - p1_geo['x']])
            
            p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
            p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index+1]]
            seg_len_proj = np.linalg.norm(np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']]))

            dx = p2_proj['x'] - p1_proj['x']
            dy = p2_proj['y'] - p1_proj['y']
            dz = p2_geo.get('elevation', 0) - p1_geo.get('elevation', 0)
            d_xy = np.sqrt(dx**2 + dy**2)
            pitch_rad = np.arctan2(dz, d_xy)
            agent.pitch = np.degrees(pitch_rad)

            ratio = agent.distance_on_segment / seg_len_proj if seg_len_proj > 0 else 0
            agent.pos = np.array([p1_geo['y'], p1_geo['x']]) + ratio * vec_geo

            agent.add_to_memory_stream(f"{action}, Speed: {agent.speed:.1f}", self.step_count)
            rewards.append(self._get_reward(agent, action, data))
            dones.append(False)
        
        # Update traffic lights and pedestrians
        for light in self.traffic_lights:
            light.step()
        
        for ped in self.pedestrians:
            # Find closest light to determine state
            # A real system would link them, but this is a simple approximation
            closest_light = min(self.traffic_lights, key=lambda l: np.linalg.norm(l.pos - ped.pos)) if self.traffic_lights else None
            light_state = closest_light.state if closest_light else 'red'
            ped.step(light_state)

        return rewards, dones

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "heading": a.heading, "pitch": a.pitch, "color": a.color, "memory_stream": a.memory_stream, "goal": a.get_goal().tolist()} for a in self.agents],
            "llm_logs": self.llm_logs,
            "messages": self.messages,
            "road_network": self.road_network_for_viz,
            "pedestrians": [{"id": p.id, "pos": p.pos.tolist(), "state": p.state} for p in self.pedestrians],
            "traffic_lights": [{"id": l.id, "pos": l.pos.tolist(), "state": l.state} for l in self.traffic_lights]
        }

def get_agent_state_vector(agent: Agent, env: "SelfDrivingCarEnv") -> np.ndarray:
    if agent.path_index >= len(agent.path) - 1:
        return np.zeros(10) # Updated size

    p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
    p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 1]]

    vec_to_next = np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']])
    heading_to_next = np.degrees(np.arctan2(vec_to_next[1], vec_to_next[0]))
    
    dist_to_next = np.linalg.norm(vec_to_next) - agent.distance_on_segment
    
    remaining_len = sum(agent.graph_proj[u][v][0]['length'] for i in range(agent.path_index, len(agent.path) - 1) for u,v in [(agent.path[i], agent.path[i+1])])
    remaining_len -= agent.distance_on_segment

    # Normalize heading difference
    heading_diff = (heading_to_next - agent.heading + 180) % 360 - 180
    
    # Find nearest pedestrian and traffic light
    nearest_ped = min(env.pedestrians, key=lambda p: np.linalg.norm(agent.pos - p.pos)) if env.pedestrians else None
    dist_to_ped = np.linalg.norm(agent.pos - nearest_ped.pos) if nearest_ped else 1.0 # Normalized distance
    
    nearest_light = min(env.traffic_lights, key=lambda l: np.linalg.norm(agent.pos - l.pos)) if env.traffic_lights else None
    light_state = 1.0 if nearest_light and nearest_light.state == 'green' else 0.0

    return np.concatenate([
        vec_to_next / (np.linalg.norm(vec_to_next) + 1e-6),
        [agent.speed / 15.0],
        [dist_to_next / 100.0],
        [remaining_len / 1000.0],
        [float(len(agent.path) - 1 - agent.path_index > 0)],
        [agent.heading / 360.0],
        [heading_diff / 180.0],
        [dist_to_ped],
        [light_state]
    ])

def get_valid_actions_mask(agent: Agent) -> np.ndarray:
    mask = np.ones(len(DISCRETE_ACTIONS), dtype=bool)
    
    # Logic to disable turning if heading is already correct
    if agent.path_index < len(agent.path) - 1:
        p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
        p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 1]]
        vec_to_next = np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']])
        heading_to_next = np.degrees(np.arctan2(vec_to_next[1], vec_to_next[0]))
        heading_diff = abs((heading_to_next - agent.heading + 180) % 360 - 180)

        if heading_diff < 5:  # If heading is mostly correct
            mask[DISCRETE_ACTIONS.index("slight_left")] = False
            mask[DISCRETE_ACTIONS.index("slight_right")] = False
        else: # If turning is needed, maybe don't accelerate
            mask[DISCRETE_ACTIONS.index("accelerate")] = False

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
EPISODES = 1000

async def train_self_driving_car(websocket: WebSocket, env: SelfDrivingCarEnv):
    global _current_websocket
    _current_websocket = websocket
    
    try:
        env = SelfDrivingCarEnv()
        dummy_agent = env.agents[0]
        dummy_obs = get_agent_state_vector(dummy_agent, env)
        obs_size = dummy_obs.shape[0]
        action_size = len(DISCRETE_ACTIONS)

        model = ActorCritic(obs_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        env.trained_policy = model

        step_buffer: list[dict] = []
        ep_counter = 0
        total_steps = 0
        current_loss = None
        
        obs_t = torch.tensor(np.array([get_agent_state_vector(agent, env) for agent in env.agents]), dtype=torch.float32)

        while ep_counter < EPISODES:
            agent_states = [get_agent_state_vector(agent, env) for agent in env.agents]
            obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)
            
            valid_masks = np.array([get_valid_actions_mask(agent) for agent in env.agents])
            masks_t = torch.from_numpy(valid_masks).bool()

            with torch.no_grad():
                dist, value = model(obs_t, valid_actions_mask=masks_t)
                actions_t = dist.sample()
                logp_t = dist.log_prob(actions_t)
                
            actions_np = actions_t.cpu().numpy()
            
            agent_actions_for_env = []
            for i, agent in enumerate(env.agents):
                action_name = DISCRETE_ACTIONS[actions_np[i]]
                agent_actions_for_env.append((action_name, None))

            if model:
                top_features = model.get_local_feature_importance(obs_t, actions_t)
                explanation = f"Action: {DISCRETE_ACTIONS[actions_np[0]]}, Causes: {', '.join(top_features)}"
                env.add_message(agent_id=env.agents[0].id, message=explanation)

            rewards, dones = env._execute_actions(agent_actions_for_env)
            
            step_buffer.append({
                "obs": obs_t, 
                "actions": actions_t, 
                "logp": logp_t, 
                "reward": torch.tensor(rewards, dtype=torch.float32), 
                "done": torch.tensor(dones, dtype=torch.float32), 
                "value": value.flatten(),
                "mask": masks_t
            })
            
            env.step_count += 1
            total_steps += len(env.agents)

            for i, done in enumerate(dones):
                if done:
                    ep_counter += 1
                    env.reset_agent(i)
            
            next_obs_list = [get_agent_state_vector(agent, env) for agent in env.agents]
            obs_t = torch.tensor(np.array(next_obs_list), dtype=torch.float32)


            if env.step_count % 16 == 0:
                current_reward = float(torch.tensor(rewards, dtype=torch.float32).mean().cpu().item())
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": current_reward, "loss": current_loss})

            if env.step_count % 8 == 0:
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
                await asyncio.sleep(0.01)

            if total_steps >= BATCH_SIZE:
                with torch.no_grad():
                    next_agent_states = [get_agent_state_vector(agent, env) for agent in env.agents]
                    next_obs_t = torch.tensor(np.array(next_agent_states), dtype=torch.float32)
                    _, next_value_t = model(next_obs_t)
                    next_value = next_value_t.squeeze()

                num_steps = len(step_buffer)
                
                values = torch.stack([b["value"] for b in step_buffer])
                rewards = torch.stack([b["reward"] for b in step_buffer])
                dones = torch.stack([b["done"] for b in step_buffer])
                
                advantages = torch.zeros_like(rewards)
                gae = 0.0
                for t in reversed(range(num_steps)):
                    delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
                    gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
                    advantages[t] = gae
                    next_value = values[t]
                
                returns = advantages + values
                
                b_obs = torch.cat([b["obs"] for b in step_buffer])
                b_actions = torch.cat([b["actions"] for b in step_buffer])
                b_logp = torch.cat([b["logp"] for b in step_buffer])
                b_adv = advantages.flatten()
                b_returns = returns.flatten()
                b_masks = torch.cat([b["mask"] for b in step_buffer])
                
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

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
                        
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        v_loss = 0.5 * ((value.flatten() - mb_returns).pow(2)).mean()
                        
                        loss = pg_loss - ENT_COEF * entropy_bonus + v_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    await asyncio.sleep(0)

                avg_reward = float(torch.stack([b["reward"] for b in step_buffer]).mean().cpu().item())
                current_loss = loss.item()
                
                step_buffer = []
                total_steps = 0
                
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": current_loss})

        await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item()}})
    except Exception as e:
        logger.error(f"Error during SelfDrivingCar training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"})


# --- Websocket runner ---
async def run_self_driving_car(websocket: WebSocket, env: SelfDrivingCarEnv):
    global _current_websocket
    _current_websocket = websocket
    
    running = True

    async def receive_commands():
        nonlocal running
        while running:
            try:
                data = await websocket.receive_text()
                if "stop" in data: running = False
            except Exception:
                running = False
            
    cmd_task = asyncio.create_task(receive_commands())

    while running:
        try:
            agent_actions_for_env = []
            if env.trained_policy:
                agent_states = [get_agent_state_vector(agent, env) for agent in env.agents]
                obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)
                
                valid_masks = np.array([get_valid_actions_mask(agent) for agent in env.agents])
                masks_t = torch.from_numpy(valid_masks).bool()

                with torch.no_grad():
                    dist, _ = env.trained_policy(obs_t, valid_actions_mask=masks_t)
                    actions_t = dist.sample()
                
                actions_np = actions_t.cpu().numpy()
                
                for i, agent in enumerate(env.agents):
                    action_name = DISCRETE_ACTIONS[actions_np[i]]
                    agent_actions_for_env.append((action_name, None))
                
                top_features = env.trained_policy.get_local_feature_importance(obs_t, actions_t)
                explanation = f"Action: {DISCRETE_ACTIONS[actions_np[0]]}, Causes: {', '.join(top_features)}"
                env.add_message(agent_id=env.agents[0].id, message=explanation)

            else:
                for agent in env.agents:
                    action_name = random.choice(DISCRETE_ACTIONS)
                    agent_actions_for_env.append((action_name, None))
                env.add_message(agent_id=env.agents[0].id, message="Action: Random, Causes: No policy trained yet.")


            rewards, dones = env._execute_actions(agent_actions_for_env)
            
            env.step_count += 1
            for i, done in enumerate(dones):
                if done:
                    env.reset_agent(i)

            state = env.get_state_for_viz()
            await websocket.send_json({"type": "run_step", "state": state})
            reward = sum(rewards) # Sum rewards for all agents
            await websocket.send_json({
                "type": "progress", "episode": env.step_count,
                "reward": reward, "loss": None
            })
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error during self-driving car run: {e}", exc_info=True)
            running = False

    if not cmd_task.done():
        cmd_task.cancel() 


