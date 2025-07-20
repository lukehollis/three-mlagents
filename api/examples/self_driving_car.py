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
NUM_AGENTS = 1
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 10
USE_LOCAL_OLLAMA = True 
MAX_EPISODE_STEPS = 1000

DISCRETE_ACTIONS = [
    "accelerate", "decelerate", "maintain_speed", "slight_left", "slight_right"
]

# --- New Feature Set (64 dimensions) ---
FEATURE_LABELS = {}
# Agent Kinematics (5)
FEATURE_LABELS.update({
    0: "Speed", 1: "Acceleration", 2: "Heading", 3: "Angular Velocity", 4: "Pitch"
})
# Path & Navigation (13)
FEATURE_LABELS.update({
    5: "Dist to Next Waypoint", 6: "Vec to Next Waypoint X", 7: "Vec to Next Waypoint Y",
    8: "Heading Error to Waypoint", 9: "Total Dist Remaining on Path", 10: "Is on Final Segment",
    11: "Path Curvature at Waypoint+1", 12: "Path Curvature at Waypoint+2",
    13: "Upcoming Elevation Change", 14: "Current Road Speed Limit",
    15: "Dist to Goal (Air)", 16: "Vec to Goal X", 17: "Vec to Goal Y"
})
# Nearby Traffic Lights (4 * 4 = 16)
for i in range(4):
    base = 18 + i * 4
    FEATURE_LABELS.update({
        base + 0: f"Light {i+1} Dist", base + 1: f"Light {i+1} Vec X",
        base + 2: f"Light {i+1} Vec Y", base + 3: f"Light {i+1} State"
    })
# Nearby Pedestrians (6 * 5 = 30)
for i in range(6):
    base = 18 + 16 + i * 5
    FEATURE_LABELS.update({
        base + 0: f"Ped {i+1} Dist", base + 1: f"Ped {i+1} Vec X", base + 2: f"Ped {i+1} Vec Y",
        base + 3: f"Ped {i+1} Speed", base + 4: f"Ped {i+1} State"
    })


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

    def get_local_feature_importance(self, obs: torch.Tensor, action: torch.Tensor, top_k=5):
        """Calculates local, instance-based feature importance using gradients."""
        # Ensure the observation tensor can have gradients calculated for it
        obs.requires_grad_(True)

        # Re-compute the forward pass to build the computation graph for the specific observation
        dist, _ = self.forward(obs)
        log_prob = dist.log_prob(action)

        # Calculate the gradients of the log-probability of the action with respect to the inputs
        grads = torch.autograd.grad(log_prob, obs)[0]
        
        feature_importance_scores = grads.abs().squeeze()
        obs_values = obs.squeeze()

        # We can now disable grad on obs as we are done with it.
        obs.requires_grad_(False)

        # Get the indices of the top_k most important features
        top_scores, top_indices = torch.topk(feature_importance_scores, k=top_k)

        # Sum of the top_k scores for normalization
        total_importance = torch.sum(top_scores)
        
        feature_details = []
        if total_importance > 0:
            for i in range(top_k):
                index = top_indices[i].item()
                importance_score = top_scores[i].item()
                percentage = (importance_score / total_importance) * 100
                feature_details.append({
                    "feature": FEATURE_LABELS.get(index, f"Unknown Feature {index}"),
                    "value": obs_values[index].item(),
                    "importance": importance_score,
                    "percentage": percentage
                })
        
        return feature_details

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
        
        self._set_new_path(start_node, goal_node, path)
        
        self.pos = np.array([self.graph.nodes[self.path[0]]['y'], self.graph.nodes[self.path[0]]['x']])
        self.heading = 0
        self.pitch = 0
        self.acceleration = 0.0
        self.angular_velocity = 0.0

        self.speed = 0.0
        self.color = random.choice(RETRO_SCIFI_COLORS)
        self.memory_stream = []
        self.episode_steps = 0

    def _set_new_path(self, start_node: int, goal_node: int, path: list):
        self.start_node = start_node
        self.goal_node = goal_node
        self.path = path
        self.path_index = 0
        self.distance_on_segment = 0.0

        # Pre-calculate projected path lengths for efficiency
        self.path_segment_lengths_proj = [self.graph_proj[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])]
        self.total_path_len_proj = sum(self.path_segment_lengths_proj)
        self.traveled_dist_proj = 0.0

    def reset(self, start_node: int, goal_node: int, path: list):
        self._set_new_path(start_node, goal_node, path)
        self.pos = np.array([self.graph.nodes[self.path[0]]['y'], self.graph.nodes[self.path[0]]['x']])
        self.heading = 0
        self.pitch = 0
        self.acceleration = 0.0
        self.angular_velocity = 0.0
        self.speed = 0.0
        self.memory_stream = []
        self.episode_steps = 0

    def _calculate_remaining_len(self):
        """Calculates the total remaining distance along the agent's path."""
        if self.path_index >= len(self.path) - 1:
            return 0.0
        
        # Sum of lengths of remaining future segments
        remaining_segments_len = sum(self.path_segment_lengths_proj[self.path_index+1:])
        
        # Length of current segment not yet traveled
        current_segment_remaining = self.path_segment_lengths_proj[self.path_index] - self.distance_on_segment
        
        return remaining_segments_len + current_segment_remaining

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

    def _get_reward(self, agent: Agent, action: str, data: Any, progress_made: float) -> float:
        reward = 0
        
        # REWARD: Progress towards goal (Increased)
        # The primary reward is for making progress along the path.
        reward += progress_made * 0.2

        # PENALTY: Collisions (less harsh)
        for ped in self.pedestrians:
            if np.linalg.norm(agent.pos - ped.pos) < 0.0002: # Collision threshold
                reward -= 50
        
        # PENALTY: Red light violation (less harsh, and only if moving)
        for light in self.traffic_lights:
            if np.linalg.norm(agent.pos - light.pos) < 0.0003:
                if light.state == 'red' and agent.speed > 1.0:
                    reward -= 20

        # REWARD: Reaching goal (Increased)
        if agent.path_index >= len(agent.path) - 1:
            return 200.0 # Large terminal reward
        
        # PENALTY: Time step penalty to encourage finishing the episode.
        reward -= 0.1
        
        # PENALTY: Unnecessary Turning
        # Penalize for turning when not necessary to encourage smooth driving.
        if "left" in action or "right" in action:
            reward -= 0.2
            
        return reward

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        rewards = []
        dones = []

        for agent, (action, data) in zip(self.agents, agent_actions):
            agent.episode_steps += 1
            # Store state before action
            last_speed = agent.speed
            last_heading = agent.heading
            old_remaining_len = agent._calculate_remaining_len()

            if action == "accelerate": agent.speed += 0.5
            elif action == "decelerate": agent.speed -= 0.5
            elif action == "slight_left": agent.heading -= 5
            elif action == "slight_right": agent.heading += 5
            
            agent.speed = np.clip(agent.speed, 0, 15)
            agent.heading %= 360

            # Update derivatives
            agent.acceleration = agent.speed - last_speed
            heading_change = (agent.heading - last_heading + 180) % 360 - 180
            agent.angular_velocity = heading_change

            if agent.path_index >= len(agent.path) - 1:
                agent.speed = 0
                progress = old_remaining_len - agent._calculate_remaining_len()
                rewards.append(self._get_reward(agent, action, data, progress))
                dones.append(True)
                continue

            if agent.episode_steps > MAX_EPISODE_STEPS:
                agent.add_to_memory_stream(f"Episode timed out after {MAX_EPISODE_STEPS} steps.", self.step_count)
                rewards.append(-20.0)  # Timeout penalty
                dones.append(True)
                continue

            dist_to_move = agent.speed
            agent.distance_on_segment += dist_to_move
            
            while agent.path_index < len(agent.path) - 1:
                segment_len = agent.path_segment_lengths_proj[agent.path_index]

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
                progress = old_remaining_len - agent._calculate_remaining_len()
                rewards.append(self._get_reward(agent, action, data, progress))
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
            progress = old_remaining_len - agent._calculate_remaining_len()
            rewards.append(self._get_reward(agent, action, data, progress))
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
    """
    Generate the 64-dimensional feature vector for the agent.
    """
    if agent.path_index >= len(agent.path) - 1:
        return np.zeros(64)

    # 1. Agent Kinematics (5 features)
    kinematics_features = np.array([
        agent.speed / 15.0,  # Max speed normalization
        agent.acceleration / 0.5,  # Max acceleration per step normalization
        agent.heading / 360.0,
        agent.angular_velocity / 5.0,  # Max turn degrees per step normalization
        np.clip(agent.pitch / 45.0, -1.0, 1.0) # Pitch normalization
    ])

    # 2. Path and Navigation (13 features)
    path_features = np.zeros(13)
    # Waypoint features
    p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
    p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 1]]
    vec_to_next = np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']])
    dist_to_next = np.linalg.norm(vec_to_next) - agent.distance_on_segment
    vec_to_next_norm = vec_to_next / (np.linalg.norm(vec_to_next) + 1e-6)
    heading_to_next = np.degrees(np.arctan2(vec_to_next_norm[1], vec_to_next_norm[0]))
    heading_error = (heading_to_next - agent.heading + 180) % 360 - 180
    
    # Path features
    remaining_len = sum(agent.graph_proj[u][v][0]['length'] for i in range(agent.path_index, len(agent.path) - 1) for u,v in [(agent.path[i], agent.path[i+1])]) - agent.distance_on_segment
    is_final_segment = float(agent.path_index == len(agent.path) - 2)

    # Curvature features
    def get_path_curvature(p_current, p_next, p_future):
        vec1 = np.array([p_next['x'] - p_current['x'], p_next['y'] - p_current['y']])
        vec2 = np.array([p_future['x'] - p_next['x'], p_future['y'] - p_next['y']])
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
        dot_product = np.dot(vec1_norm, vec2_norm)
        return (1.0 - np.clip(dot_product, -1.0, 1.0)) / 2.0 # Normalize to [0, 1]

    curvature1 = 0.0
    if agent.path_index < len(agent.path) - 2:
        p3_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 2]]
        curvature1 = get_path_curvature(p1_proj, p2_proj, p3_proj)

    curvature2 = 0.0
    if agent.path_index < len(agent.path) - 3:
        p3_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 2]]
        p4_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 3]]
        curvature2 = get_path_curvature(p2_proj, p3_proj, p4_proj)

    # Elevation
    p1_geo = agent.graph.nodes[agent.path[agent.path_index]]
    p2_geo = agent.graph.nodes[agent.path[agent.path_index + 1]]
    elevation_change = p2_geo.get('elevation', 0) - p1_geo.get('elevation', 0)

    # Goal features
    goal_pos = np.array([agent.graph.nodes[agent.goal_node]['y'], agent.graph.nodes[agent.goal_node]['x']])
    dist_to_goal = np.linalg.norm(goal_pos - agent.pos)
    vec_to_goal = (goal_pos - agent.pos) / (dist_to_goal + 1e-6)

    path_features = np.array([
        dist_to_next / 100.0, vec_to_next_norm[0], vec_to_next_norm[1],
        heading_error / 180.0, remaining_len / 1000.0, is_final_segment,
        curvature1, curvature2,
        np.clip(elevation_change / 10, -1.0, 1.0), 50.0 / 100.0, # Placeholder speed limit
        dist_to_goal / 0.01, vec_to_goal[0], vec_to_goal[1]
    ])

    # 3. Nearby Traffic Lights (4 lights * 4 features = 16 features)
    all_lights = sorted(env.traffic_lights, key=lambda l: np.linalg.norm(agent.pos - l.pos))
    light_features = []
    for i in range(4):
        if i < len(all_lights):
            light = all_lights[i]
            dist = np.linalg.norm(agent.pos - light.pos)
            vec = (light.pos - agent.pos) / (dist + 1e-6)
            state = 1.0 if light.state == 'green' else 0.0
            light_features.extend([min(dist / 0.01, 1.0), vec[0], vec[1], state])
        else:
            light_features.extend([1.0, 0, 0, -1.0])  # Padding

    # 4. Nearby Pedestrians (6 peds * 5 features = 30 features)
    all_peds = sorted(env.pedestrians, key=lambda p: np.linalg.norm(agent.pos - p.pos))
    ped_features = []
    state_map = {'waiting': 0, 'crossing': 1, 'jaywalking': 2}
    for i in range(6):
        if i < len(all_peds):
            ped = all_peds[i]
            dist = np.linalg.norm(agent.pos - ped.pos)
            vec = (ped.pos - agent.pos) / (dist + 1e-6)
            state = state_map.get(ped.state, 0)
            ped_features.extend([min(dist / 0.01, 1.0), vec[0], vec[1], ped.speed / 2.0, state / 2.0])
        else:
            ped_features.extend([1.0, 0, 0, 0, -1.0]) # Padding

    return np.concatenate([
        kinematics_features,
        path_features,
        np.array(light_features),
        np.array(ped_features)
    ]).astype(np.float32)

def get_valid_actions_mask(agent: Agent, env: "SelfDrivingCarEnv") -> np.ndarray:
    mask = np.ones(len(DISCRETE_ACTIONS), dtype=bool)
    
    # --- Heading Alignment Logic ---
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

    # --- Obstacle Avoidance Logic ---
    # Agent's current heading vector (using bearing, 0=North)
    heading_rad = np.radians(agent.heading)
    # Vector is [y, x] to match position format
    agent_heading_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)]) 

    # Check for red lights ahead
    for light in env.traffic_lights:
        if light.state == 'red':
            dist = np.linalg.norm(agent.pos - light.pos)
            # Check if light is close and in front of the agent
            if dist < 0.0003: # Approx 33 meters
                vec_to_light = light.pos - agent.pos
                vec_to_light_norm = vec_to_light / (np.linalg.norm(vec_to_light) + 1e-6)
                # Check if the light is roughly in the forward cone
                if np.dot(agent_heading_vec, vec_to_light_norm) > 0.7: 
                    mask[DISCRETE_ACTIONS.index("accelerate")] = False
                    break # One obstacle is enough

    # Check for pedestrians ahead, only if acceleration is still possible
    if mask[DISCRETE_ACTIONS.index("accelerate")]:
        for ped in env.pedestrians:
            dist = np.linalg.norm(agent.pos - ped.pos)
            # Check if pedestrian is close and in front
            if dist < 0.00025: # Approx 27 meters
                vec_to_ped = ped.pos - agent.pos
                vec_to_ped_norm = vec_to_ped / (np.linalg.norm(vec_to_ped) + 1e-6)
                if np.dot(agent_heading_vec, vec_to_ped_norm) > 0.7:
                    mask[DISCRETE_ACTIONS.index("accelerate")] = False
                    break
    return mask

# --- PPO Hyperparameters ---
BATCH_SIZE = 512
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.02
LR = 3e-4

# making this super low
EPISODES = 64

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
            
            valid_masks = np.array([get_valid_actions_mask(agent, env) for agent in env.agents])
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
                top_features_list = model.get_local_feature_importance(obs_t[[0]], actions_t[[0]])
                causes_str = ', '.join([f"{f['feature']} ({f['percentage']:.0f}%)" for f in top_features_list])
                explanation = f"Action: {DISCRETE_ACTIONS[actions_np[0]]}, Causes: {causes_str}"
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


            # if env.step_count % 64 == 0:
            #     current_reward = float(torch.tensor(rewards, dtype=torch.float32).mean().cpu().item())
            #     await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": current_reward, "loss": current_loss})

            #     state = env.get_state_for_viz()
            #     await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
            # print(f"Step {env.step_count} - Total Steps: {total_steps}")

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
                
                valid_masks = np.array([get_valid_actions_mask(agent, env) for agent in env.agents])
                masks_t = torch.from_numpy(valid_masks).bool()

                with torch.no_grad():
                    dist, _ = env.trained_policy(obs_t, valid_actions_mask=masks_t)
                    actions_t = dist.sample()
                
                actions_np = actions_t.cpu().numpy()
                
                for i, agent in enumerate(env.agents):
                    action_name = DISCRETE_ACTIONS[actions_np[i]]
                    agent_actions_for_env.append((action_name, None))
                
                top_features_list = env.trained_policy.get_local_feature_importance(obs_t[[0]], actions_t[[0]])
                causes_str = ', '.join([f"{f['feature']} ({f['percentage']:.0f}%)" for f in top_features_list])
                explanation = f"Action: {DISCRETE_ACTIONS[actions_np[0]]}, Causes: {causes_str}"
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


