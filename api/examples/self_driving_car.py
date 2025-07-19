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
    "accelerate", "decelerate"
]

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
        self.trained_policy: "ActorCritic" = None
        
        self.location_point = (40.758896, -73.985130) # Times Square
        self.graph = ox.graph_from_point(self.location_point, dist=500, network_type='drive')
        ox.add_node_elevations_opentopodata(self.graph)
        self.graph_proj = ox.project_graph(self.graph)
        
        self.road_network_for_viz = self._get_road_network_for_viz()
        self._create_agents(NUM_AGENTS)

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
        if agent.path_index >= len(agent.path) - 1:
            return 100.0
        return agent.speed * 0.1 - 0.05

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        rewards = []
        dones = []

        for agent, (action, data) in zip(self.agents, agent_actions):
            if action == "accelerate": agent.speed += 0.5
            else: agent.speed -= 0.5
            agent.speed = np.clip(agent.speed, 0, 15)
            
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
        
        return rewards, dones

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "heading": a.heading, "pitch": a.pitch, "color": a.color, "memory_stream": a.memory_stream, "goal": a.get_goal().tolist()} for a in self.agents],
            "llm_logs": self.llm_logs,
            "messages": self.messages,
            "road_network": self.road_network_for_viz
        }

def get_agent_state_vector(agent: Agent) -> np.ndarray:
    if agent.path_index >= len(agent.path) - 1:
        return np.zeros(6)

    p1_proj = agent.graph_proj.nodes[agent.path[agent.path_index]]
    p2_proj = agent.graph_proj.nodes[agent.path[agent.path_index + 1]]

    vec_to_next = np.array([p2_proj['x'] - p1_proj['x'], p2_proj['y'] - p1_proj['y']])
    dist_to_next = np.linalg.norm(vec_to_next) - agent.distance_on_segment
    
    remaining_len = sum(agent.graph_proj[u][v][0]['length'] for i in range(agent.path_index, len(agent.path) - 1) for u,v in [(agent.path[i], agent.path[i+1])])
    remaining_len -= agent.distance_on_segment

    return np.concatenate([
        vec_to_next / (np.linalg.norm(vec_to_next) + 1e-6),
        [agent.speed / 15.0],
        [dist_to_next / 100.0],
        [remaining_len / 1000.0],
        [float(len(agent.path) - 1 - agent.path_index > 0)]
    ])

def get_valid_actions_mask(agent: Agent) -> np.ndarray:
    mask = np.ones(len(DISCRETE_ACTIONS), dtype=bool)
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
        dummy_obs = get_agent_state_vector(dummy_agent)
        obs_size = dummy_obs.shape[0]
        action_size = len(DISCRETE_ACTIONS)

        model = ActorCritic(obs_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        env.trained_policy = model

        step_buffer: list[dict] = []
        ep_counter = 0
        total_steps = 0
        current_loss = None
        
        obs_t = torch.tensor(np.array([get_agent_state_vector(agent) for agent in env.agents]), dtype=torch.float32)

        while ep_counter < EPISODES:
            agent_states = [get_agent_state_vector(agent) for agent in env.agents]
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
            
            next_obs_list = [get_agent_state_vector(agent) for agent in env.agents]
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
                    next_agent_states = [get_agent_state_vector(agent) for agent in env.agents]
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
            if env.trained_policy:
                agent_states = [get_agent_state_vector(agent) for agent in env.agents]
                obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)
                
                valid_masks = np.array([get_valid_actions_mask(agent) for agent in env.agents])
                masks_t = torch.from_numpy(valid_masks).bool()

                with torch.no_grad():
                    dist, _ = env.trained_policy(obs_t, valid_actions_mask=masks_t)
                    actions_t = dist.sample()
                
                actions_np = actions_t.cpu().numpy()
                
                agent_actions_for_env = []
                for i, agent in enumerate(env.agents):
                    action_name = DISCRETE_ACTIONS[actions_np[i]]
                    agent_actions_for_env.append((action_name, None))
            else:
                agent_actions_for_env = [(random.choice(DISCRETE_ACTIONS), None) for _ in env.agents]


            rewards, dones = env._execute_actions(agent_actions_for_env)
            
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


