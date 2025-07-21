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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_current_websocket.send_json({
                "type": "debug", 
                "message": message
            }))

# --- Constants ---
GRID_SIZE_X = 50
GRID_SIZE_Z = 50
NUM_AGENTS = 4 # Start with one "Mayor" agent
AGENT_ROLES = ["Mayor", "Lead Developer", "Citizen Advocate", "Industrial Magnate"]
INITIAL_BUDGET = 100000
INITIAL_POPULATION = 0

BUILDING_TYPES = {
    "road": {"cost": 50, "income": 0, "population": 0, "color": [0.2, 0.2, 0.2], "model": "road"},
    "residential": {"cost": 500, "income": 10, "population": 50, "color": [0.2, 0.8, 0.2], "model": "house"},
    "commercial": {"cost": 1000, "income": 50, "population": 0, "color": [0.2, 0.2, 0.8], "model": "shop"},
    "industrial": {"cost": 1500, "income": 100, "population": 0, "color": [0.8, 0.8, 0.2], "model": "factory"},
    "park": {"cost": 200, "income": 0, "population": 0, "color": [0.1, 0.5, 0.1], "model": "park"}
}
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 5
USE_LOCAL_OLLAMA = True

# Simplified discrete actions: build each type of building. The location will be chosen by the LLM.
DISCRETE_ACTIONS = [f"build_{btype}" for btype in BUILDING_TYPES.keys()] + ["talk", "wait"]


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
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

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        dist, value = self.forward(obs_t)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)

        if obs_t.shape[0] == 1:
            return action.item(), log_prob.item(), value.item()
        
        return action, log_prob, value


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.role = AGENT_ROLES[agent_id % len(AGENT_ROLES)]
        self.goal = self.get_initial_goal()
        self.llm_intent = None
        self.is_thinking = False
        self.last_llm_step = -10
        self.memory_vector = np.zeros(384, dtype=np.float32)
        self.memory_stream = []

    def get_initial_goal(self):
        if self.role == "Mayor":
            return "Balance the budget, grow the population, and keep all stakeholders happy."
        elif self.role == "Lead Developer":
            return "Build a balanced and aesthetically pleasing mix of residential, commercial, and park zones."
        elif self.role == "Citizen Advocate":
            return "Advocate for more parks and residential zones, and campaign against industrial pollution."
        elif self.role == "Industrial Magnate":
            return "Expand the industrial sector to maximize city income and efficiency."
        return "Help the city grow."

    def update_memory(self, text: str):
        """Update agent's memory with a new text embedding using a moving average"""
        new_embedding = get_embedding(text)
        self.memory_vector = (self.memory_vector * 0.9) + (new_embedding.astype(np.float32) * 0.1)

    def add_to_memory_stream(self, event: str, step: int = None):
        if step is not None:
            event_entry = f"Step {step}: {event}"
        else:
            event_entry = event
        self.memory_stream.append(event_entry)
        if len(self.memory_stream) > 10:
            self.memory_stream = self.memory_stream[-10:]

    async def decide_action_llm(self, grid: np.ndarray, city_stats: Dict, messages: List[Dict], step_count: int):
        self.is_thinking = True
        log_to_frontend(f"🤖 Agent {self.id} ({self.role}): Thinking...")

        # Create a simplified representation of the grid for the LLM
        grid_summary = []
        for x in range(0, GRID_SIZE_X, 5):
            row_summary = []
            for z in range(0, GRID_SIZE_Z, 5):
                sub_grid = grid[x:x+5, z:z+5]
                unique, counts = np.unique(sub_grid, return_counts=True)
                if len(unique) == 1 and unique[0] == 0:
                    row_summary.append("E") # Empty
                else:
                    row_summary.append("B") # Buildings
            grid_summary.append("".join(row_summary))

        system_prompt = f"You are an agent in a city simulation. Your role is: {self.role}. Your ID is {self.id}."
        
        template = f"""
        Example for build:
        {{
            "action": "build",
            "data": {{ "building_type": "residential", "position": [10, 12] }}
        }}
        Example for talk:
        {{
            "action": "talk",
            "data": {{ "message": "We need more parks for our citizens!" }}
        }}
        """

        prompt = f"""You are an agent in a city simulation.
Your role: {self.role} (ID: {self.id})
Your primary goal: {self.goal}
Current city stats: {json.dumps(city_stats)}
Recent messages from other agents: {json.dumps(messages[-5:])}
Building options: {json.dumps(BUILDING_TYPES)}

The city map is {GRID_SIZE_X}x{GRID_SIZE_Z}. Here is a low-resolution summary (E=Empty, B=Built):
{chr(10).join(grid_summary)}

Your available actions are "build" or "talk".
- build: requires {{ "building_type": string, "position": [x, z] }}. Use this to advance your goals.
- talk: requires {{ "message": string }}. Use this to coordinate, debate, or state your intentions to other agents.

Based on your role, goal, and the current city state, what is your next single action?

CRITICAL: You MUST respond with valid JSON only. No explanations.
{template}
Pick ONE action and respond in the exact JSON format shown above.
"""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["build", "talk"]},
                "data": {
                    "type": "object",
                    "properties": {
                        "building_type": {"type": "string", "enum": list(BUILDING_TYPES.keys())},
                        "position": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                        "message": {"type": "string"}
                    }
                }
            },
            "required": ["action", "data"]
        }
        
        model_to_use = "gemma3n:latest" if USE_LOCAL_OLLAMA else "anthropic/claude-sonnet-4"
        response = await asyncio.wait_for(
            get_json(
                system_prompt=system_prompt,
                prompt=prompt,
                model=model_to_use,
                response_schema=action_schema,
                schema_name="sim_city_action",
                should_use_ollama=USE_LOCAL_OLLAMA
            ),
            timeout=20.0
        )

        self.llm_intent = (response.get("action"), response.get("data"))
        action_desc = response.get("action")
        if action_desc == 'build':
            log_to_frontend(f"🧠 Agent {self.id} ({self.role}): LLM decided to {action_desc} a {self.llm_intent[1].get('building_type')}")
        elif action_desc == 'talk':
             log_to_frontend(f"🧠 Agent {self.id} ({self.role}): LLM decided to {action_desc}: {self.llm_intent[1].get('message')}")
        
        # If the action was "talk", we can use that as the conversational message
        conversation_text = ""
        if response.get("action") == "talk":
            conversation_text = response.get("data", {}).get("message")
        
        if _current_websocket and conversation_text:
            await _current_websocket.send_json({
                "type": "agent_message",
                "agent_id": self.id,
                "role": self.role,
                "message": conversation_text,
                "step": step_count
            })

        self.is_thinking = False
        log_entry = {
            "agent_id": self.id, 
            "step": step_count, 
            "prompt": "...", # Truncate for brevity
            "response": response,
            "conversation": conversation_text
        }
        return log_entry

    def get_fast_action(self, trained_policy: "ActorCritic", grid: np.ndarray, city_stats: Dict) -> Tuple[str, Any]:
        # --- LLM Intent First ---
        if self.llm_intent:
            action, data = self.llm_intent
            self.llm_intent = None # Consume intent
            log_to_frontend(f"🧠 Agent {self.id} ({self.role}): Acting on LLM intent.")
            return (action, data)

        # --- Policy Decision ---
        if trained_policy:
            log_to_frontend(f"🧮 Agent {self.id} ({self.role}): Using trained policy for decision.")
            state_vector = get_agent_state_vector(self, grid, city_stats)
            action_index, _, _ = trained_policy.get_action(state_vector)
            action_name = DISCRETE_ACTIONS[action_index]
            
            if action_name == "talk":
                # Policy should not talk with generic messages. Wait instead.
                return ("wait", {})
            if action_name == "wait":
                return ("wait", {})

            # The policy chooses *what* to build. We find a random *where*.
            building_type = action_name.replace("build_", "")
            
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) > 0:
                pos = random.choice(empty_cells)
                log_to_frontend(f"🎲 Agent {self.id} ({self.role}): Policy chose to build {building_type}. Placing at random empty cell {pos.tolist()}.")
                return ("build", {"building_type": building_type, "position": [int(pos[0]), int(pos[1])]})
            else:
                log_to_frontend(f"⚠️ Agent {self.id} ({self.role}): Policy wanted to build, but no empty space left! Waiting instead.")
                return ("wait", {})

        # --- Fallback to random action if no policy and no LLM intent ---
        log_to_frontend(f"🎲 Agent {self.id} ({self.role}): No policy or LLM intent, choosing a random action.")
        action = random.choice(["build", "wait"])
        if action == "wait":
            return ("wait", {})
        
        building_type = random.choice(list(BUILDING_TYPES.keys()))
        pos = [random.randint(0, GRID_SIZE_X-1), random.randint(0, GRID_SIZE_Z-1)]
        return ("build", {"building_type": building_type, "position": pos})


# --- Environment Class ---
class SimCityEnv:
    def __init__(self):
        # Grid stores building type index (0 for empty, 1+ for buildings)
        self.grid = np.zeros((GRID_SIZE_X, GRID_SIZE_Z), dtype=int)
        self.buildings = [] # List of placed building objects
        self.llm_logs: List[Dict] = []
        self.messages: List[Dict] = []
        self.step_count = 0
        self.agents = [Agent(i) for i in range(NUM_AGENTS)]
        self.trained_policy: "ActorCritic" = None

        self.budget = INITIAL_BUDGET
        self.population = INITIAL_POPULATION
        self.rci_demand = {"residential": 0.5, "commercial": 0.3, "industrial": 0.2}

    def _update_simulation(self):
        # Calculate total income and population from all buildings
        total_income = sum(b['income'] for b in self.buildings)
        total_population = sum(b['population'] for b in self.buildings)
        
        self.budget += total_income
        self.population = total_population

        # Simple RCI demand model: Demand is influenced by population and existing buildings
        num_res = len([b for b in self.buildings if b['type'] == 'residential']) + 1
        num_com = len([b for b in self.buildings if b['type'] == 'commercial']) + 1
        num_ind = len([b for b in self.buildings if b['type'] == 'industrial']) + 1

        self.rci_demand["residential"] = np.clip(0.1 + (self.population / 500) - (num_res / (num_com + num_ind)), 0, 1)
        self.rci_demand["commercial"] = np.clip(0.1 + (num_res / (self.population + 100)) - (num_com / num_res), 0, 1)
        self.rci_demand["industrial"] = np.clip(0.1 + (num_com / (self.population + 100)) - (num_ind / num_res), 0, 1)

    def _get_reward(self, agent: Agent, action: str, data: Any) -> float:
        reward = 0.0
        # Reward for population growth and positive income change
        reward += self.population / 1000.0
        reward += self.budget / 100000.0

        if action == "wait":
            reward -= 0.1 # Small cost for doing nothing

        if action == "build" and data:
            pos = data.get("position")
            # Big penalty for trying to build on an existing structure
            if self.grid[pos[0], pos[1]] != 0:
                reward -= 5.0
            else:
                # Reward for building something useful
                reward += 1.0
                # Reward for satisfying demand
                b_type = data.get("building_type")
                if b_type in self.rci_demand:
                    reward += self.rci_demand[b_type] * 2.0

        return reward

    def _execute_actions(self, agent_actions: List[Tuple[str, Any]]):
        randomized_order = list(zip(self.agents, agent_actions))
        random.shuffle(randomized_order)

        for agent, (action, data) in randomized_order:
            if action == "wait":
                agent.add_to_memory_stream("Decided to wait.", self.step_count)
                log_to_frontend(f"⏳ Agent {agent.id} ({agent.role}) is waiting.")
                continue
            
            if action == "talk" and data and "message" in data:
                message_data = {
                    "sender_id": agent.id,
                    "role": agent.role,
                    "message": data.get("message"),
                    "step": self.step_count,
                }
                self.messages.append(message_data)
                if len(self.messages) > MAX_MESSAGES:
                    self.messages = self.messages[-MAX_MESSAGES:]
                
                # Log to frontend and have other agents update memory
                log_to_frontend(f"💬 Agent {agent.id} ({agent.role}): {data.get('message')}")
                agent.add_to_memory_stream(f"I said: {data.get('message')}", self.step_count)
                for a in self.agents:
                    if a.id != agent.id:
                        a.update_memory(f"Agent {agent.id} ({agent.role}) said: {data.get('message')}")
                continue

            if action == "build" and data:
                b_type = data.get("building_type")
                pos = data.get("position")
                
                # Clamp position to grid bounds
                pos[0] = np.clip(pos[0], 0, GRID_SIZE_X - 1)
                pos[1] = np.clip(pos[1], 0, GRID_SIZE_Z - 1)

                if b_type in BUILDING_TYPES:
                    cost = BUILDING_TYPES[b_type]["cost"]
                    if self.budget >= cost and self.grid[pos[0], pos[1]] == 0:
                        self.budget -= cost
                        
                        b_type_idx = list(BUILDING_TYPES.keys()).index(b_type) + 1
                        self.grid[pos[0], pos[1]] = b_type_idx
                        
                        building_info = BUILDING_TYPES[b_type]
                        self.buildings.append({
                            "type": b_type,
                            "pos": pos,
                            "income": building_info["income"],
                            "population": building_info["population"]
                        })
                        agent.add_to_memory_stream(f"Built {b_type} at {pos} for ${cost}", self.step_count)
                        agent.update_memory(f"I built a {b_type} at {pos}.")
                        log_to_frontend(f"🏗️ Agent {agent.id} ({agent.role}) built {b_type}. Budget: ${self.budget}")
                    elif self.grid[pos[0], pos[1]] != 0:
                        agent.add_to_memory_stream(f"Failed to build - location {pos} is occupied.", self.step_count)
                        agent.update_memory(f"I failed to build at {pos} because it was occupied.")
                        log_to_frontend(f"❌ Agent {agent.id} ({agent.role}) failed to build - location occupied.")
                    else:
                        agent.add_to_memory_stream(f"Failed to build {b_type} - not enough budget.", self.step_count)
                        agent.update_memory(f"I failed to build a {b_type} due to lack of funds.")
                        log_to_frontend(f"💰 Agent {agent.id} ({agent.role}) failed to build - not enough budget.")

    async def step(self):
        self.step_count += 1
        self._update_simulation() # Update stats before decisions
        
        llm_tasks = []
        task_to_agent = {}
        max_concurrent_llm = 2
        current_thinking_count = sum(1 for agent in self.agents if agent.is_thinking)

        for agent in self.agents:
            if current_thinking_count >= max_concurrent_llm:
                break
            
            if not agent.is_thinking and (self.step_count - agent.last_llm_step) >= LLM_CALL_FREQUENCY:
                agent.last_llm_step = self.step_count
                city_stats = {"budget": self.budget, "population": self.population, "rci_demand": self.rci_demand}
                
                async def safe_llm_call(agent_ref):
                    try:
                        log_entry = await agent_ref.decide_action_llm(self.grid, city_stats, self.messages, self.step_count)
                        if log_entry and isinstance(log_entry, dict):
                            self.llm_logs.append(log_entry)
                            if len(self.llm_logs) > MAX_LLM_LOGS:
                                self.llm_logs = self.llm_logs[-MAX_LLM_LOGS:]
                    except Exception as e:
                        log_to_frontend(f"Error in LLM call for Agent {agent_ref.id}: {e}")
                        if agent_ref:
                            agent_ref.is_thinking = False

                task = asyncio.create_task(safe_llm_call(agent))
                llm_tasks.append(task)
                current_thinking_count += 1

        # --- Fast Action Execution (Synchronous) ---
        agent_actions = []
        for agent in self.agents:
             city_stats = {"budget": self.budget, "population": self.population, "rci_demand": self.rci_demand}
             agent_actions.append(agent.get_fast_action(self.trained_policy, self.grid, city_stats))

        self._execute_actions(agent_actions)
        

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "role": a.role, "goal": a.goal, "memory_stream": a.memory_stream} for a in self.agents],
            "city_stats": {"budget": self.budget, "population": self.population, "rci_demand": self.rci_demand},
            "llm_logs": self.llm_logs,
            "grid_size": [GRID_SIZE_X, GRID_SIZE_Z],
            "building_types": BUILDING_TYPES,
            "messages": self.messages,
        }

# --- PPO Implementation ---
def get_agent_state_vector(agent: Agent, grid: np.ndarray, city_stats: Dict) -> np.ndarray:
    """Create a fixed-size state vector for the policy."""
    # 1. Grid summary (down-sampled)
    grid_size_x, grid_size_z = grid.shape
    x_bins, z_bins = 10, 10
    grid_summary = np.zeros((x_bins, z_bins))
    for i in range(x_bins):
        for j in range(z_bins):
            sub_grid = grid[
                i * (grid_size_x // x_bins) : (i + 1) * (grid_size_x // x_bins),
                j * (grid_size_z // z_bins) : (j + 1) * (grid_size_z // z_bins)
            ]
            grid_summary[i, j] = np.mean(sub_grid)
    
    grid_flat = grid_summary.flatten() / len(BUILDING_TYPES) # Normalize

    # 2. City Stats (normalized)
    stats_vec = np.array([
        np.log1p(city_stats['budget']) / 15.0, 
        np.log1p(city_stats['population']) / 10.0, 
        city_stats['rci_demand']['residential'],
        city_stats['rci_demand']['commercial'],
        city_stats['rci_demand']['industrial']
    ])

    # 3. Agent Memory
    memory_vec = agent.memory_vector

    # Pad/truncate memory vector to ensure fixed size, just in case
    fixed_mem_size = 384
    if len(memory_vec) > fixed_mem_size:
        memory_vec = memory_vec[:fixed_mem_size]
    else:
        memory_vec = np.pad(memory_vec, (0, fixed_mem_size - len(memory_vec)))

    return np.concatenate([grid_flat, stats_vec, memory_vec]).astype(np.float32)


# --- PPO Hyperparameters ---
BATCH_SIZE = 512
MINI_BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

# --- Websocket runner (Simplified for LLM-only) ---
async def run_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket
    _current_websocket = websocket
    
    running = True
    async def receive_commands():
        nonlocal running
        while running:
            try:
                data = await websocket.receive_json()
                if data.get("cmd") == "stop":
                    running = False
                    break
            except Exception:
                running = False
                break
            
    cmd_task = asyncio.create_task(receive_commands())

    while running:
        await env.step()
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state})

        # Send progress update for chart
        progress_update = {
            "type": "progress",
            "episode": env.step_count,
            "reward": env.population, # Use population as the main "score"
            "loss": None 
        }
        await websocket.send_json(progress_update)
        await asyncio.sleep(0.2) # Speed up simulation for run mode

    if not cmd_task.done():
        cmd_task.cancel()

# This function is not used by the LLM-only runner, but is kept for compatibility
async def train_simcity(websocket: WebSocket, env: SimCityEnv):
    global _current_websocket
    _current_websocket = websocket
    
    try:
        await websocket.send_json({"type": "debug", "message": "Starting SimCity training..."})
        
        # Determine obs size from a dummy agent
        dummy_agent = env.agents[0]
        city_stats = {"budget": env.budget, "population": env.population, "rci_demand": env.rci_demand}
        dummy_obs = get_agent_state_vector(dummy_agent, env.grid, city_stats)
        obs_size = dummy_obs.shape[0]
        action_size = len(DISCRETE_ACTIONS)

        model = ActorCritic(obs_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        env.trained_policy = model

        step_buffer: list[dict] = []
        ep_counter = 0
        total_steps = 0
        current_loss = None

        while ep_counter < 5000:
            env._update_simulation()
            
            # --- Collect experience from all agents ---
            city_stats = {"budget": env.budget, "population": env.population, "rci_demand": env.rci_demand}
            agent_states = [get_agent_state_vector(agent, env.grid, city_stats) for agent in env.agents]
            obs_t = torch.tensor(np.array(agent_states), dtype=torch.float32)

            with torch.no_grad():
                actions_t, logp_t, value_t = model.get_action(obs_t)
            
            actions_np = actions_t.cpu().numpy()
            
            rewards = []
            dones = []
            agent_actions_for_env = []

            for i, agent in enumerate(env.agents):
                action_name = DISCRETE_ACTIONS[actions_np[i]]
                action_data = {}
                env_action = (action_name, action_data)

                if action_name == "wait" or action_name == "talk":
                    env_action = ("wait", {}) # Policy doesn't generate talk content
                else: # Build action
                    building_type = action_name.replace("build_", "")
                    empty_cells = np.argwhere(env.grid == 0)
                    if len(empty_cells) > 0:
                        pos = random.choice(empty_cells)
                        action_data = {"building_type": building_type, "position": [int(pos[0]), int(pos[1])]}
                        env_action = ("build", action_data)
                    else:
                        env_action = ("wait", {})
                
                agent_actions_for_env.append(env_action)
                reward = env._get_reward(agent, env_action[0], action_data)
                rewards.append(reward)
                dones.append(env.budget < 0)

            env._execute_actions(agent_actions_for_env)
            
            step_buffer.append({
                "obs": obs_t, "actions": actions_t, "logp": logp_t, 
                "reward": torch.tensor(rewards, dtype=torch.float32), 
                "done": torch.tensor(dones, dtype=torch.float32), 
                "value": value_t.flatten()
            })
            
            env.step_count += 1
            total_steps += len(env.agents)
            ep_counter += len(env.agents)

            if any(dones):
                await websocket.send_json({"type": "debug", "message": f"Episode finished (Bankrupt). Resetting."})
                env = SimCityEnv()
                env.trained_policy = model

            # --- Send progress updates ---
            if env.step_count % 16 == 0:
                avg_reward = float(torch.tensor(rewards, dtype=torch.float32).mean().item())
                await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": current_loss})
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
                await asyncio.sleep(0.01)

            # --- PPO Update phase ---
            if total_steps >= BATCH_SIZE:
                with torch.no_grad():
                    city_stats = {"budget": env.budget, "population": env.population, "rci_demand": env.rci_demand}
                    next_obs_list = [get_agent_state_vector(agent, env.grid, city_stats) for agent in env.agents]
                    next_obs_t = torch.tensor(np.array(next_obs_list), dtype=torch.float32)
                    _, _, next_value = model.get_action(next_obs_t)
                    next_value = next_value.squeeze()
                
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
                        
                        pg_loss1 = -b_adv[mb_idxs] * ratio
                        pg_loss2 = -b_adv[mb_idxs] * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        v_loss = 0.5 * ((value.flatten() - b_returns[mb_idxs]).pow(2)).mean()
                        loss = pg_loss - ENT_COEF * entropy_bonus + v_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                current_loss = loss.item()
                step_buffer = []
                total_steps = 0
                await websocket.send_json({"type": "debug", "message": "Completed PPO update"})

        await websocket.send_json({"type": "trained", "model_info": {"epochs": ep_counter, "loss": loss.item()}})
    except Exception as e:
        logger.error(f"Error during SimCity training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"}) 