import asyncio
import random
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
from services.llm import get_json

# --- Constants ---
GRID_SIZE = 60
NUM_AGENTS = 30
RESOURCE_TYPES = {
    "wood": {"value": 1, "color": [0.5, 0.3, 0.1]},
    "stone": {"value": 2, "color": [0.5, 0.5, 0.5]},
    "iron": {"value": 5, "color": [0.8, 0.8, 0.9]},
    "gold": {"value": 10, "color": [0.9, 0.8, 0.2]},
    "diamond": {"value": 20, "color": [0.7, 0.9, 1.0]},
}
MAX_MESSAGES = 15

# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {res: 0 for res in RESOURCE_TYPES}
        self.goal = None # e.g., "wood" or "stone"
        self.path = []
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]

    async def decide_action(self, grid: np.ndarray, agents: List['Agent'], messages: List[Dict]) -> Tuple[str, Any]:
        # LLM-based decision making.
        recent_messages = messages[-5:] # Only use last 5 messages for brevity
        prompt = f"""You are a mining agent in a grid world. Your ID is {self.id}.
Your current position is {self.pos.tolist()}.
Your inventory is {self.inventory}.
Your current goal is to collect '{self.goal if self.goal else "anything"}'.
Recent messages from other agents: {json.dumps(recent_messages)}

The grid is {GRID_SIZE}x{GRID_SIZE}. You can see a 5x5 area around you.
Resources are encoded as numbers in the grid. 0 is empty.
Your view:
{grid[self.pos[0]-2:self.pos[0]+3, self.pos[1]-2:self.pos[1]+3].tolist()}

Your available actions are "move", "mine", "talk", or "wait".
- move: requires integer [x, y] coordinates for the next step. You can only move to an adjacent square.
- mine: requires integer [x, y] coordinates of an adjacent resource to mine.
- talk: requires a short string message for other agents.
- wait: does nothing.

Based on your state, what is your next action? If you have no goal, set one by talking. If you see your goal, move towards it or mine it. If you don't see your goal, explore randomly or ask for help.
"""

        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "mine", "talk", "wait"]},
                "data": {"oneOf": [
                    {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    {"type": "string"},
                    {"type": "object"}, # Allow empty dict for wait
                ]}
            },
            "required": ["action", "data"]
        }

        try:
            # Note: The user specified 'meta-llama/llama-4-maverick'. This may be a custom model name.
            # Using it as specified. If it fails, a known OpenRouter model can be substituted.
            response = await get_json(
                prompt=prompt,
                model="meta-llama/llama-4-maverick",
                response_schema=action_schema,
                schema_name="agent_action_decision"
            )
            action = response.get("action", "wait")
            data = response.get("data")

            # Basic validation of LLM output
            if action == "move":
                # Ensure agent only moves to adjacent squares
                target_pos = np.array(data)
                if np.sum(np.abs(target_pos - self.pos)) == 1:
                    return ("move", np.array(data))
            elif action == "mine":
                 target_pos = np.array(data)
                 if np.sum(np.abs(target_pos - self.pos)) <= 1: # Can mine current or adjacent
                    return ("mine", np.array(data))
            elif action == "talk":
                return ("talk", str(data))
            
            # Fallback for invalid data
            return ("wait", None)

        except Exception as e:
            print(f"Agent {self.id} LLM call failed: {e}")
            # Fallback to simple random walk on error
            move = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
            next_pos = self.pos + np.array(move)
            if 0 <= next_pos[0] < GRID_SIZE and 0 <= next_pos[1] < GRID_SIZE:
                return ("move", next_pos)
            return ("wait", None)


# --- Environment Class ---
class MineFarmEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.agents = [Agent(i, np.random.randint(0, GRID_SIZE, 2)) for i in range(NUM_AGENTS)]
        self.messages: List[Dict] = []
        self._spawn_resources()
        self.step_count = 0

    def _spawn_resources(self):
        for i, res_type in enumerate(RESOURCE_TYPES.keys()):
            for _ in range(random.randint(15, 25)): # number of patches
                px, py = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                for _ in range(random.randint(30, 50)): # size of patch
                    x = px + random.randint(-5, 5)
                    y = py + random.randint(-5, 5)
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        self.grid[x, y] = i + 1

    async def step(self):
        self.step_count += 1
        
        # Let agents decide in parallel
        agent_actions = await asyncio.gather(
            *[agent.decide_action(self.grid, self.agents, self.messages) for agent in self.agents]
        )

        randomized_order = list(zip(self.agents, agent_actions))
        random.shuffle(randomized_order)

        for agent, (action, data) in randomized_order:
            if action == "move" and data is not None:
                # Ensure agent stays within bounds
                agent.pos = np.clip(data, 0, GRID_SIZE - 1)
            elif action == "mine" and data is not None:
                res_pos = data
                if 0 <= res_pos[0] < GRID_SIZE and 0 <= res_pos[1] < GRID_SIZE:
                    res_idx = self.grid[res_pos[0], res_pos[1]]
                    if res_idx > 0:
                        res_name = list(RESOURCE_TYPES.keys())[int(res_idx) - 1]
                        agent.inventory[res_name] += 1
                        self.grid[res_pos[0], res_pos[1]] = 0 # Resource depleted
                        agent.goal = None # Find a new goal
            elif action == "talk" and data is not None:
                msg = {"agent_id": agent.id, "message": data, "step": self.step_count}
                self.messages.append(msg)
                if len(self.messages) > MAX_MESSAGES:
                    self.messages.pop(0)

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agents": [{"id": a.id, "pos": a.pos.tolist(), "inventory": a.inventory, "color": a.color} for a in self.agents],
            "messages": self.messages,
            "grid_size": GRID_SIZE,
            "resource_types": RESOURCE_TYPES,
        }

# --- Websocket runner ---
async def run_minefarm(websocket: WebSocket):
    env = MineFarmEnv()
    running = True

    async def receive_commands():
        nonlocal running
        try:
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
            await asyncio.sleep(0.5) # Simulation speed
        except Exception:
            running = False
            break # Exit while loop

    if not cmd_task.done():
        cmd_task.cancel() 