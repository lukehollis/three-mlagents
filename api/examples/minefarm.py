import asyncio
import random
from typing import List, Dict, Any
import numpy as np
from fastapi import WebSocket

# --- Constants ---
GRID_SIZE = 40
NUM_AGENTS = 20
RESOURCE_TYPES = {
    "wood": {"value": 1, "color": [0.5, 0.3, 0.1]},
    "stone": {"value": 2, "color": [0.5, 0.5, 0.5]},
    "iron": {"value": 5, "color": [0.8, 0.8, 0.9]},
}
MAX_MESSAGES = 10

# --- Agent Class ---
class Agent:
    def __init__(self, agent_id: int, pos: np.ndarray):
        self.id = agent_id
        self.pos = pos
        self.inventory = {res: 0 for res in RESOURCE_TYPES}
        self.goal = None # e.g., "wood" or "stone"
        self.path = []
        self.color = [random.random() * 0.8, random.random() * 0.8, random.random() * 0.8]

    def decide_action(self, grid: np.ndarray, agents: List['Agent'], messages: List[Dict]):
        # Simplified decision making. A real implementation could use an LLM.
        # Placeholder for LLM-based decision:
        # prompt = f"You are agent {self.id} at {self.pos} with inventory {self.inventory}. Your goal is {self.goal}. You see resources. Messages: {messages}. Decide: move, mine, or talk."
        # action_str = llm.predict(prompt) -> returns e.g. "talk: Looking for wood"
        
        # 1. If at a resource and has a goal for it, mine it
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0,0)]: # check current spot too
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] > 0:
                resource_idx = grid[nx, ny]
                resource_name = list(RESOURCE_TYPES.keys())[int(resource_idx) - 1]
                if self.goal == resource_name:
                    return ("mine", np.array([nx, ny]))

        # 2. If no goal, or goal is achieved, find a new goal
        if self.goal is None:
             self.goal = random.choice(list(RESOURCE_TYPES.keys()))
             return ("talk", f"I'm looking for {self.goal}")

        # 3. Simple random walk
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
            for _ in range(random.randint(8, 15)): # number of patches
                px, py = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                for _ in range(random.randint(20, 40)): # size of patch
                    x = px + random.randint(-4, 4)
                    y = py + random.randint(-4, 4)
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        self.grid[x, y] = i + 1

    def step(self):
        self.step_count += 1
        random.shuffle(self.agents)
        for agent in self.agents:
            action, data = agent.decide_action(self.grid, self.agents, self.messages)
            
            if action == "move" and data is not None:
                agent.pos = data
            elif action == "mine" and data is not None:
                res_pos = data
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
        env.step()
        state = env.get_state_for_viz()
        try:
            await websocket.send_json({"type": "run_step", "state": state})
            await asyncio.sleep(0.2) # Simulation speed
        except Exception:
            running = False
            break # Exit while loop

    if not cmd_task.done():
        cmd_task.cancel() 