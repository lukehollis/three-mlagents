# ---------------------------------------------------------------------------------
# GridWorld environment and Q-learning trainer -------------------------------------
# ---------------------------------------------------------------------------------

from typing import List, Tuple

import numpy as np
from fastapi import WebSocket

# ---------------------------------------------------------------------------------
# Simplified discrete GridWorld ----------------------------------------------------
# ---------------------------------------------------------------------------------

DEFAULT_GRID_SIZE = 5  # N x N grid
MAX_STEPS_PER_EP = 100

# Action mapping
# 0: no-op, 1: up (+z), 2: down (−z), 3: left (−x), 4: right (+x)
ACTION_DELTAS: List[Tuple[int, int]] = [
    (0, 0),  # stay
    (0, 1),  # up
    (0, -1),  # down
    (-1, 0),  # left
    (1, 0),  # right
]
NUM_ACTIONS = len(ACTION_DELTAS)

# Observation is 4-dim float vector:
#  [dx_to_goal, dy_to_goal, goal_one_hot_0, goal_one_hot_1]
OBS_SIZE = 4


class GridWorldEnv:
    """Minimal multi-goal GridWorld with one agent and two goal types."""

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Random positions – ensure they are all unique
        all_cells = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]
        np.random.shuffle(all_cells)
        self.agent_pos = all_cells[0]
        self.green_goals = [all_cells[1]]  # "plus" goals
        self.red_goals = [all_cells[2]]  # "ex" goals
        # Randomly assign current target goal type
        self.current_goal_type = np.random.choice([0, 1])  # 0 = green, 1 = red
        self.steps = 0
        return self._get_obs()

    # -------------------------------------------------------------------------
    def _get_obs(self):
        # Vector from agent to the *nearest* target goal of the required type
        if self.current_goal_type == 0:
            goal = self.green_goals[0]
        else:
            goal = self.red_goals[0]
        dx = (goal[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        dy = (goal[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)
        one_hot_goal = [1.0, 0.0] if self.current_goal_type == 0 else [0.0, 1.0]
        return np.array([dx, dy, *one_hot_goal], dtype=np.float32)

    # -------------------------------------------------------------------------
    def step(self, action_idx: int):
        delta = ACTION_DELTAS[action_idx]
        new_x = int(np.clip(self.agent_pos[0] + delta[0], 0, self.grid_size - 1))
        new_y = int(np.clip(self.agent_pos[1] + delta[1], 0, self.grid_size - 1))
        self.agent_pos = (new_x, new_y)
        self.steps += 1

        # Base step penalty
        reward = -0.01
        done = False

        # Check for goal collision
        if self.agent_pos in self.green_goals:
            if self.current_goal_type == 0:
                reward = 1.0
            else:
                reward = -1.0
            done = True
        elif self.agent_pos in self.red_goals:
            if self.current_goal_type == 1:
                reward = 1.0
            else:
                reward = -1.0
            done = True

        if self.steps >= MAX_STEPS_PER_EP:
            done = True

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# ---------------------------------------------------------------------------------


async def train_gridworld(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(
        websocket,
        "gridworld",
        total_timesteps=100_000,
        algorithm="dqn",
        progress_freq=2_000,
    )


def infer_action_gridworld(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("gridworld", obs, model_filename)
