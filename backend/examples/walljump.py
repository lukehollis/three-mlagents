# --------------------------------------------------------------
# Wall Jump Example - simplified for browser RL demo
# --------------------------------------------------------------

from typing import List

import numpy as np
from fastapi import WebSocket

# --------------------------------------------------------------
# Environment ---------------------------------------------------
# --------------------------------------------------------------

WIDTH = 20  # 1-D track length
MAX_STEPS = 150

# Actions: 0 stay, 1 forward, 2 backward, 3 jump
NUM_ACTIONS = 4
ACTION_DELTAS = [0, 1, -1, 1]  # jump also moves forward

OBS_SIZE = 4  # [dx_goal, dx_wall, wall_height, on_ground]


class WallJumpEnv:
    """Minimal 1-D wall-jump environment.

    The agent moves along the x-axis starting at 0 aiming to reach WIDTH-1.
    A wall may be present at x == WALL_X. If present (height == 1), the
    agent must jump to cross the wall. Jumps last JUMP_DURATION steps.
    """

    WALL_X = 10
    JUMP_DURATION = 3

    def __init__(self):
        self.width = WIDTH
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        self.agent_x: int = 0
        self.in_air: int = 0  # remaining steps in air (0 means grounded)
        self.wall_height: int = int(np.random.rand() < 0.7)  # 70% wall present
        self.steps: int = 0
        return self._obs()

    # ----------------------------------------------------------
    def _obs(self):
        dx_goal = (self.width - 1 - self.agent_x) / (self.width - 1)
        dx_wall = (self.WALL_X - self.agent_x) / (self.width - 1)
        wall_h = float(self.wall_height)
        on_ground = 1.0 if self.in_air == 0 else 0.0
        return np.array([dx_goal, dx_wall, wall_h, on_ground], dtype=np.float32)

    # ----------------------------------------------------------
    def step(self, action_idx: int):
        assert 0 <= action_idx < NUM_ACTIONS

        reward = -0.01  # step penalty
        done = False

        just_jumped = False
        if action_idx == 3 and self.in_air == 0:
            self.in_air = self.JUMP_DURATION
            just_jumped = True

        # Determine proposed move
        dx = ACTION_DELTAS[action_idx]
        proposed_x = int(np.clip(self.agent_x + dx, 0, self.width - 1))

        # Block movement by wall if not jumping
        crossing_wall = (self.agent_x < self.WALL_X <= proposed_x) or (
            proposed_x < self.WALL_X <= self.agent_x
        )
        if crossing_wall and self.wall_height == 1 and self.in_air == 0:
            proposed_x = self.agent_x  # cannot cross
            reward -= 0.02  # slight penalty for hitting wall

        # Penalise jumping when not needed (wall not immediately ahead)
        if just_jumped and not crossing_wall and abs(self.WALL_X - self.agent_x) > 1:
            reward -= 0.03

        self.agent_x = proposed_x

        # Update air timer
        if self.in_air > 0:
            self.in_air -= 1

        # Success condition
        if self.agent_x == self.width - 1:
            reward = 1.0
            done = True

        self.steps += 1
        if self.steps >= MAX_STEPS:
            done = True

        return self._obs(), reward, done


# --------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# --------------------------------------------------------------


async def train_walljump(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(
        websocket,
        "walljump",
        total_timesteps=150_000,
        algorithm="dqn",
        progress_freq=2_000,
    )


def infer_action_walljump(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("walljump", obs, model_filename)
