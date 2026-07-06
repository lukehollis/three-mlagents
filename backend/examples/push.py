from typing import List, Tuple

import numpy as np
from fastapi import WebSocket

# --------------------------------------------------------------
# Simple Push-Block Environment --------------------------------
# --------------------------------------------------------------

DEFAULT_GRID_SIZE = 6  # N x N grid
MAX_STEPS_PER_EP = 120

# Actions: 0 stay, 1 up, 2 down, 3 left, 4 right
ACTION_DELTAS: List[Tuple[int, int]] = [
    (0, 0),  # stay
    (0, 1),  # up (+z)
    (0, -1),  # down (−z)
    (-1, 0),  # left (−x)
    (1, 0),  # right (+x)
]
NUM_ACTIONS = len(ACTION_DELTAS)

# Observation: relative vectors (agent→box, box→goal)
OBS_SIZE = 4


class PushEnv:
    """Minimal push-block environment.

    The agent must push a movable box into a goal area located on the
    top row of the grid. Only cardinal moves are allowed.
    """

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(cells)
        self.agent_pos = cells[0]
        self.box_pos = cells[1]

        # Choose a random goal cell on the top row (y == grid_size − 1)
        goal_x = np.random.randint(0, self.grid_size)
        self.goal_pos = (goal_x, self.grid_size - 1)

        self.steps = 0
        return self._get_obs()

    # ----------------------------------------------------------
    def _get_obs(self):
        # Relative vectors normalised to [−1,1]
        dx_ab = (self.box_pos[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        dy_ab = (self.box_pos[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)
        dx_bg = (self.goal_pos[0] - self.box_pos[0]) / max(1, self.grid_size - 1)
        dy_bg = (self.goal_pos[1] - self.box_pos[1]) / max(1, self.grid_size - 1)
        return np.array([dx_ab, dy_ab, dx_bg, dy_bg], dtype=np.float32)

    # ----------------------------------------------------------
    def step(self, action_idx: int):
        dx, dy = ACTION_DELTAS[action_idx]
        new_agent_x = int(np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1))
        new_agent_y = int(np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1))

        new_box_x, new_box_y = self.box_pos

        # Dense shaping: distances before moving
        prev_dist_bg = abs(self.goal_pos[0] - self.box_pos[0]) + abs(
            self.goal_pos[1] - self.box_pos[1]
        )
        prev_dist_ab = abs(self.box_pos[0] - self.agent_pos[0]) + abs(
            self.box_pos[1] - self.agent_pos[1]
        )

        reward = -0.01  # step penalty
        done = False

        invalid_push = False

        # Attempt push if agent moves into box
        if (new_agent_x, new_agent_y) == self.box_pos:
            tentative_box_x = self.box_pos[0] + dx
            tentative_box_y = self.box_pos[1] + dy
            # If push is within bounds, move box
            if (
                0 <= tentative_box_x < self.grid_size
                and 0 <= tentative_box_y < self.grid_size
            ):
                new_box_x, new_box_y = tentative_box_x, tentative_box_y
            else:
                # Invalid push – cancel agent movement
                new_agent_x, new_agent_y = self.agent_pos
                invalid_push = True

        # Update positions
        self.agent_pos = (new_agent_x, new_agent_y)
        self.box_pos = (new_box_x, new_box_y)
        self.steps += 1

        # Shaping: reward change
        dist_bg = abs(self.goal_pos[0] - self.box_pos[0]) + abs(
            self.goal_pos[1] - self.box_pos[1]
        )
        dist_ab = abs(self.box_pos[0] - self.agent_pos[0]) + abs(
            self.box_pos[1] - self.agent_pos[1]
        )

        # Encourage reducing agent→box distance until touching, then box→goal distance
        reward += 0.05 * (prev_dist_ab - dist_ab)  # approach box
        reward += 0.3 * (prev_dist_bg - dist_bg)  # push box closer to goal

        if invalid_push:
            reward -= 0.05

        # Check goal achievement (any cell on goal strip)
        if self.box_pos[1] == self.grid_size - 1:
            reward = 1.0
            done = True

        if self.steps >= MAX_STEPS_PER_EP:
            done = True

        return self._get_obs(), reward, done


# --------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# --------------------------------------------------------------


async def train_push(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(
        websocket,
        "push",
        total_timesteps=200_000,
        algorithm="dqn",
        progress_freq=2_000,
    )


def infer_action_push(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("push", obs, model_filename)
