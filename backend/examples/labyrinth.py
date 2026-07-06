from typing import Any, Dict
import random

import asyncio
import numpy as np
import torch
import torch.nn as nn
from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ACTION_SIZE = 4  # 0=up, 1=down, 2=left, 3=right
POLICIES_DIR = "policies"
# BATCH_SIZE = 8192 - Handled by SB3 n_steps
# MINI_BATCH = 512 - Handled by SB3 batch_size
# EPOCHS = 10 - Handled by SB3 n_epochs
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 1e-4
MAX_STEPS_PER_EPISODE = 250  # Reduced for smaller maze
EPISODES = 5000
TOTAL_TIMESTEPS = MAX_STEPS_PER_EPISODE * EPISODES
LABYRINTH_WIDTH = 21  # Drastically reduced from 129
LABYRINTH_HEIGHT = 11  # Drastically reduced from 65

# -----------------------------------------------------------------------------------
# Labyrinth Environment
# -----------------------------------------------------------------------------------


class LabyrinthEnv(gym.Env):
    """A 2D ASCII labyrinth environment with a pursuer (Minotaur) and an exit."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, training_mode: bool = True):
        super().__init__()
        self.width = LABYRINTH_WIDTH
        self.height = LABYRINTH_HEIGHT
        self.grid = np.full((self.height, self.width), "#")
        self.training_mode = training_mode

        self.action_space = spaces.Discrete(ACTION_SIZE)

        # Observation space is the grid as uint8 image [0,255] for NatureCNN
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

        # Episode tracking
        self.total_episode_reward = 0.0
        self.reset()

    def _generate_labyrinth(self):
        # Iterative maze generation to handle large sizes without recursion limits.
        grid = np.full((self.height, self.width), "#", dtype="<U1")

        def is_valid(y, x):
            return 1 <= y < self.height - 1 and 1 <= x < self.width - 1

        stack = []
        start_y, start_x = 1, 1
        grid[start_y, start_x] = " "
        stack.append((start_y, start_x))

        while stack:
            y, x = stack[-1]
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)

            carved_new_path = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                nwy, nwx = y + dy // 2, x + dx // 2

                if is_valid(ny, nx) and grid[ny, nx] == "#":
                    grid[nwy, nwx] = " "
                    grid[ny, nx] = " "
                    stack.append((ny, nx))
                    carved_new_path = True
                    break

            if not carved_new_path:
                stack.pop()
        return grid

    def _get_random_empty_cell(self):
        while True:
            y = random.randint(1, self.height - 2)
            x = random.randint(1, self.width - 2)
            if self.grid[y, x] == " ":
                return (y, x)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self._generate_labyrinth()

        self.theseus_pos = self._get_random_empty_cell()

        self.minotaur_pos = self._get_random_empty_cell()
        min_dist = (self.width + self.height) / 4  # Quarter of max manhattan distance
        while (
            abs(self.theseus_pos[0] - self.minotaur_pos[0])
            + abs(self.theseus_pos[1] - self.minotaur_pos[1])
            < min_dist
        ):
            self.minotaur_pos = self._get_random_empty_cell()

        self.exit_pos = self._get_random_empty_cell()
        while (
            abs(self.theseus_pos[0] - self.exit_pos[0])
            + abs(self.theseus_pos[1] - self.exit_pos[1])
            < min_dist
        ):
            self.exit_pos = self._get_random_empty_cell()

        self.grid[self.exit_pos] = "E"
        self.steps = 0
        self.minotaur_turn_counter = 0
        self.total_episode_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1
        py, px = self.theseus_pos
        ny, nx = py, px

        if action == 0:
            ny -= 1  # Up
        elif action == 1:
            ny += 1  # Down
        elif action == 2:
            nx -= 1  # Left
        elif action == 3:
            nx += 1  # Right

        reward = -0.05  # Small time penalty

        # Penalty for hitting a wall
        if self.grid[ny, nx] == "#":
            reward -= 0.5  # Increased wall penalty
        else:
            # Reward for moving closer to the exit
            dist_to_exit_prev = abs(py - self.exit_pos[0]) + abs(px - self.exit_pos[1])
            dist_to_exit_new = abs(ny - self.exit_pos[0]) + abs(nx - self.exit_pos[1])
            reward += 0.2 * (dist_to_exit_prev - dist_to_exit_new)  # Increased reward
            self.theseus_pos = (ny, nx)

        # Penalty for moving closer to the Minotaur
        dist_to_mino_prev = abs(py - self.minotaur_pos[0]) + abs(
            px - self.minotaur_pos[1]
        )
        dist_to_mino_new = abs(self.theseus_pos[0] - self.minotaur_pos[0]) + abs(
            self.theseus_pos[1] - self.minotaur_pos[1]
        )
        reward -= 0.1 * (dist_to_mino_prev - dist_to_mino_new)  # Increased penalty

        # Move Minotaur every four steps, making it easier for the agent to learn
        self.minotaur_turn_counter += 1
        if self.minotaur_turn_counter % 4 == 0:
            self._move_minotaur()

        terminated = False
        truncated = False
        info = {}
        episode_end_reason = None

        if self.theseus_pos == self.exit_pos:
            reward = 200.0
            terminated = True
            episode_end_reason = "Reached the exit"
        elif self.theseus_pos == self.minotaur_pos:
            reward = -100.0
            terminated = True
            episode_end_reason = "Caught by Minotaur"
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            reward -= (
                5.0  # Reduced timeout penalty to be less punishing than getting caught
            )
            truncated = True
            episode_end_reason = "Timeout"

        self.total_episode_reward += reward

        if terminated or truncated:
            if episode_end_reason and self.training_mode:
                info["episode_end"] = {
                    "reason": episode_end_reason,
                    "steps": self.steps,
                    "total_reward": self.total_episode_reward,
                }

        return self._get_obs(), reward, terminated, truncated, info

    def _move_minotaur(self):
        my, mx = self.minotaur_pos

        # 20% chance of random move
        if random.random() < 0.2:
            possible_moves = []
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if self.grid[my + dy, mx + dx] != "#":
                    possible_moves.append((my + dy, mx + dx))
            if possible_moves:
                self.minotaur_pos = random.choice(possible_moves)
            return

        ty, tx = self.theseus_pos
        dy, dx = np.sign(ty - my), np.sign(tx - mx)

        # Smarter greedy move: try axis with largest distance first
        if abs(ty - my) > abs(tx - mx):
            if dy != 0 and self.grid[my + int(dy), mx] != "#":
                self.minotaur_pos = (my + int(dy), mx)
            elif dx != 0 and self.grid[my, mx + int(dx)] != "#":
                self.minotaur_pos = (my, mx + int(dx))
        else:
            if dx != 0 and self.grid[my, mx + int(dx)] != "#":
                self.minotaur_pos = (my, mx + int(dx))
            elif dy != 0 and self.grid[my + int(dy), mx] != "#":
                self.minotaur_pos = (my + int(dy), mx)

    def _get_obs(self):
        # Observation as uint8 image [0,255] for NatureCNN
        obs_grid = np.full(
            (self.height, self.width), 51, dtype=np.uint8
        )  # Path ~0.2*255

        # Walls are 0
        wall_y, wall_x = np.where(self.grid == "#")
        obs_grid[wall_y, wall_x] = 0

        # Exit is 255
        exit_y, exit_x = self.exit_pos
        obs_grid[exit_y, exit_x] = 255

        # Theseus is ~204 (0.8*255)
        ty, tx = self.theseus_pos
        obs_grid[ty, tx] = 204

        # Minotaur is ~102 (0.4*255)
        my, mx = self.minotaur_pos
        obs_grid[my, mx] = 102

        return np.expand_dims(obs_grid, axis=-1)

    def get_state_for_viz(self) -> Dict[str, Any]:
        grid_viz = self.grid.copy()
        grid_viz[self.theseus_pos] = "T"
        grid_viz[self.minotaur_pos] = "M"
        return {"grid": grid_viz.tolist(), "steps": self.steps}


# -----------------------------------------------------------------------------------
# Custom CNN for smaller input
# -----------------------------------------------------------------------------------


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for the Labyrinth environment.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_tensor = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# -----------------------------------------------------------------------------------
# Callbacks and training setup
# -----------------------------------------------------------------------------------


async def send_json_safely(websocket: WebSocket, payload: dict) -> bool:
    """Safely send JSON over a websocket, returning True on success, False on failure."""
    if websocket.application_state != WebSocketState.CONNECTED:
        print("Skipping send, websocket not connected.")
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, ConnectionClosedError) as e:
        print(f"WebSocket connection closed unexpectedly: {e}")
        return False


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points and preview simulation
# -----------------------------------------------------------------------------------


async def train_labyrinth(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "labyrinth")


async def run_simulation(websocket: WebSocket):
    env = LabyrinthEnv(training_mode=False)
    env.reset()
    while websocket.application_state == WebSocketState.CONNECTED:
        await send_json_safely(
            websocket, {"type": "state", "state": env.get_state_for_viz()}
        )
        await asyncio.sleep(0.5)
        env.reset()


def infer_action_labyrinth(obs: np.ndarray, model_filename: str | None = None) -> int:
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("labyrinth", obs, model_filename)


async def run_labyrinth(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "labyrinth",
        lambda: LabyrinthEnv(training_mode=False),
        model_filename=model_filename,
        action_transform=int,
        sleep_seconds=0.1,
    )
