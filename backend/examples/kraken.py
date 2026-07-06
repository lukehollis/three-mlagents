import numpy as np
from fastapi import WebSocket
import gymnasium as gym
from gymnasium import spaces

# Constants
GRID_SIZE = 200  # Water grid size
NUM_SHIPS = 4  # Number of pirate ships (agents)
NUM_TENTACLES = 10  # Kraken tentacles
TENTACLE_SPEED = 0.8
TENTACLE_RADIUS_MIN = 5.0
TENTACLE_RADIUS_MAX = 25.0
REWARD_DAMAGE_KRAKEN = 50.0
REWARD_SURVIVE = 1.0
PENALTY_DAMAGE = -20.0
PENALTY_SUNK = -100.0
SHIP_SPEED = 1.5
KRAKEN_SPEED = 1.0
SHOOT_RANGE = 20.0
KRAKEN_HEALTH = 500.0
SHIP_HEALTH = 100.0
MAX_STEPS = 1000


class PirateShipEnv(gym.Env):
    def __init__(self, training_mode=True):
        super().__init__()
        self.training_mode = training_mode
        # 0: no move, 1: up, 2: down, 3: left, 4: right, 5: shoot.
        self.action_space = spaces.MultiDiscrete([6] * NUM_SHIPS)
        self.observation_space = self._make_observation_space()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.ship_positions = self.np_random.uniform(0, GRID_SIZE, (NUM_SHIPS, 2))
        self.ship_healths = np.full(NUM_SHIPS, SHIP_HEALTH)
        self.kraken_position = np.array([GRID_SIZE / 2, GRID_SIZE / 2])
        self.kraken_health = KRAKEN_HEALTH
        self.tentacle_offsets = np.zeros((NUM_TENTACLES, 2))
        for i in range(NUM_TENTACLES):
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(TENTACLE_RADIUS_MIN, TENTACLE_RADIUS_MAX)
            self.tentacle_offsets[i] = [dist * np.cos(angle), dist * np.sin(angle)]
        self.tentacle_positions = self.kraken_position[None, :] + self.tentacle_offsets
        return self._get_obs(), {}

    @staticmethod
    def _make_observation_space():
        ship_low = [0.0, 0.0, -GRID_SIZE, -GRID_SIZE, 0.0, 0.0]
        ship_high = [
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
            SHIP_HEALTH,
            np.sqrt(2.0) * GRID_SIZE,
        ]
        low = np.array(ship_low * NUM_SHIPS + [0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            ship_high * NUM_SHIPS + [GRID_SIZE, GRID_SIZE, KRAKEN_HEALTH],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        obs = []
        for i in range(NUM_SHIPS):
            rel_pos = self.kraken_position - self.ship_positions[i]
            obs.extend(
                [
                    self.ship_positions[i][0],
                    self.ship_positions[i][1],
                    rel_pos[0],
                    rel_pos[1],
                    self.ship_healths[i],
                    np.linalg.norm(rel_pos),
                ]
            )
        obs.extend(
            [self.kraken_position[0], self.kraken_position[1], self.kraken_health]
        )
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)

        reward = 0
        self.steps += 1

        # Move ships
        for i, action in enumerate(actions):
            if self.ship_healths[i] <= 0:
                continue
            if action == 1:  # up
                self.ship_positions[i][1] += SHIP_SPEED
            elif action == 2:  # down
                self.ship_positions[i][1] -= SHIP_SPEED
            elif action == 3:  # left
                self.ship_positions[i][0] -= SHIP_SPEED
            elif action == 4:  # right
                self.ship_positions[i][0] += SHIP_SPEED
            elif action == 5:  # shoot
                dist = np.linalg.norm(self.kraken_position - self.ship_positions[i])
                if dist < SHOOT_RANGE:
                    self.kraken_health -= 10
                    reward += REWARD_DAMAGE_KRAKEN

            self.ship_positions[i] = np.clip(self.ship_positions[i], 0, GRID_SIZE)
            reward += REWARD_SURVIVE

        # Tentacle movement and attacks
        if np.any(self.ship_healths > 0):
            alive_ships = self.ship_positions[self.ship_healths > 0]
            for j, tentacle in enumerate(self.tentacle_positions):
                # Move tentacle towards the nearest ship
                distances = np.linalg.norm(alive_ships - tentacle, axis=1)
                nearest_ship_idx = np.argmin(distances)
                direction = alive_ships[nearest_ship_idx] - tentacle
                direction /= np.linalg.norm(direction) + 1e-8
                self.tentacle_positions[j] += direction * TENTACLE_SPEED

                # Attack ships
                for i in range(NUM_SHIPS):
                    if self.ship_healths[i] > 0:
                        dist = np.linalg.norm(
                            self.tentacle_positions[j] - self.ship_positions[i]
                        )
                        if dist < 5.0:
                            self.ship_healths[i] -= 10
                            reward += PENALTY_DAMAGE
                            if self.ship_healths[i] <= 0:
                                reward += PENALTY_SUNK

        # Move kraken towards average ship position
        if np.any(self.ship_healths > 0):
            avg_ship_pos = np.mean(self.ship_positions[self.ship_healths > 0], axis=0)
            direction = avg_ship_pos - self.kraken_position
            direction /= np.linalg.norm(direction) + 1e-8
            self.kraken_position += direction * KRAKEN_SPEED
            self.kraken_position = np.clip(self.kraken_position, 0, GRID_SIZE)
            self.tentacle_positions = (
                self.kraken_position[None, :] + self.tentacle_offsets
            )

        terminated = False
        truncated = False
        if self.kraken_health <= 0:
            terminated = True
            reward += 1000
        elif np.all(self.ship_healths <= 0):
            terminated = True
            reward -= 1000
        elif self.steps >= MAX_STEPS:
            truncated = True
            reward -= 1000

        return self._get_obs(), reward / NUM_SHIPS, terminated, truncated, {}

    def get_state_for_viz(self):
        return {
            "ships": [
                {"pos": pos.tolist(), "health": health}
                for pos, health in zip(self.ship_positions, self.ship_healths)
            ],
            "kraken": {
                "pos": self.kraken_position.tolist(),
                "health": self.kraken_health,
            },
            "tentacles": self.tentacle_positions.tolist(),
            "grid_size": GRID_SIZE,
        }


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_pirate_ship(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "kraken")


async def run_pirate_ship(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "kraken",
        PirateShipEnv,
        model_filename=model_filename,
        sleep_seconds=0.05,
    )
