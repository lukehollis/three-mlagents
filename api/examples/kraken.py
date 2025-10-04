import asyncio
import random
import numpy as np
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Constants
GRID_SIZE = 200  # Water grid size
NUM_SHIPS = 4  # Number of pirate ships (agents)
NUM_TENTACLES = 6  # Kraken tentacles
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
        self.action_space = spaces.Discrete(5)  # 0: no move, 1: forward, 2: left, 3: right, 4: shoot
        self.observation_space = spaces.Box(low=-GRID_SIZE, high=GRID_SIZE, shape=(NUM_SHIPS * 4 + 7,), dtype=np.float32)  # Positions, healths, kraken pos/health
        self.reset()

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.ship_positions = np.random.uniform(0, GRID_SIZE, (NUM_SHIPS, 2))
        self.ship_healths = np.full(NUM_SHIPS, SHIP_HEALTH)
        self.kraken_position = np.array([GRID_SIZE / 2, GRID_SIZE / 2])
        self.kraken_health = KRAKEN_HEALTH
        self.tentacle_offsets = np.random.uniform(-10, 10, (NUM_TENTACLES, 2))
        self.tentacle_positions = self.kraken_position[None, :] + self.tentacle_offsets
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(NUM_SHIPS):
            rel_pos = self.kraken_position - self.ship_positions[i]
            obs.extend([rel_pos[0], rel_pos[1], self.ship_healths[i], np.linalg.norm(rel_pos)])
        obs.extend([self.kraken_position[0], self.kraken_position[1], self.kraken_health])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        reward = 0
        done = False
        self.steps += 1

        # Move ships
        for i, action in enumerate(actions):
            if self.ship_healths[i] <= 0:
                continue
            if action == 1:  # forward
                self.ship_positions[i][1] += SHIP_SPEED
            elif action == 2:  # left
                self.ship_positions[i][0] -= SHIP_SPEED
            elif action == 3:  # right
                self.ship_positions[i][0] += SHIP_SPEED
            elif action == 4:  # shoot
                dist = np.linalg.norm(self.kraken_position - self.ship_positions[i])
                if dist < SHOOT_RANGE:
                    self.kraken_health -= 10
                    reward += REWARD_DAMAGE_KRAKEN

            # Bound positions
            self.ship_positions[i] = np.clip(self.ship_positions[i], 0, GRID_SIZE)

            reward += REWARD_SURVIVE if self.ship_healths[i] > 0 else 0

        # Kraken attacks
        for tentacle in self.tentacle_positions:
            for i in range(NUM_SHIPS):
                if self.ship_healths[i] > 0:
                    dist = np.linalg.norm(tentacle - self.ship_positions[i])
                    if dist < 5.0:
                        self.ship_healths[i] -= 10
                        reward += PENALTY_DAMAGE
                        if self.ship_healths[i] <= 0:
                            reward += PENALTY_SUNK

        # Move kraken towards average ship position
        if np.any(self.ship_healths > 0):
            avg_ship_pos = np.mean(self.ship_positions[self.ship_healths > 0], axis=0)
            direction = avg_ship_pos - self.kraken_position
            direction /= (np.linalg.norm(direction) + 1e-8)
            self.kraken_position += direction * KRAKEN_SPEED
            self.kraken_position = np.clip(self.kraken_position, 0, GRID_SIZE)
            self.tentacle_positions = self.kraken_position[None, :] + self.tentacle_offsets

        # Check done
        if self.kraken_health <= 0 or np.all(self.ship_healths <= 0) or self.steps >= MAX_STEPS:
            done = True
            reward += 1000 if self.kraken_health <= 0 else -1000

        return self._get_obs(), reward / NUM_SHIPS, done, False, {}

    def get_state_for_viz(self):
        return {
            "ships": [{"pos": pos.tolist(), "health": health} for pos, health in zip(self.ship_positions, self.ship_healths)],
            "kraken": {"pos": self.kraken_position.tolist(), "health": self.kraken_health},
            "tentacles": self.tentacle_positions.tolist(),
            "grid_size": GRID_SIZE
        }

class WebSocketCallback(BaseCallback):
    def __init__(self, websocket, verbose=0):
        super().__init__(verbose)
        self.websocket = websocket
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # Send state every few steps
        if self.n_calls % 10 == 0:
            state = self.training_env.get_attr("get_state_for_viz")[0]()
            asyncio.run(self.websocket.send_json({"type": "train_step", "state": state}))

        # Collect rewards
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        if done:
            self.episode_rewards.append(self.locals['infos'][0].get('episode', {}).get('r', 0))
            self.episode_lengths.append(self.locals['infos'][0].get('episode', {}).get('l', 0))
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths)
            asyncio.run(self.websocket.send_json({
                "type": "progress",
                "episode": self.num_timesteps // 1000,  # Approximate episode count
                "reward": float(avg_reward),
                "loss": 0.0  # Placeholder, as SB3 doesn't directly provide loss here; can be extended if needed
            }))
            self.episode_rewards = []
            self.episode_lengths = []

async def train_pirate_ship(websocket: WebSocket):
    env = make_vec_env(PirateShipEnv, n_envs=8)
    model = PPO("MlpPolicy", env, verbose=1)
    callback = WebSocketCallback(websocket)
    model.learn(total_timesteps=100000, callback=callback)
    model.save("pirate_ship_policy")
    await websocket.send_json({"type": "trained"})

async def run_pirate_ship(websocket: WebSocket):
    model = PPO.load("pirate_ship_policy")
    env = PirateShipEnv(training_mode=False)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        await websocket.send_json({"type": "run_step", "state": env.get_state_for_viz()})
        await asyncio.sleep(0.05)
