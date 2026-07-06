"""Gymnasium adapters for the Python training surface.

These wrappers intentionally keep environment dynamics in Python. The frontend
may render the same state, but learning code should use Gymnasium/SB3 contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


SMALL_GOAL = 7
LARGE_GOAL = 17
MIN_POS = 0
MAX_POS = 20
START_POS = 10


def position_to_onehot(position: int) -> np.ndarray:
    obs = np.zeros(MAX_POS - MIN_POS + 1, dtype=np.float32)
    obs[int(np.clip(position, MIN_POS, MAX_POS)) - MIN_POS] = 1.0
    return obs


class BasicMoveToGoalEnv(gym.Env):
    """Gymnasium version of the Unity ML-Agents Basic move-to-goal task."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 50):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(MAX_POS - MIN_POS + 1,),
            dtype=np.float32,
        )
        self.position = START_POS
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.position = int((options or {}).get("position", START_POS))
        self.position = int(np.clip(self.position, MIN_POS, MAX_POS))
        self.steps = 0
        return position_to_onehot(self.position), self._info()

    def step(self, action: int):
        delta = (-1, 0, 1)[int(action)]
        self.position = int(np.clip(self.position + delta, MIN_POS, MAX_POS))
        self.steps += 1

        reward = -0.01
        terminated = False
        if self.position == SMALL_GOAL:
            reward += 0.1
            terminated = True
        elif self.position == LARGE_GOAL:
            reward += 1.0
            terminated = True

        truncated = self.steps >= self.max_episode_steps and not terminated
        return (
            position_to_onehot(self.position),
            reward,
            terminated,
            truncated,
            self._info(),
        )

    def _info(self) -> dict[str, Any]:
        return {"position": self.position, "steps": self.steps}


class LegacySingleAgentGymAdapter(gym.Env):
    """Adapt a Python example with ``reset()`` and ``step(action)`` to Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_ctor: Callable[[], Any],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *,
        max_episode_steps: int | None = None,
        action_transform: Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        self.env_ctor = env_ctor
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps
        self.action_transform = action_transform or (lambda action: action)
        self.env = env_ctor()
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # Most legacy examples use numpy's module-level RNG.
            np.random.seed(seed)
        self.env = self.env_ctor()
        obs = self.env.reset()
        self.steps = 0
        return np.asarray(obs, dtype=np.float32), self._info()

    def step(self, action: Any):
        env_action = self.action_transform(action)
        result = self.env.step(env_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            self.steps += 1
            return (
                np.asarray(obs, dtype=np.float32),
                reward,
                terminated,
                truncated,
                info,
            )

        obs, reward, done = result
        self.steps += 1
        hit_time_limit = (
            self.max_episode_steps is not None and self.steps >= self.max_episode_steps
        )
        terminated = bool(done and not hit_time_limit)
        truncated = bool(hit_time_limit)
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            self._info(),
        )

    def _info(self) -> dict[str, Any]:
        info = {"steps": self.steps}
        state_for_viz = getattr(self.env, "get_state_for_viz", None)
        if callable(state_for_viz):
            info["state"] = state_for_viz()
        return info


def make_basic_env() -> gym.Env:
    return BasicMoveToGoalEnv()


def make_ball3d_env() -> gym.Env:
    from examples.ball3d import Ball3DEnv, MAX_STEPS_PER_EP, NUM_ACTIONS, OBS_SIZE

    return LegacySingleAgentGymAdapter(
        Ball3DEnv,
        spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32),
        spaces.Discrete(NUM_ACTIONS),
        max_episode_steps=MAX_STEPS_PER_EP,
        action_transform=lambda action: int(action),
    )


def make_gridworld_env() -> gym.Env:
    from examples.gridworld import GridWorldEnv, MAX_STEPS_PER_EP, NUM_ACTIONS, OBS_SIZE

    return LegacySingleAgentGymAdapter(
        GridWorldEnv,
        spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32),
        spaces.Discrete(NUM_ACTIONS),
        max_episode_steps=MAX_STEPS_PER_EP,
        action_transform=lambda action: int(action),
    )


def make_push_env() -> gym.Env:
    from examples.push import MAX_STEPS_PER_EP, NUM_ACTIONS, OBS_SIZE, PushEnv

    return LegacySingleAgentGymAdapter(
        PushEnv,
        spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32),
        spaces.Discrete(NUM_ACTIONS),
        max_episode_steps=MAX_STEPS_PER_EP,
        action_transform=lambda action: int(action),
    )


def make_walljump_env() -> gym.Env:
    from examples.walljump import MAX_STEPS, NUM_ACTIONS, OBS_SIZE, WallJumpEnv

    return LegacySingleAgentGymAdapter(
        WallJumpEnv,
        spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32),
        spaces.Discrete(NUM_ACTIONS),
        max_episode_steps=MAX_STEPS,
        action_transform=lambda action: int(action),
    )


def make_brick_break_env() -> gym.Env:
    from examples.brick_break import BrickBreakEnv

    probe = BrickBreakEnv()
    obs_shape = np.asarray(probe.reset(), dtype=np.float32).shape
    return LegacySingleAgentGymAdapter(
        BrickBreakEnv,
        spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32),
        spaces.Discrete(3),
        max_episode_steps=2000,
        action_transform=lambda action: int(action),
    )


def make_bicycle_env() -> gym.Env:
    from examples.bicycle import BicycleEnv

    probe = BicycleEnv()
    obs_shape = np.asarray(probe.reset(), dtype=np.float32).shape
    return LegacySingleAgentGymAdapter(
        BicycleEnv,
        spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32),
        spaces.Discrete(3),
        max_episode_steps=2000,
        action_transform=lambda action: int(action),
    )


def make_glider_env() -> gym.Env:
    from examples.glider import GliderEnv

    probe = GliderEnv()
    obs_shape = np.asarray(probe.reset(), dtype=np.float32).shape
    return LegacySingleAgentGymAdapter(
        GliderEnv,
        spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32),
        spaces.Discrete(5),
        max_episode_steps=4000,
        action_transform=lambda action: int(action),
    )


def make_labyrinth_env() -> gym.Env:
    from examples.labyrinth import LabyrinthEnv

    return LabyrinthEnv(training_mode=True)


def make_astrodynamics_env() -> gym.Env:
    from examples.astrodynamics import AstrodynamicsEnv

    return AstrodynamicsEnv()


def make_kraken_env() -> gym.Env:
    from examples.kraken import PirateShipEnv

    return PirateShipEnv()


def make_ant_env() -> gym.Env:
    import gymnasium as gym

    return gym.make("Ant-v5", exclude_current_positions_from_observation=True)


def make_swimmer_env() -> gym.Env:
    import gymnasium as gym

    return gym.make("Swimmer-v5", exclude_current_positions_from_observation=True)
