"""WebSocket compatibility helpers for Python-first SB3 training."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketState
from stable_baselines3.common.callbacks import BaseCallback

from .registry import get_task
from .training import TrainConfig, predict_action, train_task


class WebSocketProgressCallback(BaseCallback):
    """Send coarse SB3 learning progress to an already accepted FastAPI socket."""

    def __init__(
        self,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        *,
        total_timesteps: int,
        progress_freq: int = 2_000,
    ):
        super().__init__()
        self.websocket = websocket
        self.loop = loop
        self.total_timesteps = max(1, total_timesteps)
        self.progress_freq = max(1, progress_freq)
        self._last_emit = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_emit < self.progress_freq:
            return True
        self._last_emit = self.num_timesteps
        payload = {
            "type": "progress",
            "episode": int(self.num_timesteps),
            "reward": None,
            "loss": None,
            "timesteps": int(self.num_timesteps),
            "progress": min(1.0, self.num_timesteps / self.total_timesteps),
            "algorithm": self.model.__class__.__name__,
        }
        asyncio.run_coroutine_threadsafe(self.websocket.send_json(payload), self.loop)
        return True


async def train_task_for_websocket(
    websocket: WebSocket,
    task_id: str,
    *,
    total_timesteps: int | None = None,
    algorithm: str | None = None,
    seed: int = 1,
    n_envs: int | None = None,
    eval_episodes: int | None = None,
    eval_freq: int = 10_000,
    progress_freq: int = 2_000,
) -> dict[str, Any]:
    task = get_task(task_id)
    config = TrainConfig(
        task_id=task_id,
        total_timesteps=total_timesteps,
        algorithm=algorithm,
        seed=seed,
        n_envs=n_envs,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        verbose=0,
    )
    effective_timesteps = total_timesteps or task.total_timesteps
    loop = asyncio.get_running_loop()
    callback = WebSocketProgressCallback(
        websocket,
        loop,
        total_timesteps=effective_timesteps,
        progress_freq=progress_freq,
    )

    await websocket.send_json(
        {
            "type": "progress",
            "episode": 0,
            "reward": None,
            "loss": None,
            "timesteps": 0,
            "progress": 0.0,
            "algorithm": algorithm or "default",
            "task_id": task.id,
        }
    )
    result = await asyncio.to_thread(train_task, config, callback=callback)
    payload = {
        "type": "trained",
        "file_url": f"/policies/{result.model_filename}",
        "model_filename": result.model_filename,
        "timestamp": result.run_id,
        "session_uuid": result.run_id.rsplit("_", 1)[-1],
        "algorithm": result.algorithm,
        "mean_reward": result.mean_reward,
        "std_reward": result.std_reward,
        "eval_episodes": result.eval_episodes,
        "run_dir": result.run_dir,
        "metadata_path": result.metadata_path,
    }
    await websocket.send_json(payload)
    return asdict(result)


def predict_discrete_action(
    task_id: str,
    obs: np.ndarray | list[float],
    model_filename: str | None = None,
) -> int:
    action = predict_action(task_id, np.asarray(obs, dtype=np.float32), model_filename)
    if isinstance(action, list):
        if len(action) != 1:
            raise ValueError(
                f"Expected one discrete action for {task_id}, got {action}"
            )
        return int(action[0])
    return int(action)


def predict_policy_action(
    task_id: str,
    obs: np.ndarray | list[float],
    model_filename: str | None = None,
) -> int | list[float] | list[int]:
    """Return an SB3 action for discrete, continuous, or multi-discrete tasks."""

    return predict_action(task_id, np.asarray(obs, dtype=np.float32), model_filename)


async def run_policy_for_websocket(
    websocket: WebSocket,
    task_id: str,
    env_factory: Callable[[], Any],
    *,
    model_filename: str | None = None,
    action_transform: Callable[[Any], Any] | None = None,
    sleep_seconds: float = 0.03,
) -> None:
    """Run a saved SB3 policy in a visualization environment.

    The environment can expose either the legacy ``reset``/``step`` API used by
    the Three.js demos or the Gymnasium API. State is streamed from
    ``get_state_for_viz`` when available.
    """

    env = env_factory()
    episode = 0
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    transform = action_transform or (lambda action: action)

    while websocket.application_state == WebSocketState.CONNECTED:
        action = transform(predict_policy_action(task_id, obs, model_filename))
        result = env.step(action)
        if len(result) == 5:
            next_obs, _, terminated, truncated, _ = result
            done = bool(terminated or truncated)
        else:
            next_obs, _, done = result

        state_for_viz = getattr(env, "get_state_for_viz", None)
        payload: dict[str, Any] = {"type": "run_step", "episode": episode + 1}
        if callable(state_for_viz):
            payload["state"] = state_for_viz()
        await websocket.send_json(payload)
        await asyncio.sleep(sleep_seconds)

        if done:
            episode += 1
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        else:
            obs = next_obs
        await asyncio.sleep(0)


async def send_error(websocket: WebSocket, exc: Exception) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "message": str(exc)})
