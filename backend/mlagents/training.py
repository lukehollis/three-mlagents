"""Stable-Baselines3 training and evaluation utilities."""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import stable_baselines3
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed

from .registry import TaskSpec, get_task, make_env


POLICIES_DIR = Path("policies")
RUNS_DIR = Path("runs")

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


@dataclass(frozen=True)
class TrainConfig:
    task_id: str
    total_timesteps: int | None = None
    algorithm: str | None = None
    seed: int = 1
    n_envs: int | None = None
    eval_episodes: int | None = None
    eval_freq: int = 10_000
    deterministic_eval: bool = True
    policy: str | None = None
    run_name: str | None = None
    save_policy: bool = True
    verbose: int = 1


@dataclass(frozen=True)
class TrainResult:
    task_id: str
    algorithm: str
    run_id: str
    model_filename: str
    model_path: str
    run_dir: str
    mean_reward: float
    std_reward: float
    eval_episodes: int
    total_timesteps: int
    metadata_path: str


def make_vector_env(
    task_id: str,
    *,
    n_envs: int,
    seed: int,
    monitor_dir: Path | None = None,
) -> VecEnv:
    env_fns = []
    for rank in range(n_envs):
        env_seed = seed + rank

        def _init(env_seed: int = env_seed, rank: int = rank):
            env = make_env(task_id)
            env.reset(seed=env_seed)
            monitor_file = str(monitor_dir / f"{rank}") if monitor_dir else None
            return Monitor(env, filename=monitor_file)

        env_fns.append(_init)
    return DummyVecEnv(env_fns)


def make_eval_env(task_id: str, *, seed: int) -> gymnasium.Env:
    env = make_env(task_id)
    env.reset(seed=seed)
    return Monitor(env)


def train_task(
    config: TrainConfig,
    *,
    callback: BaseCallback | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> TrainResult:
    task = get_task(config.task_id)
    if not task.trainable:
        raise ValueError(
            f"Task '{task.id}' is not trainable through Gymnasium/SB3 yet."
        )

    algorithm_name = (config.algorithm or task.default_algorithm).lower()
    if algorithm_name not in ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm_name}'. Use one of {sorted(ALGORITHMS)}."
        )

    total_timesteps = config.total_timesteps or task.total_timesteps
    n_envs = config.n_envs or task.n_envs
    if algorithm_name in {"dqn", "sac", "td3"}:
        n_envs = 1

    eval_episodes = config.eval_episodes or task.eval_episodes
    run_id = config.run_name or _make_run_id(task.id, algorithm_name)
    run_dir = RUNS_DIR / task.id / run_id
    monitor_dir = run_dir / "monitor"
    eval_dir = run_dir / "eval"
    tb_dir = run_dir / "tb"
    for path in (POLICIES_DIR, run_dir, monitor_dir, eval_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    set_random_seed(config.seed)
    train_env = make_vector_env(
        task.id, n_envs=n_envs, seed=config.seed, monitor_dir=monitor_dir
    )
    eval_env = make_eval_env(task.id, seed=config.seed + 10_000)

    try:
        policy = config.policy or _default_policy(task)
        algo_cls = ALGORITHMS[algorithm_name]
        kwargs = _default_model_kwargs(
            algorithm_name,
            train_env=train_env,
            task=task,
            total_timesteps=total_timesteps,
            tensorboard_log=str(tb_dir),
            verbose=config.verbose,
        )
        if model_kwargs:
            kwargs.update(model_kwargs)

        model = algo_cls(policy, train_env, seed=config.seed, **kwargs)
        callbacks: list[BaseCallback] = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(eval_dir),
                eval_freq=max(1, config.eval_freq // max(1, n_envs)),
                n_eval_episodes=eval_episodes,
                deterministic=config.deterministic_eval,
                verbose=config.verbose,
                warn=True,
            )
        ]
        if callback is not None:
            callbacks.append(callback)

        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )

        model_filename = f"{task.policy_prefix}_{run_id}.zip"
        model_path = POLICIES_DIR / model_filename
        if config.save_policy:
            model.save(model_path)

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=config.deterministic_eval,
            return_episode_rewards=True,
            warn=True,
        )
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        metadata = {
            "task": task.card(),
            "config": asdict(config),
            "algorithm": algorithm_name,
            "run_id": run_id,
            "model_filename": model_filename,
            "model_path": str(model_path),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_lengths": [int(length) for length in episode_lengths],
            "software": {
                "python": platform.python_version(),
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return TrainResult(
            task_id=task.id,
            algorithm=algorithm_name,
            run_id=run_id,
            model_filename=model_filename,
            model_path=str(model_path),
            run_dir=str(run_dir),
            mean_reward=mean_reward,
            std_reward=std_reward,
            eval_episodes=eval_episodes,
            total_timesteps=total_timesteps,
            metadata_path=str(metadata_path),
        )
    finally:
        train_env.close()
        eval_env.close()


def evaluate_model(
    task_id: str,
    model_filename_or_path: str,
    *,
    episodes: int | None = None,
    deterministic: bool = True,
    seed: int = 10_001,
) -> dict[str, Any]:
    task = get_task(task_id)
    model = load_model(task, model_filename_or_path)
    eval_env = make_eval_env(task.id, seed=seed)
    try:
        n_eval_episodes = episodes or task.eval_episodes
        rewards, lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
            warn=True,
        )
        return {
            "task_id": task.id,
            "model": str(_resolve_model_path(task, model_filename_or_path)),
            "episodes": n_eval_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "episode_rewards": [float(r) for r in rewards],
            "episode_lengths": [int(length) for length in lengths],
        }
    finally:
        eval_env.close()


def load_model(
    task: TaskSpec, model_filename_or_path: str | None = None
) -> BaseAlgorithm:
    model_path = _resolve_model_path(task, model_filename_or_path)
    algorithm_name = (
        _infer_algorithm_from_metadata(task, model_path) or task.default_algorithm
    )
    algo_cls = ALGORITHMS[algorithm_name]
    return algo_cls.load(model_path)


def predict_action(
    task_id: str, obs: np.ndarray, model_filename: str | None = None
) -> int | list[float]:
    task = get_task(task_id)
    model = load_model(task, model_filename)
    obs = np.asarray(obs, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return int(action.item())
        return action.tolist()
    return int(action)


def latest_model_filename(task_id: str) -> str:
    task = get_task(task_id)
    matches = sorted(POLICIES_DIR.glob(f"{task.policy_prefix}_*.zip"), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No SB3 policy zip found for task '{task.id}'.")
    return matches[0].name


def _resolve_model_path(task: TaskSpec, model_filename_or_path: str | None) -> Path:
    if model_filename_or_path is None:
        model_filename_or_path = latest_model_filename(task.id)
    path = Path(model_filename_or_path)
    if path.exists():
        return path
    if not path.is_absolute():
        policy_path = POLICIES_DIR / path
        if policy_path.exists():
            return policy_path
        path = policy_path
    raise FileNotFoundError(f"Model not found: {path}")


def _infer_algorithm_from_metadata(task: TaskSpec, model_path: Path) -> str | None:
    stem = model_path.name.removesuffix(".zip")
    for metadata_path in (RUNS_DIR / task.id).glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            metadata.get("model_filename") == model_path.name
            or metadata.get("run_id") in stem
        ):
            algorithm = metadata.get("algorithm") or metadata.get("config", {}).get(
                "algorithm"
            )
            return (algorithm or task.default_algorithm).lower()
    return None


def _default_policy(task: TaskSpec) -> str:
    return "CnnPolicy" if task.observation == "image" else "MlpPolicy"


def _default_model_kwargs(
    algorithm_name: str,
    *,
    train_env: VecEnv,
    task: TaskSpec,
    total_timesteps: int,
    tensorboard_log: str,
    verbose: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "tensorboard_log": tensorboard_log,
        "verbose": verbose,
    }
    if algorithm_name == "dqn":
        env = train_env.envs[0]
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("DQN requires a discrete action space.")
        return {
            **common,
            "learning_rate": 3e-4,
            "buffer_size": max(25_000, min(500_000, total_timesteps)),
            "learning_starts": min(2_000, max(100, total_timesteps // 20)),
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.25,
            "exploration_final_eps": 0.03,
            "policy_kwargs": {"net_arch": [128, 128]},
        }
    if algorithm_name == "ppo":
        n_steps = 1024 if task.research_tier == "foundation" else 2048
        policy_kwargs: dict[str, Any] = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]}
        }
        if task.observation == "image":
            if task.id != "labyrinth":
                raise ValueError(
                    f"Task '{task.id}' needs task-specific CNN policy settings."
                )
            from examples.labyrinth import CustomCNN

            policy_kwargs = {
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            }
        return {
            **common,
            "learning_rate": 3e-4,
            "n_steps": n_steps,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": policy_kwargs,
        }
    if algorithm_name in {"sac", "td3"}:
        return {
            **common,
            "learning_rate": 3e-4,
            "buffer_size": max(100_000, min(1_000_000, total_timesteps)),
            "learning_starts": min(10_000, max(1_000, total_timesteps // 20)),
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    return common


def _make_run_id(task_id: str, algorithm_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task_id}_{algorithm_name}_{timestamp}_{uuid.uuid4().hex[:8]}"
