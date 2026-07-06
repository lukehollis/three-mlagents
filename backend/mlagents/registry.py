"""Research task registry for Python training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import gymnasium as gym

from . import envs


Interface = Literal["gymnasium", "pettingzoo", "mlagents-llapi", "external"]
ResearchTier = Literal["foundation", "benchmark", "frontier", "roadmap"]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    family: str
    interface: Interface
    research_tier: ResearchTier
    default_algorithm: str
    policy_prefix: str
    total_timesteps: int
    eval_episodes: int = 20
    n_envs: int = 1
    reward_threshold: float | None = None
    tags: tuple[str, ...] = ()
    observation: str = "vector"
    action: str = "discrete"
    publication_role: str = "supporting"
    status: str = "standardized"
    notes: str = ""
    env_factory: Callable[[], gym.Env] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def trainable(self) -> bool:
        return self.interface == "gymnasium" and self.env_factory is not None

    def card(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("env_factory", None)
        data["trainable"] = self.trainable
        return data


TASKS: dict[str, TaskSpec] = {
    "basic": TaskSpec(
        id="basic",
        title="Basic Move-To-Goal",
        family="control",
        interface="gymnasium",
        research_tier="foundation",
        default_algorithm="dqn",
        policy_prefix="basic_policy",
        total_timesteps=25_000,
        eval_episodes=50,
        n_envs=1,
        reward_threshold=0.85,
        tags=("sparse-reward", "tabular-state", "unity-ml-agents"),
        publication_role="unit sanity check for action/observation plumbing",
        env_factory=envs.make_basic_env,
    ),
    "ball3d": TaskSpec(
        id="ball3d",
        title="3D Ball Balance",
        family="continuous-control",
        interface="gymnasium",
        research_tier="foundation",
        default_algorithm="ppo",
        policy_prefix="ball3d_policy",
        total_timesteps=150_000,
        eval_episodes=30,
        n_envs=8,
        reward_threshold=150.0,
        tags=("physics", "stability", "unity-ml-agents"),
        publication_role="browser/Unity parity smoke benchmark",
        env_factory=envs.make_ball3d_env,
    ),
    "gridworld": TaskSpec(
        id="gridworld",
        title="GridWorld Goal-Conditioned Navigation",
        family="navigation",
        interface="gymnasium",
        research_tier="foundation",
        default_algorithm="dqn",
        policy_prefix="gridworld_policy",
        total_timesteps=100_000,
        eval_episodes=100,
        n_envs=1,
        reward_threshold=0.75,
        tags=("goal-conditioned", "procedural-layout", "discrete-control"),
        publication_role="generalization and seed-control baseline",
        env_factory=envs.make_gridworld_env,
    ),
    "push": TaskSpec(
        id="push",
        title="Push Block",
        family="navigation",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="dqn",
        policy_prefix="push_policy",
        total_timesteps=200_000,
        eval_episodes=100,
        n_envs=1,
        reward_threshold=0.65,
        tags=("object-manipulation", "sparse-reward", "planning"),
        publication_role="single-agent manipulation transfer task",
        env_factory=envs.make_push_env,
    ),
    "walljump": TaskSpec(
        id="walljump",
        title="Wall Jump",
        family="navigation",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="dqn",
        policy_prefix="walljump_policy",
        total_timesteps=150_000,
        eval_episodes=100,
        n_envs=1,
        reward_threshold=0.7,
        tags=("conditional-skill", "exploration", "procedural-wall"),
        publication_role="conditional-control benchmark",
        env_factory=envs.make_walljump_env,
    ),
    "brickbreak": TaskSpec(
        id="brickbreak",
        title="Brick Break",
        family="arcade",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="ppo",
        policy_prefix="brickbreak_policy",
        total_timesteps=500_000,
        eval_episodes=50,
        n_envs=8,
        tags=("arcade", "partial-observability-lite", "long-horizon"),
        publication_role="small arcade control benchmark before ALE/Procgen",
        env_factory=envs.make_brick_break_env,
    ),
    "bicycle": TaskSpec(
        id="bicycle",
        title="Bicycle Balance and Navigation",
        family="continuous-control",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="ppo",
        policy_prefix="bicycle_policy",
        total_timesteps=500_000,
        eval_episodes=50,
        n_envs=8,
        tags=("underactuated-control", "stability", "navigation"),
        publication_role="control-system benchmark",
        env_factory=envs.make_bicycle_env,
    ),
    "glider": TaskSpec(
        id="glider",
        title="Dynamic Soaring Glider",
        family="aerospace",
        interface="gymnasium",
        research_tier="frontier",
        default_algorithm="ppo",
        policy_prefix="glider_policy",
        total_timesteps=1_000_000,
        eval_episodes=50,
        n_envs=8,
        tags=("aerodynamics", "energy-management", "long-horizon"),
        publication_role="domain-specific continuous physics case study",
        env_factory=envs.make_glider_env,
    ),
    "labyrinth": TaskSpec(
        id="labyrinth",
        title="Labyrinth / NetHack-Inspired Navigation",
        family="games",
        interface="gymnasium",
        research_tier="frontier",
        default_algorithm="ppo",
        policy_prefix="labyrinth_policy",
        total_timesteps=2_000_000,
        eval_episodes=100,
        n_envs=8,
        tags=("pixels", "maze", "memory", "exploration"),
        observation="image",
        publication_role="first serious game-like benchmark in this repo",
        env_factory=envs.make_labyrinth_env,
    ),
    "astrodynamics": TaskSpec(
        id="astrodynamics",
        title="Orbital Rendezvous and Docking",
        family="aerospace",
        interface="gymnasium",
        research_tier="frontier",
        default_algorithm="ppo",
        policy_prefix="astrodynamics_policy",
        total_timesteps=2_000_000,
        eval_episodes=50,
        n_envs=8,
        tags=("orbital-mechanics", "safety", "long-horizon"),
        publication_role="physics-heavy scientific case study",
        env_factory=envs.make_astrodynamics_env,
    ),
    "kraken": TaskSpec(
        id="kraken",
        title="Kraken Fleet Combat",
        family="games",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="ppo",
        policy_prefix="kraken_policy",
        total_timesteps=1_000_000,
        eval_episodes=50,
        n_envs=8,
        tags=("multi-unit-control", "coordination", "combat"),
        action="multi-discrete",
        publication_role="compact multi-unit control benchmark",
        env_factory=envs.make_kraken_env,
    ),
    "ant": TaskSpec(
        id="ant",
        title="MuJoCo Ant",
        family="continuous-control",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="ppo",
        policy_prefix="ant_policy",
        total_timesteps=3_000_000,
        eval_episodes=20,
        n_envs=8,
        tags=("mujoco", "locomotion", "external-standard"),
        action="continuous",
        publication_role="external control baseline",
        env_factory=envs.make_ant_env,
    ),
    "worm": TaskSpec(
        id="worm",
        title="MuJoCo Swimmer / Worm",
        family="continuous-control",
        interface="gymnasium",
        research_tier="benchmark",
        default_algorithm="ppo",
        policy_prefix="worm_policy",
        total_timesteps=2_000_000,
        eval_episodes=20,
        n_envs=8,
        tags=("mujoco", "locomotion", "external-standard"),
        action="continuous",
        publication_role="external control baseline",
        env_factory=envs.make_swimmer_env,
    ),
    "foodcollector": TaskSpec(
        id="foodcollector",
        title="Food Collector",
        family="multi-agent",
        interface="pettingzoo",
        research_tier="roadmap",
        default_algorithm="ippo",
        policy_prefix="foodcollector_policy",
        total_timesteps=2_000_000,
        tags=("multi-agent", "mixed-action", "competitive-cooperative"),
        action="hybrid",
        publication_role="PettingZoo conversion target",
        status="needs PettingZoo ParallelEnv wrapper before paper-grade training",
        notes="Do not force through single-agent SB3; use PettingZoo plus SuperSuit/RLlib/CleanRL IPPO/MAPPO.",
    ),
    "intersection": TaskSpec(
        id="intersection",
        title="Traffic Intersection",
        family="multi-agent",
        interface="pettingzoo",
        research_tier="frontier",
        default_algorithm="mappo",
        policy_prefix="intersection_policy",
        total_timesteps=5_000_000,
        tags=("multi-agent", "safety", "traffic", "social-dilemma"),
        publication_role="safety-critical MARL benchmark",
        status="needs PettingZoo ParallelEnv wrapper and safety metrics",
    ),
    "minecraft": TaskSpec(
        id="minecraft",
        title="Minecraft-Inspired Crafting World",
        family="open-ended-games",
        interface="pettingzoo",
        research_tier="frontier",
        default_algorithm="hierarchical-rl-plus-llm",
        policy_prefix="minecraft_policy",
        total_timesteps=10_000_000,
        tags=("crafting", "open-ended", "llm-agents", "multi-agent"),
        publication_role="open-ended agentic-game case study",
        status="needs PettingZoo wrapper, scripted baselines, and LLM ablation harness",
    ),
    "simcity": TaskSpec(
        id="simcity",
        title="SimCity Collaborative Construction",
        family="open-ended-games",
        interface="pettingzoo",
        research_tier="frontier",
        default_algorithm="hierarchical-rl-plus-llm",
        policy_prefix="simcity_policy",
        total_timesteps=10_000_000,
        tags=("collaboration", "llm-agents", "economy", "multi-agent"),
        publication_role="LLM/RL collaboration benchmark",
        status="needs PettingZoo wrapper and reproducible LLM transcript evaluation",
    ),
    "fish": TaskSpec(
        id="fish",
        title="Fish Schooling",
        family="multi-agent",
        interface="pettingzoo",
        research_tier="roadmap",
        default_algorithm="ippo",
        policy_prefix="fish_policy",
        total_timesteps=3_000_000,
        tags=("swarm", "predator-prey", "multi-agent"),
        publication_role="swarm behavior benchmark",
        status="needs PettingZoo wrapper and population-level metrics",
    ),
    "self-driving-car": TaskSpec(
        id="self-driving-car",
        title="Self-Driving Car Routing",
        family="safety",
        interface="pettingzoo",
        research_tier="frontier",
        default_algorithm="mappo",
        policy_prefix="self_driving_car_policy",
        total_timesteps=5_000_000,
        tags=("traffic", "interpretability", "safety", "multi-agent"),
        publication_role="interpretable safety case study",
        status="needs PettingZoo wrapper, scenario splits, and safety/regret metrics",
    ),
}


def list_tasks(*, include_roadmap: bool = True) -> list[TaskSpec]:
    tasks = list(TASKS.values())
    if not include_roadmap:
        tasks = [task for task in tasks if task.trainable]
    return sorted(tasks, key=lambda task: (task.family, task.id))


def list_task_cards(*, include_roadmap: bool = True) -> list[dict[str, Any]]:
    return [task.card() for task in list_tasks(include_roadmap=include_roadmap)]


def get_task(task_id: str) -> TaskSpec:
    normalized = task_id.lower().replace("_", "-")
    aliases = {
        "brick-break": "brickbreak",
        "food-collector": "foodcollector",
        "self_driving_car": "self-driving-car",
    }
    key = aliases.get(normalized, normalized)
    if key not in TASKS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {', '.join(sorted(TASKS))}"
        )
    return TASKS[key]


def make_env(task_id: str) -> gym.Env:
    task = get_task(task_id)
    if not task.trainable or task.env_factory is None:
        raise ValueError(f"Task '{task_id}' is not a Gymnasium/SB3 trainable task yet.")
    return task.env_factory()
