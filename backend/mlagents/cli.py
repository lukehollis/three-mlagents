"""Command-line research runner for Three ML-Agents."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .registry import list_task_cards, make_env
from .training import TrainConfig, evaluate_model, train_task


def main() -> None:
    parser = argparse.ArgumentParser(prog="three-mlagents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered tasks")
    list_parser.add_argument("--trainable-only", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print one environment's spaces"
    )
    inspect_parser.add_argument("task")

    train_parser = subparsers.add_parser("train", help="Train one task with SB3")
    train_parser.add_argument("task")
    train_parser.add_argument("--algorithm", "-a")
    train_parser.add_argument("--timesteps", "-t", type=int)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--n-envs", type=int)
    train_parser.add_argument("--eval-episodes", type=int)
    train_parser.add_argument("--eval-freq", type=int, default=10_000)
    train_parser.add_argument("--run-name")
    train_parser.add_argument("--quiet", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an SB3 policy zip")
    eval_parser.add_argument("task")
    eval_parser.add_argument("model")
    eval_parser.add_argument("--episodes", type=int)
    eval_parser.add_argument("--seed", type=int, default=10_001)
    eval_parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        print(
            json.dumps(
                list_task_cards(include_roadmap=not args.trainable_only), indent=2
            )
        )
        return

    if args.command == "inspect":
        env = make_env(args.task)
        try:
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "observation_space": repr(env.observation_space),
                        "action_space": repr(env.action_space),
                    },
                    indent=2,
                )
            )
        finally:
            env.close()
        return

    if args.command == "train":
        result = train_task(
            TrainConfig(
                task_id=args.task,
                total_timesteps=args.timesteps,
                algorithm=args.algorithm,
                seed=args.seed,
                n_envs=args.n_envs,
                eval_episodes=args.eval_episodes,
                eval_freq=args.eval_freq,
                run_name=args.run_name,
                verbose=0 if args.quiet else 1,
            )
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "evaluate":
        result = evaluate_model(
            args.task,
            args.model,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
