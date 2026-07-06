---
name: gymnasium-sb3-task
description: Add or modify a supported Gymnasium/SB3 task in backend/mlagents.
argument-hint: "[task-id] [new|modify|verify]"
allowed-tools: Read, Write, Glob, Bash
context: repo
agent: three-mlagents-gymnasium-integrator
---

# Gymnasium/SB3 Task Skill

Use this skill for single-agent tasks that should train through the Python
backend.

## Implementation Path

1. Read the environment proposal and existing local patterns.
2. Put dynamics in `backend/examples` only when they are reusable outside the
   training wrapper.
3. Expose a Gymnasium environment or adapter in `backend/mlagents/envs.py`.
4. Register the task in `backend/mlagents/registry.py`.
5. Route training and evaluation through `backend/mlagents/training.py` and
   `backend/mlagents/websocket_training.py`.
6. Add or update tests in `backend/tests`.
7. Update frontend metadata only after backend capabilities are real.

## Library Use

Use Gymnasium spaces and API contracts directly. Use Stable-Baselines3 for
baseline training and evaluation. Do not write custom PPO/DQN loops unless the
user explicitly asks for a research algorithm implementation and the code is
separated from the environment surface.

Algorithm defaults:

- `DQN`: plain `spaces.Discrete` tasks with low-dimensional observations.
- `PPO`: discrete, continuous, `MultiDiscrete`, or vectorized tasks where a
  robust general baseline is appropriate.
- `SAC` or `TD3`: continuous-control tasks only when off-policy training is part
  of the documented experiment plan.

Do not use DQN for continuous or `MultiDiscrete` actions. Do not flatten a
multi-agent task into one SB3 policy unless the compromise is explicitly
documented in the task card and UI.

## Correctness Checklist

- `reset(seed=...)` calls `super().reset(seed=seed)` for native Gymnasium envs.
- Observations are inside `observation_space` and use stable dtypes.
- Actions are validated or transformed at the adapter boundary.
- `terminated` means the MDP ended; `truncated` means a cutoff such as a time
  limit.
- `info` includes useful metrics and visualization state when needed.
- Time limits are explicit and documented.
- Vectorized training uses the registry `n_envs` setting where appropriate.
- Policy files, metadata, monitor logs, and TensorBoard logs are written through
  the shared training code.

## Required Verification

Run the focused backend checks for changed code:

```bash
uvx ruff check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uvx ruff format --check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uv run --project backend python -m unittest discover -s backend/tests
uv run --project backend three-mlagents inspect <task-id>
```

For a new trainable task, also run a short train/evaluate pass with a fixed seed
and report that it is a sanity check, not final benchmark evidence.
