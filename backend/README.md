# Three ML-Agents Backend

The backend package owns training and evaluation. Frontend scenes render policies
and interactive state, but experiments should be run through Python.

## Commands

```bash
uv run three-mlagents list --trainable-only
uv run three-mlagents inspect gridworld
uv run three-mlagents train basic --algorithm dqn --timesteps 25000 --seed 1
uv run three-mlagents evaluate basic policies/<model>.zip --episodes 50
uv run uvicorn main:app --reload
```

## Runtime Layout

- `mlagents/registry.py`: task cards, default algorithms, benchmark status.
- `mlagents/envs.py`: Gymnasium adapters for Python-owned environments.
- `mlagents/training.py`: Stable-Baselines3 train/evaluate/load helpers.
- `mlagents/websocket_training.py`: compatibility layer for existing demo WebSockets.
- `runs/`: per-run metadata, monitor logs, eval logs, and TensorBoard logs.
- `policies/`: saved SB3 policy zip files served to the browser/API.

## HTTP Endpoints

- `GET /tasks`: list task cards.
- `GET /tasks/{task_id}`: inspect one task card.
- `POST /tasks/{task_id}/train`: run a headless SB3 training job.
- `POST /tasks/{task_id}/evaluate`: evaluate a saved SB3 policy.

The legacy demo WebSockets still exist, but the standardized single-agent demo
trainers now call the SB3-backed Python research layer.
