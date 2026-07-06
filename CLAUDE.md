# CLAUDE.md

This file gives coding agents the working context for this repository. Read it
before making changes. The project is intentionally moving toward publishable,
scientific reinforcement learning work, so prefer real implementations,
standard libraries, reproducible verification, and clear research boundaries.

## Project Summary

Three ML-Agents is a browser-first visualization project with a Python-first
reinforcement learning backend. The original inspiration is Unity ML-Agents, but
new training work should live in Python using open scientific RL standards.

The browser is not the training source of truth. React and Three.js scenes are
interactive visualizations and WebSocket clients. Training, evaluation, task
metadata, saved policies, and experiment metadata belong in `backend/`.

Current scientific baseline stack:

- Python >= 3.11
- FastAPI for HTTP/WebSocket service
- Gymnasium environments and adapters
- Stable-Baselines3 for supported single-agent training/evaluation
- PyTorch through SB3
- TensorBoard/monitor logs for run metadata

Do not reintroduce browser-side training, hand-rolled PPO loops, ONNX inference,
or Unity ML-Agents bridge code unless explicitly asked and scientifically
justified.

## Repository Layout

```text
.
├── README.md
├── CLAUDE.md
├── backend/
│   ├── main.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── mlagents/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── envs.py
│   │   ├── registry.py
│   │   ├── training.py
│   │   └── websocket_training.py
│   ├── examples/
│   ├── policies/
│   ├── services/
│   └── tests/
└── client/
    ├── package.json
    ├── vite.config.js
    ├── public/
    └── src/
        ├── App.jsx
        ├── components/
        └── examples/
```

Important package name: the backend research package is `mlagents`, located at
`backend/mlagents`. The old package name `three_mlagents` should not be used.

The public project, GitHub Pages base path, and CLI command still use the
hyphenated project name `three-mlagents`; do not mechanically rename those URL
or command strings.

## Backend Architecture

`backend/mlagents` is the canonical training/evaluation package.

- `registry.py`: task registry, task cards, trainable/roadmap status, default
  algorithms, research tier, policy prefixes, timestep budgets, and eval
  episodes.
- `envs.py`: Gymnasium adapters and environment factories. This is where legacy
  Python example dynamics are wrapped to comply with Gymnasium reset/step
  contracts.
- `training.py`: SB3 train/evaluate/load/predict utilities, run directories,
  policy saving, metadata writing, monitor logs, TensorBoard logs, and CLI
  support.
- `websocket_training.py`: compatibility layer that lets existing demo
  WebSocket routes call the shared SB3 training path.
- `cli.py`: command line runner exposed as `three-mlagents`.

`backend/main.py` exposes:

- `/health`
- `/tasks`
- `/tasks/{task_id}`
- `/tasks/{task_id}/train`
- `/tasks/{task_id}/evaluate`
- legacy demo WebSockets such as `/ws/basic`, `/ws/ball3d`, `/ws/glider`, etc.

Supported training routes must use the shared `mlagents.training` and
`mlagents.websocket_training` APIs. Do not add one-off training loops inside
`backend/examples/*`.

## Frontend Architecture

The client is a Vite + React + Three.js app.

- `client/src/examples`: route-level scene implementations.
- `client/src/components`: shared UI/3D helpers.
- `client/src/config.js`: API/WebSocket base URLs.
- `client/vite.config.js`: GitHub Pages base path `/three-mlagents/`.

The frontend should:

- render environment state,
- open WebSocket connections for demo state/training progress,
- pass selected `model_filename` back to backend run commands when applicable,
- show roadmap status for unsupported research demos,
- avoid claiming unsupported Train/Run functionality.

The frontend should not:

- implement training,
- contain ML-Agents bridge code,
- contain ONNX runtime inference,
- advertise roadmap demos as paper-grade evals.

## Task Status and Research Boundaries

There are two classes of examples.

### Standardized Gymnasium/SB3 Tasks

These tasks are intended to train/evaluate through `backend/mlagents`:

- `basic`
- `ball3d`
- `gridworld`
- `push`
- `walljump`
- `brickbreak`
- `bicycle`
- `glider`
- `labyrinth`
- `astrodynamics`
- `kraken`
- `ant`
- `worm`

The precise source of truth is `backend/mlagents/registry.py`, not this file.
If you add or change a task, update the registry and tests.

### Roadmap / Non-Standardized Demos

These demos may visualize simulation state, but should not expose fake
paper-grade training:

- `foodcollector`
- `fish`
- `intersection`
- `minecraft`
- `simcity`
- `self-driving-car`

Roadmap multi-agent work should move to PettingZoo-compatible environments
before being treated as real MARL research. For simultaneous-action settings,
prefer PettingZoo `ParallelEnv` semantics. For serious MARL baselines, use
appropriate open tooling such as PettingZoo/SuperSuit with a well-documented
IPPO/MAPPO/RLlib/CleanRL-style pipeline. Do not force multi-agent problems
through single-agent SB3 wrappers without explicitly documenting the modeling
compromise.

Geospatial and LLM demos may require optional dependencies or API keys. If a
dependency is optional, keep the route failure graceful and explicit.

## Development Commands

Backend setup:

```bash
cd backend
uv sync
```

Backend server:

```bash
cd backend
uv run uvicorn main:app --reload
```

Backend CLI:

```bash
cd backend
uv run three-mlagents list --trainable-only
uv run three-mlagents inspect gridworld
uv run three-mlagents train basic --algorithm dqn --timesteps 25000 --seed 1
uv run three-mlagents evaluate basic policies/<model>.zip --episodes 50
```

Frontend setup:

```bash
cd client
pnpm install
```

Frontend server:

```bash
cd client
pnpm run dev
```

Production build:

```bash
cd client
pnpm run build
```

## Verification Checklist

Run the narrowest useful checks while developing, then run the broader checks
before handing work back.

Backend lint and format:

```bash
uvx ruff check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uvx ruff format --check backend/examples backend/mlagents backend/main.py backend/services backend/tests
```

Backend unit tests:

```bash
uv run --project backend python -m unittest discover -s backend/tests
```

Backend import smoke after package changes:

```bash
uv run --project backend python - <<'PY'
import importlib
for name in [
    "mlagents",
    "mlagents.registry",
    "mlagents.envs",
    "mlagents.training",
    "mlagents.websocket_training",
    "mlagents.cli",
]:
    module = importlib.import_module(name)
    print(name, module.__file__)
PY
```

CLI smoke:

```bash
uv run --project backend three-mlagents list --trainable-only
```

Frontend build:

```bash
cd client
pnpm run build
```

If browser route verification tooling exists in the working tree, run it against
live backend/frontend servers. Use `VITE_API_BASE_URL` and `VITE_WS_BASE_URL` to
point the client at the local backend.

## SB3 and Gymnasium Standards

Use Gymnasium's current API:

- `reset(seed=...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- distinguish true termination from time-limit truncation.

Use SB3 through `mlagents.training`:

- choose algorithms based on action space and task class,
- preserve model metadata,
- save policies as SB3 `.zip` files under `backend/policies`,
- write run metadata under `backend/runs`,
- prefer deterministic evaluation unless testing exploration behavior.

Do not save or serve ONNX models for supported tasks. The current policy format
is SB3 `.zip`.

## Generated Files and Git Hygiene

Do not commit generated experiment artifacts:

- `backend/policies/*` except `.gitkeep`
- `backend/runs/`
- TensorBoard event files
- monitor CSVs
- Python caches
- Vite `dist/`
- Playwright reports and test results
- local `.env*` files except `.env.template`

Relevant ignore files:

- root `.gitignore`
- `backend/.gitignore`
- `client/.gitignore`

When you run builds/tests, clean generated artifacts before final response if
they are not meant to be committed.

## Coding Rules for Future Agents

- Keep training in Python.
- Prefer existing local patterns and `backend/mlagents` helpers.
- Do not add toy-only implementations when the user asks for a real feature.
- Do not reintroduce dead bridge code, unused custom training loops, or fake UI
  Train/Run controls.
- Keep roadmap status explicit in both backend registry and frontend UI.
- Add tests when changing registry, adapters, training helpers, or API behavior.
- Use `rg` for searches.
- Use `apply_patch` for manual file edits.
- Avoid destructive git commands. This worktree may contain user changes.
- If you rename packages or files, update imports, docs, packaging metadata,
  tests, and lockfiles in the same change.

## Known Caveats

- Some roadmap visualizations depend on optional geospatial or LLM stacks and
  may render a graceful backend-unavailable state if those packages/API keys are
  absent.
- Vite builds may warn about large map/deck.gl chunks. That is not a test
  failure by itself.
- The GitHub Pages route base is `/three-mlagents/`; keep it unless the deploy
  target changes.

