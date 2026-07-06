# Three ML-Agents Agent Pack

This directory is a repo-local agent operating pack for building Three
ML-Agents environments. It follows the same broad shape as image-blaster's
agent setup: small role files, deeper skill playbooks, shared rules, settings,
and reusable templates. This repository uses `.agents` rather than `.claude`
so the pack is tool-neutral and can be consumed by any coding agent.

This pack is metadata and process guidance only. It does not add runtime Python
or JavaScript code.

## Directory Layout

```text
.agents/
|-- agents/       # Frontmatter-driven role entrypoints
|-- roles/        # Human-readable role routing notes
|-- rules/        # Repository-wide non-negotiables
|-- skills/       # Task playbooks with concrete checklists
|-- templates/    # Spec and review templates
`-- settings.json # Machine-readable index for the pack
```

## Role Routing

Use these roles in sequence for new environments:

1. `three-mlagents-environment-architect`
   Define the task as an RL problem before code is written.
2. `three-mlagents-gymnasium-integrator`
   Implement the backend Gymnasium/SB3 surface in `backend/mlagents`.
3. `three-mlagents-frontend-visualizer`
   Add or update the React/Three route without moving training into the client.
4. `three-mlagents-evaluation-scientist`
   Verify scientific correctness, library usage, formatting, tests, and UI
   behavior.

Use `three-mlagents-roadmap-marl-planner` before implementation whenever the
request is multi-agent, LLM-agent, geospatial, traffic, ecological, self-driving,
or simulator-heavy. Those tasks often need PettingZoo, external simulators, or a
separate MARL pipeline instead of plain single-agent SB3.

## Current Research Stack

The backend package is `backend/mlagents`. Do not reintroduce the old
`backend/three_mlagents` package or `three_mlagents` imports.

The canonical single-agent stack is:

- Gymnasium environments and adapters.
- Stable-Baselines3 for supported single-agent training and evaluation.
- PyTorch through SB3.
- FastAPI and WebSockets for serving training/evaluation and demo state.
- TensorBoard/Monitor logs for experiment records.

The frontend stack is:

- Vite, React, and Three.js.
- WebSocket clients and visual state rendering.
- No browser-side training source of truth.

## Environment Priority

Treat `backend/mlagents/registry.py` as the source of truth. As of this pack,
the standardized Gymnasium/SB3 tasks are `basic`, `ball3d`, `gridworld`,
`push`, `walljump`, `brickbreak`, `bicycle`, `glider`, `labyrinth`,
`astrodynamics`, `kraken`, `ant`, and `worm`.

For game-like work, prioritize scientifically useful coverage:

- Foundation: `basic`, `ball3d`, `gridworld`.
- In-repo game/control benchmarks: `brickbreak`, `labyrinth`, `kraken`,
  `push`, `walljump`, `bicycle`.
- External standards: MuJoCo/Gymnasium tasks where optional dependencies are
  correctly declared and graceful failures are documented.
- Roadmap MARL/simulator work: `foodcollector`, `fish`, `intersection`,
  `simcity`, `self-driving-car`, and any new simultaneous-agent task.

Do not label roadmap tasks as publishable evals until the environment API,
baselines, seeds, metrics, and dependency story are complete.

## Required Verification

For code changes, the final pass should normally include:

```bash
uvx ruff check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uvx ruff format --check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uv run --project backend python -m unittest discover -s backend/tests
uv run --project backend three-mlagents list --trainable-only
cd client && pnpm run build
```

If an environment route was changed, verify the backend and frontend together
with live servers. Smoke tests are useful but are not enough when the request is
for a full implementation.

## Source Pattern

The structure was modeled after image-blaster's `.claude` tree, which separates
agent entrypoints, skills, rules, scripts/hooks, and settings. The content here
is specific to Three ML-Agents and its Python-first scientific RL direction.
