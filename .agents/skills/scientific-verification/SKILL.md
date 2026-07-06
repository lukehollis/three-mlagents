---
name: scientific-verification
description: Verify ML correctness, library usage, formatting, tests, dead code, and client/backend integration.
argument-hint: "[scope]"
allowed-tools: Read, Write, Glob, Bash
context: repo
agent: three-mlagents-evaluation-scientist
---

# Scientific Verification Skill

Use this skill before declaring work complete.

## Backend Checks

Run the broad backend checks unless the change is documentation-only:

```bash
uvx ruff check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uvx ruff format --check backend/examples backend/mlagents backend/main.py backend/services backend/tests
uv run --project backend python -m unittest discover -s backend/tests
uv run --project backend three-mlagents list --trainable-only
```

For a changed trainable task, also inspect the task and run a short fixed-seed
train/evaluate sanity pass. If full benchmark training is too expensive, state
that clearly and do not claim final performance.

## Frontend Checks

Run:

```bash
cd client && pnpm run build
```

For route or WebSocket changes, verify with live backend and frontend servers.
Confirm that trainable tasks call real backend routes and roadmap tasks do not
show fake training controls.

## Scientific Review

Check:

- Gymnasium API compliance and dtype/bounds correctness.
- SB3 algorithm compatibility with observation and action spaces.
- seed handling and reproducibility.
- termination versus truncation semantics.
- reward signs, units, and scaling.
- episode metrics and evaluation protocol.
- task registry metadata, research tier, and publication role.
- optional dependency failures are graceful.
- frontend claims match backend capability.

## Dead-Code And Artifact Review

Search for:

```bash
rg "three_mlagents|backend/three_mlagents|onnxruntime|hand-rolled|fake policy|TODO" .
rg "console\\.log|debugger" client/src
git status --short
```

Do not remove unrelated user changes. Do not commit generated policies, caches,
build output, TensorBoard runs, or dependency directories.

## Completion Standard

Smoke tests can support the evidence, but they do not satisfy a full
implementation request by themselves. Completion requires code, docs, tests, and
client/backend behavior to agree.
