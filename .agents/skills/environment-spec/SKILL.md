---
name: environment-spec
description: Create a rigorous RL environment specification before implementation.
argument-hint: "[task-id] [new|modify]"
allowed-tools: Read, Write, Glob, Bash
context: repo
agent: three-mlagents-environment-architect
---

# Environment Specification Skill

Use this skill before adding or materially changing an environment.

## Required Inputs

Read:

- `CLAUDE.md`
- `backend/mlagents/registry.py`
- `backend/mlagents/envs.py`
- relevant `backend/examples/*.py`
- relevant `client/src/examples/*.jsx`
- `.agents/templates/environment-proposal.md`

## Specification Requirements

Define the environment as a Markov decision process or partially observable
decision process:

- agent count and control frequency;
- observation space with shape, dtype, bounds, and normalization;
- action space with type, shape, legal values, and action masking if applicable;
- transition dynamics and stochasticity;
- reward terms with units and signs;
- terminated conditions for task success/failure;
- truncated conditions for time limits or external cutoffs;
- reset options and seed behavior;
- info dictionary fields, metrics, and frontend state payload;
- baseline algorithm and why it matches the action space;
- evaluation protocol, seeds, episode count, and reward threshold;
- optional dependencies and graceful failure behavior.

For Gymnasium tasks, follow the current reset/step contract:
`reset(seed=...) -> (observation, info)` and
`step(action) -> (observation, reward, terminated, truncated, info)`.

## Game And Evaluation Priority

Use the registry as source of truth. For new game-like environments, prefer
benchmarks that add scientific coverage rather than duplicate existing tasks:

- small deterministic sanity tasks for plumbing only;
- sparse-reward navigation and manipulation;
- arcade-style long-horizon control;
- memory/exploration tasks with explicit partial observability;
- physics-heavy control tasks with measurable units;
- external standard environments when dependencies and licenses are acceptable;
- PettingZoo-style MARL tasks for simultaneous or turn-based multi-agent games.

Do not treat a visual demo as a solved benchmark until it has a reproducible
backend environment, train/evaluate API, baseline policy, metrics, and tests.

## Output

Fill `.agents/templates/environment-proposal.md` or include equivalent content
in the implementation notes. The proposal should be specific enough that a
different engineer can implement the backend and frontend without inventing the
scientific contract mid-stream.
