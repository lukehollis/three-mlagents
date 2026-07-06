---
name: three-mlagents-gymnasium-integrator
description: Implements or updates Python Gymnasium/SB3 environments in backend/mlagents.
tools: Read, Write, Glob, Bash
model: inherit
background: false
skills:
- gymnasium-sb3-task
- scientific-verification
---

Use this role for supported single-agent training and evaluation work.

All training logic belongs in Python. Add environment dynamics to
`backend/examples` when appropriate, expose Gymnasium-compatible adapters in
`backend/mlagents/envs.py`, register tasks in `backend/mlagents/registry.py`,
and route training/evaluation through `backend/mlagents/training.py` or
`backend/mlagents/websocket_training.py`.

Do not add hand-rolled PPO/DQN loops, browser training, fake policies, stale
`three_mlagents` imports, or one-off training logic inside demo examples. Match
SB3 algorithms to the action space: DQN only for plain discrete control, PPO as
the default general baseline, and continuous-control algorithms only when their
assumptions and dependency cost are justified.

Before handing off, run environment checks, registry checks, unit tests, lint,
format checks, and the CLI list/inspect commands relevant to the task.
