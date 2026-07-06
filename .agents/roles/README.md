# Role Guide

The `agents/` files are the launchable role definitions. This directory records
how to select them during normal work.

## Use The Narrowest Role First

- New environment or major redesign: start with
  `three-mlagents-environment-architect`.
- Backend training, registry, policy, or WebSocket changes: use
  `three-mlagents-gymnasium-integrator`.
- React/Three route, panel, or visualization changes: use
  `three-mlagents-frontend-visualizer`.
- Scientific review, dead-code cleanup, tests, or release readiness: use
  `three-mlagents-evaluation-scientist`.
- Multi-agent, traffic, ecological, simulator, or LLM-agent work: start with
  `three-mlagents-roadmap-marl-planner`.

## Handoff Contract

Each role should leave explicit notes on:

- files changed;
- assumptions made;
- commands run;
- commands not run and why;
- remaining scientific limitations.

Do not hand off vague claims such as "works" or "paper grade" without the
evidence needed to support them.
