# Scientific ML Rules

These rules apply to all backend and environment work.

## Non-Negotiables

- Training and evaluation live in Python.
- `backend/mlagents` is the backend package. Do not use `three_mlagents`
  imports.
- Use Gymnasium for single-agent environments.
- Use Stable-Baselines3 for supported single-agent baselines.
- Use PettingZoo-compatible APIs for true multi-agent work.
- Use the shared training and WebSocket helpers instead of route-local training
  loops.
- Do not reintroduce ONNX/browser policy execution as the source of truth.
- Do not claim paper-grade results from smoke tests.

## Environment Standards

Every trainable environment needs:

- clear observation and action spaces;
- correct reset/step contracts;
- deterministic seeding support;
- explicit reward, termination, and truncation semantics;
- registry metadata, default algorithm, and evaluation episodes;
- tests that cover reset, step, registry, and API behavior;
- frontend state contract if the route visualizes live state.

## Claims And Documentation

Use precise labels:

- "standardized" for backend train/evaluate tasks with tests;
- "roadmap" for visual or conceptual demos without a real baseline;
- "sanity check" for short training runs;
- "benchmark" only when the protocol, seeds, metrics, and run budget are
  documented.

When adding or upgrading dependencies, verify current official documentation
before changing the scientific stack.
