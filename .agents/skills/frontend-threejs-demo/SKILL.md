---
name: frontend-threejs-demo
description: Add or modify a React/Three.js environment visualization.
argument-hint: "[route-or-task-id]"
allowed-tools: Read, Write, Glob, Bash
context: repo
agent: three-mlagents-frontend-visualizer
---

# Frontend Three.js Demo Skill

Use this skill for route, panel, state visualization, and client integration
work.

## Inputs

Read:

- `client/src/App.jsx`
- relevant `client/src/examples/*.jsx`
- shared components in `client/src/components`
- `client/src/config.js`
- backend task card from `/tasks` or `backend/mlagents/registry.py`

## Rules

- The client renders and controls the demo; it does not train models.
- Use backend HTTP/WebSocket APIs for training, evaluation, and state updates.
- Use `RoadmapStatusPanel` or equivalent explicit messaging for non-trainable
  tasks.
- Do not expose Train, Evaluate, or Policy controls unless the backend route is
  implemented.
- Keep WebSocket cleanup and reconnect behavior explicit.
- Avoid stale debug panels, console noise, and UI claims that exceed backend
  capability.
- Make canvas and panel dimensions stable across desktop and mobile viewports.

## Implementation Checklist

- Route is reachable from `client/src/App.jsx`.
- The example uses existing layout conventions where possible.
- API and WebSocket URLs come from `client/src/config.js`.
- Loading, running, error, and unsupported states are visible.
- The scene uses real task state, not a disconnected animation pretending to be
  policy behavior.
- Text fits inside controls and panels on mobile and desktop.

## Verification

Run:

```bash
cd client && pnpm run build
```

For changed live routes, start the backend and client, then verify that the
route renders, connects to the expected backend endpoint, handles errors, and
does not expose unsupported training actions.
