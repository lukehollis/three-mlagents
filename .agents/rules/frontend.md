# Frontend Rules

The frontend is a visualization and control surface for Python training.

## Required Behavior

- Use Vite, React, and Three.js patterns already present in `client/src`.
- Use `client/src/config.js` for API and WebSocket base URLs.
- Render task state from backend responses or documented local simulation state.
- Clean up animation loops, event listeners, and WebSocket connections.
- Show clear unsupported/roadmap state for non-trainable demos.
- Keep layout stable and readable on mobile and desktop.

## Forbidden Behavior

- Do not train policies in the browser.
- Do not add fake Train/Evaluate buttons.
- Do not imply roadmap demos have validated backend training.
- Do not hide backend errors behind silent animations.
- Do not leave debug console output or `debugger` statements.
- Do not add large visual rewrites unrelated to the task.

## Route Readiness

A route is ready only when:

- it is reachable from app navigation;
- it renders without console-breaking errors;
- it uses correct API/WebSocket endpoints;
- controls map to real backend capabilities;
- unsupported tasks use roadmap status instead of fake execution.
