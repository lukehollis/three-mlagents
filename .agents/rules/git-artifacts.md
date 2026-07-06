# Git And Artifact Rules

Keep source, docs, tests, lockfiles, and small curated fixtures tracked. Keep
generated artifacts out of git unless the user explicitly asks to version them.

## Ignore Or Avoid Committing

- `node_modules/`
- `.venv/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`
- `client/dist/`
- backend policy archives such as `*.zip`
- TensorBoard and monitor run directories
- temporary evaluation logs and screenshots
- local environment files containing secrets

## Before Commit

Run:

```bash
git status --short
```

Review every changed file. Do not revert or stage unrelated user work. If a
generated artifact appears, either remove it from the commit or update
`.gitignore` if it is a repeatable output of the current workflow.
