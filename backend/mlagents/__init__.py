"""Python-first research tooling for Three ML-Agents.

The browser examples are visualization surfaces. Training, evaluation, and
benchmark bookkeeping live in this package so experiments can be reproduced
without a React runtime.
"""

from .registry import TaskSpec, get_task, list_tasks, make_env

__all__ = ["TaskSpec", "get_task", "list_tasks", "make_env"]
