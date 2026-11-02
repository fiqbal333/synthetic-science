"""Backend interface. `run(tasks, ctx)` must leave every task DONE or FAILED, or raise PendingDispatch."""
from __future__ import annotations

from pathlib import Path


class PendingDispatch(Exception):
    """Raised when tasks were written as packets and must be executed outside the engine (subagent / batch API)."""

    def __init__(self, tasks, instructions_path: Path):
        self.tasks = tasks
        self.instructions_path = Path(instructions_path)
        super().__init__(f"{len(tasks)} task(s) await dispatch; see {instructions_path}")


class Backend:
    name = "base"
    executes_code = False

    def __init__(self, cfg: dict, run_dir, store):
        self.cfg, self.run_dir, self.store = cfg, Path(run_dir), store

    def run(self, tasks: list, ctx: dict) -> list:
        """Execute or defer. ctx carries engine objects the mock backend may use (hidden truth never reaches an LLM)."""
        raise NotImplementedError

    def usage(self, task) -> dict:
        """Realized usage for a completed task, if the backend knows it."""
        return {}
