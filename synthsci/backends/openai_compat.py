"""Self-hosted / OpenAI-compatible backend stub. Same contract as the Anthropic stub: pinned ids, dry-run request
files, transcripts and usage when live. Text-only task types only; code-executing types are Stage 2."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import CONFIG_DIR, load_yaml
from ..util import read_text, write_json
from .api_anthropic import TEXT_ONLY, _extract_files
from .base import Backend, PendingDispatch


class OpenAICompatBackend(Backend):
    name = "openai_compat"

    def __init__(self, cfg, run_dir, store, live: bool = False):
        super().__init__(cfg, run_dir, store)
        mpath = Path(cfg.get("models", "models.yaml"))
        self.models = load_yaml(mpath if mpath.exists() else CONFIG_DIR / mpath)["openai_compat"]
        self.live = live and bool(os.environ.get("OPENAI_API_KEY"))

    def model_id(self, tier: str) -> str:
        mid = self.models.get(tier, {}).get("id", "UNPINNED")
        if mid == "UNPINNED":
            raise RuntimeError(f"model id for tier {tier!r} is UNPINNED in configs/models.yaml")
        return mid

    def run(self, tasks: list, ctx: dict) -> list:
        pending = [t for t in tasks if self.store.status(t) != "done"]
        for t in pending:
            if t.type not in TEXT_ONLY:
                raise NotImplementedError(f"openai_compat backend: task type {t.type!r} executes code (Stage 2)")
            d = self.store.task_dir(t)
            req = {"model": self.model_id(t.tier), "messages": [{"role": "user", "content": read_text(d / "prompt.md")}], "temperature": 0.0}
            write_json(d / "api_request.json", {"base_url": self.models.get("base_url"), **{k: v for k, v in req.items() if k != "messages"}})
            if self.live:
                self._call(t, req, d)
        if not self.live:
            raise PendingDispatch(pending, self.run_dir / "dispatch" / "api_dry_run.md")
        return tasks

    def _call(self, task, req, d):
        import urllib.request
        body = json.dumps(req).encode("utf-8")
        http = urllib.request.Request(f"{self.models['base_url'].rstrip('/')}/chat/completions", data=body,
                                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        with urllib.request.urlopen(http, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        write_json(d / "transcript.json", {"request": req, "response": data})
        write_json(d / "usage.json", data.get("usage", {}) | {"model": data.get("model")})
        _extract_files(text, d / "output", task.outputs)

    def usage(self, task) -> dict:
        p = self.store.task_dir(task) / "usage.json"
        return json.loads(read_text(p)) if p.exists() else {}
