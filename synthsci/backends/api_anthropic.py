"""Direct-API backend stub, Anthropic first (pinned model ids, retries, full transcripts, machine-captured usage).

Status: STUB. It builds a request per packet, refuses UNPINNED model ids, writes a dry-run request file, and when
`ANTHROPIC_API_KEY` is set and `live=True` it calls the Messages API for text-only task types (desk, review, decision,
lab_decision, handbook, prior_probe, legacy_abstracts), saving the full transcript and machine-captured usage.
Code-executing task types (research, revision, analyst, verification, decisive_test) need a sandboxed tool loop and are
NotImplemented here (Stage 2; the subagent backend covers them in Stage 1).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..config import CONFIG_DIR, load_yaml
from ..util import read_text, write_json, write_text
from .base import Backend, PendingDispatch

TEXT_ONLY = {"desk", "review", "decision", "lab_decision", "handbook", "prior_probe", "legacy_abstracts", "prescreen"}


class AnthropicBackend(Backend):
    name = "anthropic"
    executes_code = False

    def __init__(self, cfg, run_dir, store, live: bool = False):
        super().__init__(cfg, run_dir, store)
        mpath = Path(cfg.get("models", "models.yaml"))
        self.models = load_yaml(mpath if mpath.exists() else CONFIG_DIR / mpath)["anthropic"]
        self.live = live and bool(os.environ.get("ANTHROPIC_API_KEY"))

    def model_id(self, tier: str) -> str:
        mid = self.models.get(tier, {}).get("id", "UNPINNED")
        if mid == "UNPINNED":
            raise RuntimeError(f"model id for tier {tier!r} is UNPINNED in configs/models.yaml; a live run refuses unpinned ids")
        return mid

    def build_request(self, task) -> dict:
        d = self.store.task_dir(task)
        prompt = read_text(d / "prompt.md")
        attachments = []
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".md", ".csv", ".py", ".json") and p.name not in ("prompt.md", "task.json") and "output" not in p.parts:
                if p.suffix == ".csv" and p.stat().st_size > 200_000:
                    continue  # data are read by code, not by the model
                attachments.append(f"\n\n----- FILE: {p.relative_to(d)} -----\n{read_text(p)}")
        return {"model": self.model_id(task.tier), "max_tokens": 8000, "temperature": float(self.models.get("temperature", 0.0)),
                "system": "Follow the task exactly. Write the requested files' contents as fenced blocks headed by their file names.",
                "messages": [{"role": "user", "content": prompt + "".join(attachments)}]}

    def run(self, tasks: list, ctx: dict) -> list:
        pending = [t for t in tasks if self.store.status(t) not in ("done",)]
        for t in pending:
            d = self.store.task_dir(t)
            if t.type not in TEXT_ONLY:
                raise NotImplementedError(f"API backend: task type {t.type!r} executes code; use the subagent backend (Stage 1) or the Stage 2 tool loop")
            req = self.build_request(t)
            write_json(d / "api_request.json", {k: v for k, v in req.items() if k != "messages"} | {"prompt_chars": len(req["messages"][0]["content"])})
            if not self.live:
                continue
            self._call(t, req, d)
        if not self.live:
            raise PendingDispatch(pending, self.run_dir / "dispatch" / "api_dry_run.md")
        return tasks

    def _call(self, task, req: dict, d: Path) -> None:
        import anthropic  # imported lazily; not a hard dependency for offline runs
        client = anthropic.Anthropic()
        delays = self.models.get("backoff_seconds", [5, 15, 60, 180, 600, 1800])
        for attempt in range(int(self.models.get("max_retries", 6)) + 1):
            try:
                resp = client.messages.create(**req)
                break
            except Exception as e:  # noqa: BLE001
                if attempt >= len(delays):
                    write_json(d / "api_error.json", {"error": str(e)[:500]})
                    return
                time.sleep(delays[attempt])
        text = "".join(getattr(b, "text", "") for b in resp.content)
        write_json(d / "transcript.json", {"request": req, "response_model": resp.model, "stop_reason": resp.stop_reason, "text": text})
        write_json(d / "usage.json", {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                                      "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens, "model": resp.model})
        _extract_files(text, d / "output", task.outputs)

    def usage(self, task) -> dict:
        p = self.store.task_dir(task) / "usage.json"
        return json.loads(read_text(p)) if p.exists() else {}


def _extract_files(text: str, out: Path, names: list[str]) -> None:
    """Pull fenced blocks headed by a file name out of a text-only response."""
    import re
    out.mkdir(exist_ok=True)
    for name in names:
        m = re.search(rf"{re.escape(name)}.*?```[a-zA-Z]*\n(.*?)```", text, re.S)
        if m:
            write_text(out / name, m.group(1))
