"""Claude Code subagent backend: the engine writes task packets and an operator session runs one subagent per task.

The engine writes task packets; this backend never calls a model. It writes `dispatch/batch_<n>.md` with the verbatim
stub per task and `dispatch/dispatch_instructions.md`, then raises PendingDispatch so the run stops at the boundary.
The operator session runs the agents, then `python -m synthsci ingest --run <dir>` validates the outputs
(schema, nonce, leak scan, forensic re-run) and writes DONE/FAILED; `python -m synthsci resume --run <dir>` continues.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..config import CONFIG_DIR, load_yaml
from ..orchestration.tasks import PROMPTS, Task, TaskStore, render
from ..scorer import forensic_rerun
from ..util import append_jsonl, read_json, read_text, write_json, write_text
from .base import Backend, PendingDispatch

# the interpreter that runs the engine; the dispatch instructions tell the operator to call ingest and resume with it
PYTHON = Path(sys.executable).as_posix()

TIER_ORDER = {"haiku": 0, "sonnet": 1, "opus": 2}


class SubagentBackend(Backend):
    name = "subagent"
    executes_code = True

    def __init__(self, cfg, run_dir, store):
        super().__init__(cfg, run_dir, store)
        mpath = Path(cfg.get("models", "models.yaml"))
        self.models = load_yaml(mpath if mpath.exists() else CONFIG_DIR / mpath).get("subagent", {})
        self.max_concurrent = int(self.models.get("max_concurrent", 5))
        self.max_retries = int(self.models.get("max_retries", 2))

    def run(self, tasks: list, ctx: dict) -> list:
        pending = [t for t in tasks if self.store.status(t) not in ("done", "failed")]
        failed_retry = [t for t in tasks if self.store.status(t) == "failed" and self.store.retries(t) < self.max_retries]
        for t in failed_retry:
            self._reset_for_retry(t)
        pending += failed_retry
        if not pending:
            return tasks
        n = self._write_dispatch(pending, ctx)
        raise PendingDispatch(pending, self.run_dir / "dispatch" / "dispatch_instructions.md")

    def usage(self, task) -> dict:
        p = self.store.task_dir(task) / "usage.json"
        return read_json(p) if p.exists() else {}

    def _reset_for_retry(self, t: Task) -> None:
        d = self.store.task_dir(t)
        n = self.store.retries(t) + 1
        att = d / f"attempt_{n}"
        att.mkdir(exist_ok=True)
        for name in ("validation.json", "FAILED", "AGENT_FINISHED"):
            p = d / name
            if p.exists():
                p.rename(att / name)
        out = d / "output"
        if out.exists():
            for f in out.iterdir():
                f.rename(att / f.name)

    def _write_dispatch(self, tasks: list, ctx: dict) -> int:
        ddir = self.run_dir / "dispatch"
        ddir.mkdir(exist_ok=True)
        existing = sorted(ddir.glob("batch_*.md"))
        n = len(existing) + 1
        stub_tpl = read_text(PROMPTS / "dispatch_stub.md")
        # the environment line is configuration (backend.environment_note; fallback models.yaml subagent.environment_note), not
        # prompt text: an API or self-hosted backend has its own file-writing rule or none
        note = (self.cfg.get("backend") or {}).get("environment_note")
        if note is None:
            note = self.models.get("environment_note", "")
        stub_tpl = stub_tpl.replace("{{ENVIRONMENT_NOTE}}", str(note or "").strip()).replace("  ", " ").rstrip() + "\n"
        lines = [f"# Dispatch batch {n} ({len(tasks)} tasks)", "",
                 "Run at most {} subagents at a time. For each task start ONE fresh general-purpose agent with the model shown and the verbatim stub.".format(self.max_concurrent),
                 "Never reuse an agent across tasks. A refusal or an empty output is an outcome: do not coach the agent; mark it by running ingest.", ""]
        rows = []
        for t in sorted(tasks, key=lambda t: (TIER_ORDER.get(t.tier, 1), t.type, t.id)):
            d = self.store.task_dir(t)
            stub = stub_tpl.replace("{{PROMPT_PATH}}", str(d / "prompt.md")).replace("{{TASK_DIR}}", str(d)) \
                .replace("{{OUTPUT_DIR}}", str(d / "output")).replace("{{NONCE}}", t.nonce())
            model = self.models.get(t.tier, {}).get("model", t.tier)
            rows.append({"task_id": t.id, "type": t.type, "tier": t.tier, "model": model, "call_type": t.call_type, "task_dir": str(d), "stub": stub})
            lines += [f"## {t.id}  (type {t.type}; tier {t.tier}; Agent model = `{model}`)", "", "```", stub, "```", ""]
        write_text(ddir / f"batch_{n}.md", "\n".join(lines))
        write_json(ddir / f"batch_{n}.json", rows)
        write_text(ddir / "dispatch_instructions.md", self._instructions(n, tasks))
        return n

    def _instructions(self, n: int, tasks: list) -> str:
        by = {}
        for t in tasks:
            by[t.type] = by.get(t.type, 0) + 1
        tier_note = "\n".join(f"- `{k}`: {v} task(s)" for k, v in sorted(by.items()))
        return f"""# Dispatch instructions (run {self.run_dir.name}; current batch: batch_{n}.md)

This run is paused at a dispatch boundary. The engine has written one packet per task under `{self.run_dir / 'tasks'}\\<task_id>\\`:
`prompt.md` (the full prompt), the agent-visible inputs (`profile.md`, `menu.md`, `inputs\\...`), an empty `output\\` folder and `task.json`
(ids, nonce, expected outputs). Nothing in a packet names the truth, the planted problems, the scenario or the purpose of the study; do not add
any of that in your own words when you launch an agent.

## How to run a batch

1. Open `dispatch\\batch_{n}.md`. It lists every task with its model tier and a **verbatim stub**. Tasks in this batch: {len(tasks)}
   ({tier_note.replace(chr(10), '; ')}).
2. For each task, launch ONE fresh general-purpose subagent (`Agent` tool, `subagent_type: general-purpose`) with
   `model` set to the tier shown (haiku / sonnet / opus) and the stub pasted **exactly as written** as the whole prompt. Run at most
   {self.max_concurrent} agents concurrently. Never reuse an agent for a second task. Do not read the packet's inputs into your own context.
3. When an agent returns, record its reported usage into `<task_dir>\\usage.json` as
   `{{"total_tokens": <int>, "duration_ms": <int>, "tool_uses": <int>, "model": "<tier>", "at": "<ISO time>"}}` (the subagent path exposes no
   machine-readable usage; transcribe what the Agent result reports). Then move on; do not validate by hand.
4. After the batch (or after a pause: stop launching, then continue later from the first task
   without an `AGENT_FINISHED` marker), run:

       {PYTHON} -m synthsci ingest --run "{self.run_dir}"

   Ingest validates every finished task (expected files present, JSON schema, nonce, leak scan of agent text, forensic re-run of
   `analysis.py` where a dataset is attached) and writes `DONE` or `FAILED` (with `validation.json`). Tasks without `AGENT_FINISHED` stay pending.
   It appends a record to `dispatch\\ingest_log.jsonl` (also `dispatch\\last_ingest.json`): `reported_but_missing` lists tasks whose
   agent returned (usage.json recorded) but left neither the marker nor any output; `pending_after_batch` lists tasks of the current
   batch still pending while the rest of the batch has been ingested. Re-launch each of those with its verbatim stub (a fresh agent),
   then run ingest again; do not take an agent's hand-back as evidence that files exist.
5. Resume the run:

       {PYTHON} -m synthsci resume --run "{self.run_dir}"

   The engine re-enters the paused phase, picks up DONE outputs, re-dispatches FAILED tasks (at most {self.max_retries} retries each;
   a task that fails again is recorded as a failed call and the world moves on: an abandoned paper, a review by the aggregation rule, and so on),
   and writes the next `dispatch\\batch_<n+1>.md` when it reaches the next boundary. Repeat until the run prints `DONE`.

## Batch sizes and tiers

| Task type | Items per call | Tier (baseline) |
|---|---|---|
| research / revision | 1 paper | sonnet |
| desk | up to 10 submissions | sonnet |
| review | up to 4 papers (6 for AI-written review) | sonnet |
| decision | up to 8 manuscripts | sonnet |
| prescreen | 4 papers | haiku |
| lab_decision | 1 lab, all decisions of the quarter | sonnet |
| analyst | 1 analysis | sonnet |
| verification | 2 claims | sonnet |
| decisive_test | 1 test | sonnet |
| handbook / prior_probe | 1 writer | opus (writer 0), sonnet (writers 1-2) |
| legacy_abstracts | 10 abstracts | haiku |

## Rules that protect the design

- Verbatim stub; nothing else in the agent's prompt. The stub points the agent at `prompt.md` with absolute paths (Haiku agents misread relative paths).
- Never open `landscape\\truth.json`, `ledger\\`, or `state\\` while dispatching, and never paste anything from them to an agent.
- Record usage before running ingest (forensics read it). Validate only after the agent has returned (no race).
- Model tier is recorded per task in `task.json`; the Agent tool's `model` field is the only place the tier is set.
"""


# ---------------------------------------------------------------------- ingest
def ingest(run_dir: Path) -> dict:
    """Validate every task that has an AGENT_FINISHED marker (or any output) and write DONE/FAILED.

    Agent self-reports are unreliable (in an early test run a reviewer reported outputs that never landed). The operator
    records `usage.json` when an agent returns, so a task with `usage.json` but neither `AGENT_FINISHED` nor any
    output is recorded as `reported_but_missing`; a task with outputs but no marker as `finished_without_marker`
    (validated as usual); the rest as `pending`. Every ingest appends one record to `dispatch/ingest_log.jsonl`
    and writes the same counts and lists to `dispatch/last_ingest.json`."""
    run_dir = Path(run_dir)
    store = TaskStore(run_dir)
    from ..config import load_yaml as _ly
    cfg = _ly(run_dir / "config.resolved.yaml")
    counts = {"done": 0, "failed": 0, "pending": 0, "skipped": 0}
    anomalies = {"reported_but_missing": [], "finished_without_marker": [], "pending_tasks": [], "table_notes": {}}
    for d in sorted(store.dir.iterdir()):
        if not d.is_dir() or not (d / "task.json").exists():
            continue
        if (d / "DONE").exists() or (d / "FAILED").exists():
            counts["skipped"] += 1
            continue
        rec = read_json(d / "task.json")
        has_marker = (d / "AGENT_FINISHED").exists()
        has_output = any((d / "output").rglob("*"))
        if not has_marker and not has_output:
            counts["pending"] += 1
            anomalies["pending_tasks"].append(rec["task_id"])
            if (d / "usage.json").exists():   # the agent returned (usage was recorded) but left nothing behind
                anomalies["reported_but_missing"].append(rec["task_id"])
            continue
        if not has_marker:
            anomalies["finished_without_marker"].append(rec["task_id"])
        t = Task(type=rec["type"], quarter=rec["quarter"], identity="", tier=rec["tier"], run_id=rec["run_id"], call_type=rec["call_type"],
                 template=rec["template"], outputs=rec["outputs"], schema=rec["schema"], meta=rec["meta"], who=rec.get("who"), fixed_id=rec["task_id"])
        checks = []
        if t.type in ("research", "revision", "analyst", "decisive_test"):
            checks.append(lambda task, dd, cfg=cfg: _forensic_check(task, dd, cfg))
        if t.type in ("research", "revision"):
            checks.append(_robustness_check)
        if (d / "AGENT_FINISHED").exists():
            fin = read_text(d / "AGENT_FINISHED").strip()
            if t.nonce() not in fin:
                checks.append(lambda task, dd: ["AGENT_FINISHED nonce mismatch"])
        rep = store.validate(t, extra_checks=checks)
        rep["stray_files"] = _stray_files(d)   # flagged, not failed: files the agent wrote outside output/ and scratch/
        rep["table_notes"] = _table_notes(d)   # flagged, not failed: non-regression rows in table.csv
        if rep["table_notes"]:
            anomalies["table_notes"][rec["task_id"]] = rep["table_notes"]
        store.mark(t, rep["ok"], rep)
        counts["done" if rep["ok"] else "failed"] += 1
    batches = sorted((run_dir / "dispatch").glob("batch_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    latest = batches[-1] if batches else None
    if latest is not None:   # tasks of the latest batch left pending while the rest of that batch has been ingested
        ids = [b["task_id"] for b in read_json(latest)]
        settled = [i for i in ids if (store.dir / i / "DONE").exists() or (store.dir / i / "FAILED").exists()]
        anomalies["pending_after_batch"] = [i for i in ids if i in anomalies["pending_tasks"]] if settled else []
        anomalies["batch"] = latest.stem
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **counts, **anomalies}
    (run_dir / "dispatch").mkdir(exist_ok=True)
    append_jsonl(run_dir / "dispatch" / "ingest_log.jsonl", [record])
    write_json(run_dir / "dispatch" / "last_ingest.json", record)
    return record


def _table_notes(d: Path) -> dict[str, list[str]]:
    """`table_check` notes for every table.csv under output/ (batched packets: output/<id>/table.csv), keyed by path."""
    from ..scorer import table_check
    out = {}
    for p in sorted((d / "output").rglob("table.csv")):
        notes = table_check(p)
        if notes:
            out[p.relative_to(d / "output").as_posix()] = notes
    return out


ENGINE_FILES = {"prompt.md", "task.json", "AGENT_FINISHED", "usage.json", "validation.json", "DONE", "FAILED", "LEAK", "forensic.json",
                "api_request.json", "transcript.json", "api_error.json", "mock_error.txt"}
ENGINE_DIRS = {"output", "scratch", "inputs", "authors", "analyses", "drafts", "registry", "open_claims"}


def _stray_files(d: Path) -> list[str]:
    """Files inside the task folder that neither the engine nor the contract placed there (helper scripts, copied data)."""
    out = []
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        top = rel.split("/")[0]
        if rel in ENGINE_FILES or top in ENGINE_DIRS or top.startswith("attempt_") or p.suffix == ".md" and "/" not in rel:
            continue
        out.append(rel)
    return sorted(out)


def _forensic_check(task: Task, d: Path, cfg: dict) -> list[str]:
    out = d / "output"
    data_files = {}
    for csv in list((d / "inputs").rglob("data.csv")) + list((d / "authors").rglob("data.csv")) + list((d / "analyses").rglob("data.csv")):
        data_files[str(csv.relative_to(d)).replace("\\", "/")] = csv
    if not data_files:
        return []  # data not kept: forensics recorded as unknown by the engine
    res = forensic_rerun(out, data_files, cfg["datasets"]["forensics"]["mode"], cfg["datasets"]["forensics"]["tolerance"], task_root=d)
    write_json(d / "forensic.json", res)
    return []  # not_reproducible is a flag, not a validation failure: the audit reports it as an outcome


def _robustness_check(task: Task, d: Path) -> list[str]:
    try:
        claims = json.loads(read_text(d / "output" / "claims.json"))
    except Exception:  # noqa: BLE001
        return []
    if task.call_type.endswith("_t1") and len(claims.get("robustness_models", [])) < 4:
        return ["T1 contract: fewer than four robustness models"]
    return []
