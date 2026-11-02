"""Task packets, manifest, markers, validation and ingest.

An event-driven queue: tasks are appended as the clock advances. Ids are content hashes of
(run id, quarter, type, identity); every task has a nonce, a template hash, DONE/FAILED markers and a validation report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from ..config import PKG_ROOT
from ..leakscan import agent_allow, scan_text
from ..util import append_jsonl, canonical_json, read_json, read_text, sha256_text, short_id, write_json, write_text

PROMPTS = PKG_ROOT / "prompts"
SCHEMAS = PKG_ROOT / "schemas"


@dataclass
class Task:
    type: str
    quarter: int
    identity: str                 # what makes the task unique inside (run, quarter, type)
    tier: str
    run_id: str
    call_type: str                # tariff class
    template: str                 # prompt template file name
    variables: dict = field(default_factory=dict)      # template variables
    inputs: dict = field(default_factory=dict)         # {relative path in task dir: text or source Path}
    outputs: list = field(default_factory=list)        # expected output file names
    schema: str | None = None                          # JSON schema for the main JSON output
    meta: dict = field(default_factory=dict)           # engine-side bookkeeping (never hidden truth)
    who: str | None = None
    fixed_id: str | None = None                        # set when a task is reconstructed from its task.json

    @property
    def id(self) -> str:
        if self.fixed_id:
            return self.fixed_id
        return f"{self.type}-{short_id(self.run_id, self.quarter, self.type, self.identity, n=10)}"

    def nonce(self) -> str:
        return short_id("nonce", self.run_id, self.id, n=12)


def render(template_name: str, variables: dict) -> tuple[str, str]:
    """{{> include}} and {{VAR}} substitution; a missing variable is an error. Returns (text, template sha)."""
    tpl = read_text(PROMPTS / template_name)
    tsha = sha256_text(tpl)
    def inc(m):
        return read_text(PROMPTS / m.group(1).strip())
    text = re.sub(r"\{\{>\s*([^}]+)\}\}", inc, tpl)
    missing = []
    def sub(m):
        k = m.group(1).strip()
        if k not in variables:
            missing.append(k)
            return m.group(0)
        return str(variables[k])
    text = re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", sub, text)
    if missing:
        raise KeyError(f"template {template_name}: missing variables {sorted(set(missing))}")
    return text, tsha


def _scan_view(p: Path) -> str:
    """The text of a packet file as the leak scan should read it. A JSON file is read as its keys and decoded string values,
    one per line: JSON escaping (\\" and \\n inside a quoted code string) otherwise hides agent text from the
    fragment match, which collects agent JSON the same decoded way. The raw text is used if the file does not parse."""
    raw = read_text(p)
    if p.suffix != ".json":
        return raw
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw
    out: list[str] = []

    def walk(o):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                out.append(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif o is not None:
            out.append(str(o))
    walk(obj)
    return "\n".join(out)


class TaskStore:
    def __init__(self, run_dir: Path, leak_mode: str = "strict"):
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.jsonl"
        self.leak_mode = leak_mode

    def task_dir(self, task: Task) -> Path:
        return self.dir / task.id

    def status(self, task: Task) -> str:
        d = self.task_dir(task)
        if (d / "DONE").exists():
            return "done"
        if (d / "FAILED").exists():
            return "failed"
        if (d / "prompt.md").exists():
            return "pending"
        return "new"

    def materialize(self, task: Task, agent_visible_check: bool = True) -> Path:
        """Write the packet (idempotent): task.json, prompt.md, inputs; scan every agent-visible file."""
        d = self.task_dir(task)
        if (d / "prompt.md").exists():
            return d
        d.mkdir(parents=True, exist_ok=True)
        (d / "output").mkdir(exist_ok=True)
        (d / "scratch").mkdir(exist_ok=True)   # helper files go here, never to a system temp folder
        for rel, src in task.inputs.items():
            dst = d / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(src, Path):
                import shutil
                shutil.copy(src, dst)
            else:
                write_text(dst, str(src))
        variables = dict(task.variables)
        variables.setdefault("TASK_DIR", str(d))
        variables.setdefault("OUTPUT_DIR", str(d / "output"))
        variables.setdefault("NONCE", task.nonce())
        prompt, tsha = render(task.template, variables)
        problems = []
        if agent_visible_check:
            problems = self._scan_mixed(prompt, [str(d)])
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix in (".md", ".json", ".txt") and p.name not in ("task.json",):
                    problems += [f"{p.relative_to(d)}: {x}" for x in self._scan_mixed(_scan_view(p), [str(d)])]
        if problems:
            write_json(d / "LEAK", {"problems": problems})
            raise RuntimeError(f"leak scan failed for task {task.id}: {problems[:5]}")
        write_text(d / "prompt.md", prompt)
        rec = {"task_id": task.id, "type": task.type, "quarter": task.quarter, "tier": task.tier, "run_id": task.run_id,
               "call_type": task.call_type, "template": task.template, "template_sha256": tsha,
               "prompt_sha256": sha256_text(prompt), "nonce": task.nonce(), "outputs": task.outputs, "schema": task.schema,
               "meta": task.meta, "who": task.who, "task_dir": str(d)}
        write_json(d / "task.json", rec)
        append_jsonl(self.manifest, [{"event": "created", **{k: rec[k] for k in ("task_id", "type", "quarter", "tier", "call_type", "nonce")}}])
        return d

    def _agent_lines(self) -> set:
        """Normalized lines of agent-written text (outputs of DONE tasks in this run). Such text was already scanned in
        'design' mode at ingest; when the engine quotes it in a later packet it is held to that standard, not to the
        strict list meant for text we wrote (an author may honestly write "impossible values" about its data)."""
        done = sorted(self.dir.glob("*/DONE"))
        transforms = getattr(self, "fragment_transforms", [])
        key = (len(done), len(transforms))
        if getattr(self, "_agent_lines_key", None) != key:
            frags = set()
            def add(s):
                for raw in [str(s)] + [t(str(s)) for t in transforms]:   # e.g. the handbook's de-identified quotes
                    for ln in raw.splitlines():
                        n = " ".join(ln.split()).lower()
                        if len(n) >= 12:
                            frags.add(n)
            def walk(o):
                if isinstance(o, str):
                    add(o)
                elif isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            for m in done:
                for p in (m.parent / "output").rglob("*"):   # rglob: batched packets write to output/<study id>/
                    if not p.is_file() or p.suffix not in (".md", ".json", ".txt", ".csv"):
                        continue
                    if p.suffix == ".json":
                        try:
                            walk(json.loads(read_text(p)))   # string fields, as the engine renders them (e.g. reviews.md)
                        except ValueError:
                            add(read_text(p))
                    else:
                        add(read_text(p))
            # longest first, so a containing fragment is removed before its parts
            self._agent_lines_cache, self._agent_lines_key = sorted(frags, key=len, reverse=True), key
        return self._agent_lines_cache

    def _task_roots(self) -> list[str]:
        """This run's tasks folder and, for a run made by `synthsci extend`, the source run's tasks folder (extension.json)."""
        roots = [str(self.dir)]
        ext = self.dir.parent / "extension.json"
        if ext.exists():
            try:
                src = json.loads(read_text(ext)).get("source")
                if src:
                    roots.append(str(Path(src) / "tasks"))
            except ValueError:
                pass
        return roots

    def _scan_mixed(self, text: str, allowed_paths: list[str]) -> list[str]:
        """Strict scan (which includes the design list) for engine-written text. Agent-written fragments quoted in it are
        removed first: they already passed the design scan (with the agent allow-list) at ingest. Only lines with a hit
        are examined, so the fragment search stays cheap."""
        if self.leak_mode != "strict":
            return scan_text(text, self.leak_mode, allowed_paths=allowed_paths)
        if not scan_text(text, "strict", allowed_paths=allowed_paths):
            return []
        frags = self._agent_lines()
        rest, quoted = [], []
        for ln in text.splitlines():
            n = " ".join(ln.split()).lower()
            if not scan_text(n, "strict", allowed_paths=allowed_paths):
                continue
            for f in frags:
                if f in n:
                    n = n.replace(f, " ")
                    quoted.append(f)
            if scan_text(n, "strict", allowed_paths=allowed_paths):
                # records quote agent text cut short (a decision gist [:60]/[:80] in "Criticisms received"): a piece that is
                # the start of an agent fragment is that agent's text
                # a quote can itself contain parentheses, and a record label ("criticisms received: ") can precede it, so
                # pieces are also taken between ; and | only, with and without a leading label
                pieces = re.split(r"[;|()]", n) + re.split(r"[;|]", n)
                pieces += [p.split(": ", k)[k] for p in list(pieces) for k in range(1, p.count(": ") + 1)]
                for piece in sorted(pieces, key=len, reverse=True):
                    s = piece.strip(" .")
                    if len(s) >= 12 and s in n and scan_text(s, "strict", allowed_paths=allowed_paths) and any(f.startswith(s) for f in frags):
                        n = n.replace(s, " ")
                        quoted.append(s)
            rest.append(n)
        out = scan_text("\n".join(rest), "strict", allowed_paths=allowed_paths)
        if quoted:   # defence in depth: quoted agent text still gets the design scan (with the agent allow-list)
            # an agent's own text may name task folders of this run (or, for an extended run, of its source run),
            # e.g. a script's scratch path; such paths say nothing about the design and are not leaks
            q_allowed = list(allowed_paths) + self._task_roots()
            out += [x for x in scan_text("\n".join(quoted), "design", allowed_paths=q_allowed, extra_allow=agent_allow())
                    if x not in out]
        return out

    def mark(self, task: Task, ok: bool, report: dict) -> None:
        d = self.task_dir(task)
        write_json(d / "validation.json", report)
        marker = d / ("DONE" if ok else "FAILED")
        other = d / ("FAILED" if ok else "DONE")
        other.unlink(missing_ok=True)
        write_text(marker, canonical_json({"ok": ok, "outputs": report.get("output_hashes", {})}) + "\n")
        append_jsonl(self.manifest, [{"event": "done" if ok else "failed", "task_id": task.id, "reason": report.get("reason", "")}])

    def retries(self, task: Task) -> int:
        d = self.task_dir(task)
        return len(list(d.glob("attempt_*")))

    # ------------------------------------------------------------ validation of agent outputs
    def validate(self, task: Task, extra_checks=None) -> dict:
        d = self.task_dir(task)
        out = d / "output"
        report = {"task_id": task.id, "ok": True, "problems": [], "output_hashes": {}}
        for name in task.outputs:
            p = out / name
            if not p.exists():
                report["problems"].append(f"missing output {name}")
                continue
            report["output_hashes"][name] = sha256_text(read_text(p)) if p.suffix in (".md", ".json", ".csv", ".py", ".txt") else ""
        if task.schema and not report["problems"]:
            schema = read_json(SCHEMAS / task.schema)
            for main in [n for n in task.outputs if n.endswith(".json")]:   # every JSON output (batched packets have one per study)
                try:
                    obj = json.loads(read_text(out / main))
                    jsonschema.validate(obj, schema)
                    if "nonce" in obj and obj["nonce"] != task.nonce():
                        report["problems"].append(f"{main}: nonce mismatch")
                except Exception as e:  # noqa: BLE001
                    report["problems"].append(f"{main}: schema: {type(e).__name__}: {str(e)[:200]}")
        # agent-written text that later enters other agents' prompts must not carry design tokens
        for name in task.outputs:
            p = out / name
            if p.exists() and p.suffix in (".md", ".json"):
                hits = scan_text(read_text(p), "design", allowed_paths=[str(d)], extra_allow=agent_allow())
                if hits:
                    report["problems"].append(f"{name}: design tokens {hits[:3]}")
        if extra_checks and not report["problems"]:
            for chk in extra_checks:
                res = chk(task, d)
                if res:
                    report["problems"].extend(res if isinstance(res, list) else [res])
        report["ok"] = not report["problems"]
        report["reason"] = "; ".join(report["problems"])[:500]
        return report

    def read_output(self, task: Task, name: str):
        p = self.task_dir(task) / "output" / name
        if name.endswith(".json"):
            return json.loads(read_text(p))
        return read_text(p)
