"""Reuse finished agent outputs from a source run in a new run, only where the task prompt is byte-identical.

Usage: python tools/reuse_identical.py <source run dir> <new run dir> [<source scenario name> <new scenario name>]
The source is a set-aside attempt of the same run (restarted from scratch), or the paired baseline under common random
numbers (S0-150 for S0P-150), whose pre-treatment tasks match byte for byte up to the
scenario folder name.
For every pending task of the new run (no AGENT_FINISHED) whose prompt.md and task.json equal the source run's and whose
source copy has AGENT_FINISHED: copy output/, AGENT_FINISHED and usage.json (marked reused_from), so no tokens are spent twice.
Writes REUSED.md in the new run listing what was copied. Prints a line per task.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(old, new, old_name=None, new_name=None):
    """old_name/new_name: a scenario folder name that differs between the two runs (e.g. S0-150 vs S0P-150);
    the old run's prompt.md and task.json are compared after substituting it, so only paths may differ."""
    old, new = Path(old).resolve(), Path(new).resolve()

    def h(p: Path) -> str:
        b = p.read_bytes()
        if old_name and p.name == "task.json":   # task_dir and the prompt hash follow the path; the prompt is compared directly
            d = json.loads(b.decode("utf-8"))
            d.pop("prompt_sha256", None)
            d.pop("task_dir", None)
            return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
        if old_name and str(p).startswith(str(old)):   # path segments only (the shared run_id namespace must not change)
            b = b.replace(f"\\{old_name}\\".encode(), f"\\{new_name}\\".encode())
        return hashlib.sha256(b).hexdigest()
    rows = []
    for td in sorted((new / "tasks").iterdir()):
        if not td.is_dir() or (td / "AGENT_FINISHED").exists():
            continue
        od = old / "tasks" / td.name
        if not (od / "AGENT_FINISHED").exists() or not (td / "prompt.md").exists():
            continue
        same = all((od / f).exists() and h(od / f) == h(td / f) for f in ("prompt.md", "task.json"))
        print(f"{td.name}: {'REUSE' if same else 'different, not reused'}")
        if not same:
            continue
        if (td / "output").exists():
            shutil.rmtree(td / "output")
        shutil.copytree(od / "output", td / "output")
        shutil.copy2(od / "AGENT_FINISHED", td / "AGENT_FINISHED")
        if (od / "usage.json").exists():
            u = json.loads((od / "usage.json").read_text(encoding="utf-8"))
            u["reused_from"] = str(od)
            (td / "usage.json").write_text(json.dumps(u), encoding="ascii")
        rows.append(td.name)
    if rows:
        (new / "REUSED.md").write_text(
            "# Outputs reused from a source run (byte-identical prompt.md and task.json)\n\nSource: `" + str(old) + "`\n\n"
            + "\n".join(f"- {r}" for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
