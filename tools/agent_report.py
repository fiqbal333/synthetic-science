"""Who did what in a live run: role x model x calls x tokens, from each task's task.json (role, tier) and usage.json.

Writes <run>/agent_report.md and prints it. Retries (attempt_*) and packets of set-aside attempts (_void/) are counted as wasted calls.
Usage: python tools/agent_report.py runs/S0-150/1/llm
"""
import collections
import json
import sys
from pathlib import Path

ROLE = {
    "research": "Author (single study)", "research_batch": "Author (shared call, 2-3 studies)", "revision": "Author (revision)",
    "desk": "Editor (desk screen)", "review": "Reviewer", "review_ai_written": "Reviewer (AI-written)", "prescreen": "AI pre-screener",
    "decision": "Editor (decision)", "handbook": "Handbook writer", "lab_decision": "Lab head", "analyst": "AI analyst",
    "verification": "Verifier", "decisive_test": "Decisive-test executor", "legacy_abstracts": "Legacy abstract writer",
    "prior_probe": "Prior probe",
}


def main(run: Path) -> str:
    rows = collections.defaultdict(lambda: {"calls": 0, "tokens": 0, "tool_uses": 0, "ms": 0, "wasted": 0})
    void = [p for p in (run / "_void").rglob("task.json")] if (run / "_void").exists() else []
    for tj in sorted((run / "tasks").glob("*/task.json")) + void:
        d = tj.parent
        t = json.loads(tj.read_text(encoding="utf-8"))
        u = d / "usage.json"
        if not u.exists():
            continue
        us = json.loads(u.read_text(encoding="utf-8"))
        tier = us.get("model") or t.get("tier")
        key = (ROLE.get(t["type"], t["type"]), tier)
        r = rows[key]
        r["calls"] += 1
        r["tokens"] += int(us.get("total_tokens") or 0)
        r["tool_uses"] += int(us.get("tool_uses") or 0)
        r["ms"] += int(us.get("duration_ms") or 0)
        if "_void" in d.parts or not (d / "DONE").exists() or list(d.glob("attempt_*")):
            r["wasted"] += 1 if ("_void" in d.parts or not (d / "DONE").exists()) else 0
    lines = [f"# Agent report: {run.as_posix()}", "",
             "Every agent call in this run by role and model. Tokens are the subagent totals reported at completion "
             "(they include the ~45k fixed overhead per call). 'Wasted' = packet of a set-aside attempt or task never accepted.", "",
             "| Role | Model | Calls | Tokens | Mean tokens/call | Tool uses | Agent minutes | Wasted calls |",
             "|---|---|---|---|---|---|---|---|"]
    tot = {"calls": 0, "tokens": 0, "tool_uses": 0, "ms": 0, "wasted": 0}
    for (role, tier), r in sorted(rows.items(), key=lambda kv: -kv[1]["tokens"]):
        lines.append(f"| {role} | {tier} | {r['calls']} | {r['tokens']:,} | {r['tokens'] // max(r['calls'], 1):,} | {r['tool_uses']} | "
                     f"{r['ms'] / 60000:.1f} | {r['wasted']} |")
        for k in tot:
            tot[k] += r[k]
    lines.append(f"| **Total** | | **{tot['calls']}** | **{tot['tokens']:,}** | {tot['tokens'] // max(tot['calls'], 1):,} | {tot['tool_uses']} | "
                 f"{tot['ms'] / 60000:.1f} | {tot['wasted']} |")
    lines += ["", "Note: calls whose usage was recorded on a retried task folder count once per recorded usage; the project ledger "
              "itemises every call including wasted ones."]
    out = "\n".join(lines) + "\n"
    (run / "agent_report.md").write_text(out, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1]).resolve()))
