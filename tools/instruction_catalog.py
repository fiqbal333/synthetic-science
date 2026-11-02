"""Catalogue every instruction given to the agents in the live runs (read-only on runs/).

Usage: python tools/instruction_catalog.py [--out analysis/instructions] [--runs <runs folder>]
(--runs defaults to the environment variable SYNTHSCI_RUNS, else runs/ under the repository root.)
Every agent task folder already holds the exact prompt the agent received (prompt.md) and task.json (template name and
sha256, prompt sha256, role, model tier, quarter). This tool gathers them into:
  index.csv              one row per task: run, world, seed, quarter, role, task type, model, template, template sha,
                         prompt sha, prompt words, path to prompt.md
  templates.md           each role template (synthsci/prompts/*.md) with its sha256, which worlds and roles used it,
                         how many prompts were rendered from it, and whether every use matches the current template
  examples/<world>/<role>.md   one full rendered prompt per world and role (the earliest live one), with the material the
                         agent was handed listed beside it
  README.md              what is where, and what is not in prompt.md (the one-paragraph dispatch stub that points the
                         agent to its folder: configs/backends dispatch_stub; and the input files, which sit next to it)
Warm-start (quarter < 1, rule agents) and set-aside attempts (runs/_void/) are excluded; copies of quarters 1-8 inside -x20 runs are
counted once, under the original run. In the example prompts, the machine-specific folder that held runs/ is shown as `<repo>`;
nothing else in the text is changed (index.csv keeps each prompt's sha256 as recorded by the engine).
"""
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from synthsci.util import read_text, sha256_text  # noqa: E402

WORLD = {"S0-150": "paper", "S0P-150": "paper + career pressure", "R-b-150": "repository"}
# an absolute path up to and including the runs folder, with either separator (the prompts name absolute task folders)
ABS_RUNS = re.compile(r"[A-Za-z]:[\\/](?:[^\s`\"'()<>|\\/]+[\\/])*?runs(?=[\\/])")


def redact(text: str) -> str:
    """Show the machine-specific folder that held runs/ as <repo>, so example prompts carry no local paths."""
    return ABS_RUNS.sub(lambda m: "<repo>" + m.group(0)[-5] + "runs", text)


def main(args):
    out = W / (args[args.index("--out") + 1] if "--out" in args else "analysis/instructions")
    runs = Path(args[args.index("--runs") + 1] if "--runs" in args else (os.environ.get("SYNTHSCI_RUNS") or W / "runs"))
    if not runs.is_dir():
        sys.exit(f"STOP: run folder {runs} not found; give it with --runs or SYNTHSCI_RUNS")
    rel = lambda p: "runs/" + p.relative_to(runs).as_posix()   # path as it sits under runs/, whatever the machine
    (out / "examples").mkdir(parents=True, exist_ok=True)
    rows, first = [], {}
    uses = defaultdict(Counter)
    shas = defaultdict(set)
    for tj in sorted(runs.glob("*/*/llm/tasks/*/task.json")):
        run = tj.parents[4].name   # runs/<run>/<seed>/llm/tasks/<task>/task.json, e.g. S0-150-x20
        base = run.replace("-x20", "")
        if base not in WORLD:
            continue
        t = json.loads(tj.read_text(encoding="utf-8"))
        q = int(t.get("quarter", 0))
        if q < 1 and t.get("type") != "handbook":
            continue
        if run.endswith("-x20") and q <= 8:
            continue   # copied from the original run by `synthsci extend`
        if not run.endswith("-x20") and q > 8:
            continue
        pm = tj.parent / "prompt.md"
        if not pm.exists():
            continue
        text = pm.read_text(encoding="utf-8")
        world, seed = WORLD[base], tj.parents[3].name
        role = t.get("type")   # task type = role; `who` holds the agent or lab id, kept in its own column
        who = t.get("who") or ""
        tmpl = t.get("template", "")
        rows.append({"world": world, "run": f"{run}/{seed}", "seed": seed, "quarter": q, "role": role, "who": who,
                     "task_type": t.get("type"), "model": t.get("tier"), "template": tmpl,
                     "template_sha256": t.get("template_sha256", ""), "prompt_sha256": t.get("prompt_sha256", ""),
                     "prompt_words": len(text.split()), "prompt_path": str(Path(rel(pm)))})
        uses[tmpl][(world, role)] += 1
        shas[tmpl].add(t.get("template_sha256", ""))
        key = (world, role)
        if key not in first or (q, str(pm)) < first[key][0]:
            first[key] = ((q, str(pm)), pm, text, tj.parent)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines = ["# Agent instruction templates", "",
             "Each prompt an agent received was rendered from one of these templates (synthsci/prompts/) by filling in "
             "that agent's own material. `index.csv` lists every rendered prompt with its template and sha256.", ""]
    for tmpl in sorted(uses):
        p = W / "synthsci" / "prompts" / tmpl
        cur = sha256_text(read_text(p)) if p.exists() else "(file not found)"   # the engine's own fingerprint (render())
        used = sorted(s for s in shas[tmpl] if s)
        same = "yes" if used == [cur] else f"no: {len(used)} template version(s) used, current sha {cur[:12]}"
        lines += [f"## {tmpl}", "", f"- current sha256: `{cur}`", f"- every use rendered from the current version: {same}",
                  "- used by: " + "; ".join(f"{w} / {r}: {n:,}" for (w, r), n in sorted(uses[tmpl].items())), ""]
    (out / "templates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for (world, role), (_, pm, text, td) in sorted(first.items()):
        d = out / "examples" / world.replace(" ", "_").replace("+", "plus")
        d.mkdir(parents=True, exist_ok=True)
        handed = sorted(str(p.relative_to(td)) for p in td.rglob("*") if p.is_file()
                        and p.name not in ("prompt.md", "task.json", "AGENT_FINISHED", "DONE", "FAILED", "usage.json", "validation.json")
                        and "output" not in p.relative_to(td).parts and "scratch" not in p.relative_to(td).parts
                        and not p.relative_to(td).parts[0].startswith("attempt_"))
        (d / f"{role}.md").write_text(
            f"# Example instruction: {world}, {role}\n\nSource: `{Path(rel(pm))}`\n\nMaterial handed to the agent in its "
            f"folder: {', '.join(handed) if handed else 'none'}\n\n---\n\n{redact(text)}", encoding="utf-8")

    n = Counter((r["world"], r["role"]) for r in rows)
    readme = ["# Instructions given to the agents", "",
              f"{len(rows):,} live agent prompts across the eight runs (quarters 1-20, plus handbooks). Built by "
              "`tools/instruction_catalog.py` (read-only on runs/).", "",
              "- `index.csv`: every prompt, with world, run, quarter, role, model, template and sha256 fingerprints, and the "
              "path to the exact text in its task folder.",
              "- `templates.md`: the role templates and whether every prompt came from the current version.",
              "- `examples/<world>/<role>.md`: one full prompt per world and role, and the material handed with it.", "",
              "Not in prompt.md: (1) the short dispatch stub each worker was launched with, which only points it to its "
              "folder and tells it where to write (configs/backends; the printed stub is in each run's dispatch/batch_N.md); "
              "(2) the material files (profile, menu, data, reviews), which sit beside prompt.md in the same task folder.", "",
              "| World | Role | Prompts |", "|---|---|---|"] + [f"| {w} | {r} | {k:,} |" for (w, r), k in sorted(n.items())]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"{len(rows):,} prompts; {len(uses)} templates; {len(first)} world-role examples -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
