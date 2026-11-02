"""Resumable driver for a fast-mode one-at-a-time parameter sweep, given a design YAML (name, baseline worlds, dials and levels, seeds, out_root).
No design file is shipped; tools/fastcal_validate.py uses this file's --worker mode to run its validation runs.

    python tools/sweep.py --design <design.yaml>                                         # run everything not yet done
    python tools/sweep.py --design ... --seeds 1001-1010 --workers 6                     # a subset
    python tools/sweep.py --design ... --list                                            # print the cells and run counts

Layout: <out_root>/<cell>/<scenario>/<seed>/fast  (cell = baseline | <dial>-<level>). One subprocess per run (this file in
--worker mode, which calls synthsci.engine.run in-process). A run is done when its folder holds SWEEP_OK; a folder without it
is removed and the run is started from scratch. Failures are logged to <out_root>/_sweep/failures.jsonl and never stop the
sweep; a repository run whose paired paper run failed is skipped and logged. Fast mode only: no model is called.
Never edits synthsci/; never touches runs outside out_root.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PAPER, REPO = "S0-150", "R-b-150"
KEEP = {"audit", "budget", "handbook", "log", "run.json", "config.resolved.yaml", "SWEEP_OK"}


def load_yaml(p: Path) -> dict:
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _set_dotted(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    if parts[-1] not in cur:
        raise KeyError(f"params_patch key not in params.yaml: {dotted}")
    cur[parts[-1]] = value


def cells(design: dict) -> list[dict]:
    """baseline first, then every dial x level, each with worlds and overrides (params_patch resolved to a file)."""
    import yaml
    # optional design keys (absent in the 2026-10-15 design, so its cells and config hashes are unchanged):
    #   global_overrides: applied to every cell first (a cell's own overrides win)
    #   base_params:      rule-parameter file that params_patch levels start from (default configs/fastmode/params.yaml)
    #   generated_dir:    where patched parameter files are written (default configs/sweeps/generated)
    glob = dict(design.get("global_overrides") or {})
    out = [{"cell": "baseline", "dial": "baseline", "level": "baseline", "worlds": design["baseline"]["worlds"],
            "overrides": glob | dict(design["baseline"].get("overrides") or {})}]
    gen_dir = WS / design.get("generated_dir", "configs/sweeps/generated")
    base_params = load_yaml(WS / design.get("base_params", "configs/fastmode/params.yaml"))
    for dial, spec in design["dials"].items():
        for level, ov in spec["levels"].items():
            ov = glob | dict(ov or {})
            patch = ov.pop("params_patch", None)
            if patch:
                p = copy.deepcopy(base_params)
                for k, v in patch.items():
                    _set_dotted(p, k, v)
                p["status"] = f"{base_params.get('status')} + sweep patch {design['name']} {dial}-{level}: {patch}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                fp = gen_dir / f"params-{dial}-{level}.yaml"
                txt = yaml.safe_dump(p, sort_keys=False, allow_unicode=True)
                if not fp.exists() or fp.read_text(encoding="utf-8") != txt:
                    fp.write_text(txt, encoding="utf-8")
                ov["fastmode.params"] = str(fp)
            out.append({"cell": f"{dial}-{level}", "dial": dial, "level": level, "worlds": spec["worlds"], "overrides": ov})
    return out


def runs_for(design: dict, cl: list[dict], seeds: list[int]) -> list[dict]:
    root = (WS / design["out_root"]).resolve()
    jobs = []
    for s in seeds:
        for c in cl:
            w = c["worlds"]
            if w in ("both", "paper"):
                jobs.append({"cell": c["cell"], "dial": c["dial"], "level": c["level"], "scenario": PAPER, "seed": s,
                             "overrides": c["overrides"], "dir": root / c["cell"] / PAPER / str(s) / "fast", "needs": None, "paired": None})
            if w in ("both", "repo"):
                pc = c["cell"] if w == "both" else "baseline"
                pdir = root / pc / PAPER / str(s) / "fast"
                jobs.append({"cell": c["cell"], "dial": c["dial"], "level": c["level"], "scenario": REPO, "seed": s,
                             "overrides": c["overrides"], "dir": root / c["cell"] / REPO / str(s) / "fast", "needs": pdir, "paired": pdir})
    return jobs


def _log(root: Path, name: str, row: dict) -> None:
    d = root / "_sweep"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def prune(rd: Path) -> None:
    for p in rd.iterdir():
        if p.name not in KEEP:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    hb = rd / "handbook"
    if hb.exists():
        for q in hb.iterdir():
            if q.is_dir():
                for f in q.iterdir():
                    if f.name != "score.json":
                        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)


def worker(args) -> int:
    sys.path.insert(0, str(WS))
    os.chdir(WS)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    from synthsci.engine import run
    rd = Path(job["dir"])
    out_root = rd.parents[2]           # <root>/<cell>
    t0 = time.time()
    run(job["scenario"], int(job["seed"]), "fast", None, job["overrides"] or None, out_root,
        Path(job["paired"]) if job.get("paired") else None, resume=False, quiet=True)
    if not (rd / "audit" / "audit.json").exists():
        raise RuntimeError("run returned without audit/audit.json")
    if job.get("prune"):
        prune(rd)
    (rd / "SWEEP_OK").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}),
                                 encoding="utf-8")
    return 0


def drive(args) -> int:
    design = load_yaml(WS / args.design)
    root = (WS / design["out_root"]).resolve()
    seeds = _seeds(args.seeds) if args.seeds else list(range(design["seeds"]["first"], design["seeds"]["first"] + design["seeds"]["n"]))
    cl = cells(design)
    if args.cells:   # e.g. --cells baseline (extra seeds for one cell); the paired paper runs of repository cells must exist
        keep = set(args.cells.split(","))
        cl = [c for c in cl if c["cell"] in keep]
    jobs = runs_for(design, cl, seeds)
    if args.list:
        for c in cl:
            print(f"{c['cell']:24s} {c['worlds']:6s} {c['overrides']}")
        print(f"{len(cl)} cells, {len(jobs)} runs over {len(seeds)} seeds")
        return 0
    nworkers = args.workers or int(design.get("workers", 4))
    timeout = int(design.get("timeout_seconds", 1800))
    prune_flag = bool(design.get("prune_after_run", False))
    (root / "_sweep" / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "_sweep" / "stdout").mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (j["dir"] / "SWEEP_OK").exists()]
    print(f"{len(jobs)} runs in design, {len(jobs) - len(pending)} already done, {len(pending)} to run with {nworkers} workers", flush=True)
    failed_dirs: set[str] = set()
    running: list[tuple[subprocess.Popen, dict, float, object]] = []
    t_start = time.time()
    n_ok = n_fail = n_skip = 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    while pending or running:
        # reap
        for item in running[:]:
            proc, job, t0, fh = item
            rc = proc.poll()
            if rc is None and time.time() - t0 > timeout:
                proc.kill()
                rc = -9
            if rc is None:
                continue
            fh.close()
            running.remove(item)
            secs = round(time.time() - t0, 1)
            ok = rc == 0 and (job["dir"] / "SWEEP_OK").exists()
            row = {"cell": job["cell"], "scenario": job["scenario"], "seed": job["seed"], "rc": rc, "seconds": secs,
                   "status": "ok" if ok else "failed", "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failed_dirs.add(str(job["dir"]))
                try:
                    row["stderr_tail"] = Path(fh.name).read_text(encoding="utf-8", errors="replace")[-2000:]
                except Exception:  # noqa: BLE001
                    pass
                _log(root, "failures.jsonl", row)
            _log(root, "runs.jsonl", row)
            print(f"[{time.time() - t_start:7.0f}s] {row['status']:6s} {job['cell']:22s} {job['scenario']:8s} {job['seed']} ({secs}s)  "
                  f"ok {n_ok} fail {n_fail} skip {n_skip} left {len(pending)} running {len(running)}", flush=True)
        # launch
        while len(running) < nworkers:
            pick = None
            for j in pending:
                if j["needs"] is None or (j["needs"] / "SWEEP_OK").exists():
                    pick = j
                    break
                if str(j["needs"]) in failed_dirs:
                    pick = j
                    break
            if pick is None:
                break
            pending.remove(pick)
            if pick["needs"] is not None and str(pick["needs"]) in failed_dirs:
                n_skip += 1
                row = {"cell": pick["cell"], "scenario": pick["scenario"], "seed": pick["seed"], "status": "skipped",
                       "reason": f"paired paper run failed: {pick['needs']}", "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                _log(root, "failures.jsonl", row)
                _log(root, "runs.jsonl", row)
                failed_dirs.add(str(pick["dir"]))
                continue
            if pick["dir"].exists():
                shutil.rmtree(pick["dir"], ignore_errors=True)
            jf = root / "_sweep" / "jobs" / f"{pick['cell']}-{pick['scenario']}-{pick['seed']}.json"
            jf.write_text(json.dumps({"scenario": pick["scenario"], "seed": pick["seed"], "overrides": pick["overrides"],
                                      "dir": str(pick["dir"]), "paired": str(pick["paired"]) if pick["paired"] else None,
                                      "prune": prune_flag}), encoding="utf-8")
            fh = open(root / "_sweep" / "stdout" / f"{pick['cell']}-{pick['scenario']}-{pick['seed']}.txt", "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--job", str(jf)],
                                    cwd=str(WS), stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((proc, pick, time.time(), fh))
        if pending and not running:
            # nothing launchable and nothing running: unmet dependencies that will never be met
            for j in pending:
                n_skip += 1
                _log(root, "failures.jsonl", {"cell": j["cell"], "scenario": j["scenario"], "seed": j["seed"], "status": "skipped",
                                              "reason": f"dependency never completed: {j['needs']}"})
            pending = []
        time.sleep(1.0)
    wall = time.time() - t_start
    _log(root, "sessions.jsonl", {"design": args.design, "seeds": [seeds[0], seeds[-1]], "cells": args.cells or "all", "ok": n_ok, "failed": n_fail,
                                  "skipped": n_skip, "wall_seconds": round(wall), "workers": nworkers,
                                  "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"done: ok {n_ok}, failed {n_fail}, skipped {n_skip}, wall {wall / 60:.1f} min", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="configs/sweeps/2026-10-15-sweep.yaml")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cells", default=None, help="comma-separated cell names to run (default: all)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--job", default=None)
    a = ap.parse_args()
    if a.worker:
        try:
            return worker(a)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1
    return drive(a)


if __name__ == "__main__":
    sys.exit(main())
