"""Validation runs and scoring for the fast-mode recalibration against the eight 8-quarter live runs.

    python tools/fastcal_validate.py run   [--params configs/fastmode/params-stage1.yaml] [--root runs/fastcal-2026-10-15]
                                           [--seeds 2001-2030] [--twins 1-3] [--workers 6]
    python tools/fastcal_validate.py score [--root ...] [--out <md>]

`run`: fast runs of the live configurations (S0-150, S0P-150, R-b-150 budget-paired with the fast S0-150 at the same seed),
unchanged except `fastmode.params`, at the main seeds (<root>/main/...) and at the live seeds (<root>/twins/...). One
subprocess per run through tools/sweep.py's worker (the engine in-process; no model is called). Resumable: a run folder with
SWEEP_OK is done. After a run, bulky folders are removed and audit/, ledger/, budget/, handbook/*/score.json, log/,
run.json and config.resolved.yaml are kept (the criteria read the audit and the ledger).

`score`: every criterion of the frozen note, the fast median over the main seeds against its pass interval, plus the share
of seeds inside the live min-max, the fast twins beside the live runs, and fast S0P-150 beside fast S0-150. Writes CSVs and a
markdown table under <root>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
KEEP = {"audit", "budget", "handbook", "log", "ledger", "run.json", "config.resolved.yaml", "SWEEP_OK"}

# (id, world, measure, c or 'deg', degenerate bound, T/H) - the pass criteria, fixed before any fitting.
# Ids P* are journal-world criteria and R* repository-world criteria.
CRITERIA = [
    ("P1", "P", "accept_rate", 0.05, None, "H"), ("P2", "P", "r1_accept_share", "deg", ("le", 3 / 231), "T"),
    ("P3", "P", "accepted_without_revision", "deg", ("le", 3 / 15), "H"), ("P4", "P", "r1_reject_share", 0.05, None, "T"),
    ("P5", "P", "r2_accept_share", 0.05, None, "T"), ("P6", "P", "desk_reject_rate", "deg", ("le", 3 / 202), "T"),
    ("P7", "P", "new_units", 0.05, None, "H"), ("P8", "P", "rec_reject", 0.05, None, "T"),
    ("P9", "P", "rec_major_revision", 0.05, None, "T"), ("P10", "P", "rec_minor_revision", 0.05, None, "T"),
    ("P11", "P", "rec_accept", "deg", ("le", 3 / 909), "T"), ("P12", "P", "reviewer_kappa", 0.05, None, "H"),
    ("P13", "P", "flaw_noticed_rate", "deg", ("ge", 1 - 3 / 414), "T"), ("P14", "P", "false_alarm_rate", "deg", ("ge", 1 - 3 / 495), "T"),
    ("P15", "P", "detection_lift", "deg", ("abs", 3 / 414 + 3 / 495), "H"), ("P16", "P", "flaw_found_flagged", 0.05, None, "T"),
    ("P17", "P", "flaw_found_clean", 0.05, None, "T"), ("P18", "P", "support_share", 0.05, None, "H"),
    ("P19", "P", "hb_acc_q0", 0.05, None, "H"), ("P20", "P", "hb_acc_final", 0.05, None, "H"), ("P21", "P", "hb_acc_change", 0.05, None, "H"),
    ("P22", "P", "nk_B1_change", 0.05, None, "H"), ("P23", "P", "nk_B2_change", 0.05, None, "H"),
    ("P24", "P", "data", 0.05, None, "H"), ("P25", "P", "compute", 0.05, None, "H"),
    ("R1", "R", "support_share", 0.80, None, "T"), ("R2", "R", "vote_endorse", 0.80, None, "T"), ("R3", "R", "vote_challenge", 0.80, None, "T"),
    ("R4", "R", "vote_abstain", 0.80, None, "T"), ("R5", "R", "unverified_endorsement_share", 0.80, None, "H"),
    ("R6", "R", "verif_powered_share", 0.80, None, "H"), ("R7", "R", "verif_reproduced_share", 0.80, None, "H"),
    ("R8", "R", "filed_share", 0.80, None, "T"), ("R9", "R", "new_units", 0.80, None, "H"), ("R10", "R", "hb_acc_q0", 0.80, None, "H"),
    ("R11", "R", "hb_acc_final", 0.80, None, "H"), ("R12", "R", "hb_acc_change", 0.80, None, "H"), ("R13", "R", "nk_B1_change", 0.80, None, "H"),
    ("R14", "R", "nk_B2_change", 0.80, None, "H"), ("R15", "R", "data", 0.80, None, "H"), ("R16", "R", "compute", 0.80, None, "H"),
]
DIAG_P = ["inherited_units", "live_published", "run_coverage", "new_false_share", "n_r1_decisions", "n_r2_decisions", "n_reviews"]
DIAG_R = ["inherited_units", "run_coverage", "new_false_share", "compute_ratio_to_paired", "n_votes", "n_verifications", "n_entries"]


def _seeds(spec: str) -> list[int]:
    out = []
    for part in (spec or "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _prune(rd: Path) -> None:
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


def run(a) -> int:
    root = (WS / a.root).resolve()
    params = str((WS / a.params).resolve())
    jobs = []
    for cell, seeds in (("main", _seeds(a.seeds)), ("twins", _seeds(a.twins))):
        for s in seeds:
            for scen in ("S0-150", "S0P-150"):
                jobs.append({"scenario": scen, "seed": s, "dir": root / cell / scen / str(s) / "fast", "paired": None, "needs": None})
            pdir = root / cell / "S0-150" / str(s) / "fast"
            jobs.append({"scenario": "R-b-150", "seed": s, "dir": root / cell / "R-b-150" / str(s) / "fast", "paired": pdir, "needs": pdir})
    (root / "_jobs").mkdir(parents=True, exist_ok=True)
    (root / "_stdout").mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (j["dir"] / "SWEEP_OK").exists()]
    print(f"{len(jobs)} runs, {len(jobs) - len(pending)} done, {len(pending)} to run with {a.workers} workers; params {params}", flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    running, failed, t0 = [], set(), time.time()
    n_ok = n_fail = 0
    while pending or running:
        for item in running[:]:
            proc, job, ts, fh = item
            rc = proc.poll()
            if rc is None:
                continue
            fh.close(); running.remove(item)
            ok = rc == 0 and (job["dir"] / "SWEEP_OK").exists()
            if ok:
                n_ok += 1
                _prune(job["dir"])
            else:
                n_fail += 1; failed.add(str(job["dir"]))
            row = {"scenario": job["scenario"], "seed": job["seed"], "dir": str(job["dir"]), "rc": rc, "ok": ok, "seconds": round(time.time() - ts, 1)}
            with open(root / "_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{time.time() - t0:6.0f}s] {'ok  ' if ok else 'FAIL'} {job['scenario']:8s} {job['seed']:5d} ({row['seconds']}s) ok {n_ok} fail {n_fail} left {len(pending)}", flush=True)
        while len(running) < a.workers:
            pick = next((j for j in pending if j["needs"] is None or (j["needs"] / "SWEEP_OK").exists() or str(j["needs"]) in failed), None)
            if pick is None:
                break
            pending.remove(pick)
            if pick["needs"] is not None and str(pick["needs"]) in failed:
                n_fail += 1; failed.add(str(pick["dir"]))
                continue
            if pick["dir"].exists():
                shutil.rmtree(pick["dir"], ignore_errors=True)
            jf = root / "_jobs" / f"{pick['dir'].parents[2].name}-{pick['scenario']}-{pick['seed']}.json"
            jf.write_text(json.dumps({"scenario": pick["scenario"], "seed": pick["seed"], "overrides": ({"fastmode.params": params} | ({"fastmode.prehistory_params": a.prehistory} if a.prehistory else {})),
                                      "dir": str(pick["dir"]), "paired": str(pick["paired"]) if pick["paired"] else None, "prune": False}), encoding="utf-8")
            fh = open(root / "_stdout" / f"{jf.stem}.txt", "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-X", "utf8", str(WS / "tools" / "sweep.py"), "--worker", "--job", str(jf)],
                                    cwd=str(WS), stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((proc, pick, time.time(), fh))
        if pending and not running:
            break
        time.sleep(1.0)
    print(f"done: ok {n_ok}, failed {n_fail}, wall {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0 if n_fail == 0 else 1


def _collect(root: Path, cell: str) -> dict:
    from fastcal_measures import measures
    out = {}
    for scen in ("S0-150", "S0P-150", "R-b-150"):
        d = root / cell / scen
        if not d.exists():
            continue
        for sd in sorted(d.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            rd = sd / "fast"
            if (rd / "SWEEP_OK").exists():
                out[(scen, int(sd.name))] = measures(rd)
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def score(a) -> int:
    from fastcal_measures import live_table
    root = (WS / a.root).resolve()
    live = live_table()
    main = _collect(root, "main")
    twins = _collect(root, "twins")
    liveP = [v for v in live.values() if v["regime"] == "paper"]
    liveR = [v for v in live.values() if v["regime"] == "repository"]
    rows, csv = [], ["id,world,measure,TH,live_min,live_max,lo,hi,fast_median,fast_p25,fast_p75,n_seeds,share_in_live_range,pass,s0p_median"]
    for cid, w, m, c, deg, th in CRITERIA:
        lv = [_num(v.get(m)) for v in (liveP if w == "P" else liveR)]
        lv = [x for x in lv if x is not None]
        scen = "S0-150" if w == "P" else "R-b-150"
        fv = [_num(v.get(m)) for (s_, _), v in main.items() if s_ == scen]
        fv = [x for x in fv if x is not None]
        sv = [_num(v.get(m)) for (s_, _), v in main.items() if s_ == "S0P-150"] if w == "P" else []
        sv = [x for x in sv if x is not None]
        lmin, lmax = min(lv), max(lv)
        if c == "deg":
            kind, b = deg
            lo, hi = (-np.inf, b) if kind == "le" else ((b, np.inf) if kind == "ge" else (-b, b))
        else:
            tol = c * (lmax - lmin)
            lo, hi = lmin - tol, lmax + tol
        med = float(np.median(fv)) if fv else None
        ok = med is not None and lo <= med <= hi
        inr = float(np.mean([lmin <= x <= lmax for x in fv])) if fv else None
        smed = float(np.median(sv)) if sv else None
        rows.append({"id": cid, "world": w, "measure": m, "TH": th, "live_min": lmin, "live_max": lmax, "lo": lo, "hi": hi, "median": med,
                     "p25": float(np.quantile(fv, .25)) if fv else None, "p75": float(np.quantile(fv, .75)) if fv else None, "n": len(fv),
                     "in_range": inr, "pass": ok, "s0p_median": smed, "s0p_pass": (smed is not None and lo <= smed <= hi) if w == "P" else None})
        csv.append(",".join(str(x) for x in (cid, w, m, th, lmin, lmax, lo, hi, med, rows[-1]["p25"], rows[-1]["p75"], len(fv), inr, ok, smed)))
    (root / "criteria_results.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    def f(x, m):
        if x is None:
            return "n/a"
        if x in (np.inf, -np.inf):
            return "-" if x < 0 else "+inf"
        if m in ("data", "compute"):
            return f"{x:,.0f}"
        if m in ("new_units", "nk_B1_change", "nk_B2_change"):
            return f"{x:.1f}"
        return f"{x:.3f}"
    md = ["| # | Measure | T/H | Live min-max | Pass interval | Fast median [IQR] (30 seeds) | Seeds inside live min-max | Result | Fast S0P median |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo = "" if r["lo"] == -np.inf else f(r["lo"], r["measure"])
        hi = "" if r["hi"] == np.inf else f(r["hi"], r["measure"])
        iv = f"[{lo}, {hi}]" if lo and hi else (f"<= {hi}" if hi else f">= {lo}")
        inr = "n/a" if r["in_range"] is None else format(r["in_range"], ".2f")
        res = "pass" if r["pass"] else "FAIL"
        s0p = "" if r["world"] == "R" else f(r["s0p_median"], r["measure"])
        md.append(f"| {r['id']} | {r['measure']} | {r['TH']} | {f(r['live_min'], r['measure'])}-{f(r['live_max'], r['measure'])} | {iv} | "
                  f"{f(r['median'], r['measure'])} [{f(r['p25'], r['measure'])}, {f(r['p75'], r['measure'])}] | {inr} | **{res}** | {s0p} |")
    # twins
    tw = ["| Live run | Measure | Live | Fast twin (same seed) |", "|---|---|---|---|"]
    for key, lvv in live.items():
        scen, s = key.split(" s")
        t = twins.get((scen, int(s)))
        if not t:
            continue
        for cid, w, m, *_ in CRITERIA:
            if (w == "P") != (lvv["regime"] == "paper"):
                continue
            if m in ("accept_rate", "r1_accept_share", "accepted_without_revision", "r2_accept_share", "new_units", "rec_reject", "reviewer_kappa",
                     "support_share", "vote_endorse", "verif_powered_share", "hb_acc_final", "nk_B1_change"):
                tw.append(f"| {key} | {m} | {f(_num(lvv.get(m)), m)} | {f(_num(t.get(m)), m)} |")
    diag = ["| World | Diagnostic | Live values | Fast median [IQR] |", "|---|---|---|---|"]
    for w, keys, lvs, scen in (("P", DIAG_P, liveP, "S0-150"), ("R", DIAG_R, liveR, "R-b-150")):
        for m in keys:
            fv = [_num(v.get(m)) for (s_, _), v in main.items() if s_ == scen]
            fv = [x for x in fv if x is not None]
            lvx = ", ".join(f(_num(v.get(m)), m) for v in lvs)
            diag.append(f"| {w} | {m} | {lvx} | {f(float(np.median(fv)), m) if fv else 'n/a'} [{f(float(np.quantile(fv, .25)), m) if fv else ''}, {f(float(np.quantile(fv, .75)), m) if fv else ''}] |")
    n_pass = sum(r["pass"] for r in rows)
    head = [f"Criteria: {n_pass}/{len(rows)} pass. Main seeds: S0-150 {sum(1 for k in main if k[0] == 'S0-150')}, S0P-150 {sum(1 for k in main if k[0] == 'S0P-150')}, "
            f"R-b-150 {sum(1 for k in main if k[0] == 'R-b-150')} runs. Fast S0P = fast S0 on the paper criteria: {all((r['s0p_pass'] == r['pass']) for r in rows if r['world'] == 'P')}.", ""]
    text = "\n".join(head + ["### Criteria", ""] + md + ["", "### Fast twins at the live seeds (not scored)", ""] + tw + ["", "### Diagnostics (not scored)", ""] + diag) + "\n"
    out = Path(a.out) if a.out else root / "criteria_results.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score"))
    ap.add_argument("--params", default="configs/fastmode/params-stage1.yaml")
    ap.add_argument("--root", default="runs/fastcal-2026-10-15")
    ap.add_argument("--prehistory", default="fastmode/params.yaml", help="warm-start rule file; empty = same as --params")
    ap.add_argument("--seeds", default="2001-2030")
    ap.add_argument("--twins", default="1-3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    sys.exit(main())
