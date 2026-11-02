"""Command line: run / resume / ingest / audit / montecarlo / pair / regen-dataset / freeze.

    python -m synthsci run --scenario S0 --seed 1 --mode mock
    python -m synthsci run --scenario R-b --seed 1 --mode mock          # pairs with runs/S0/1/mock
    python -m synthsci run --scenario S0 --seed 1 --mode llm --backend subagent   # writes packets, stops at the boundary
    python -m synthsci ingest --run runs/S0/1/llm                          # validate outputs, write DONE/FAILED
    python -m synthsci resume --run runs/S0/1/llm
    python -m synthsci extend --run runs/S0-150/1/llm --to 20               # -> runs/S0-150-x20/1/llm, then resume it
    python -m synthsci montecarlo --scenario S0 --seeds 1-100 --mode fast
    python -m synthsci pair --paper S0 --repo R-b --seeds 1-3 --mode mock
    python -m synthsci audit --run runs/S0/1/mock
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config as C


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            import yaml
            out[k] = yaml.safe_load(v)
        except Exception:  # noqa: BLE001
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="synthsci")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scenario", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--mode", default="mock", choices=C.MODES)
    r.add_argument("--backend", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--set", nargs="*", help="config overrides key=value (dotted keys)")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--allow-unpaired", action="store_true", help="budget-pairing override: run a pairing scenario on the fixed budget when the paired spend is missing (logged)")
    rs = sub.add_parser("resume")
    rs.add_argument("--run", required=True)
    ex = sub.add_parser("extend", help="copy a finished run to <scenario>-x<H>/<seed>/<mode>, reopen its close-out, keep the original horizon's rules")
    ex.add_argument("--run", required=True, help="the finished run folder (never modified)")
    ex.add_argument("--to", type=int, required=True, help="the new horizon in quarters")
    ex.add_argument("--out-root", default=None, help="root of the new folder (default: runs/)")
    ex.add_argument("--paired-run", default=None, help="repository runs: the EXTENDED paper run to pair with (default <out-root>/<paired>-x<H>/<seed>/<mode>)")
    ing = sub.add_parser("ingest")
    ing.add_argument("--run", required=True)
    au = sub.add_parser("audit")
    au.add_argument("--run", required=True)
    mc = sub.add_parser("montecarlo")
    mc.add_argument("--scenario", required=True)
    mc.add_argument("--seeds", required=True)
    mc.add_argument("--mode", default="fast", choices=C.MODES)
    mc.add_argument("--out", default=None)
    mc.add_argument("--set", nargs="*")
    mc.add_argument("--quiet", action="store_true")
    mc.add_argument("--allow-unpaired", action="store_true")
    ag = sub.add_parser("aggregate")
    ag.add_argument("--scenario", required=True)
    ag.add_argument("--seeds", required=True)
    ag.add_argument("--mode", default="fast")
    ag.add_argument("--out", default=None)
    pr = sub.add_parser("pair")
    pr.add_argument("--paper", required=True)
    pr.add_argument("--repo", required=True)
    pr.add_argument("--seeds", required=True)
    pr.add_argument("--mode", default="mock")
    pr.add_argument("--out", default=None)
    rg = sub.add_parser("regen-dataset")
    rg.add_argument("--run", required=True)
    rg.add_argument("--dataset", required=True)
    rg.add_argument("--dest", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--verify", action="store_true")
    g5 = sub.add_parser("gate5")
    g5.add_argument("--scenario", default="S0"); g5.add_argument("--seed", type=int, default=1); g5.add_argument("--n", type=int, default=200)
    ff = sub.add_parser("flaw-floor")
    ff.add_argument("--scenario", default="S0"); ff.add_argument("--seed", type=int, default=1); ff.add_argument("--n", type=int, default=20)
    pp = sub.add_parser("prior-probe")
    pp.add_argument("--scenario", default="S0"); pp.add_argument("--seed", type=int, default=1); pp.add_argument("--out", default=None)
    sp = sub.add_parser("score-probe")
    sp.add_argument("--run", required=True)
    co = sub.add_parser("cost")
    co.add_argument("--scenario", required=True)
    co.add_argument("--set", nargs="*")
    co.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "cost":
        from .cost import estimate, table
        import json as _json
        est = estimate(C.resolve(a.scenario, _overrides(a.set)))
        print(_json.dumps(est, indent=2) if a.json else table(est))
        return 0

    if a.cmd in ("gate5", "flaw-floor", "prior-probe", "score-probe"):
        from . import gates
        import json as _json
        if a.cmd == "gate5":
            res = gates.gate5(a.scenario, a.seed, a.n, C.WS_ROOT / "runs" / "_gates" / f"gate5-{a.scenario}-s{a.seed}.json")
        elif a.cmd == "flaw-floor":
            res = gates.flaw_floor(a.scenario, a.seed, a.n, C.WS_ROOT / "runs" / "_gates" / f"flawfloor-{a.scenario}-s{a.seed}.json")
            res = {k: v for k, v in res.items() if k != "details"}
        elif a.cmd == "prior-probe":
            res = {"run_dir": str(gates.prior_probe(a.scenario, a.seed, Path(a.out) if a.out else None))}
        else:
            res = gates.score_probe(Path(a.run).resolve())
        print(_json.dumps(res, indent=2, default=str))
        return 0

    if a.cmd == "run":
        from .engine import run
        t0 = time.time()
        rd = run(a.scenario, a.seed, a.mode, a.backend, _overrides(a.set), Path(a.out) if a.out else None, resume=not a.no_resume, quiet=a.quiet,
                 allow_unpaired=a.allow_unpaired)
        if not a.quiet:
            print(f"{time.time() - t0:.1f}s -> {rd}")
        return 0
    if a.cmd == "resume":
        return _resume(Path(a.run).resolve())
    if a.cmd == "extend":
        from .extend import extend
        import json as _json
        res = extend(Path(a.run), a.to, Path(a.out_root) if a.out_root else None, Path(a.paired_run) if a.paired_run else None)
        print(_json.dumps({k: res[k] for k in ("run_dir", "run_id", "task_namespace", "source_manifest_sha256", "source_unchanged", "reopen")},
                          indent=2, default=str))
        return 0
    if a.cmd == "ingest":
        from .backends.subagent import ingest
        rep = ingest(Path(a.run).resolve())
        print(f"ingested: {rep['done']} done, {rep['failed']} failed, {rep['pending']} pending")
        return 0
    if a.cmd == "audit":
        from .audit import run_audit
        run_audit(Path(a.run).resolve(), quiet=False)
        return 0
    if a.cmd == "montecarlo":
        from .audit import aggregate
        from .engine import run, run_dir_for
        dirs = []
        t0 = time.time()
        for s in _seeds(a.seeds):
            dirs.append(run(a.scenario, s, a.mode, None, _overrides(a.set), Path(a.out) if a.out else None, quiet=True, allow_unpaired=a.allow_unpaired))
            if not a.quiet:
                print(f"seed {s} done ({time.time() - t0:.0f}s)")
        cfg = C.resolve(a.scenario, _overrides(a.set))
        root = Path(a.out) if a.out else C.WS_ROOT / "runs"
        out = aggregate(dirs, root / "_aggregates" / f"{cfg['scenario']}-{a.mode}-distribution.json")
        print(f"{out['n_runs']} runs aggregated -> {root / '_aggregates'}")
        return 0
    if a.cmd == "aggregate":
        from .audit import aggregate
        from .engine import run_dir_for
        root = Path(a.out) if a.out else C.WS_ROOT / "runs"
        dirs = [run_dir_for({"scenario": a.scenario}, s, a.mode, root) for s in _seeds(a.seeds)]
        dirs = [d for d in dirs if (d / "audit" / "audit.json").exists()]
        out = aggregate(dirs, root / "_aggregates" / f"{a.scenario}-{a.mode}-distribution.json")
        print(f"{out['n_runs']} runs aggregated -> {root / '_aggregates' / (a.scenario + '-' + a.mode + '-distribution.json')}")
        return 0
    if a.cmd == "pair":
        from .audit import pair_regimes
        from .engine import run_dir_for
        root = Path(a.out) if a.out else C.WS_ROOT / "runs"
        seeds = _seeds(a.seeds)
        pd_ = {s: run_dir_for({"scenario": a.paper}, s, a.mode, root) for s in seeds}
        rd_ = {s: run_dir_for({"scenario": a.repo}, s, a.mode, root) for s in seeds}
        out = pair_regimes(pd_, rd_, root / "_aggregates" / f"paired-{a.repo}-vs-{a.paper}-{a.mode}.json")
        print(f"paired {len(out['seeds'])} seeds -> {root / '_aggregates'}")
        return 0
    if a.cmd == "regen-dataset":
        return _regen(Path(a.run).resolve(), a.dataset, Path(a.dest))
    if a.cmd == "freeze":
        from .freeze import freeze, verify
        return verify() if a.verify else freeze()
    return 1


def _resume(rd: Path) -> int:
    from .engine import run
    from .util import read_json
    meta = read_json(rd / "run.json")
    import yaml
    from .util import read_text
    cfg = yaml.safe_load(read_text(rd / "config.resolved.yaml"))
    out_root = rd.parents[2]
    # the folder given is the run (an extended run's folder is not <scenario>/<seed>/<mode>)
    run(cfg, meta["seed"], meta["mode"], meta["backend"], None, out_root, Path(meta["paired_run_dir"]) if meta.get("paired_run_dir") else None,
        resume=True, run_dir=rd)
    return 0


def _regen(rd: Path, dataset_id: str, dest: Path) -> int:
    from .util import read_jsonl
    from .engine import World
    import yaml
    from .util import read_text, read_json
    cfg = yaml.safe_load(read_text(rd / "config.resolved.yaml"))
    meta = read_json(rd / "run.json")
    rows = [r for r in read_jsonl(rd / "ledger" / "datasets.jsonl") if r["dataset_id"] == dataset_id]
    if not rows:
        print("dataset not found in ledger")
        return 1
    row = rows[0]
    w = World(cfg, meta["seed"], meta["mode"], "mock", rd, None)
    res = w.gen.generate(row["question_id"], row["meta"]["dataset_seed"], dataset_id, upper_half=row["meta"]["frame"].get("upper_half", False), write_dir=dest)
    ok = res["meta"]["data_sha256"] == row["meta"]["data_sha256"]
    print(f"regenerated {dataset_id} -> {dest}; hash {'matches' if ok else 'DIFFERS'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
