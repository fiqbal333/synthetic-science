"""Equivalence check for `synthsci extend` in any mode: an extended run must equal a run made at the longer horizon in one go (the suite runs it in fast mode, tests/test_extend.py; mock mode takes ~9 min).

    python tools/extend_equivalence_check.py <scratch dir> [mode=mock] [seed=3]

Builds smoke-P and smoke-R (careers on) to quarter 4, extends both to 8 and resumes them (the repository run paired with
the extended paper run), runs the same extended configs in one go, and compares every ledger table (run_id and the
extension rows excluded). Prints 'identical', 'same rows, different order' or 'CONTENT DIFFERS' per table. Exit 1 on a
content difference.
"""
import json
import shutil
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
import yaml  # noqa: E402

from synthsci import config as C  # noqa: E402
from synthsci.cli import main as cli  # noqa: E402
from synthsci.engine import run  # noqa: E402
from synthsci.extend import extend, extended_config  # noqa: E402
from synthsci.util import read_json, read_jsonl, read_text  # noqa: E402

SHORT, LONG = 4, 8
TABLES = ("hypotheses", "papers", "claims", "votes", "verifications", "reviews", "decisions", "datasets", "budget", "events")


def main(argv) -> int:
    root = Path(argv[0]).resolve()
    mode = argv[1] if len(argv) > 1 else "mock"
    seed = int(argv[2]) if len(argv) > 2 else 3
    if root.exists():
        shutil.rmtree(root)
    ov = {"datasets": {"keep_csv": False}, "horizon_quarters": SHORT, "long_horizon": {"turnover": {"min_horizon_quarters": 1}}}
    pin = {"horizon_quarters": LONG, "long_horizon": {"pin_short_rules": True, "extended_from": [SHORT]}}
    stored = lambda d: yaml.safe_load(read_text(d / "config.resolved.yaml"))
    out = {}
    pairs = []
    shorts = {}
    for name in ("smoke-P", "smoke-R"):
        shorts[name] = run(C.deep_merge(C.resolve(name), ov), seed, mode, out_root=root / "short", quiet=True)
    for name in ("smoke-P", "smoke-R"):
        info = extend(shorts[name], LONG, root / "ext")
        cli(["resume", "--run", info["run_dir"]])
        cfg = C.deep_merge(C.deep_merge(C.resolve(name), ov), pin)
        assert C.config_hash(cfg) == C.config_hash(extended_config(stored(shorts[name]), LONG))
        straight = run(cfg, seed, mode, out_root=root / "straight", quiet=True)
        pairs.append((name, Path(info["run_dir"]), straight, info["reopen"]))
    bad = False
    for name, a, b, rep in pairs:
        res = {"reopen": rep}
        for t in TABLES:
            ra = [{k: v for k, v in r.items() if k != "run_id"} for r in read_jsonl(a / "ledger" / f"{t}.jsonl")
                  if not (r.get("event") == "extension_reopened" or (r.get("event") == "unfinished_at_horizon" and r.get("quarter") == SHORT))]
            rb = [{k: v for k, v in r.items() if k != "run_id"} for r in read_jsonl(b / "ledger" / f"{t}.jsonl")]
            canon = lambda rows: sorted(json.dumps(x, sort_keys=True) for x in rows)
            res[t] = f"identical ({len(ra)} rows)" if ra == rb else (f"same rows, different order ({len(ra)})" if canon(ra) == canon(rb) else "CONTENT DIFFERS")
            bad |= res[t] == "CONTENT DIFFERS"
        res["state"] = "identical" if read_json(a / "state" / "state.json") == read_json(b / "state" / "state.json") else "DIFFERS"
        res["population"] = "identical" if read_json(a / "state" / "world.json") == read_json(b / "state" / "world.json") else "DIFFERS"
        aa, ab = read_json(a / "audit" / "audit.json"), read_json(b / "audit" / "audit.json")
        res["audit_knowledge"] = "identical" if all(aa[k] == ab[k] for k in ("field", "theory", "handbook")) else "DIFFERS"
        bad |= "DIFFERS" in (res["state"], res["population"], res["audit_knowledge"])
        out[name] = res
    print(json.dumps({"mode": mode, "seed": seed, "result": out}, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
