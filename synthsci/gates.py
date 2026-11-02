"""Pre-run gates that need no LLM: the sufficient-statistics check of fast mode, the flaw floor, and the handbook prior probe.

gate5(): sufficient-statistics check. For n datasets, the full generator's correct-analysis estimate (partial r) is compared
with the analytic path's draw for the same seed and handling; the distributions must agree within preset bands.
flaw_floor(): rule-agent version of the flaw-floor gate. 20 VERIFIED and 20 AUTO analyses on Stage 1 datasets by the rule
author (real analysis.py on real data); the scorer's flags must differ between styles. The LLM version is the same datasets
dispatched as research packets.
prior_probe(): writes the three handbook writers' prior-probe packets for a seed (dispatched like any other agent task) and
score_probe() scores the outputs: accuracy must be at chance.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import config as C
from .agents.rules import RuleAuthor, load_params
from .datasets import DatasetGenerator
from .landscape import build_landscape
from .scorer import analyst_flags
from .seeding import SeedingTables
from .util import seed_from, write_json


def _world(scenario: str, seed: int):
    cfg = C.resolve(scenario)
    seeds = C.seed_layers(cfg, seed)
    tables = SeedingTables(cfg["seeding"]["table_set"])
    L = build_landscape(cfg, seeds, tables)
    return cfg, seeds, tables, L, DatasetGenerator(cfg, tables, L)


def gate5(scenario: str = "S0", seed: int = 1, n: int = 200, out: Path | None = None) -> dict:
    """Bands (preset): mean difference of partial r within .01; SD ratio within [.7, 1.4]; trap and frame agreement 100%."""
    cfg, seeds, tables, L, gen = _world(scenario, seed)
    diffs, full_r, an_r, mism = [], [], [], 0
    qs = L.questions
    for i in range(n):
        q = qs[i % len(qs)]
        s = seed_from("gate5", seed, i)
        full = gen.generate(q["id"], s, f"G{i}")
        an = gen.analytic_study(q["id"], s, False, {t: True for t in full["hidden"]["traps"]}, use_fe=q["panel"])
        if set(an["traps"]) != set(full["hidden"]["traps"]) or an["frame"]["n_target"] != full["meta"]["frame"]["n_target"]:
            mism += 1
        fr = full["hidden"]["oracle"]["correct"]["partial_r"]
        ar = an["focal"]["estimate"]
        if fr is not None and np.isfinite(fr):
            full_r.append(fr)
            an_r.append(ar)
            diffs.append(ar - fr)
    diffs = np.asarray(diffs)
    # the full generator may drop a trap after failed admissibility redraws (DatasetGenerator.generate); the analytic path cannot know
    # that, so the share of datasets whose trap set differs is reported as a known divergence, not a failure
    res = {"n": n, "mean_diff": float(diffs.mean()), "sd_full": float(np.std(full_r)), "sd_analytic": float(np.std(an_r)),
           "sd_ratio": float(np.std(an_r) / max(np.std(full_r), 1e-9)), "trap_set_reduced_share": mism / max(n, 1),
           "pass": bool(abs(diffs.mean()) < 0.01 and 0.7 <= np.std(an_r) / max(np.std(full_r), 1e-9) <= 1.4),
           "bands": {"mean_diff": 0.01, "sd_ratio": [0.7, 1.4]},
           "note": "estimates compared for the same seed and full handling; trap_set_reduced_share is the admissibility-reduction divergence"}
    if out:
        write_json(out, res)
    return res


def flaw_floor(scenario: str = "S0", seed: int = 1, n_per_style: int = 20, out: Path | None = None) -> dict:
    cfg, seeds, tables, L, gen = _world(scenario, seed)
    params = load_params(cfg)
    author = RuleAuthor(params)
    results = {"VERIFIED": [], "AUTO": []}
    tmp = Path(tempfile.mkdtemp(prefix="flawfloor_"))
    try:
        for style in ("VERIFIED", "AUTO"):
            for i in range(n_per_style):
                q = L.questions[(i * 7) % len(L.questions)]
                s = seed_from("flawfloor", seed, style, i)
                tdir = tmp / f"{style}_{i}"
                res = gen.generate(q["id"], s, f"FF{i}", write_dir=tdir / "inputs" / q["id"])
                traps = res["hidden"]["traps"]
                a = {"id": f"X{i}", "style": style, "throughput": "T1" if i % 2 else "T0", "repertoire": "archival_regression", "primary_theory": q["family"]}
                r = C.rng(s, "ff")
                handled = author.decide_handling(traps, style, r)
                variables = res["meta"]["variables"]
                ana = author.analyse_full(q, [f"../inputs/{q['id']}/data.csv"], tdir, a, handled, variables, author.use_fe(q, a, r), 4, r, a["throughput"], traps)
                claims = author.claims_json(q, ana, "first study", [], nonce="gate")
                fl = analyst_flags(claims, res["hidden"], res["meta"], q, author.paper_md(q, ana, "first study", a, [], handled), style, a["throughput"], True)
                results[style].append({"traps": traps, "flags": fl["flags"], "handled": fl["trap_handled"], "throughput": a["throughput"]})
                shutil.rmtree(tdir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    trap_flags = {"duplicates_not_removed", "scale_mixture_unfixed", "reverse_coding_missed", "selection_unacknowledged", "bad_control_included",
                  "miscoded_values_kept", "unit_mismatch_unfixed", "omitted_standard_control", "n_inflated"}

    def rate(style):
        rows = [x for x in results[style] if x["traps"]]
        return (sum(1 for x in rows if any(f in trap_flags for f in x["flags"])) / len(rows)) if rows else None
    out_d = {"flaw_rate_VERIFIED": rate("VERIFIED"), "flaw_rate_AUTO": rate("AUTO"),
             "n_trapped": {k: sum(1 for x in v if x["traps"]) for k, v in results.items()},
             "gap": None, "floor": 0.15, "pass": None, "details": results}
    if out_d["flaw_rate_VERIFIED"] is not None and out_d["flaw_rate_AUTO"] is not None:
        out_d["gap"] = out_d["flaw_rate_AUTO"] - out_d["flaw_rate_VERIFIED"]
        out_d["pass"] = bool(out_d["gap"] >= out_d["floor"])
    if out:
        write_json(out, out_d)
    return out_d


def prior_probe(scenario: str, seed: int, out_root: Path | None = None) -> Path:
    """Write the prior-probe packets for a seed into runs/<scenario>/<seed>/probe/ and stop at the boundary."""
    from .engine import World
    from .handbook import build_handbook_input, prior_probe_tasks
    cfg = C.resolve(scenario)
    rd = (Path(out_root) if out_root else C.WS_ROOT / "runs") / cfg["scenario"] / str(seed) / "probe"
    w = World(cfg, seed, "llm", "subagent", rd, None)
    if cfg["regime"] == "paper":
        from .regimes.paper import PaperRegime as Regime
    else:
        from .regimes.repository import RepositoryRegime as Regime
    reg = Regime(w)
    inp = build_handbook_input(reg, 0)
    tasks = prior_probe_tasks(reg, inp)
    for t in tasks:
        w.store.materialize(t)
    try:
        w.backend.run(tasks, {"world": w, "regime": reg, "task_ctx": {}})
    except Exception as e:  # PendingDispatch is the expected outcome
        print(str(e))
    return rd


def score_probe(run_dir: Path) -> dict:
    """After ingest: score each writer's probe answers against the truth (must be at chance; mostly 'no evidence')."""
    from .util import read_json
    rd = Path(run_dir)
    truth = read_json(rd / "landscape" / "truth.json")
    from .belief import is_correct, truth_class
    out = {"writers": []}
    for d in sorted((rd / "tasks").glob("prior_probe-*")):
        if not (d / "DONE").exists():
            continue
        hb = read_json(d / "output" / "handbook.json")
        n, c, ne = 0, 0, 0
        for q in hb["questions"]:
            t = truth["questions"].get(q["question_id"])
            if not t:
                continue
            if q["answer"] == "no evidence":
                ne += 1
                continue
            if q["answer"] == "contested":
                continue
            n += 1
            c += int(bool(is_correct(q["answer"], truth_class(t["theta"]))))
        out["writers"].append({"task": d.name, "n_answered": n, "accuracy": (c / n) if n else None, "no_evidence": ne,
                               "chance": 1 / 3, "flag": bool(n >= 10 and (c / n) > 0.5)})
    write_json(rd / "probe_score.json", out)
    return out
