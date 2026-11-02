"""Both regimes end to end with the mock backend, fast mode, byte identity of reruns, the paired budget, and resume."""
import json
from pathlib import Path

import pytest

from synthsci import config as C
from synthsci.engine import run
from synthsci.util import hash_tree, read_json, read_jsonl


@pytest.fixture(scope="module")
def smoke_runs(tmp_runs):
    p = run("smoke-P", 1, "mock", out_root=tmp_runs, quiet=True)
    r = run("smoke-R", 1, "mock", out_root=tmp_runs, quiet=True)
    return p, r


def test_paper_regime_artifacts(smoke_runs):
    p, _ = smoke_runs
    for f in ("run.json", "config.resolved.yaml", "landscape/truth.json", "landscape/deid_map.json", "population/scientists.json",
              "ledger/hypotheses.jsonl", "ledger/papers_snapshot.json", "ledger/reviews.jsonl", "ledger/decisions.jsonl", "ledger/budget.jsonl",
              "budget/spend_by_quarter.json", "audit/audit.json", "audit/summary.md", "handbook/Q0/panel.json", "handbook/Q4/score.json", "tasks/manifest.jsonl"):
        assert (p / f).exists(), f
    papers = [x for x in read_json(p / "ledger" / "papers_snapshot.json") if not x["legacy"]]
    assert papers, "no papers were produced"
    assert all(x["status"] in ("published", "rejected", "desk_rejected", "abandoned", "unfinished") for x in papers), [x["status"] for x in papers]
    assert any(x["status"] == "published" for x in papers)
    assert all(x["reproducible"] is True for x in papers if x["reproducible"] is not None)
    hyps = [h for h in read_jsonl(p / "ledger" / "hypotheses.jsonl") if not h["legacy"]]
    assert all(h["truth_check"] in ("true_discovery", "false_discovery", "missed_effect", "correct_null", "true_discovery_negligible", "correct_null_negligible", "unknown") for h in hyps)
    man = read_jsonl(p / "tasks" / "manifest.jsonl")
    created = {m["task_id"] for m in man if m["event"] == "created"}
    done = {m["task_id"] for m in man if m["event"] == "done"}
    assert created == done, "every task must end DONE in mock mode"
    # no data.csv left behind in mock mode
    assert not list((p / "tasks").rglob("data.csv"))


def test_repository_regime_artifacts_and_equal_budget(smoke_runs):
    p, r = smoke_runs
    for f in ("population/labs.json", "ledger/claims.jsonl", "ledger/votes.jsonl", "ledger/registry_snapshot.json", "ledger/reputation.json",
              "budget/summary.json", "audit/audit.json", "handbook/Q4/score.json"):
        assert (r / f).exists(), f
    meta = read_json(r / "run.json")
    assert meta["pairing_fallback"] is None, meta["pairing_fallback"]
    sp = read_json(p / "budget" / "spend_by_quarter.json")
    sr = read_json(r / "budget" / "spend_by_quarter.json")
    allot = read_json(r / "budget" / "summary.json")["allotments"]
    # paired budgets: R's compute does not exceed P's compute (the lab-decision calls are debited first and may
    # exceed a thin early allotment; the carry-over rule settles it within the run); R's data floats under the ceiling of
    # data_ceiling_multiple x P's realized data (2 in smoke-R); the per-quarter allotments are P's compute and that multiple of P's data
    dcm = float(C.resolve("smoke-R")["budget"]["data_ceiling_multiple"])
    tot_p = {res: sum(v[res] for v in sp.values()) for res in ("data", "compute")}
    tot_r = {res: sum(v[res] for v in sr.values()) for res in ("data", "compute")}
    for res in ("data", "compute"):
        assert tot_p[res] > 0 and tot_r[res] > 0, (res, tot_r, tot_p)          # both fields spend (no silent zero tallies)
    assert tot_r["compute"] <= tot_p["compute"] * 1.05 + 1e-6, (tot_r, tot_p)
    assert tot_r["data"] <= dcm * tot_p["data"] * 1.003 + 1e-6, (tot_r, tot_p)
    assert abs(allot["1"]["data"] - dcm * sp["1"]["data"]) < 1e-6 and abs(allot["1"]["compute"] - sp["1"]["compute"]) < 1e-6
    # one certified-unit definition in both audits: the inherited literature is reported apart from the run's units
    for rd in (p, r):
        fa = read_json(rd / "audit" / "audit.json")["field"]
        assert fa["certification"]["definition"] == fa["certification_by_origin"]["definition"]
        assert fa["certification_by_origin"]["inherited"]["n_units"] > 0 and fa["certification_by_origin"]["inherited"]["n_unknown"] == 0
        assert fa["certification"]["n_certified_units"] == fa["certification_by_origin"]["new_in_run"]["n_units"]
    # the same legacy library (same seed) scores the same in both regimes; the paper world's inherited table also holds the
    # warm-start pre-history's publications (the repository regime has no pre-history), reported as n_from_prehistory
    ip = read_json(p / "audit" / "audit.json")["field"]["certification_by_origin"]["inherited"]
    ir = read_json(r / "audit" / "audit.json")["field"]["certification_by_origin"]["inherited"]
    assert ip["n_from_prehistory"] > 0 and ir["n_from_prehistory"] == 0
    assert (ip["n_units"] - ip["n_from_prehistory"]) == ir["n_units"]
    from synthsci.audit import _is_focal, _stated_truth_check
    lib = {}
    for rd, prefix in ((p, "L"), (r, "LC")):
        rows = [h for h in read_jsonl(rd / "ledger" / "hypotheses.jsonl") if h.get("legacy") and h.get("certified") and _is_focal(h) and str(h["unit_id"]).startswith(prefix)]
        lib[prefix] = sorted(_stated_truth_check(h) for h in rows if h.get("theta") is not None)
    assert lib["L"] and lib["L"] == lib["LC"]
    # every admission event names the binding resource of each deferred request
    adm_events = [e for e in read_jsonl(r / "ledger" / "events.jsonl") if e["event"] == "budget_admission"]
    assert adm_events and all("deferred_by_binding" in e and "remaining_before" in e for e in adm_events)
    assert all(d["binding"] in ("compute", "data", "both") for e in adm_events for d in e["deferred_requests"])
    # verification verdicts carry their power (minimum detectable effect at the draw's N)
    ver = read_jsonl(r / "ledger" / "verifications.jsonl")
    assert ver and all("mde_partial_r" in v and "powered" in v for v in ver)
    assert all(int(q) >= 1 for q in sp) and all(int(q) >= 1 for q in sr)      # the warm-start pre-history is never charged
    # the audit's budget section is rebuilt from the ledger and agrees with the engine's summary
    for rd, tot in ((p, tot_p), (r, tot_r)):
        a = read_json(rd / "audit" / "audit.json")["budget"]
        assert abs(a["total"]["data"] - tot["data"]) < 1e-6 and abs(a["total"]["compute"] - tot["compute"]) < 1e-6 and not a["summary_stale"]
    claims = read_jsonl(r / "ledger" / "claims.jsonl")
    assert any(c["event"] == "filed" for c in claims)
    statuses = {c["status"] for c in claims}
    assert statuses <= {"proposed", "contested", "endorsed", "superseded", "retracted", "lapsed", "withdrawn", "endorsed_legacy"}
    # internal consistency: every endorsed run claim under variant b has >= k reproductions
    snap = read_json(r / "ledger" / "registry_snapshot.json")
    for c in snap:
        if c["status"] == "endorsed" and not c["legacy"]:
            reps = [v for v in c["verifications"] if v["result"] == "reproduced"]
            assert len({v["lab"] for v in reps}) >= 2, c["claim_id"]
    # same world and population as the paired paper run
    assert read_json(p / "landscape" / "truth.json") == read_json(r / "landscape" / "truth.json")
    assert read_json(p / "population" / "scientists.json") == read_json(r / "population" / "scientists.json")
    assert read_json(p / "landscape" / "deid_map.json") == read_json(r / "landscape" / "deid_map.json")


def test_fast_mode_byte_identical(tmp_path):
    a = run("smoke-P", 5, "fast", out_root=tmp_path / "a", quiet=True)
    b = run("smoke-P", 5, "fast", out_root=tmp_path / "b", quiet=True)
    inc = ("ledger/", "audit/", "landscape/", "population/", "handbook/", "budget/", "state/")
    assert hash_tree(a, include=inc) == hash_tree(b, include=inc)
    c = run("smoke-P", 6, "fast", out_root=tmp_path / "a", quiet=True)
    assert hash_tree(c, include=("ledger/",)) != hash_tree(a, include=("ledger/",))


def test_fast_repository_runs(tmp_path):
    run("smoke-P", 2, "fast", out_root=tmp_path, quiet=True)
    r = run("smoke-R", 2, "fast", out_root=tmp_path, quiet=True)
    a = read_json(r / "audit" / "audit.json")
    assert a["regime"] == "repository" and "failure_indices" in a


def test_long_horizon_fast(tmp_path):
    rd = run("S0", 11, "fast", overrides={"population": 20, "horizon_quarters": 24, "landscape.legacy.size": 30, "landscape.fringe_pool": 40,
                                          "handbook.snapshots": [8, 16, 24]}, out_root=tmp_path, quiet=True)
    ev = read_jsonl(rd / "ledger" / "events.jsonl")
    assert any(e["event"] == "career" for e in ev)
    assert (rd / "handbook" / "Q24" / "score.json").exists()


def test_pressure_world_shares_random_numbers_with_baseline(tmp_path):
    """S0P-150 differs from S0-150 only in what agents see. With `crn_with`, the rule agents (who do not read the
    pressure lines) produce the same warm start and the same run, so the paired worlds enter quarter 1 identical."""
    from synthsci.engine import crn_namespace
    # the live Stage 1 configs: the pressure world names its tasks exactly as its baseline does
    assert crn_namespace(C.resolve("S0P-150"), 1, "llm") == C.run_id(C.resolve("S0-150"), 1, "llm")
    ov = {"datasets": {"keep_csv": False}, "horizon_quarters": 2}
    a = run(C.deep_merge(C.resolve("S0-150"), ov), 4, "fast", out_root=tmp_path / "a", quiet=True)
    b = run(C.deep_merge(C.resolve("S0P-150"), ov), 4, "fast", out_root=tmp_path / "b", quiet=True)
    for f in ("hypotheses", "decisions", "reviews", "events"):
        strip = lambda r: {k: v for k, v in r.items() if k != "run_id"}
        assert [strip(r) for r in read_jsonl(a / "ledger" / f"{f}.jsonl")] == [strip(r) for r in read_jsonl(b / "ledger" / f"{f}.jsonl")], f
    pop = lambda d: sorted((s["id"], s["stage"], s.get("active", True)) for s in read_json(d / "state" / "world.json")["scientists"])
    assert pop(a) == pop(b)


def test_resume_keeps_evolved_population(tmp_path):
    """With careers moving (turnover) and a warm start, a run resumed after an annual step keeps its entrants,
    exits and promotions, and reaches the same ledger and population as an uninterrupted run."""
    from synthsci.engine import World, run_dir_for
    from synthsci.regimes.paper import PaperRegime
    over = {"datasets": {"keep_csv": False}, "horizon_quarters": 6}
    cfg = C.deep_merge(C.resolve("S0-150"), over)   # 150 authors, careers on, 16-quarter warm start (Stage 1 world)
    rd = run_dir_for(cfg, 11, "fast", tmp_path / "x")
    w = World(cfg, 11, "fast", "mock", rd, None)
    reg = PaperRegime(w)
    reg.setup(); reg.mark_phase(0, "setup")
    stop = 5   # after the quarter-4 annual step (careers move, fringe refilled), so both restore paths matter
    for q in range(int(reg.state["quarter"]), stop):
        reg.state["quarter"] = q
        for ph in reg.phases_for(q):
            reg.prehistory = q <= 0 and ph != "handbook0"
            w.budget.suspended = reg.prehistory
            getattr(reg, ph)(q)
            reg.mark_phase(q, ph)
        if q >= 1:
            w.budget.close_quarter(q)
    reg.state["quarter"] = stop
    reg.save_state(); w.ledger.flush(); w.budget.write()
    seeded = {s["id"] for s in read_json(rd / "population" / "scientists.json")}
    evolved = {s["id"] for s in w.scientists}
    assert evolved - seeded, "the warm start should have brought in entrants"
    resumed = run(cfg, 11, "fast", out_root=tmp_path / "x", quiet=True)
    full = run(cfg, 11, "fast", out_root=tmp_path / "y", quiet=True)
    key = lambda h: (h["unit_id"], h["status"], h.get("certified"), round(h.get("estimate") or 0.0, 6))
    hr = sorted(set(map(key, read_jsonl(resumed / "ledger" / "hypotheses.jsonl"))))
    hf = sorted(set(map(key, read_jsonl(full / "ledger" / "hypotheses.jsonl"))))
    assert hr == hf
    pop = lambda d: sorted((s["id"], s["stage"], s.get("active", True)) for s in read_json(d / "state" / "world.json")["scientists"])
    assert pop(resumed) == pop(full)
    assert read_json(resumed / "audit" / "audit.json")["field"]["net_knowledge"] == read_json(full / "audit" / "audit.json")["field"]["net_knowledge"]


def test_resume_mid_run_matches(tmp_path):
    """A run resumed from a saved state (after quarter 2) reaches the same ledger as an uninterrupted fast run."""
    from synthsci.engine import World, run_dir_for
    from synthsci.regimes.paper import PaperRegime
    cfg = C.resolve("smoke-P")
    cfg = C.deep_merge(cfg, {"datasets": {"keep_csv": False}})
    rd = run_dir_for(cfg, 9, "fast", tmp_path / "x")
    w = World(cfg, 9, "fast", "mock", rd, None)
    reg = PaperRegime(w)
    reg.setup(); reg.mark_phase(0, "setup")
    for q in range(int(reg.state["quarter"]), 3):   # the warm-start pre-history, quarter 0 (handbook) and quarters 1-2
        reg.state["quarter"] = q
        for ph in reg.phases_for(q):
            reg.prehistory = q <= 0 and ph != "handbook0"
            w.budget.suspended = reg.prehistory
            getattr(reg, ph)(q)
            reg.mark_phase(q, ph)
        if q >= 1:
            w.budget.close_quarter(q)
    reg.state["quarter"] = 3
    reg.save_state(); w.ledger.flush(); w.budget.write()
    resumed = run(cfg, 9, "fast", out_root=tmp_path / "x", quiet=True)
    full = run(cfg, 9, "fast", out_root=tmp_path / "y", quiet=True)
    hr = [h for h in read_jsonl(resumed / "ledger" / "hypotheses.jsonl")]
    hf = [h for h in read_jsonl(full / "ledger" / "hypotheses.jsonl")]
    key = lambda h: (h["unit_id"], h["status"], h.get("certified"), round(h.get("estimate") or 0.0, 6))
    assert sorted(set(map(key, hr))) == sorted(set(map(key, hf)))
    assert read_json(resumed / "audit" / "audit.json")["field"]["net_knowledge"] == read_json(full / "audit" / "audit.json")["field"]["net_knowledge"]
