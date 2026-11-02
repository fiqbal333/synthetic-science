"""The compute-only pairing rule with a data ceiling, one certified-unit
definition across regimes with the inherited literature reported apart, lab-requested dataset sizes and verification draws at
the claim's N with an MDE per verdict, the claim-sign regression test, the handbook `agree` field, the dispatch stub's
environment line as config, the scorer's clustering check, and the calibrated parameter file."""
import json
from pathlib import Path

import pytest

from synthsci import config as C
from synthsci.engine import World, run
from synthsci.util import read_json, read_jsonl, read_text, write_json, write_text


# ====================================================================== claim sign (open item from the smoke batch)
def test_claim_sign_is_the_direction_of_a_significant_estimate():
    from synthsci.scorer import claim_sign
    assert claim_sign({"p": 0.01, "estimate": -0.05, "predicted_sign": "positive", "verdict": "not_supported"}) == "negative"
    assert claim_sign({"p": 0.01, "estimate": 0.05}) == "positive"
    assert claim_sign({"p": 0.20, "estimate": 0.05}) == "none"
    assert claim_sign({"p": None, "estimate": 0.05}) == "none" and claim_sign({"p": 0.01, "estimate": 0.0}) == "none"
    assert claim_sign({"p": float("nan"), "estimate": 0.05}) == "none"


def test_legacy_claims_and_drafts_take_the_sign_from_the_estimate(tmp_path):
    """Repository legacy claims: a significant wrong-sign legacy result is a claim of that sign, not 'no reliable relation';
    the mock analyst's draft entries follow the same rule (both import scorer.claim_sign)."""
    from synthsci.regimes.repository import RepositoryRegime
    from synthsci.backends import mock as mock_mod
    from synthsci.regimes import repository as repo_mod
    from synthsci.scorer import claim_sign
    assert mock_mod._claim_sign is claim_sign and repo_mod._claim_sign is claim_sign
    cfg = C.resolve("smoke-R", {"budget.rule": "fixed"})
    w = World(cfg, 1, "fast", "mock", tmp_path / "r", None)
    reg = RepositoryRegime(w)
    checked, against = 0, 0
    for c in reg.state["claims"].values():
        if not c.get("legacy"):
            continue
        ev = c["evidence"][0]
        if ev.get("p") is not None and ev["p"] < 0.05:
            assert c["sign"] == ("positive" if ev["estimate"] > 0 else "negative"), c
            checked += 1
            if c["kind"] == "focal" and c["sign"] != w.landscape.question(c["question_id"])["prediction_sign"]:
                against += 1   # significant against the theory's prediction: a claim of that sign, never 'no reliable relation'
        else:
            assert c["sign"] == "none"
    assert checked > 0 and against > 0


# ====================================================================== pairing budget rule (compute balanced, data under a ceiling) and reporting
def test_pairing_balances_compute_and_caps_data_at_the_multiple(tmp_path):
    from synthsci.budget import Budget
    from synthsci.ledger import Ledger
    paired = tmp_path / "p"
    write_json(paired / "budget" / "spend_by_quarter.json", {"1": {"data": 1000, "compute": 200000}})
    cfg = C.resolve("R-b")
    b = Budget(cfg, tmp_path / "r", Ledger(tmp_path / "r", "r", "fast", "repository"), paired)
    assert b.allotment(1) == {"data": 2000.0, "compute": 200000.0}
    adm, dfr = b.admit(1, [{"id": "v", "class": "verifications", "lab": "L1", "compute": 150000, "data": 100},
                           {"id": "a", "class": "analyses", "lab": "L2", "compute": 100000, "data": 100},
                           {"id": "d", "class": "analyses", "lab": "L3", "compute": 1000, "data": 5000}])
    assert [r["id"] for r in adm] == ["v"] and {r["id"]: r["binding"] for r in dfr} == {"a": "compute", "d": "data"}
    ev = b.last_admission
    assert ev["deferred_by_binding"] == {"compute": 1, "data": 1, "both": 0} and ev["remaining_after_admission"] == {"data": 1900.0, "compute": 50000.0}


def test_cost_estimate_states_the_rule():
    from synthsci.cost import estimate, table
    est = estimate(C.resolve("R-b-150"))
    assert est["repository_regime_planning"]["data_ceiling_multiple"] == 4.0   # the reported repository world's data ceiling
    assert "compute capped at P's realized compute" in table(est) and "4 x P's data" in table(est)


# ====================================================================== audit comparability across regimes
def _row(uid, q, est, p, theta, verdict, legacy=False, certified=True, quarter=1, kind=None, hyp_id="H1", moderator=None):
    return {"unit_id": uid, "question_id": q, "estimate": est, "p": p, "theta": theta, "verdict": verdict, "legacy": legacy, "certified": certified,
            "certified_quarter": quarter if certified else None, "quarter": quarter, "kind": kind, "hyp_id": hyp_id, "moderator": moderator,
            "truth_check": None, "paper_id": uid.split(":")[0], "iv": "x", "dv": "y"}


def test_audit_scores_what_a_unit_states_and_reports_inherited_apart():
    from synthsci.audit import _is_focal, _stated_truth_check, certification_by_origin, self_correction, _dedupe
    # a significant wrong-sign legacy result: the author's verdict said 'not supported', the stated direction is negative
    leg = _row("L0001:H1", "Q1", -0.05, 0.001, 0.04, "not_supported", legacy=True, quarter=0, kind="focal")
    leg["truth_check_stated"] = _stated_truth_check(leg)
    assert leg["truth_check_stated"] == "false_discovery"
    # a run paper later states a positive relation on the same question: the inherited false unit counts as corrected
    new = _row("P0001:H1", "Q1", 0.05, 0.01, 0.04, "supported", quarter=3)
    new["truth_check_stated"] = _stated_truth_check(new)
    mod = _row("P0001:H2", "Q1", 0.02, 0.5, 0.0, "not_supported", quarter=3, hyp_id="H2", moderator="size")
    mod["truth_check_stated"] = _stated_truth_check(mod)
    assert _is_focal(leg) and _is_focal(new) and not _is_focal(mod)
    rows = [leg, new, mod]
    o = certification_by_origin(rows)
    assert o["inherited"] == {"n_units": 1, "n_true": 0, "n_false": 1, "n_missed_effect": 0, "n_unknown": 0, "n_from_prehistory": 0, "true_share": 0.0,
                              "false_share": 1.0, "false_units_corrected_in_run": 1, "share_false_corrected_in_run": 1.0, "n_rows_all_kinds": 1}
    assert o["new_in_run"]["n_units"] == 1 and o["new_in_run"]["n_true"] == 1 and o["new_in_run"]["n_rows_all_kinds"] == 2
    assert self_correction(rows)["n_false_certified_units"] == 0            # the inherited false unit is not the run's own
    # duplicated legacy rows (pre-decision-38 ledgers) collapse to one per unit
    assert len(_dedupe([leg, dict(leg), new])) == 2


def test_audit_backfills_legacy_truth_for_old_runs(tmp_path):
    from synthsci.audit import _fill_legacy_truth
    write_json(tmp_path / "legacy" / "papers.json", [{"paper_id": "L0001", "question_id": "Q1", "hidden": {"theta": 0.0}}])
    q = {"id": "Q1", "iv_var": "x", "dv_var": "y", "secondary_dv_var": None}
    rows = [{"unit_id": "L0001:H1", "paper_id": "L0001", "question_id": "Q1", "legacy": True, "iv": "x", "dv": "y", "moderator": None,
             "estimate": 0.05, "p": 0.01, "verdict": "supported", "truth_check": None}]
    out = _fill_legacy_truth(rows, tmp_path, [q])
    assert out[0]["theta"] == 0.0 and out[0]["truth_check"] == "false_discovery"


def test_paper_regime_scores_the_starting_literature_at_setup(tmp_path):
    from synthsci.regimes.paper import PaperRegime, legacy_theta
    cfg = C.resolve("smoke-P")
    w = World(cfg, 1, "fast", "mock", tmp_path / "p", None)
    reg = PaperRegime(w)
    rows = [h for h in w.ledger.read("hypotheses") if h.get("legacy")]
    assert rows and all(h["truth_check"] not in (None, "unknown") and h["theta"] is not None for h in rows)
    assert len({h["unit_id"] for h in rows}) == len(rows)
    lp = w.legacy[0]
    q = w.landscape.question(lp["question_id"])
    assert legacy_theta(lp, lp["hypotheses"][0], q) == lp["hidden"]["theta"]
    # the roll-up units state the direction of a significant estimate (the repository regime's rule), the author's verdict beside it
    for qid in {lp["question_id"] for lp in w.legacy}:
        for u in reg.certified_units(qid):
            assert "author_verdict" in u and u["verdict"] in ("supported", "not_supported")


# ====================================================================== lab-requested sizes, verification N and MDE
def test_lab_requested_dataset_size_is_honoured_within_the_budget(tmp_path):
    from synthsci.regimes.repository import RepositoryRegime
    cfg = C.resolve("smoke-R", {"budget.rule": "fixed", "budget.fixed.data_per_quarter": 3000, "budget.fixed.compute_per_quarter": 4000000})
    w = World(cfg, 3, "fast", "mock", tmp_path / "r", None)
    reg = RepositoryRegime(w)
    reg.repo_update(1)
    lid, lab = sorted(reg.state["labs"].items())[0]
    ctx = reg._lab_ctx(lab, 1)
    assert ctx["members_available"], "the first lab has no free member at quarter 1"
    m = ctx["members_available"][0][0]
    q1, q2 = ctx["menu"][0]["question_id"], ctx["menu"][1]["question_id"]
    dec = {"nonce": "fast", "agenda": [{"question_id": q1, "data": "new", "lead": m, "brief": "b", "n": 500}], "filings": [], "votes": [], "verifications": [], "responses": []}
    if len(ctx["members_available"]) > 1:
        dec["agenda"].append({"question_id": q2, "data": "new", "lead": ctx["members_available"][1][0], "brief": "b", "n": 999999})
    reg._ingest_lab_decision(lab, ctx, dec, 1)
    an = [a for a in reg.state["analyses"].values()]
    a1 = next(a for a in an if a["question_id"] == q1)
    assert a1["n_requested"] == 500 and a1["n_override"] == 500 and a1["n_est"] == 500
    ev = [e for e in w.ledger.read("events") if e["event"] == "dataset_size_request"]
    assert ev and ev[0]["n_granted"] == 500 and ev[0]["reason"] == "as requested"
    if len(an) > 1:
        a2 = next(a for a in an if a["question_id"] == q2)
        assert a2["n_requested"] == 999999 and a2["n_override"] <= 3000   # clamped to the data allotment, never above n_cap
    reg.budget_admit(1)
    reg.draws(1)
    dep = reg._deposit(a1["deposits"][0])
    assert dep["n_override"] == 500 and 400 <= dep["n"] <= 500   # the frame is the requested size (panels round to whole firms)
    # the registry view and the lab card state the cost of a new dataset
    view = reg.registry_view(q1, 1, lid)
    assert "Typical sample sizes" in view and "costs its number of observations" in view
    from synthsci.regimes import texts
    card = texts.lab_card(lab, reg.lab_members(lid), lab["record"] | {"points": 0}, {"slots": 1, "available_members": [m], "verifications": 2, "data_budget": 3000})
    assert "a new dataset costs its number of observations" in card
    prompt = read_text(Path(__file__).resolve().parents[1] / "synthsci" / "prompts" / "lab_decision.md")
    assert '"n": null | <integer>' in prompt and "costs its own number of observations" in prompt
    schema = read_json(Path(__file__).resolve().parents[1] / "synthsci" / "schemas" / "lab_decision.schema.json")
    assert "n" in schema["properties"]["agenda"]["items"]["properties"]


def test_verification_power_and_claim_sized_fresh_draws():
    from synthsci.scorer import verification_power
    p = verification_power(500, 4, 0.10, "positive", 250)
    assert 0.12 < p["mde_partial_r"] < 0.13 and p["powered"] is False and p["n_ratio"] == 2.0 and "claimed" in p["powered_for"]
    assert verification_power(5000, 4, 0.10, "positive", 250)["powered"] is True
    assert verification_power(5000, 4, None, "none", 250)["powered"] is True and verification_power(200, 4, None, "none", 250)["powered"] is False
    assert verification_power(None, 4, 0.1, "positive", 250)["mde_partial_r"] is None


@pytest.fixture(scope="module")
def fast_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair")
    p = run("smoke-P", 4, "fast", out_root=root, quiet=True)
    r = run("smoke-R", 4, "fast", out_root=root, quiet=True)
    return p, r


def test_verification_ledger_carries_mde_and_the_audit_reports_power(fast_pair):
    p, r = fast_pair
    ver = read_jsonl(r / "ledger" / "verifications.jsonl")
    assert ver and all(v.get("mde_partial_r") is not None and v.get("claim_n") is not None and "n_basis" not in v for v in ver)
    a = read_json(r / "audit" / "audit.json")
    vp = a["verification_power"]
    assert vp["n_with_mde"] == len(ver) and vp["median_mde_partial_r"] > 0 and vp["powered_share"] is not None
    # admission events name the binding resource and the request rows carry the N basis of a fresh verification draw
    reqs = [b for b in read_jsonl(r / "ledger" / "budget.jsonl") if b.get("resource") == "request"]
    assert reqs and all(("binding" in b) == (b["admitted"] == "deferred") for b in reqs)
    st = read_json(r / "state" / "state.json")
    bases = {x.get("n_basis") for x in st["budget"].get("deferred", [])} if "budget" in st else set()
    ev = [e for e in read_jsonl(r / "ledger" / "events.jsonl") if e["event"] == "budget_admission"]
    assert ev and all(set(e["deferred_by_binding"]) == {"compute", "data", "both"} for e in ev)
    # the per-unit resource reporting (data and compute per certified unit) is in both audits
    for rd in (p, r):
        pu = read_json(rd / "audit" / "audit.json")["budget"]["per_unit"]
        assert "data_per_correct_handbook_answer" in pu and pu["data_used"] >= 0
    assert read_json(r / "audit" / "audit.json")["budget"]["per_unit"]["paired_world"]["compute_ratio"] <= 1.05


def test_verification_batches_are_one_lab_and_distinct_claims():
    from synthsci.regimes.repository import verification_batches
    reqs = [{"id": "a", "lab": "L1", "claim_id": "C1"}, {"id": "b", "lab": "L2", "claim_id": "C1"}, {"id": "c", "lab": "L1", "claim_id": "C1"},
            {"id": "d", "lab": "L1", "claim_id": "C2"}, {"id": "e", "lab": "L1", "claim_id": "C3"}]
    batches = verification_batches(reqs)
    assert sum(len(b) for b in batches) == len(reqs) and all(1 <= len(b) <= 2 for b in batches)
    for b in batches:
        assert len({r["lab"] for r in b}) == 1 and len({r["claim_id"] for r in b}) == len(b)
    assert [r["id"] for b in batches for r in b] == ["a", "d", "c", "e", "b"]


# ====================================================================== handbook `agree`
def test_rule_handbook_agree_means_prediction_agreement():
    from synthsci.agents.rules import RuleHandbook, load_params
    import numpy as np
    hb = RuleHandbook(load_params(C.resolve("S0")))
    rules = {"Q1": {"B1": {"belief": "positive", "confidence": 0.9}, "B2": {"belief": "positive", "confidence": 0.9}, "B3": {"belief": "positive", "confidence": 0.9},
                    "B5": {"belief": "positive", "confidence": 0.9}, "n_units": 3},
             "Q2": {"B1": {"belief": "null", "confidence": 0.9}, "B2": {"belief": "null", "confidence": 0.9}, "B3": {"belief": "null", "confidence": 0.9},
                    "B5": {"belief": "null", "confidence": 0.9}, "n_units": 3}}
    out = hb.write(rules, {"A": ["Q1", "Q2"]}, np.random.default_rng(0), nonce="x", predictions={"Q1": "positive", "Q2": "positive"})
    by = {q["question_id"]: q for q in out["questions"]}
    assert by["Q1"]["agree"] == (by["Q1"]["answer"] == "positive") and by["Q2"]["agree"] == (by["Q2"]["answer"] == "positive")
    assert by["Q2"]["agree"] is False
    prompt = read_text(Path(__file__).resolve().parents[1] / "synthsci" / "prompts" / "handbook_writer.md")
    assert "the direction the theory predicts" in prompt


def test_handbook_score_reports_prediction_agreement(fast_pair):
    p, _ = fast_pair
    s = read_json(p / "handbook" / "Q4" / "score.json")
    assert "agree_with_prediction_share" in s and "agree_field_consistent_share" in s and "correct_answers" in s
    assert s["agree_field_consistent_share"] in (None, 1.0)   # the rule writer's agree field is exactly prediction agreement


# ====================================================================== dispatch stub environment line
def test_environment_note_is_config(tmp_path):
    from synthsci.backends.subagent import SubagentBackend
    from synthsci.orchestration.tasks import Task, TaskStore
    base = C.resolve("S0")
    assert "Bash tool does not work" in base["backend"]["environment_note"]
    stub = read_text(Path(__file__).resolve().parents[1] / "synthsci" / "prompts" / "dispatch_stub.md")
    assert "{{ENVIRONMENT_NOTE}}" in stub and "Bash" not in stub
    for note, expect, absent in (("Use the X tool only.", "Use the X tool only.", "Bash"), ("", "Do not read or write anywhere else.", "Bash"), (None, "Bash tool does not work", "{{")):
        cfg = C.resolve("S0", {"backend.environment_note": note}) if note is not None else base
        store = TaskStore(tmp_path / f"run{abs(hash(note))}")
        be = SubagentBackend(cfg, store.run_dir, store)
        t = Task(type="desk", quarter=1, identity="d", tier="sonnet", run_id="r", call_type="desk_screen", template="desk_screen.md",
                 variables={"SUBMISSIONS": "none", "STRICTNESS_SENTENCE": "s", "COUNT_CLAUSE": ""}, outputs=["desk.json"], meta={})
        store.materialize(t)
        be._write_dispatch([t], {})
        b = read_json(store.run_dir / "dispatch" / "batch_1.json")[0]
        assert expect in b["stub"] and absent not in b["stub"] and "{{" not in b["stub"]


# ====================================================================== scorer clustering check
def test_se_type_naming_clustering_is_not_flagged():
    from synthsci.scorer import analyst_flags
    q = {"id": "Q", "iv_var": "x", "dv_var": "y", "panel": True, "moderator_vars": [], "controls_pool": ["ln_assets"]}
    hidden = {"traps": [], "oracle": {"correct": {"estimate": 0.1, "se": 0.02, "n": 100}}, "theta": 0.1}
    meta = {"n_correct": 100, "variables": {"controls": ["ln_assets"]}}
    for se_type, flagged in (("cluster-robust by firm", False), ("firm-clustered", False), ("cluster", False), ("HC1", True), ("robust", True)):
        claims = {"hypotheses": [{"hyp_id": "H1", "iv": "x", "dv": "y", "moderator": None, "estimate": 0.1, "se": 0.02, "p": 0.001, "n": 100, "verdict": "supported",
                                  "se_type": se_type, "controls": ["ln_assets"], "estimator": "fixed effects"}], "sample_changes": [], "n_models_run": 1,
                  "robustness_models": [], "executed_code": "df.describe()", "commands_run": 1}
        fl = analyst_flags(claims, hidden, meta, q, "## Discussion\n\nmay be associated\n", "VERIFIED", "T0", True)
        assert ("se_not_clustered" in fl["flags"]) == flagged, (se_type, fl["flags"])


# ====================================================================== calibrated parameters
def test_params_and_tariffs_are_calibrated_with_provenance():
    from synthsci.agents.rules import load_params
    from synthsci.config import CONFIG_DIR, load_yaml
    p = load_params(C.resolve("S0"))
    assert p["status"].startswith("calibrated")
    assert p["lab_head"]["vote"]["stated_probability"]["endorse"] == 60 and p["reviewer"]["recommendation_cuts"]["minor"] == 80
    assert p["author"]["p_handle"]["VERIFIED"]["omitted_standard_control"] == 0.90 and p["author"]["n_models_run"]["VERIFIED"] == [4, 14]
    t = load_yaml(CONFIG_DIR / "tariffs.yaml")
    assert t["status"].startswith("measured") and all("source" in v for v in t["calls"].values())
    assert t["batch_tariff_multiplier"] == {1: 1.0, 2: 1.2, 3: 1.45}
