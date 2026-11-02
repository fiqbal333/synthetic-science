"""Truth scorer, belief rules, and the paired budget's admission rule."""
import numpy as np

from synthsci.belief import all_rules, b1_vote, b2_pooled, b3_most_attended, is_correct, truth_class
from synthsci.scorer import truth_check, causal_language, claims_text_mismatch


def test_primary_rule():
    assert truth_check("supported", 0.03, 0.0) == "false_discovery"
    assert truth_check("supported", -0.03, 0.05) == "false_discovery"
    assert truth_check("supported", 0.03, 0.05) == "true_discovery"
    assert truth_check("not_supported", 0.01, 0.05) == "missed_effect"
    assert truth_check("not_supported", 0.01, 0.0) == "correct_null"
    assert truth_check("partially", 0.03, 0.01) == "true_discovery_negligible"


def test_truth_class_and_correctness():
    assert truth_class(0.0) == "null" and truth_class(0.01) == "negligible" and truth_class(-0.05) == "negative"
    assert is_correct("null", "negligible") and is_correct("positive", "negligible")
    assert is_correct("contested", "null") is None
    assert is_correct("positive", "negative") is False


def _unit(est, n, verdict="supported", att=0, q=1, uid=None, dep=None):
    return {"estimate": est, "se": 1 / np.sqrt(n), "n": n, "verdict": verdict, "attention": att, "quarter": q, "unit_id": uid or f"u{est}{n}", "deposit_id": dep}


def test_b1_vote_hand_computation():
    units = [_unit(0.05, 500), _unit(0.04, 800), _unit(-0.01, 300, "not_supported")]
    out = b1_vote(units)
    assert out["belief"] == "positive" and abs(out["confidence"] - 2 / 3) < 1e-9
    assert b1_vote([_unit(0.05, 500), _unit(-0.05, 500)])["belief"] == "contested"
    assert b1_vote([])["belief"] == "uncovered"


def test_b2_pooled_hand_computation():
    units = [_unit(0.10, 2000), _unit(0.08, 3000)]
    out = b2_pooled(units)
    z = np.array([np.arctanh(0.10), np.arctanh(0.08)])
    w = np.array([1997, 2997])
    zbar = (w * z).sum() / w.sum()
    assert abs(out["estimate"] - np.tanh(zbar)) < 0.005
    assert out["belief"] == "positive"
    nulls = [_unit(0.001, 40000, "not_supported"), _unit(-0.002, 40000, "not_supported")]
    assert b2_pooled(nulls)["belief"] == "null"
    assert b2_pooled([_unit(0.05, 200), _unit(-0.05, 200)])["belief"] == "contested"


def test_b2_one_per_deposit():
    units = [_unit(0.10, 2000, dep="D1", uid="a"), _unit(0.11, 2000, dep="D1", uid="b", q=2), _unit(0.02, 2000, dep="D2", uid="c")]
    assert b2_pooled(units, one_per_deposit=True)["n_units"] == 2
    assert b2_pooled(units, one_per_deposit=False)["n_units"] == 3


def test_b3_and_all_rules():
    units = [_unit(0.05, 500, att=4, uid="x"), _unit(-0.05, 500, att=1, uid="y")]
    assert b3_most_attended(units)["unit_id"] == "x" and abs(b3_most_attended(units)["confidence"] - 0.8) < 1e-9
    r = all_rules(units)
    assert set(r) == {"B1", "B2", "B3", "B5"}


def test_keyword_rules():
    assert causal_language("## Discussion\n\nX causes Y. Higher X increases Y. X drives Y.\n") is True
    assert causal_language("## Discussion\n\nX may be associated with Y; the design cannot support causal claims.\n") is False
    claims = {"hypotheses": [{"estimate": 0.0481}]}
    assert claims_text_mismatch(claims, "the estimate is 0.048 (SE 0.01)") is False
    assert claims_text_mismatch(claims, "the estimate is 0.12") is True


def test_budget_pairing_and_admission(tmp_path, cfg):
    from synthsci.budget import Budget
    from synthsci.ledger import Ledger
    from synthsci.util import write_json
    import copy
    paired = tmp_path / "paired"
    write_json(paired / "budget" / "spend_by_quarter.json", {"1": {"data": 1000, "compute": 250000}, "2": {"data": 500, "compute": 100000}})
    rcfg = copy.deepcopy(cfg)
    rcfg["regime"] = "repository"
    led = Ledger(tmp_path / "run", "r", "mock", "repository")
    b = Budget(rcfg, tmp_path / "run", led, paired)
    # pairing rule: compute is balanced one to one, data floats under data_ceiling_multiple (2.0) x the paired data
    assert b.rule == "pairing" and b.allotment(1) == {"data": 2000.0, "compute": 250000.0}
    reqs = [{"id": "a1", "class": "analyses", "lab": "L1", "compute": 80000, "data": 600},
            {"id": "a2", "class": "analyses", "lab": "L2", "compute": 80000, "data": 600},
            {"id": "v1", "class": "verifications", "lab": "L3", "compute": 37500, "data": 300},
            {"id": "a3", "class": "analyses", "lab": "L3", "compute": 80000, "data": 400}]
    adm, dfr = b.admit(1, reqs)
    # priority class first; compute (250k) admits v1 + a1 + a2 (197.5k), a3's compute (80k) exceeds the 52.5k left (its data 400 fits the 500 left)
    assert [r["id"] for r in adm] == ["v1", "a1", "a2"] and [r["id"] for r in dfr] == ["a3"]
    assert dfr[0]["binding"] == "compute" and dfr[0]["short_by"]["compute"] == 27500 and dfr[0]["short_by"]["data"] == 0
    assert b.last_admission["deferred_by_binding"] == {"compute": 1, "data": 0, "both": 0}
    assert b.last_admission["remaining_before"] == {"data": 2000.0, "compute": 250000.0}
    b.debit(1, "data", 1500, "dataset", "x", "L1")
    b.debit(1, "compute", 237500, "analyst_t0", "y", "L1")
    b.close_quarter(1)
    assert b.allotment(2)["data"] == 2 * 500 + 500 and b.allotment(2)["compute"] == 100000 + 12500   # carry-over of the unspent allotment
    # a request larger than the data ceiling is deferred with data (or both) as the binding resource, never silently
    adm2, dfr2 = b.admit(2, [{"id": "big", "class": "analyses", "lab": "L1", "compute": 10000, "data": 5000}])
    assert [r["id"] for r in adm2] == ["a3"] and [r["id"] for r in dfr2] == ["big"] and dfr2[0]["binding"] == "data"   # the deferred a3 is served first
    rows = [r for r in led.read("budget") if r.get("resource") == "request" and r.get("admitted") == "deferred"]
    assert {r["binding"] for r in rows} == {"compute", "data"}
    pcfg = copy.deepcopy(cfg)
    bp = Budget(pcfg, tmp_path / "run2", Ledger(tmp_path / "run2", "p", "mock", "paper"), None)
    assert bp.rule == "uncapped"
    # the multiple is a knob
    rcfg2 = copy.deepcopy(rcfg)
    rcfg2["budget"]["data_ceiling_multiple"] = 1.0
    b1 = Budget(rcfg2, tmp_path / "run3", Ledger(tmp_path / "run3", "r", "mock", "repository"), paired)
    assert b1.allotment(1)["data"] == 1000.0
