"""Rule-agent behaviours added for the fast-mode recalibration against the live Stage 1 runs (configs/fastmode/params-stage1.yaml).

Each new behaviour is reachable only through a new parameter; absent (as in configs/fastmode/params.yaml) the rules behave
and draw random numbers exactly as before, so earlier fast-mode runs stay reproducible."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from synthsci.agents.rules import RuleAuthor, RuleEditor, RuleLabHead, RuleReviewer, UNPROMPTED_KINDS
from synthsci.config import CONFIG_DIR, load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def old():
    return load_yaml(CONFIG_DIR / "fastmode" / "params.yaml")


def _item(pid="P0001", se=0.01, verdict="not_supported", flags=()):
    return {"paper_id": pid, "hypotheses": [{"iv": "x", "dv": "y", "n": 4000, "se": se, "estimate": 0.02, "verdict": verdict}],
            "visible_problems": list(flags), "relation_to_prior": "first study", "n_prior": 0}


# ------------------------------------------------------------------ editor: round-1 accept threshold
def test_editor_round1_accept_threshold(old):
    p = copy.deepcopy(old)
    p["editor"]["intercept_round1"] = 10.0          # every paper's index far above the old accept threshold (band / 2)
    reviews = [{"recommendation": "minor_revision", "validity_probability": 80.0, "issues": []}] * 3
    old_dec = [RuleEditor(p).decide({"paper_id": "P1"}, reviews, 1, 0, np.random.default_rng(i))["decision"] for i in range(50)]
    assert set(old_dec) == {"accept"}               # the old rule accepts in round 1
    p["editor"]["accept_cut_round1"] = 99.0
    new_dec = [RuleEditor(p).decide({"paper_id": "P1"}, reviews, 1, 0, np.random.default_rng(i))["decision"] for i in range(50)]
    assert set(new_dec) == {"revise"}               # with the separate threshold the live editors' behaviour is reachable
    # the round-2 rule is untouched
    assert RuleEditor(p).decide({"paper_id": "P1"}, reviews, 2, 0, np.random.default_rng(0))["decision"] == "accept"


# ------------------------------------------------------------------ reviewer: shared paper shift, unprompted objections
def test_reviewer_paper_shift_is_shared_by_the_papers_referees(old):
    p = copy.deepcopy(old)
    p["reviewer"]["validity_sd"] = 1.0
    p["reviewer"]["validity_paper_sd"] = 20.0
    rv = RuleReviewer(p)
    rng = np.random.default_rng(3)
    within, means = [], []
    for j in range(40):
        it = _item(pid=f"P{j:04d}", se=0.01 + j / 1000)
        v = [rv.review(it, rng)["validity_probability"] for _ in range(3)]
        within.append(np.std(v)); means.append(np.mean(v))
    assert np.mean(within) < 2.0                    # the referees of one paper agree (shared shift, small own noise)
    assert np.std(means) > 10.0                     # papers differ
    # the shift persists into round 2 (same paper, same focal row) and differs for the same id in another run
    a = rv.review(_item("P0001", 0.0123), np.random.default_rng(1))["validity_probability"]
    b = rv.review(_item("P0001", 0.0123), np.random.default_rng(2))["validity_probability"]
    c = rv.review(_item("P0001", 0.0456), np.random.default_rng(1))["validity_probability"]
    assert abs(a - b) < 6 and abs(a - c) > 0.01


def test_reviewer_unprompted_objection(old):
    p = copy.deepcopy(old)
    rv_old = RuleReviewer(p)
    kinds_old = {i["kind"] for s in range(200) for i in rv_old.review(_item(), np.random.default_rng(s))["issues"]}
    assert not kinds_old & set(UNPROMPTED_KINDS)     # old rule: a clean paper never draws a methods objection
    p["reviewer"]["p_unprompted_issue"] = 1.0
    p["reviewer"]["p_unprompted_major"] = 1.0
    rv = RuleReviewer(p)
    for s in range(50):
        out = rv.review(_item(), np.random.default_rng(s))
        assert any(i["kind"] in UNPROMPTED_KINDS and i["severity"] == "major" for i in out["issues"])


def test_new_reviewer_keys_at_zero_draw_exactly_as_before(old):
    """Absent keys and zero values give the same random stream as the old rule (sweep reproducibility)."""
    p0 = copy.deepcopy(old)
    p1 = copy.deepcopy(old)
    p1["reviewer"].update({"validity_paper_sd": 0.0, "p_unprompted_issue": 0.0, "p_unprompted_major": 0.9})
    p1["editor"]["accept_cut_round1"] = None
    for s in range(30):
        it = _item(flags=["se_not_clustered"] if s % 2 else [])
        a = RuleReviewer(p0).review(it, np.random.default_rng(s))
        b = RuleReviewer(p1).review(it, np.random.default_rng(s))
        assert a == b
        ra, rb = np.random.default_rng(s), np.random.default_rng(s)
        assert RuleEditor(p0).decide({"paper_id": "P"}, [a] * 3, 1, 2, ra) == RuleEditor(p1).decide({"paper_id": "P"}, [b] * 3, 1, 2, rb)


# ------------------------------------------------------------------ lab head: dataset-size requests
def _lab_ctx(old):
    q = {"id": "Q-1", "frame_quantiles": {"p50": 2000.0}}
    return {"lab": {"id": "L01", "primary_theory": "rbv"}, "members_available": [("A1", 0, "VERIFIED", "T0"), ("A2", 0, "VERIFIED", "T0")],
            "verification_capacity": 0, "menu": [{"question_id": "Q-1", "slot": "affinity", "family": "rbv", "count": 0, "panel": True,
                                                    "question": q, "catalogue": []},
                                                   {"question_id": "Q-2", "slot": "random", "family": "rbv", "count": 0, "panel": True,
                                                    "question": {"id": "Q-2", "frame_quantiles": {"p50": 500.0}}, "catalogue": []}],
            "drafts": [], "open_claims": [], "challenges": [], "variant": "b", "pooling": True, "autonomous_filing": False,
            "author_params": old["author"], "nonce": "t"}


def test_lab_head_names_dataset_size(old):
    p = copy.deepcopy(old)
    dec = RuleLabHead(p).decide(_lab_ctx(p), np.random.default_rng(0))
    assert all("n" not in a for a in dec["agenda"])            # old rule: never names a size (the frame draw stands)
    p["lab_head"]["p_request_n"] = 1.0
    p["lab_head"]["request_n_multiple"] = {2.5: 1.0}
    dec = RuleLabHead(p).decide(_lab_ctx(p), np.random.default_rng(0))
    p50 = {"Q-1": 2000.0, "Q-2": 500.0}
    assert dec["agenda"] and all(a["n"] == int(2.5 * p50[a["question_id"]]) for a in dec["agenda"])


# ------------------------------------------------------------------ analyst: verdict from the found direction
class _Gen:
    def analytic_study(self, qid, seed, upper_half, handled, use_fe, n_override=None):
        return {"question_id": qid, "n": 5000, "n_reported": 5000, "df": 4990, "traps": [],
                "focal": {"estimate": -0.08, "se": 0.01}, "moderators": {}, "secondary": {"estimate": 0.0, "se": 0.01},
                "subsample": {"estimate": -0.05, "se": 0.016}, "hidden": {}}


def test_analyst_reports_found_sign(old):
    ra = RuleAuthor(old)
    q = {"id": "Q-1", "iv_var": "x", "dv_var": "y", "prediction_sign": "positive"}
    var = {"controls": [], "moderators": [], "secondary_dv": None}
    a = ra.analyse_analytic(_Gen(), q, 1, {"throughput": "T0"}, {}, True, 2, var, False)
    assert a["hypotheses"][0]["verdict"] == "not_supported"     # predicted positive, found significantly negative
    b = ra.analyse_analytic(_Gen(), q, 1, {"throughput": "T0"}, {}, True, 2, var, False, predict_sign=False)
    assert b["hypotheses"][0]["verdict"] == "supported" and b["hypotheses"][0]["predicted_sign"] is None
    assert b["hypotheses"][1]["verdict"] == "supported"         # the subsample entry, also significant and negative


def test_stage1_params_file_parses_and_uses_the_new_keys():
    p = ROOT / "configs" / "fastmode" / "params-stage1.yaml"
    if not p.exists():
        pytest.skip("params-stage1.yaml not written yet")
    d = load_yaml(p)
    assert d["status"].startswith("calibrated-stage1")
    assert d["editor"]["accept_cut_round1"] > d["editor"]["revise_band"] / 2
    assert d["reviewer"]["validity_paper_sd"] > 0 and d["reviewer"]["p_unprompted_issue"] > 0
    assert d["author"]["analyst_reports_found_sign"] is True and d["lab_head"]["p_request_n"] > 0
    old = load_yaml(CONFIG_DIR / "fastmode" / "params.yaml")
    assert set(old) <= set(d) <= set(old) | {"fast_honours_deposit_n"}   # same blocks (+ the switch that keeps a deposit's requested size)
    assert "accept_cut_round1" not in old["editor"] and "validity_paper_sd" not in old["reviewer"]


# ------------------------------------------------------------------ earlier fast-mode runs stay reproducible
def test_old_params_reproduce_a_sweep_run(tmp_path):
    """Fast S0-150 seed 1001 with configs/fastmode/params.yaml reproduces the audit of a stored fast-mode parameter-sweep run
    (its baseline cell). Skipped when that stored run is not present (run outputs are not part of the repository)."""
    ref = ROOT / "runs" / "sweep-2026-10-15" / "baseline" / "S0-150" / "1001" / "fast" / "audit" / "audit.json"
    if not ref.exists():
        pytest.skip("sweep outputs not present")
    from synthsci.engine import run
    run("S0-150", 1001, "fast", None, None, tmp_path, None, resume=False, quiet=True)
    got = json.loads((tmp_path / "S0-150" / "1001" / "fast" / "audit" / "audit.json").read_text(encoding="utf-8").replace("NaN", "null"))
    want = json.loads(ref.read_text(encoding="utf-8").replace("NaN", "null"))
    for k in ("field", "realism", "timing", "careers", "handbook"):
        assert json.dumps(got.get(k), sort_keys=True, default=str) == json.dumps(want.get(k), sort_keys=True, default=str), k


# ------------------------------------------------------------------ the warm start keeps its own rule file
def _q0_state(rd: Path) -> dict:
    """Population at quarter 0, the legacy library and the literature the warm start left (handbook Q0 input, run path masked)."""
    mask = lambda b: b.replace(str(rd).encode("utf-8"), b"<RUN>").replace(str(rd).replace("\\", "\\\\").encode("utf-8"), b"<RUN>")
    return {"pop": (rd / "population" / "scientists_Q0.json").read_bytes(), "legacy": (rd / "legacy" / "papers.json").read_bytes(),
            "hb0": mask((rd / "handbook" / "Q0" / "input.json").read_bytes())}


def test_prehistory_params_keep_the_warm_start(tmp_path):
    """Fast S0-150 with live params = params-stage1.yaml and pre-history = params.yaml starts quarter 1 from exactly the
    state of a run with params.yaml throughout; with params-stage1.yaml throughout it does not."""
    stage1 = ROOT / "configs" / "fastmode" / "params-stage1.yaml"
    if not stage1.exists():
        pytest.skip("params-stage1.yaml not present")
    from synthsci.engine import run
    short = {"horizon_quarters": 1}
    a = run("S0-150", 7, "fast", None, short, tmp_path / "a", None, resume=False, quiet=True)
    b = run("S0-150", 7, "fast", None, short | {"fastmode.params": str(stage1), "fastmode.prehistory_params": "fastmode/params.yaml"},
            tmp_path / "b", None, resume=False, quiet=True)
    c = run("S0-150", 7, "fast", None, short | {"fastmode.params": str(stage1)}, tmp_path / "c", None, resume=False, quiet=True)
    sa, sb, sc = _q0_state(a), _q0_state(b), _q0_state(c)
    assert sa["pop"] == sb["pop"] and sa["legacy"] == sb["legacy"] and sa["hb0"] == sb["hb0"]
    assert sa["hb0"] != sc["hb0"] or sa["pop"] != sc["pop"]       # the new rules throughout change the warm start


# ------------------------------------------------------------------ fast analyses keep the deposit's requested size
def test_fast_analysis_honours_requested_deposit_size(tmp_path):
    """With lab heads naming dataset sizes, the fast analytic study must have the deposit's N; the old path
    redrew the frame from the dataset seed and ignored it."""
    stage1 = ROOT / "configs" / "fastmode" / "params-stage1.yaml"
    if not stage1.exists():
        pytest.skip("params-stage1.yaml not present")
    import yaml
    from synthsci.engine import run
    from synthsci.util import read_jsonl
    out = {}
    for flag in (False, True):
        p = yaml.safe_load(stage1.read_text(encoding="utf-8"))
        p["fast_honours_deposit_n"] = flag
        pf = tmp_path / f"p{int(flag)}.yaml"
        pf.write_text(yaml.safe_dump(p), encoding="utf-8")
        rd = run("R-b-150", 7, "fast", None, {"horizon_quarters": 3, "fastmode.params": str(pf), "budget.rule": "fixed"},
                 tmp_path / f"r{int(flag)}", None, resume=False, quiet=True)
        req = {e["analysis_id"]: e["n_granted"] for e in read_jsonl(rd / "ledger" / "events.jsonl") if e.get("event") == "dataset_size_request"}
        rows = [(h["n"], req[h["analysis_id"]]) for h in read_jsonl(rd / "ledger" / "hypotheses.jsonl")
                if not h.get("legacy") and h.get("analysis_id") in req and h.get("kind") == "focal" and h.get("n")]
        assert rows, "no fresh analysis with a requested size"
        out[flag] = float(np.median([abs(n / g - 1) for n, g in rows]))
    assert out[True] < 0.25 < out[False]       # within the panel-balance factor of the request only with the fix
