"""Measures for the fast-mode recalibration against the eight 8-quarter live runs (criteria in tools/fastcal_validate.py).

One function, `measures(run_dir)`, reads a finished run's audit/audit.json and its ledger (quarter >= 1 only, i.e. the live
part of a live run and the same window of a fast run; the warm-start pre-history is excluded in both) and returns the
criterion measures. The same code scores the eight live Stage 1 runs and the fast validation runs, so live and fast are
measured identically. Read-only.

    python tools/fastcal_measures.py --live            # the live ranges of the criteria measures, as a table
    python tools/fastcal_measures.py <run_dir> ...     # print the measures of any runs
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

LIVE = [("S0-150", 1), ("S0-150", 2), ("S0-150", 3), ("S0P-150", 1), ("S0P-150", 2), ("R-b-150", 1), ("R-b-150", 2), ("R-b-150", 3)]
KINDS_FLAW = ("data", "specification", "inference")


# The audit extraction shared with the fast-mode sweep summaries: one row of headline measures per run.
def _g(d, dotted, default=None):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if isinstance(cur, float) and math.isnan(cur):
        return default
    return cur


def row_from_audit(a: dict) -> dict:
    hb = a.get("handbook") or {}
    qs = sorted(int(k[1:]) for k in hb if k.startswith("Q") and isinstance(hb[k], dict))
    qf = f"Q{qs[-1]}" if qs else None
    paper = a.get("regime") == "paper"
    r = {
        "regime": a.get("regime"), "horizon": a.get("horizon_quarters"),
        "hb_final_q": qs[-1] if qs else None,
        "hb_acc_q0": _g(hb, "Q0.accuracy"), "hb_acc_final": _g(hb, f"{qf}.accuracy") if qf else None,
        "hb_correct_q0": _g(hb, "Q0.correct_answers"), "hb_correct_final": _g(hb, f"{qf}.correct_answers") if qf else None,
        "hb_scored_final": _g(hb, f"{qf}.n_scored") if qf else None,
    }
    r["hb_acc_change"] = (r["hb_acc_final"] - r["hb_acc_q0"]) if None not in (r["hb_acc_final"], r["hb_acc_q0"]) else None
    for b in ("B1", "B2", "B3", "B5"):
        e, q0 = _g(a, f"field.net_knowledge.{b}"), _g(a, f"field.net_knowledge_quarter0.{b}")
        r[f"nk_{b}_q0"], r[f"nk_{b}_end"] = q0, e
        r[f"nk_{b}_change"] = (e - q0) if None not in (e, q0) else None
        r[f"acc_{b}"] = _g(a, f"field.accuracy.{b}.accuracy")
    r["covered_B1"] = _g(a, "field.accuracy.B1.covered")
    r["nulls_believed_effect_B1"] = _g(a, "field.knowledge_of_nulls.B1.believed_effect")
    r["nulls_believed_null_B2"] = _g(a, "field.knowledge_of_nulls.B2.believed_null")
    r["run_coverage"] = _g(a, "field.coverage.core_covered_share")
    r["prec_median_abs_error"] = _g(a, "field.precision.median_abs_error")
    r["prec_ci_coverage"] = _g(a, "field.precision.coverage")
    r["prec_n_questions"] = _g(a, "field.precision.n_questions")
    nw, inh = "field.certification_by_origin.new_in_run", "field.certification_by_origin.inherited"
    r["new_units"] = _g(a, f"{nw}.n_units")
    r["new_true"] = _g(a, f"{nw}.n_true")
    r["new_false"] = _g(a, f"{nw}.n_false")
    r["new_missed"] = _g(a, f"{nw}.n_missed_effect")
    r["new_from_prehistory"] = _g(a, f"{nw}.n_from_prehistory")
    r["new_false_share"] = _g(a, f"{nw}.false_share")
    r["inherited_units"] = _g(a, f"{inh}.n_units")
    r["inherited_false_share"] = _g(a, f"{inh}.false_share")
    r["inherited_false_corrected"] = _g(a, f"{inh}.false_units_corrected_in_run")
    r["self_corr_share"] = _g(a, "field.self_correction.share_corrected")
    r["median_q_to_first_cert"] = _g(a, "field.certification.median_quarters_to_first_certification")
    r["file_drawer_share_audit"] = _g(a, "field.certification.true_findings_never_certified_share")
    if paper:
        st, pu, un = (_g(a, f"timing.live_authored.{k}") for k in ("started", "published", "unfinished_at_horizon"))
        r["live_started"], r["live_published"], r["live_unfinished"] = st, pu, un
        r["live_unfinished_share"] = (un / st) if st else None
        r["accept_rate"] = _g(a, "realism.accept_rate_mean")
        r["sub_to_accept_median_months"] = _g(a, "timing.submission_to_acceptance_months.median")
        r["accepted_without_revision"] = _g(a, "timing.share_accepted_without_revision")
        r["reviewer_kappa"] = _g(a, "author.certifier_accuracy.kappa_recommendation")
        r["certifier_auc"] = _g(a, "author.certifier_accuracy.validity_auc_vs_truth")
    else:
        r["certifier_auc"] = _g(a, "author.certifier_accuracy.endorse_auc_vs_truth")
        r["verif_powered_share"] = _g(a, "verification_power.powered_share")
        r["lapsed_true_claims"] = _g(a, "failure_indices.free_riding.lapsed_true_claims")
        r["unverified_endorsement_share"] = _g(a, "failure_indices.herding.unverified_endorsement_share")
    r["support_share"] = _g(a, "realism.support_share")
    r["fully_supported_share"] = _g(a, "realism.fully_supported_share")
    for k in ("tenure_exit", "tenured", "promoted_full", "not_promoted_full", "promoted_assistant"):
        r[f"career_{k}"] = _g(a, f"careers.run.counts.{k}", 0)
    r["compute"] = _g(a, "budget.total.compute")
    r["data"] = _g(a, "budget.total.data")
    r["calls"] = _g(a, "budget.calls.calls")
    r["compute_ratio_to_paired"] = _g(a, "budget.per_unit.paired_world.compute_ratio")
    r["data_ratio_to_paired"] = _g(a, "budget.per_unit.paired_world.data_ratio")
    r["deferrals_data"] = _g(a, "budget.deferrals_by_binding.data")
    r["deferrals_compute"] = _g(a, "budget.deferrals_by_binding.compute")
    r["deferrals_both"] = _g(a, "budget.deferrals_by_binding.both")
    r["deferred_outstanding"] = _g(a, "budget.deferred_outstanding")
    r["data_per_correct_answer"] = _g(a, "budget.per_unit.data_per_correct_handbook_answer")
    r["compute_per_correct_answer"] = _g(a, "budget.per_unit.compute_per_correct_handbook_answer")
    return r


def _jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _share(k: int, n: int):
    return (k / n) if n else None


def measures(rd: Path) -> dict:
    rd = Path(rd)
    a = json.loads((rd / "audit" / "audit.json").read_text(encoding="utf-8").replace("NaN", "null"))
    base = row_from_audit(a)
    regime = a.get("regime")
    m: dict = {"regime": regime}
    # ---- knowledge, certification, budget (audit; identical extraction to the sweep)
    for k in ("hb_acc_q0", "hb_acc_final", "hb_acc_change", "nk_B1_change", "nk_B2_change", "acc_B2", "new_units", "new_false_share",
              "run_coverage", "support_share", "data", "compute", "accept_rate", "accepted_without_revision", "reviewer_kappa",
              "verif_powered_share", "inherited_units", "live_published", "unverified_endorsement_share", "compute_ratio_to_paired"):
        m[k] = base.get(k)
    led = rd / "ledger"
    if regime == "paper":
        m["flaw_noticed_rate"] = ((a.get("author") or {}).get("certifier_accuracy") or {}).get("flaw_noticed_rate")
        dec = [d for d in _jl(led / "decisions.jsonl") if int(d.get("quarter", 1)) >= 1]
        desk = [d for d in dec if d["stage"] == "desk"]
        r1 = [d for d in dec if d["stage"] == "round1"]
        r2 = [d for d in dec if d["stage"] == "round2"]
        m["desk_reject_rate"] = _share(sum(d["decision"] == "desk_reject" for d in desk), len(desk))
        m["r1_accept_share"] = _share(sum(d["decision"] == "accept" for d in r1), len(r1))
        m["r1_reject_share"] = _share(sum(d["decision"] == "reject" for d in r1), len(r1))
        m["r1_revise_share"] = _share(sum(d["decision"] == "revise" for d in r1), len(r1))
        m["r2_accept_share"] = _share(sum(d["decision"] == "accept" for d in r2), len(r2))
        m["n_r1_decisions"], m["n_r2_decisions"], m["n_desk"] = len(r1), len(r2), len(desk)
        rv = [x for x in _jl(led / "reviews.jsonl") if int(x.get("quarter", 1)) >= 1]
        rc = collections.Counter(x["recommendation"] for x in rv)
        for k in ("reject", "major_revision", "minor_revision", "accept"):
            m[f"rec_{k}"] = _share(rc.get(k, 0), len(rv))
        clean = [x for x in rv if not x.get("analyst_flags")]
        flagged = [x for x in rv if x.get("analyst_flags")]
        m["false_alarm_rate"] = _share(sum(any(k in KINDS_FLAW for k in x.get("issue_kinds", [])) for x in clean), len(clean))
        fn = _share(sum(any(k in KINDS_FLAW for k in x.get("issue_kinds", [])) for x in flagged), len(flagged))
        m["detection_lift"] = (fn - m["false_alarm_rate"]) if (fn is not None and m["false_alarm_rate"] is not None) else None
        m["flaw_found_flagged"] = _share(sum(bool(x.get("flaw_found")) for x in flagged), len(flagged))
        m["flaw_found_clean"] = _share(sum(bool(x.get("flaw_found")) for x in clean), len(clean))
        m["n_reviews"] = len(rv)
    else:
        votes = [v for v in _jl(led / "votes.jsonl") if int(v.get("quarter", 1)) >= 1 and v.get("vote") in ("endorse", "abstain", "challenge")]
        vc = collections.Counter(v["vote"] for v in votes)
        for k in ("endorse", "abstain", "challenge"):
            m[f"vote_{k}"] = _share(vc.get(k, 0), len(votes))
        m["n_votes"] = len(votes)
        vf = [v for v in _jl(led / "verifications.jsonl") if int(v.get("quarter", 1)) >= 1]
        m["verif_reproduced_share"] = _share(sum(v.get("result") == "reproduced" for v in vf), len(vf))
        m["n_verifications"] = len(vf)
        hy = [h for h in _jl(led / "hypotheses.jsonl") if not h.get("legacy") and h.get("analysis_id")]
        filed = [h for h in hy if h.get("status") not in ("dropped", "never_filed")]
        m["filed_share"] = _share(len(filed), len(hy))
        m["n_entries"] = len(hy)
    return m


def live_table() -> dict:
    out = {}
    for scen, s in LIVE:
        out[f"{scen} s{s}"] = measures(WS / "runs" / scen / str(s) / "llm")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    rows = live_table() if a.live else {r: measures(Path(r)) for r in a.runs}
    keys = sorted({k for v in rows.values() for k in v})
    print("measure," + ",".join(rows))
    for k in keys:
        vals = []
        for v in rows.values():
            x = v.get(k)
            vals.append("" if x is None else (f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else str(x)))
        print(k + "," + ",".join(vals))
