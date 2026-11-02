"""Planning cost estimate per quarter and per run from the config and the tariff table.

All planning shares are design defaults: 10% desk rejection, 45% revise and 90% of those revised,
three reviewers per paper, the same three on revision, batches of 10 (desk), 4/6 (review), 8 (decision).
Repository: lab calls = labs; analyses = the same slot capacity as the paper regime; verifications = k x 60% of
analyses (two per call); the pairing rule then caps R's compute at P's realized spend (compute only; data floats
under `budget.data_ceiling_multiple` x P's realized data and is reported, not planned, so the estimate prints the ceiling).
"""
from __future__ import annotations

import math

from . import config as C
from .budget import Budget
from .population import effective_availability


def estimate(cfg: dict) -> dict:
    from .ledger import Ledger
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="cost_"))
    # the estimate needs the tariff table only; a pairing scenario would look for its paired run's spend, which a plan does not have
    b = Budget(C.deep_merge(cfg, {"budget": {"rule": "fixed"}}), tmp, Ledger(tmp, "cost", "fast", cfg["regime"]), None)
    T = lambda k: b.tariff(k)
    pp = cfg["population_profile"]
    N = int(cfg["population"])
    H = int(cfg["horizon_quarters"])
    ea = effective_availability(cfg)
    p = ea["availability_p"]
    mult = math.exp(0.5 * pp["productivity_log_sd"] ** 2)
    slots = 1.0 + float(pp["throughput"]["t1_share"])
    # expected new studies per start-quarter: at a productivity target the realized rate is the target rate (the drawn p is
    # raised only to offset slots held through review); otherwise every free slot is drawn at p
    starts = N * ea["starts_per_author_quarter"] if ea["derived"] else N * slots * min(0.95, p * mult)
    start_quarters = max(1, H - int(pp["starts_stop_before_end"]))
    jc = cfg["journal"]
    B = int(jc.get("research_batch_size", 1))
    lean = jc.get("paper_contract", "full") == "lean"
    t1 = float(pp["throughput"]["t1_share"])
    research_tariff = (1 - t1) * T("research_t0") + t1 * T("research_t1")
    research_calls = starts / B
    research_tokens = research_calls * research_tariff * b.call_scale(B, lean)
    revisions = starts * 0.9 * 0.45 * 0.9
    revision_tokens = revisions * ((1 - t1) * T("revision_t0") + t1 * T("revision_t1")) * b.call_scale(1, lean)
    sent_out = starts * 0.9
    desk_calls = math.ceil(starts / 10) if starts else 0
    rev_batch = 6 if jc["reviewer_assistance"] == "ai_written" else int(jc["reviewer_cap"])
    review_calls = 3 * (sent_out + revisions) / rev_batch
    review_tokens = review_calls * (T("review_ai_written") if jc["reviewer_assistance"] == "ai_written" else T("review"))
    prescreen_tokens = (sent_out / 4) * T("prescreen") if jc["reviewer_assistance"] == "prescreen" else 0.0
    decision_calls = (sent_out + revisions) / 8
    per_q_P = research_tokens + revision_tokens + desk_calls * T("desk_screen") + review_tokens + prescreen_tokens + decision_calls * T("decision")
    calls_q_P = research_calls + revisions + desk_calls + review_calls + decision_calls
    hb = len(cfg["handbook"].get("snapshots", [])) + (1 if cfg["handbook"].get("quarter0", True) else 0)
    extra = hb * 3 * T("handbook") + math.ceil(cfg["landscape"]["legacy"]["size"] / 10) * T("legacy_abstracts")
    run_P = per_q_P * start_quarters + (H - start_quarters) * (review_tokens + decision_calls * T("decision")) * 0.5 + extra
    # repository regime
    rc = cfg["repository"]
    labs = max(1, round(N / rc["lab_mean_size"]))
    Ba = int(rc.get("analyst_batch_size", 1))
    analyses = starts
    analyst_tokens = (analyses / Ba) * ((1 - t1) * T("analyst_t0") + t1 * T("analyst_t1")) * b.call_scale(Ba, False)
    verifications = analyses * 0.6 * int(rc["reproductions_k"])
    per_q_R = labs * T("lab_decision") + analyst_tokens + (verifications / 2) * T("verification")
    calls_q_R = labs + analyses / Ba + verifications / 2
    dcm = float(cfg["budget"].get("data_ceiling_multiple", 2.0))
    return {"scenario": cfg["scenario"], "population": N, "horizon_quarters": H, "availability_p": p, "derived_from_target": ea["derived"],
            "papers_per_author_year_target": ea.get("papers_per_author_year"), "expected_starts_per_quarter": starts,
            "research_batch_size": B, "paper_contract": jc.get("paper_contract", "full"), "analyst_batch_size": Ba, "labs": labs,
            "paper_regime": {"calls_per_quarter": calls_q_P, "tokens_per_quarter": per_q_P, "tokens_per_run_incl_handbooks": run_P,
                             "research_calls_per_quarter": research_calls, "review_calls_per_quarter": review_calls},
            "repository_regime_planning": {"calls_per_quarter": calls_q_R, "tokens_per_quarter_uncapped": per_q_R,
                                           "compute_cap": "P's realized compute spend at the same seed (compute is the balanced resource)",
                                           "data_ceiling_multiple": dcm,
                                           "note": f"under the pairing rule R's compute is capped at P's realized spend and R's data floats under "
                                                   f"{dcm:g} x P's realized data; this row is the compute demand before capping"},
            "tariff_status": b.tariffs.get("status")}


def table(est: dict) -> str:
    P, R = est["paper_regime"], est["repository_regime_planning"]
    return "\n".join([
        f"Cost estimate for {est['scenario']} ({est['population']} authors, {est['horizon_quarters']} quarters; tariffs: {est['tariff_status']})",
        f"  availability p per slot: {est['availability_p']:.3f}" + (f" (derived from {est['papers_per_author_year_target']} papers/author-year)" if est["derived_from_target"] else " (design default)"),
        f"  expected new studies per quarter: {est['expected_starts_per_quarter']:.1f}; research batch {est['research_batch_size']}; paper contract {est['paper_contract']}",
        f"  P: {P['calls_per_quarter']:.0f} calls/quarter ({P['research_calls_per_quarter']:.1f} research, {P['review_calls_per_quarter']:.1f} review); "
        f"{P['tokens_per_quarter'] / 1e6:.2f}M tariff tokens/quarter; about {P['tokens_per_run_incl_handbooks'] / 1e6:.1f}M per run incl. handbooks and legacy abstracts",
        f"  R (planning demand, {est['labs']} labs, analyst batch {est['analyst_batch_size']}): {R['calls_per_quarter']:.0f} calls/quarter, "
        f"{R['tokens_per_quarter_uncapped'] / 1e6:.2f}M tokens/quarter before the pairing cap (compute capped at P's realized "
        f"compute; data floats under {R['data_ceiling_multiple']:g} x P's data)",
    ])
