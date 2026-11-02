"""20-quarter (x20) results: compare the paper, pressure and repository worlds (descriptive only).

Usage (from the repository root or anywhere):
  set PYTHONDONTWRITEBYTECODE=1
  python analysis/x20/compute_results.py          (run folders: runs/, or set SYNTHSCI_RUNS)

READS ONLY (nothing under runs/ is written, moved or re-run):
  runs/{S0-150,S0P-150,R-b-150}-x20/<seed>/llm/   audit/audit.json, run.json, extension.json, budget/{summary,spend_by_quarter}.json,
      handbook/Q*/score.json, ledger/{hypotheses.jsonl, papers_snapshot.json, papers.jsonl, claims.jsonl, votes.jsonl,
      verifications.jsonl, events.jsonl, questions_Q8.json, questions_Q20.json}, population/scientists*.json
  runs/{S0-150,S0P-150,R-b-150}/<seed>/llm/audit/audit.json   (the original 8-quarter runs: the interim snapshot at Q8)
  synthsci/audit.py, synthsci/scorer.py are imported for their definitions only (no audit is re-run, no file written).
WRITES: analysis/x20/*.csv, analysis/x20/figures/*.png, analysis/x20/run_log.txt
Every reported 20-quarter number comes from these outputs.
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
W = HERE.parents[1]
sys.path.insert(0, str(W))
# run folders: runs/ under the repository root, or the folder named by SYNTHSCI_RUNS (e.g. the unpacked run archive)
RUNS_ROOT = Path(os.environ.get("SYNTHSCI_RUNS") or (W / "runs"))
if not RUNS_ROOT.is_dir():
    sys.exit(f"STOP: run folder {RUNS_ROOT} not found; unpack the run archive there or set SYNTHSCI_RUNS to its location")

import numpy as np
import pandas as pd

from synthsci.audit import TRUE_CHECKS, _cert_q, _corrected_by, _dedupe, _is_focal, _stated_truth_check  # definitions only
from synthsci.scorer import claim_sign

OUT = HERE
FIG = HERE / "figures"
RUNS = [("S0-150", 1), ("S0-150", 2), ("S0-150", 3), ("S0P-150", 1), ("S0P-150", 2),
        ("R-b-150", 1), ("R-b-150", 2), ("R-b-150", 3)]
WORLDS = ["S0-150", "S0P-150", "R-b-150"]
WORLD_LABEL = {"S0-150": "Paper (S0)", "S0P-150": "Paper + career pressure (S0P)", "R-b-150": "Repository (R)"}
SNAPS = ["Q0", "Q4", "Q8", "Q16", "Q20"]
TRAPS = ["duplicates", "scale_mixture", "reverse_coded", "selection_on_outcome", "bad_control", "miscoded_values",
         "unit_mismatch", "omitted_standard_control"]
TRAP_FLAG = {"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed", "reverse_coded": "reverse_coding_missed",
             "selection_on_outcome": "selection_unacknowledged", "bad_control": "bad_control_included",
             "miscoded_values": "miscoded_values_kept", "unit_mismatch": "unit_mismatch_unfixed",
             "omitted_standard_control": "omitted_standard_control"}   # scorer flag raised when each planted problem is not handled
STAGES = ["doctoral", "assistant", "associate", "full"]
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def x20(s, k):
    return RUNS_ROOT / f"{s}-x20" / str(k) / "llm"


def orig(s, k):
    return RUNS_ROOT / s / str(k) / "llm"


def ld(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def isnum(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def div(a, b):
    return (a / b) if (isnum(a) and isnum(b) and b) else None


def stc(h):
    return h.get("truth_check_stated") or h.get("truth_check")


def sign_word(h):
    s = claim_sign(h)
    return "null" if s == "none" else s


# ======================================================================== load
A20, A8, LED = {}, {}, {}
for key in RUNS:
    s, k = key
    d = x20(s, k)
    A20[key] = ld(d / "audit" / "audit.json")
    A8[key] = ld(orig(s, k) / "audit" / "audit.json")
    hyps = _dedupe(jl(d / "ledger" / "hypotheses.jsonl"))
    for h in hyps:
        h["truth_check_stated"] = _stated_truth_check(h)
    LED[key] = {
        "hyps": hyps,
        "events": jl(d / "ledger" / "events.jsonl"),
        "spend": ld(d / "budget" / "spend_by_quarter.json"),
        "bsum": ld(d / "budget" / "summary.json"),
        "ext": ld(d / "extension.json"),
        "run": ld(d / "run.json"),
        "roll8": ld(d / "ledger" / "questions_Q8.json"),
        "roll20": ld(d / "ledger" / "questions_Q20.json"),
        "hb": {q: ld(d / "handbook" / q / "score.json") for q in SNAPS if (d / "handbook" / q / "score.json").exists()},
    }
    if s.startswith("S0"):
        LED[key]["papers"] = ld(d / "ledger" / "papers_snapshot.json")
    else:
        LED[key]["analyses"] = jl(d / "ledger" / "papers.jsonl")
        LED[key]["claims"] = jl(d / "ledger" / "claims.jsonl")
        LED[key]["votes"] = jl(d / "ledger" / "votes.jsonl")
        LED[key]["verifs"] = jl(d / "ledger" / "verifications.jsonl")
    pop = {}
    for p in sorted((d / "population").glob("scientists*.json")):
        lab = p.stem.replace("scientists", "").lstrip("_") or "seed"
        pop[lab] = ld(p)
    LED[key]["pop"] = pop
    log(f"loaded {s}-x20 s{k}: horizon {A20[key]['horizon_quarters']}, extended from {g(LED[key]['ext'], 'source_horizon')}, "
        f"source unchanged {g(LED[key]['ext'], 'source_unchanged')}")


def regime(key):
    return "paper" if key[0].startswith("S0") else "repository"


def run_units(key, upto=20):
    """The audit's certified units (one definition in both regimes): focal, certified, non-legacy, certified at quarter 1..upto."""
    return [h for h in LED[key]["hyps"] if not h.get("legacy") and h.get("certified") and _is_focal(h) and 1 <= _cert_q(h) <= upto]


# ======================================================================== 1-2, 6: knowledge, truth, speed (per run)
rows = []
per_q_rows = []
for key in RUNS:
    s, k = key
    a, a8 = A20[key], A8[key]
    L = LED[key]
    r = {"world": s, "seed": k, "regime": regime(key)}
    units = run_units(key)
    units8 = run_units(key, 8)
    # consistency with the audit's own count
    r["certified_units"] = len(units)
    r["certified_units_audit"] = g(a, "field", "certification", "n_certified_units")
    if r["certified_units"] != r["certified_units_audit"]:
        log(f"WARNING {key}: my certified units {r['certified_units']} != audit {r['certified_units_audit']}")
    r["certified_units_q1_8_from_x20_ledger"] = len(units8)
    r["certified_units_q9_20"] = len(units) - len(units8)
    r["certified_rows_all_kinds"] = g(a, "field", "certification", "n_certified_rows_all_kinds")
    tcs = collections.Counter(stc(h) for h in units)
    r["units_true"] = sum(v for kk, v in tcs.items() if kk in TRUE_CHECKS)
    r["units_false_discovery"] = tcs.get("false_discovery", 0)
    r["units_missed_effect"] = tcs.get("missed_effect", 0)
    r["units_unknown"] = sum(v for kk, v in tcs.items() if kk in (None, "unknown"))
    r["units_true_effect"] = tcs.get("true_discovery", 0) + tcs.get("true_discovery_negligible", 0)
    r["units_correct_null"] = tcs.get("correct_null", 0) + tcs.get("correct_null_negligible", 0)
    r["true_share"] = div(r["units_true"], len(units))
    r["false_discovery_share"] = div(r["units_false_discovery"], len(units))
    r["missed_effect_share"] = div(r["units_missed_effect"], len(units))
    # the pre-certification pool: every focal analysis written in the run (certified or not)
    pool = [h for h in L["hyps"] if not h.get("legacy") and _is_focal(h) and not h.get("prehistory") and int(h.get("quarter") or 0) >= 1]
    pc = collections.Counter(stc(h) for h in pool)
    r["focal_analyses_live"] = len(pool)
    r["focal_analyses_live_true_share"] = div(sum(v for kk, v in pc.items() if kk in TRUE_CHECKS), len(pool))
    r["focal_analyses_live_false_discovery_share"] = div(pc.get("false_discovery", 0), len(pool))
    # on the authored-by-live-agents split (paper world: warm-start pipeline papers were researched by rule agents)
    r["certified_units_live_authored"] = sum(1 for h in units if not h.get("prehistory"))
    r["certified_units_prehistory_authored"] = sum(1 for h in units if h.get("prehistory"))
    r["self_correction_false_units"] = g(a, "field", "self_correction", "n_false_certified_units")
    r["self_correction_share_corrected"] = g(a, "field", "self_correction", "share_corrected")
    if regime(key) == "paper":
        P = L["papers"]
        pub = [p for p in P if not p.get("legacy") and (p.get("quarter_published") or 0) >= 1]
        r["papers_published_in_run"] = len(pub)
        r["papers_published_live_authored"] = sum(1 for p in pub if not p.get("prehistory"))
        r["papers_published_warmstart_pipeline"] = sum(1 for p in pub if p.get("prehistory"))
        r["papers_published_q1_8"] = sum(1 for p in pub if p["quarter_published"] <= 8)
        r["papers_published_q9_20"] = sum(1 for p in pub if p["quarter_published"] >= 9)
        r["audit_published_in_run_total"] = g(a, "timing", "published_in_run_total")
        live = [p for p in P if not p.get("legacy") and not p.get("prehistory") and (p.get("quarter_started") or 0) >= 1]
        r["studies_started_live"] = len(live)
        r["studies_unfinished_at_horizon"] = sum(1 for p in live if p.get("status") == "unfinished")
        r["status_counts_live"] = json.dumps(dict(collections.Counter(p.get("status") for p in live)))
        # speed: start to publication (quarters) and submission to acceptance (audit, months)
        sp = [p["quarter_published"] - p["quarter_started"] for p in pub if p.get("quarter_started") is not None]
        spl = [p["quarter_published"] - p["quarter_started"] for p in pub if p.get("quarter_started") is not None and not p.get("prehistory")]
        r["start_to_certification_median_q"] = float(np.median(sp)) if sp else None
        r["start_to_certification_median_q_live"] = float(np.median(spl)) if spl else None
        r["start_to_certification_n_live"] = len(spl)
        r["submission_to_acceptance_median_months"] = g(a, "timing", "submission_to_acceptance_months", "median")
        r["submission_to_acceptance_n"] = g(a, "timing", "submission_to_acceptance_months", "n")
        r["submission_to_first_decision_median_months"] = g(a, "timing", "submission_to_first_decision_months", "median")
        for q in range(1, 21):
            per_q_rows.append({"world": s, "seed": k, "quarter": q,
                               "papers_published": sum(1 for p in pub if p["quarter_published"] == q),
                               "papers_published_live_authored": sum(1 for p in pub if p["quarter_published"] == q and not p.get("prehistory")),
                               "certified_units": sum(1 for h in units if _cert_q(h) == q),
                               "certified_units_true": sum(1 for h in units if _cert_q(h) == q and stc(h) in TRUE_CHECKS),
                               "studies_started": sum(1 for p in live if p.get("quarter_started") == q)})
    else:
        C = L["claims"]
        latest = {}
        for c in C:
            latest[c["claim_id"]] = c
        runc = [c for c in latest.values() if not c.get("legacy")]
        r["claims_filed"] = len(runc)
        r["claims_endorsed_now"] = sum(1 for c in runc if c.get("status") == "endorsed")
        for stt in ("proposed", "contested", "lapsed", "withdrawn", "retracted", "superseded"):
            r[f"claims_{stt}_now"] = sum(1 for c in runc if c.get("status") == stt)
        r["analyses_run"] = len(L["analyses"])
        r["papers_published_in_run"] = None
        sp = [_cert_q(h) - int(h["quarter"]) for h in units if h.get("quarter") is not None]
        r["start_to_certification_median_q"] = float(np.median(sp)) if sp else None
        r["start_to_certification_median_q_live"] = r["start_to_certification_median_q"]
        r["start_to_certification_n_live"] = len(sp)
        allc = [h for h in L["hyps"] if not h.get("legacy") and h.get("certified")]
        spa = [_cert_q(h) - int(h["quarter"]) for h in allc if h.get("quarter") is not None]
        r["analysis_to_endorsement_median_q_all_kinds"] = float(np.median(spa)) if spa else None
        r["studies_started_live"] = len(L["analyses"])
        for q in range(1, 21):
            per_q_rows.append({"world": s, "seed": k, "quarter": q, "papers_published": None, "papers_published_live_authored": None,
                               "certified_units": sum(1 for h in units if _cert_q(h) == q),
                               "certified_units_true": sum(1 for h in units if _cert_q(h) == q and stc(h) in TRUE_CHECKS),
                               "studies_started": sum(1 for x in L["analyses"] if x.get("quarter") == q)})
    r["median_quarters_to_first_certification_audit"] = g(a, "field", "certification", "median_quarters_to_first_certification")
    # resources
    comp, data = g(a, "budget", "total", "compute"), g(a, "budget", "total", "data")
    r["compute_tariff_tokens"] = comp
    r["data_observations"] = data
    r["realized_tokens"] = g(a, "budget", "total", "realized_tokens")
    r["debited_calls"] = g(a, "budget", "calls", "calls")
    r["certified_units_per_1k_compute"] = div(len(units), comp / 1000 if comp else None)
    r["certified_units_per_1M_compute"] = div(len(units), comp / 1e6 if comp else None)
    r["true_units_per_1M_compute"] = div(r["units_true"], comp / 1e6 if comp else None)
    r["certified_units_per_1M_realized_tokens"] = div(len(units), r["realized_tokens"] / 1e6 if r["realized_tokens"] else None)
    r["compute_per_certified_unit"] = g(a, "budget", "per_unit", "compute_per_certified_unit")
    r["data_per_certified_unit"] = g(a, "budget", "per_unit", "data_per_certified_unit")
    r["compute_per_correct_handbook_answer"] = g(a, "budget", "per_unit", "compute_per_correct_handbook_answer")
    r["data_per_correct_handbook_answer"] = g(a, "budget", "per_unit", "data_per_correct_handbook_answer")
    r["compute_ratio_to_paired_paper"] = g(a, "budget", "per_unit", "paired_world", "compute_ratio")
    r["data_ratio_to_paired_paper"] = g(a, "budget", "per_unit", "paired_world", "data_ratio")
    src = g(a, "budget", "per_unit", "paired_world", "source")
    if isinstance(src, str):   # the audit records an absolute path; keep it from runs/ on, so no machine path is written
        s_ = src.replace("\\", "/")
        src = s_[s_.rfind("/runs/") + 1:] if "/runs/" in s_ else s_
    r["paired_world_source"] = src
    r["deferrals_data_bound"] = g(a, "budget", "deferrals_by_binding", "data")
    r["deferrals_compute_bound"] = g(a, "budget", "deferrals_by_binding", "compute")
    r["deferred_outstanding"] = g(a, "budget", "deferred_outstanding")
    r["budget_summary_stale"] = g(a, "budget", "summary_stale")
    # 3. handbook
    for q in SNAPS:
        sc = L["hb"].get(q) or {}
        r[f"hb_{q}_accuracy"] = sc.get("accuracy")
        r[f"hb_{q}_answered"] = sc.get("n_answered")
        r[f"hb_{q}_scored"] = sc.get("n_scored")
        r[f"hb_{q}_correct"] = sc.get("correct_answers")
        r[f"hb_{q}_confidently_wrong"] = sc.get("confidently_wrong")
        r[f"hb_{q}_brier"] = sc.get("brier")
        r[f"hb_{q}_inter_writer_agreement"] = sc.get("inter_writer_agreement")
    r["hb_accuracy_change_Q0_Q20"] = (r["hb_Q20_accuracy"] - r["hb_Q0_accuracy"]) if isnum(r["hb_Q20_accuracy"]) and isnum(r["hb_Q0_accuracy"]) else None
    r["hb_correct_change_Q0_Q20"] = (r["hb_Q20_correct"] - r["hb_Q0_correct"]) if isnum(r["hb_Q20_correct"]) and isnum(r["hb_Q0_correct"]) else None
    # field (belief rules), whole literature
    for rule in ("B1", "B2", "B3", "B5"):
        r[f"net_knowledge_{rule}_Q0"] = g(a, "field", "net_knowledge_quarter0", rule)
        r[f"net_knowledge_{rule}_Q20"] = g(a, "field", "net_knowledge", rule)
        r[f"field_accuracy_{rule}_Q20"] = g(a, "field", "accuracy", rule, "accuracy")
    r["core_covered_share_run_units"] = g(a, "field", "coverage", "core_covered_share")
    r["core_covered_3plus_share_run_units"] = g(a, "field", "coverage", "core_covered_3plus_share")
    r["fringe_certified"] = g(a, "field", "coverage", "fringe_certified")
    r["theory_rank_agreement_B1"] = g(a, "theory", "rank_agreement", "B1")
    r["theory_rank_agreement_B2"] = g(a, "theory", "rank_agreement", "B2")
    r["precision_median_abs_error_B2"] = g(a, "field", "precision", "median_abs_error")
    r["precision_coverage_B2"] = g(a, "field", "precision", "coverage")
    r["null_questions_believed_effect_B1"] = g(a, "field", "knowledge_of_nulls", "B1", "believed_effect")
    r["support_share"] = g(a, "realism", "support_share")
    r["certifier_auc"] = g(a, "author", "certifier_accuracy", "validity_auc_vs_truth") if regime(key) == "paper" else g(a, "author", "certifier_accuracy", "endorse_auc_vs_truth")
    r["success_truth_spearman"] = g(a, "author", "success_truth_spearman")
    r["gini_visible"] = g(a, "author", "gini_visible")
    rows.append(r)

RES = pd.DataFrame(rows)

# ======================================================================== 4. planted false inherited literature
fate_rows = []
fate_detail = []
for key in RUNS:
    s, k = key
    L = LED[key]
    hyps = L["hyps"]
    units_all = [h for h in hyps if h.get("certified") and _is_focal(h)]
    # the quarter-0 inherited literature: legacy focal units certified at the start. A repository legacy claim that a later
    # claim superseded loses its `certified` flag (status 'superseded'), so the audit's inherited table drops it from the
    # denominator; it is kept here so that formal supersession counts as an overturn.
    inh = [h for h in hyps if h.get("legacy") and _is_focal(h) and (h.get("certified") or h.get("status") == "superseded")]
    inh_false = [h for h in inh if stc(h) == "false_discovery"]
    claims_latest = {}
    for c in L.get("claims", []):
        claims_latest.setdefault(c["claim_id"], []).append(c)
    for T in (8, 20):
        roll = L["roll8"] if T == 8 else L["roll20"]
        hb = L["hb"].get(f"Q{T}") or {}
        perq = hb.get("per_question") or {}
        upto = [u for u in units_all if u.get("legacy") or _cert_q(u) <= T]
        # every live focal analysis (certified or not) written by quarter T
        live_an = [h for h in hyps if not h.get("legacy") and _is_focal(h) and not h.get("prehistory") and 1 <= int(h.get("quarter") or 0) <= T]
        n_over = n_chal = n_ret = 0
        n_bel = {"B1": 0, "B2": 0, "B3": 0, "B5": 0}
        n_hb_core = n_hb_bel = 0
        for fu in inh_false:
            fs = sign_word(fu)
            over = _corrected_by(fu, upto)
            chal = [h for h in live_an if h.get("question_id") == fu.get("question_id") and sign_word(h) != fs]
            # formal retraction / supersession: repository regime only (claims ledger status); the paper regime has no such step
            ret = False
            if regime(key) == "repository":
                for c in claims_latest.get(fu.get("claim_id") or fu.get("unit_id"), []):
                    if c.get("status") in ("superseded", "retracted") and int(c.get("quarter") or 0) <= T:
                        ret = True
            over = bool(over) or ret          # repository: formal supersession / retraction of the legacy claim is an overturn
            n_over += over
            n_chal += bool(chal) or over
            n_ret += ret
            qrec = roll["questions"].get(fu.get("question_id"), {})
            for rule in n_bel:
                n_bel[rule] += (g(qrec, "rules", rule, "belief") == fs)
            if fu.get("question_id") in perq:
                n_hb_core += 1
                n_hb_bel += (perq[fu["question_id"]].get("answer") == fs)
            if T == 20:
                fate_detail.append({"world": s, "seed": k, "unit_id": fu.get("unit_id"), "question_id": fu.get("question_id"), "core": fu.get("core"),
                                    "stated": fs, "theta": fu.get("theta"), "challenged_by_Q20": bool(chal) or bool(over),
                                    "overturned_by_Q20": bool(over), "retracted_or_superseded_by_Q20": ret,
                                    "field_belief_B1_Q20": g(qrec, "rules", "B1", "belief"), "field_belief_B2_Q20": g(qrec, "rules", "B2", "belief"),
                                    "handbook_answer_Q20": (perq.get(fu.get("question_id")) or {}).get("answer")})
        n = len(inh_false)
        fate_rows.append({"world": s, "seed": k, "snapshot": f"Q{T}", "inherited_units": len(inh), "inherited_false_units": n,
                          "challenged_by_a_live_analysis_or_unit": n_chal, "overturned_by_a_later_certified_unit": n_over,
                          "formally_retracted_or_superseded": n_ret if regime(key) == "repository" else None,
                          "still_believed_B1": n_bel["B1"], "still_believed_B2": n_bel["B2"], "still_believed_B3": n_bel["B3"],
                          "still_believed_B5": n_bel["B5"],
                          "on_core_questions": n_hb_core, "handbook_states_false_direction": n_hb_bel,
                          "share_overturned": div(n_over, n), "share_still_believed_B1": div(n_bel["B1"], n),
                          "share_handbook_states_false_of_core": div(n_hb_bel, n_hb_core)})
    # cross-check with the audit at Q20 and Q8 (original run)
    a_over = g(A20[key], "field", "certification_by_origin", "inherited", "false_units_corrected_in_run")
    my = [f for f in fate_rows if f["world"] == s and f["seed"] == k and f["snapshot"] == "Q20"][0]
    if a_over != my["overturned_by_a_later_certified_unit"]:
        log(f"NOTE {key}: overturned {my['overturned_by_a_later_certified_unit']} here vs audit {a_over} "
            f"(audit drops superseded legacy claims from its inherited set: audit n_units {g(A20[key], 'field', 'certification_by_origin', 'inherited', 'n_units')}, here {my['inherited_units']})")
    # every legacy claim the repository superseded, true or false (a true inherited unit can be superseded too)
    if regime(key) == "repository":
        for c in L.get("claims", []):
            if c.get("legacy") and c.get("status") in ("superseded", "retracted"):
                log(f"{key}: legacy claim {c['claim_id']} {c['status']} at Q{c.get('quarter')} ({c.get('event')}); its truth check {c.get('truth_check')}")
    a8 = g(A8[key], "field", "certification_by_origin", "inherited", "false_units_corrected_in_run")
    my8 = [f for f in fate_rows if f["world"] == s and f["seed"] == k and f["snapshot"] == "Q8"][0]
    log(f"{key}: inherited false {my['inherited_false_units']} (audit {g(A20[key], 'field', 'certification_by_origin', 'inherited', 'n_false')}); "
        f"overturned Q8 {my8['overturned_by_a_later_certified_unit']} (original 8q audit {a8}), Q20 {my['overturned_by_a_later_certified_unit']} (audit {a_over})")
FATE = pd.DataFrame(fate_rows)
pd.DataFrame(fate_detail).to_csv(OUT / "planted_false_literature_units.csv", index=False)

# ======================================================================== 5. planted data flaws in certified / published work
trap_rows = []
for key in RUNS:
    s, k = key
    L = LED[key]
    units = run_units(key)
    groups = {"certified_units": units,
              "certified_units_live_authored": [h for h in units if not h.get("prehistory")]}
    if regime(key) == "paper":
        groups["live_studies_all"] = [p for p in L["papers"] if not p.get("legacy") and not p.get("prehistory") and (p.get("quarter_started") or 0) >= 1]
    else:
        groups["live_studies_all"] = [x for x in L["analyses"] if int(x.get("quarter") or 0) >= 1]
    for gname, items in groups.items():
        with_trap = [x for x in items if x.get("traps")]
        planted = sum(len(x["traps"]) for x in with_trap)
        caught = sum(1 for x in with_trap for t in x["traps"] if TRAP_FLAG.get(t) not in (x.get("flags") or []))
        missed_units = [x for x in with_trap if any(TRAP_FLAG.get(t) in (x.get("flags") or []) for t in x["traps"])]
        rr = {"world": s, "seed": k, "set": gname, "n_items": len(items), "items_with_planted_flaw": len(with_trap),
              "flaws_planted": planted, "flaws_caught": caught, "flaws_missed": planted - caught,
              "share_caught": div(caught, planted), "items_with_a_missed_flaw": len(missed_units)}
        if gname.startswith("certified"):
            rr["items_with_missed_flaw_false_discovery"] = sum(1 for x in missed_units if stc(x) == "false_discovery")
        for t in TRAPS:
            pl = sum(1 for x in with_trap if t in x["traps"])
            rr[f"{t}_planted"] = pl
            rr[f"{t}_caught"] = sum(1 for x in with_trap if t in x["traps"] and TRAP_FLAG[t] not in (x.get("flags") or []))
        trap_rows.append(rr)
TRAPDF = pd.DataFrame(trap_rows)

# ======================================================================== 7. compute parity by quarter
par_rows = []
for k in (1, 2, 3):
    R, P = LED[("R-b-150", k)], LED[("S0-150", k)]
    ps = R["bsum"].get("paired_spend") or {}
    cr = cp = dr = dp = 0.0
    for q in range(1, 21):
        rq, pq = R["spend"].get(str(q), {}), P["spend"].get(str(q), {})
        cr += rq.get("compute", 0.0); cp += pq.get("compute", 0.0)
        dr += rq.get("data", 0.0); dp += pq.get("data", 0.0)
        pps = ps.get(str(q), {})
        par_rows.append({"seed": k, "quarter": q, "R_compute": rq.get("compute"), "P_compute": pq.get("compute"),
                         "R_compute_over_P": div(rq.get("compute"), pq.get("compute")),
                         "R_data": rq.get("data"), "P_data": pq.get("data"),
                         "R_cum_compute": cr, "P_cum_compute": cp, "cum_compute_ratio": div(cr, cp),
                         "R_cum_data": dr, "P_cum_data": dp, "cum_data_ratio": div(dr, dp),
                         "R_paired_spend_record_compute": pps.get("compute"), "R_paired_spend_record_data": pps.get("data"),
                         "paired_record_matches_S0_x20": (abs((pps.get("compute") or 0) - (pq.get("compute") or 0)) < 0.5
                                                          and abs((pps.get("data") or 0) - (pq.get("data") or 0)) < 0.5) if pps else None,
                         "R_allotment_compute": g(R["bsum"], "allotments", str(q), "compute"),
                         "R_allotment_data": g(R["bsum"], "allotments", str(q), "data")})
PAR = pd.DataFrame(par_rows)
for k in (1, 2, 3):
    sub = PAR[PAR.seed == k]
    log(f"parity s{k}: paired-spend record matches S0-x20 spend in {int(sub.paired_record_matches_S0_x20.fillna(False).sum())}/20 quarters; "
        f"cumulative compute ratio Q8 {sub[sub.quarter == 8].cum_compute_ratio.iloc[0]:.3f}, Q20 {sub[sub.quarter == 20].cum_compute_ratio.iloc[0]:.3f}")

# ======================================================================== 8. careers
car_rows = []
prod_rows = []
for key in RUNS:
    s, k = key
    a = A20[key]
    L = LED[key]
    cnt = g(a, "careers", "run", "counts") or {}
    cr = {"world": s, "seed": k}
    for kk, v in cnt.items():
        cr[f"n_{kk}"] = v
    cr["mean_credit_tenured"] = g(a, "careers", "run", "mean_credit_tenured")
    cr["mean_credit_tenure_exit"] = g(a, "careers", "run", "mean_credit_tenure_exit")
    # quarter split of career events (run): Q1-8 vs Q9-20
    ev = [e for e in L["events"] if e.get("event") == "career" and int(e.get("quarter", 1)) >= 1]
    for kk in ("promoted_assistant", "tenured", "promoted_full", "tenure_exit"):
        cr[f"n_{kk}_q1_8"] = sum(1 for e in ev if e.get("what") == kk and e["quarter"] <= 8)
        cr[f"n_{kk}_q9_20"] = sum(1 for e in ev if e.get("what") == kk and e["quarter"] >= 9)
    # productivity by career stage: stage from the population snapshot at the start of each four-quarter block
    pop = L["pop"]
    snap0 = "Q0" if "Q0" in pop else "seed"
    blocks = [(1, 4, snap0), (5, 8, "Q4"), (9, 12, "Q8"), (13, 16, "Q12"), (17, 20, "Q16")]
    units = run_units(key)
    if regime(key) == "paper":
        starts = [(p.get("lead"), p["quarter_started"]) for p in L["papers"] if not p.get("legacy") and not p.get("prehistory") and (p.get("quarter_started") or 0) >= 1]
    else:
        starts = [(x.get("lead"), int(x["quarter"])) for x in L["analyses"] if int(x.get("quarter") or 0) >= 1]
    py = collections.Counter(); led_n = collections.Counter(); cred = collections.Counter()
    for q0, q1, lab in blocks:
        if lab not in pop:
            log(f"WARNING {key}: no population snapshot {lab}")
            continue
        st = {x["id"]: x["stage"] for x in pop[lab]}
        for x in pop[lab]:
            if x.get("active", True):
                py[x["stage"]] += 1.0   # one person-year per active scientist per block
        for who, q in starts:
            if q0 <= q <= q1 and who in st:
                led_n[st[who]] += 1
        for h in units:
            if q0 <= _cert_q(h) <= q1:
                au = h.get("authors") or [h.get("author_id")]
                for x in au:
                    if x in st:
                        cred[st[x]] += 1.0 / len(au)
    for stg in STAGES:
        prod_rows.append({"world": s, "seed": k, "stage": stg, "active_person_years": py[stg], "studies_led": led_n[stg],
                          "studies_led_per_person_year": div(led_n[stg], py[stg]),
                          "certified_unit_credit": cred[stg], "certified_unit_credit_per_person_year": div(cred[stg], py[stg])})
    car_rows.append(cr)
CAR = pd.DataFrame(car_rows).fillna(0)
PROD = pd.DataFrame(prod_rows)

# ======================================================================== 9. herding, replication and verification activity
herd_rows = []
for key in RUNS:
    s, k = key
    L = LED[key]
    a = A20[key]
    if regime(key) == "paper":
        st = [p for p in L["papers"] if not p.get("legacy") and not p.get("prehistory") and (p.get("quarter_started") or 0) >= 1]
    else:
        st = [x for x in L["analyses"] if int(x.get("quarter") or 0) >= 1]
    qc = collections.Counter(x.get("question_id") for x in st)
    n = len(st)
    shares = np.array(list(qc.values()), dtype=float) / n if n else np.array([])
    rel = collections.Counter((x.get("relation_to_prior") or "none") for x in st)
    hr = {"world": s, "seed": k, "studies_or_analyses": n, "distinct_questions": len(qc),
          "share_on_core": div(sum(1 for x in st if x.get("core", str(x.get("question_id", "")).startswith("Q-"))), n),
          "question_hhi": float((shares ** 2).sum()) if n else None,
          "top5_question_share": float(np.sort(shares)[::-1][:5].sum()) if n else None,
          "relation_to_prior": json.dumps(dict(rel.most_common())),
          "replication_share": div(sum(v for kk, v in rel.items() if "replicat" in kk.lower()), n),
          "contradiction_share": div(sum(v for kk, v in rel.items() if "contradict" in kk.lower()), n)}
    if regime(key) == "repository":
        V = [v for v in L["verifs"] if int(v.get("quarter", 1)) >= 1]
        res = collections.Counter(v.get("result") for v in V)
        hr.update({"verifications": len(V), "reproduced": res.get("reproduced", 0), "not_reproduced": res.get("not_reproduced", 0),
                   "inconclusive": res.get("inconclusive", 0),
                   "verification_powered_share": g(a, "verification_power", "powered_share"),
                   "votes_endorse": sum(1 for v in L["votes"] if v.get("vote") == "endorse"),
                   "votes_challenge": sum(1 for v in L["votes"] if v.get("vote") == "challenge"),
                   "votes_abstain": sum(1 for v in L["votes"] if v.get("vote") == "abstain"),
                   "verification_offers": sum(1 for v in L["votes"] if v.get("vote") == "offer"),
                   "herding_cascade_slope": g(a, "failure_indices", "herding", "cascade_slope"),
                   "unverified_endorsement_share": g(a, "failure_indices", "herding", "unverified_endorsement_share"),
                   "late_vote_accuracy": g(a, "failure_indices", "herding", "late_vote_accuracy"),
                   "early_vote_accuracy": g(a, "failure_indices", "herding", "early_vote_accuracy"),
                   "lapsed_true_claims": g(a, "failure_indices", "free_riding", "lapsed_true_claims"),
                   "single_deposit_share_of_endorsed": g(a, "failure_indices", "shared_data", "single_deposit_share_of_endorsed")})
    else:
        hr.update({"referee_reports": g(a, "author", "certifier_accuracy", "n_reviews"),
                   "reviewer_kappa": g(a, "author", "certifier_accuracy", "kappa_recommendation"),
                   "accept_rate_mean": g(a, "realism", "accept_rate_mean")})
    herd_rows.append(hr)
HERD = pd.DataFrame(herd_rows)

# ======================================================================== interim (Q8, original runs) vs final (Q20)
INTERIM = [
    ("handbook_accuracy_final", lambda a: g(a, "handbook", f"Q{a['horizon_quarters']}", "accuracy")),
    ("handbook_correct_final", lambda a: g(a, "handbook", f"Q{a['horizon_quarters']}", "correct_answers")),
    ("handbook_answered_final", lambda a: g(a, "handbook", f"Q{a['horizon_quarters']}", "n_answered")),
    ("handbook_confidently_wrong_final", lambda a: g(a, "handbook", f"Q{a['horizon_quarters']}", "confidently_wrong")),
    ("handbook_brier_final", lambda a: g(a, "handbook", f"Q{a['horizon_quarters']}", "brier")),
    ("new_certified_units", lambda a: g(a, "field", "certification", "n_certified_units")),
    ("new_units_true", lambda a: g(a, "field", "certification_by_origin", "new_in_run", "n_true")),
    ("new_units_false", lambda a: g(a, "field", "certification_by_origin", "new_in_run", "n_false")),
    ("new_units_missed_effect", lambda a: g(a, "field", "certification_by_origin", "new_in_run", "n_missed_effect")),
    ("new_units_true_share", lambda a: g(a, "field", "certification", "true_share_of_certified_units")),
    ("published_in_run_total_P", lambda a: g(a, "timing", "published_in_run_total")),
    ("live_authored_published_P", lambda a: g(a, "timing", "live_authored", "published")),
    ("net_knowledge_B1", lambda a: g(a, "field", "net_knowledge", "B1")),
    ("net_knowledge_B2", lambda a: g(a, "field", "net_knowledge", "B2")),
    ("net_knowledge_B5", lambda a: g(a, "field", "net_knowledge", "B5")),
    ("field_accuracy_B1", lambda a: g(a, "field", "accuracy", "B1", "accuracy")),
    ("field_accuracy_B2", lambda a: g(a, "field", "accuracy", "B2", "accuracy")),
    ("core_covered_share_run_units", lambda a: g(a, "field", "coverage", "core_covered_share")),
    ("inherited_false_overturned", lambda a: g(a, "field", "certification_by_origin", "inherited", "false_units_corrected_in_run")),
    ("compute_tariff_tokens", lambda a: g(a, "budget", "total", "compute")),
    ("data_observations", lambda a: g(a, "budget", "total", "data")),
    ("compute_ratio_to_paired_paper", lambda a: g(a, "budget", "per_unit", "paired_world", "compute_ratio")),
    ("data_ratio_to_paired_paper", lambda a: g(a, "budget", "per_unit", "paired_world", "data_ratio")),
    ("median_quarters_to_first_certification", lambda a: g(a, "field", "certification", "median_quarters_to_first_certification")),
    ("tenure_exits", lambda a: g(a, "careers", "run", "counts", "tenure_exit") or 0),
    ("promoted_full", lambda a: g(a, "careers", "run", "counts", "promoted_full") or 0),
    ("tenured", lambda a: g(a, "careers", "run", "counts", "tenured") or 0),
]
int_rows = []
for key in RUNS:
    for name, fn in INTERIM:
        v8, v20 = fn(A8[key]), fn(A20[key])
        int_rows.append({"world": key[0], "seed": key[1], "metric": name, "Q8_interim": v8, "Q20_final": v20,
                         "change": (v20 - v8) if isnum(v8) and isnum(v20) else None})
INT = pd.DataFrame(int_rows)

# ======================================================================== paired contrasts
con_rows = []
CON_METRICS = ["certified_units", "units_true", "units_false_discovery", "units_missed_effect", "true_share", "papers_published_in_run",
               "hb_Q20_accuracy", "hb_Q20_correct", "hb_accuracy_change_Q0_Q20", "hb_Q20_confidently_wrong", "hb_Q20_brier",
               "net_knowledge_B1_Q20", "net_knowledge_B2_Q20", "compute_tariff_tokens", "data_observations",
               "studies_started_live", "support_share"]
for treat, base, seeds in (("S0P-150", "S0-150", (1, 2)), ("R-b-150", "S0-150", (1, 2, 3))):
    for m in CON_METRICS:
        for k in seeds:
            t = RES[(RES.world == treat) & (RES.seed == k)][m].iloc[0]
            b = RES[(RES.world == base) & (RES.seed == k)][m].iloc[0]
            con_rows.append({"contrast": f"{treat} - {base}", "metric": m, "seed": k, "treatment": t, "baseline": b,
                             "difference": (t - b) if isnum(t) and isnum(b) else None})
for m in ["n_promoted_assistant", "n_tenured", "n_promoted_full", "n_tenure_exit", "n_not_promoted_full", "mean_credit_tenured", "mean_credit_tenure_exit"]:
    for k in (1, 2):
        t = CAR[(CAR.world == "S0P-150") & (CAR.seed == k)][m].iloc[0] if m in CAR else None
        b = CAR[(CAR.world == "S0-150") & (CAR.seed == k)][m].iloc[0] if m in CAR else None
        con_rows.append({"contrast": "S0P-150 - S0-150", "metric": m, "seed": k, "treatment": t, "baseline": b,
                         "difference": (t - b) if isnum(t) and isnum(b) else None})
CON = pd.DataFrame(con_rows)


# ======================================================================== world summaries
def world_summary(df, id_cols=("world", "seed"), extra_group=()):
    out = []
    num = [c for c in df.columns if c not in id_cols and c not in extra_group and pd.api.types.is_numeric_dtype(df[c])]
    for keys, sub in df.groupby(["world"] + list(extra_group), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in num:
            v = [float(x) for x in sub[c] if isnum(x)]
            row = dict(zip(["world"] + list(extra_group), keys))
            row.update({"metric": c, "n_seeds": len(v), "mean": float(np.mean(v)) if v else None, "min": min(v) if v else None,
                        "max": max(v) if v else None,
                        "per_seed": "; ".join(f"s{int(s_)}={_f(x)}" for s_, x in zip(sub["seed"], sub[c]))})
            out.append(row)
    return pd.DataFrame(out)


def _f(x):
    if not isnum(x):
        return "n/a"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if x.is_integer():
        return str(int(x))
    return f"{x:.3f}"


RES.to_csv(OUT / "results_by_run.csv", index=False)
world_summary(RES.drop(columns=["regime"])).to_csv(OUT / "results_by_world.csv", index=False)
HB = []
for key in RUNS:
    for q in SNAPS:
        sc = LED[key]["hb"].get(q) or {}
        HB.append({"world": key[0], "seed": key[1], "snapshot": q, "quarter": int(q[1:]), "accuracy": sc.get("accuracy"),
                   "answered_of_48": sc.get("n_answered"), "scored": sc.get("n_scored"), "correct": sc.get("correct_answers"),
                   "confidently_wrong": sc.get("confidently_wrong"), "brier": sc.get("brier"),
                   "inter_writer_agreement": sc.get("inter_writer_agreement"),
                   "agreement_B1": g(sc, "agreement_with_rules", "B1"), "agreement_B2": g(sc, "agreement_with_rules", "B2")})
HBDF = pd.DataFrame(HB)
HBDF.to_csv(OUT / "handbook_trajectory.csv", index=False)
world_summary(HBDF.drop(columns=["quarter"]), extra_group=("snapshot",)).to_csv(OUT / "handbook_trajectory_by_world.csv", index=False)
PAR.to_csv(OUT / "parity_by_quarter.csv", index=False)
pd.DataFrame(per_q_rows).to_csv(OUT / "knowledge_by_quarter.csv", index=False)
FATE.to_csv(OUT / "planted_false_literature.csv", index=False)
world_summary(FATE, extra_group=("snapshot",)).to_csv(OUT / "planted_false_literature_by_world.csv", index=False)
TRAPDF.to_csv(OUT / "data_flaws.csv", index=False)
world_summary(TRAPDF, extra_group=("set",)).to_csv(OUT / "data_flaws_by_world.csv", index=False)
CAR.to_csv(OUT / "careers.csv", index=False)
world_summary(CAR).to_csv(OUT / "careers_by_world.csv", index=False)
PROD.to_csv(OUT / "productivity_by_stage.csv", index=False)
world_summary(PROD, extra_group=("stage",)).to_csv(OUT / "productivity_by_stage_by_world.csv", index=False)
HERD.to_csv(OUT / "herding_and_verification.csv", index=False)
world_summary(HERD).to_csv(OUT / "herding_and_verification_by_world.csv", index=False)
INT.to_csv(OUT / "interim_q8_vs_final_q20.csv", index=False)
world_summary(INT.pivot_table(index=["world", "seed"], columns="metric", values="change", aggfunc="first").reset_index()).to_csv(
    OUT / "interim_change_by_world.csv", index=False)
CON.to_csv(OUT / "paired_contrasts.csv", index=False)

# ======================================================================== headline table (markdown, for the note)
WS = world_summary(RES.drop(columns=["regime"]))


def cell(metric, world, nd=3):
    r = WS[(WS.metric == metric) & (WS.world == world)]
    if r.empty or r["n_seeds"].iloc[0] == 0:
        return "n/a"
    m, lo, hi = r["mean"].iloc[0], r["min"].iloc[0], r["max"].iloc[0]
    fm = lambda x: (f"{x:,.0f}" if abs(x) >= 1000 else (f"{x:.{nd}f}" if nd else f"{x:.1f}" if not float(x).is_integer() else f"{int(x)}"))
    return f"{fm(m)} ({fm(lo)} to {fm(hi)})"


def fcell(metric, world, snapshot, nd=3, df=None):
    d = world_summary(FATE, extra_group=("snapshot",)) if df is None else df
    r = d[(d.metric == metric) & (d.world == world) & (d.snapshot == snapshot)]
    if r.empty or r["n_seeds"].iloc[0] == 0:
        return "n/a"
    m, lo, hi = r["mean"].iloc[0], r["min"].iloc[0], r["max"].iloc[0]
    return f"{m:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})"


FWS = world_summary(FATE, extra_group=("snapshot",))
TWS = world_summary(TRAPDF, extra_group=("set",))
HEAD = [
    ("Knowledge units certified in run (focal; P published, R endorsed)", "certified_units", 0),
    ("  of which certified Q9-20", "certified_units_q9_20", 0),
    ("Papers published in run (P; all incl. warm-start pipeline)", "papers_published_in_run", 0),
    ("  of which written by live authors (P)", "papers_published_live_authored", 0),
    ("Certified units per 1M compute tariff tokens", "certified_units_per_1M_compute", 3),
    ("True share of certified units", "true_share", 3),
    ("False discoveries among certified units (n)", "units_false_discovery", 0),
    ("Missed effects among certified units (n)", "units_missed_effect", 0),
    ("Certified units not scorable (no estimate) (n)", "units_unknown", 0),
    ("True share of all live focal analyses (pre-certification)", "focal_analyses_live_true_share", 3),
    ("Handbook accuracy Q0", "hb_Q0_accuracy", 3),
    ("Handbook accuracy Q8", "hb_Q8_accuracy", 3),
    ("Handbook accuracy Q20", "hb_Q20_accuracy", 3),
    ("Handbook correct answers Q20 (of 48)", "hb_Q20_correct", 0),
    ("Handbook answered Q20 (of 48)", "hb_Q20_answered", 0),
    ("Handbook confidently wrong Q20", "hb_Q20_confidently_wrong", 0),
    ("Handbook Brier Q20", "hb_Q20_brier", 3),
    ("Net knowledge B1, Q0 -> Q20", None, 0),
    ("Median quarters, study start to certification (live-authored)", "start_to_certification_median_q_live", 1),
    ("Median months, submission to acceptance (P)", "submission_to_acceptance_median_months", 1),
    ("Compute used (tariff tokens)", "compute_tariff_tokens", 0),
    ("Data used (observations)", "data_observations", 0),
    ("Compute per correct Q20 handbook answer", "compute_per_correct_handbook_answer", 0),
    ("Data per correct Q20 handbook answer", "data_per_correct_handbook_answer", 0),
    ("Compute ratio to paired paper run (R)", "compute_ratio_to_paired_paper", 3),
    ("Data ratio to paired paper run (R)", "data_ratio_to_paired_paper", 3),
]
md = ["| Metric | " + " | ".join(WORLD_LABEL[w] for w in WORLDS) + " |", "|---|---|---|---|"]
for lab, m, nd in HEAD:
    if m is None:
        cells = []
        for w in WORLDS:
            sub = RES[RES.world == w]
            cells.append("; ".join(f"s{s_}: {int(a_)}->{int(b_)}" for s_, a_, b_ in zip(sub.seed, sub.net_knowledge_B1_Q0, sub.net_knowledge_B1_Q20)))
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
        continue
    md.append(f"| {lab} | " + " | ".join(cell(m, w, nd) for w in WORLDS) + " |")
for lab, m, snap in (("Inherited false units (n)", "inherited_false_units", "Q20"),
                     ("  overturned by a later certified unit, by Q8", "overturned_by_a_later_certified_unit", "Q8"),
                     ("  overturned by a later certified unit, by Q20", "overturned_by_a_later_certified_unit", "Q20"),
                     ("  of which formally superseded (R only), by Q20", "formally_retracted_or_superseded", "Q20"),
                     ("  challenged by any live analysis, by Q8", "challenged_by_a_live_analysis_or_unit", "Q8"),
                     ("  challenged by any live analysis, by Q20", "challenged_by_a_live_analysis_or_unit", "Q20"),
                     ("  still believed under B1 at Q8", "still_believed_B1", "Q8"),
                     ("  still believed under B1 at Q20", "still_believed_B1", "Q20")):
    md.append(f"| {lab} | " + " | ".join(fcell(m, w, snap, 1, FWS) for w in WORLDS) + " |")
for lab, m in (("Planted data flaws in certified units (n)", "flaws_planted"), ("  share caught", "share_caught")):
    cells = []
    for w in WORLDS:
        r = TWS[(TWS.metric == m) & (TWS.world == w) & (TWS.set == "certified_units")]
        cells.append("n/a" if r.empty or r["n_seeds"].iloc[0] == 0 else f"{r['mean'].iloc[0]:.{3 if m == 'share_caught' else 1}f} ({r['min'].iloc[0]:.{3 if m == 'share_caught' else 0}f} to {r['max'].iloc[0]:.{3 if m == 'share_caught' else 0}f})")
    md.append(f"| {lab} | " + " | ".join(cells) + " |")
CW = world_summary(CAR)
for lab, m in (("Promotions to assistant (run)", "n_promoted_assistant"), ("Tenured (run)", "n_tenured"),
               ("Promoted to full (run)", "n_promoted_full"), ("Tenure exits (run)", "n_tenure_exit")):
    cells = []
    for w in WORLDS:
        r = CW[(CW.metric == m) & (CW.world == w)]
        cells.append("n/a" if r.empty else f"{r['mean'].iloc[0]:.1f} ({r['min'].iloc[0]:.0f} to {r['max'].iloc[0]:.0f})")
    md.append(f"| {lab} | " + " | ".join(cells) + " |")
(OUT / "headline_table.md").write_text("Mean (min to max) across seeds; S0 n=3, S0P n=2, R n=3. Generated by compute_results.py.\n\n" + "\n".join(md) + "\n", encoding="utf-8")

# ======================================================================== figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
    log("figure font:", fam)
    FIG.mkdir(parents=True, exist_ok=True)
    COL = {"S0-150": "#2a6fd6", "S0P-150": "#e0662f", "R-b-150": "#1a9e6f"}
    MK = {"S0-150": "o", "S0P-150": "s", "R-b-150": "^"}
    GRID = dict(color="#e3e3e0", lw=0.7)

    # F1 handbook trajectory
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for w in WORLDS:
        for metric, ax in (("accuracy", axes[0]), ("correct", axes[1])):
            sub = HBDF[HBDF.world == w]
            for k_, ss in sub.groupby("seed"):
                ax.plot(ss.quarter, ss[metric], color=COL[w], alpha=0.3, lw=1, marker=MK[w], ms=3)
            mean = sub.groupby("quarter")[metric].mean()
            ax.plot(mean.index, mean.values, color=COL[w], lw=2.4, marker=MK[w], ms=6, label=f"{WORLD_LABEL[w]} (mean)")
    axes[0].set_ylabel("Accuracy of scored answers"); axes[1].set_ylabel("Correct answers (of 48 questions)")
    for ax in axes:
        ax.set_xticks([0, 4, 8, 16, 20]); ax.set_xlabel("Quarter"); ax.grid(axis="y", **GRID)
        ax.axvline(8, color="#999", lw=0.8, ls=":")
    axes[0].set_title("Handbook accuracy"); axes[1].set_title("Handbook correct answers")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, 0.005, "Thin lines: single seeds. Dotted line: end of the original 8-quarter runs.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig1_handbook_trajectory.png", dpi=160); plt.close(fig)

    # F2 knowledge units over time (cumulative certified units)
    KQ = pd.DataFrame(per_q_rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for w in WORLDS:
        sub = KQ[KQ.world == w].sort_values("quarter")
        cums = []
        for k_, ss in sub.groupby("seed"):
            c = ss.certified_units.cumsum().values
            cums.append(c)
            ax.plot(ss.quarter, c, color=COL[w], alpha=0.3, lw=1)
        ax.plot(range(1, 21), np.mean(cums, axis=0), color=COL[w], lw=2.4, marker=MK[w], ms=4, label=WORLD_LABEL[w])
    ax.set_xlabel("Quarter"); ax.set_ylabel("Cumulative certified units (focal)"); ax.grid(axis="y", **GRID)
    ax.axvline(8, color="#999", lw=0.8, ls=":"); ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.set_title("Knowledge certified in the run: P published focal results, R endorsed focal claims")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_knowledge_units_over_time.png", dpi=160); plt.close(fig)

    # F3 true/false composition
    fig, ax = plt.subplots(figsize=(10, 4.2))
    labels, xs = [], []
    x = 0
    for w in WORLDS:
        for _, r_ in RES[RES.world == w].iterrows():
            b = 0
            for col, c_, lab in (("units_true_effect", "#1a9e6f", "True effect"), ("units_correct_null", "#8fd3b6", "Correct null"),
                                 ("units_missed_effect", "#f2b13c", "Missed effect (false negative)"),
                                 ("units_false_discovery", "#d43d3d", "False discovery (false positive)"),
                                 ("units_unknown", "#bdbdb8", "Not scorable (no estimate)")):
                v = r_[col]
                ax.bar(x, v, bottom=b, color=c_, width=0.7, label=lab if x == 0 else None, edgecolor="white", lw=0.5)
                b += v
            ax.text(x, b + 0.3, f"{int(b)}", ha="center", fontsize=8)
            labels.append(f"{w.split('-')[0]}\ns{r_['seed']}"); xs.append(x)
            x += 1
        x += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Certified units (Q1-20)"); ax.grid(axis="y", **GRID)
    ax.set_title("What the certified units state, against the hidden truth")
    ax.set_ylim(0, max(RES.certified_units) * 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "fig3_truth_composition.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # F4 planted-false-literature fate
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xs, labels = [], []
    x = 0
    for w in WORLDS:
        for k_ in sorted(FATE[FATE.world == w].seed.unique()):
            for snap, off, hatch in (("Q8", -0.2, None), ("Q20", 0.2, None)):
                r_ = FATE[(FATE.world == w) & (FATE.seed == k_) & (FATE.snapshot == snap)].iloc[0]
                n = r_.inherited_false_units
                over = r_.overturned_by_a_later_certified_unit
                chal = r_.challenged_by_a_live_analysis_or_unit - over
                rest = n - over - chal
                alpha = 0.55 if snap == "Q8" else 1.0
                axes[0].bar(x + off, over, width=0.38, color="#1a9e6f", alpha=alpha, label="Overturned (later certified unit)" if x == 0 and snap == "Q20" else None)
                axes[0].bar(x + off, chal, bottom=over, width=0.38, color="#f2b13c", alpha=alpha, label="Challenged by a live analysis only (not overturned)" if x == 0 and snap == "Q20" else None)
                axes[0].bar(x + off, rest, bottom=over + chal, width=0.38, color="#c9c9c4", alpha=alpha, label="Unchallenged" if x == 0 and snap == "Q20" else None)
                axes[1].bar(x + off, r_.share_still_believed_B1, width=0.38, color=COL[w], alpha=alpha)
            labels.append(f"{w.split('-')[0]}\ns{k_}"); xs.append(x)
            x += 1
        x += 0.5
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.grid(axis="y", **GRID)
    axes[0].set_ylabel("Inherited false units"); axes[0].set_title("Fate of the inherited false units (left bar Q8, right bar Q20)")
    axes[1].set_ylabel("Share still believed (B1 vote count)"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Inherited false units still believed by the field (Q8 light, Q20 dark)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "fig4_planted_false_literature.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # F5 compute parity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k_, ls in zip((1, 2, 3), ("-", "--", ":")):
        sub = PAR[PAR.seed == k_]
        axes[0].plot(sub.quarter, sub.R_compute_over_P, color=COL["R-b-150"], ls=ls, marker="o", ms=3, label=f"seed {k_}")
        axes[1].plot(sub.quarter, sub.cum_compute_ratio, color=COL["R-b-150"], ls=ls, lw=2, label=f"compute, seed {k_}")
        axes[1].plot(sub.quarter, sub.cum_data_ratio, color="#7a5cc4", ls=ls, lw=2, label=f"data, seed {k_}")
    for ax in axes:
        ax.axhline(1, color="#555", lw=0.8); ax.set_xlabel("Quarter"); ax.grid(axis="y", **GRID); ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.axvline(8, color="#999", lw=0.8, ls=":")
    axes[0].set_ylabel("R compute / paired S0 compute (that quarter)"); axes[0].set_title("Compute parity per quarter")
    axes[1].set_ylabel("Cumulative ratio, R / paired S0"); axes[1].set_title("Cumulative compute and data ratios")
    axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.36))
    fig.tight_layout(); fig.savefig(FIG / "fig5_compute_parity.png", dpi=160); plt.close(fig)

    # F6 careers: pressure vs not (paired by seed), plus R for reference
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cats = [("n_promoted_assistant", "Promoted to\nassistant"), ("n_tenured", "Tenured"), ("n_promoted_full", "Promoted\nto full"),
            ("n_tenure_exit", "Tenure\nexit")]
    order = [("S0-150", 1), ("S0P-150", 1), ("S0-150", 2), ("S0P-150", 2)]
    wdt = 0.2
    for i, (w, k_) in enumerate(order):
        r_ = CAR[(CAR.world == w) & (CAR.seed == k_)].iloc[0]
        vals = [r_.get(c, 0) for c, _ in cats]
        axes[0].bar(np.arange(len(cats)) + (i - 1.5) * wdt, vals, width=wdt, color=COL[w], alpha=1.0 if k_ == 1 else 0.55,
                    label=f"{WORLD_LABEL[w].split(' (')[0]}, seed {k_}", edgecolor="white", lw=0.5)
    axes[0].set_xticks(range(len(cats))); axes[0].set_xticklabels([c[1] for c in cats], fontsize=8)
    axes[0].set_ylabel("Career events, quarters 1-20"); axes[0].set_title("Career outcomes, pressure vs baseline (paired by seed)")
    axes[0].legend(frameon=False, fontsize=7.5); axes[0].grid(axis="y", **GRID)
    for i, w in enumerate(WORLDS):
        sub = PROD[PROD.world == w].groupby("stage")["studies_led_per_person_year"].mean().reindex(STAGES)
        axes[1].bar(np.arange(len(STAGES)) + (i - 1) * 0.27, sub.values, width=0.27, color=COL[w], label=WORLD_LABEL[w])
    axes[1].set_xticks(range(len(STAGES))); axes[1].set_xticklabels(STAGES)
    axes[1].set_ylabel("Studies led per active person-year"); axes[1].set_title("Productivity by career stage (seed mean)")
    axes[1].legend(frameon=False, fontsize=7.5); axes[1].grid(axis="y", **GRID)
    fig.tight_layout(); fig.savefig(FIG / "fig6_careers.png", dpi=160); plt.close(fig)
    log("figures written:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
except Exception as e:  # noqa: BLE001
    import traceback
    log("FIGURES FAILED:", repr(e))
    traceback.print_exc()

(OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("done")
