"""The knowledge audit: deterministic, from the ledger alone, no agent calls. Field, theory and author levels,
the repository failure indices, regime pairing by seed and Monte Carlo aggregation.

`run_audit(run_dir)` reads runs/<scenario>/<seed>/<mode>/ledger/* and writes audit/audit.json + audit/summary.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

from .belief import all_rules, is_correct, score_belief, truth_class
from .util import read_json, read_jsonl, write_json, write_text

RULES = ("B1", "B2", "B3", "B5")
TRUE_CHECKS = {"true_discovery", "correct_null", "true_discovery_negligible", "correct_null_negligible"}

# One definition of a certified unit for both regimes (an early test run showed that the repository audit counted the
# starting-literature claims as certified units while the paper audit counted the hypothesis rows of published papers):
#   a certified unit is a certified estimate of a question's FOCAL relation: in the paper regime the focal hypothesis of a
#   published paper, in the repository regime an endorsed claim on the focal relation (same iv, dv, no moderator);
#   the starting literature (the legacy library and papers published in the warm-start pre-history, certified at quarter <= 0)
#   is INHERITED and never counted among the run's certified units; it is reported in its own table, with the share of its
#   false units the run corrected. Non-focal certified rows (moderations, secondary outcomes) are reported as a count only.
UNIT_DEFINITION = ("certified estimate of a question's focal relation, certified in the run (quarter >= 1): P = the focal hypothesis of a "
                   "published paper, R = an endorsed claim on the focal relation; a unit states the direction of its estimate when p < .05 "
                   "and no effect otherwise, in both regimes (not the author's support verdict); the starting literature is reported separately as inherited")


def _stated_truth_check(h: dict) -> str | None:
    """The truth check on what the unit STATES: the direction of a significant estimate (p < .05), else no effect, scored
    against theta, in both regimes. The paper regime's author verdict ('supported' = the predicted direction) read a
    significant wrong-sign result as 'no effect', the repository regime's claim sign read it as a claim of that sign, so the
    same legacy study was 'missed_effect' in one audit and 'false_discovery' in the other."""
    from .scorer import claim_sign, truth_check
    theta = h.get("theta")
    if theta is None:
        return h.get("truth_check")
    sign = claim_sign(h)
    return truth_check("supported" if sign != "none" else "not_supported", h.get("estimate"), float(theta))


def _is_focal(h: dict) -> bool:
    if h.get("moderator"):
        return False
    kind = h.get("kind")
    if kind:
        return kind == "focal"
    return h.get("hyp_id") in (None, "H1")   # LLM claims.json rows carry no kind; H1 is the focal hypothesis by contract


def _dedupe(hyps: list[dict]) -> list[dict]:
    """Runs made with an early engine version re-appended the legacy rows on every resume; keep the
    first row per legacy unit id."""
    out, seen = [], set()
    for h in hyps:
        if h.get("legacy") and h.get("certified"):
            key = h.get("unit_id")
            if key in seen:
                continue
            seen.add(key)
        out.append(h)
    return out


def _fill_legacy_truth(hyps: list[dict], rd: Path, questions: list[dict]) -> list[dict]:
    """Runs made with an early engine version wrote the paper regime's legacy rows without a truth check; compute it from legacy/papers.json
    (hidden theta per legacy paper) with the same rule the engine now applies at setup."""
    from .scorer import truth_check
    lp_path = rd / "legacy" / "papers.json"
    if not lp_path.exists() or all(h.get("truth_check") is not None for h in hyps if h.get("legacy")):
        return hyps
    legacy = {p["paper_id"]: p for p in read_json(lp_path)}
    q_of = {q["id"]: q for q in questions}
    for h in hyps:
        if not h.get("legacy") or h.get("truth_check") is not None:
            continue
        lp = legacy.get(h.get("paper_id") or str(h.get("unit_id", "")).split(":")[0])
        q = q_of.get(h.get("question_id"))
        if not lp or not q:
            continue
        if h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator"):
            theta = float(lp["hidden"]["theta"])
        elif h.get("moderator"):
            theta = 0.0
        else:
            theta = float(lp["hidden"].get("theta_secondary", 0.0)) if h["dv"] == q.get("secondary_dv_var") else 0.0
        h["theta"], h["truth_check"] = theta, truth_check(h.get("verdict"), h.get("estimate"), theta)
    return hyps


def _gini(x) -> float | None:
    x = np.sort(np.asarray([v for v in x if v is not None], dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return None
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def _spearman(a, b) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(sstats.spearmanr(a, b).correlation)


def _auc(scores, labels) -> float | None:
    s, l = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    if len(s) < 2 or l.all() or (~l).all():
        return None
    ranks = sstats.rankdata(s)
    n1, n0 = l.sum(), (~l).sum()
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_audit(run_dir: Path, quiet: bool = True) -> dict:
    rd = Path(run_dir)
    run = read_json(rd / "run.json")
    cfg = _load_cfg(rd)
    regime = run["regime"]
    H = int(run["horizon_quarters"])
    truth = read_json(rd / "landscape" / "truth.json")
    questions = read_json(rd / "landscape" / "questions.json")
    fringe = read_json(rd / "landscape" / "fringe.json")
    core_ids = [q["id"] for q in questions]
    fam_of = {q["id"]: q["family"] for q in questions + fringe}
    pred_of = {q["id"]: q["prediction_sign"] for q in questions + fringe}
    theta_of = {qid: truth["questions"].get(qid, truth["fringe"].get(qid, {})).get("theta", 0.0) for qid in fam_of}
    hyps = _fill_legacy_truth(_dedupe(read_jsonl(rd / "ledger" / "hypotheses.jsonl")), rd, questions + fringe)
    for h in hyps:   # the certification measures score what a unit states; `truth_check` keeps the author-verdict reading
        h["truth_check_stated"] = _stated_truth_check(h)
    rollups = {int(p.stem.split("Q")[1]): read_json(p) for p in (rd / "ledger").glob("questions_Q*.json")}
    final = rollups[max(rollups)] if rollups else {"questions": {}}
    first = rollups.get(min(rollups)) if rollups else None
    all_events = read_jsonl(rd / "ledger" / "events.jsonl")
    events = [e for e in all_events if int(e.get("quarter", 1)) >= 1]   # the warm-start pre-history (quarters <= 0) is not the run's work
    # the budget tallies are rebuilt from the ledger (scoring from files) and compared with the engine's summary
    from .budget import totals_from_ledger
    summary = read_json(rd / "budget" / "summary.json") if (rd / "budget" / "summary.json").exists() else {}
    led = totals_from_ledger(rd)
    budget = dict(summary)
    budget["summary_total"] = summary.get("total")
    budget["total"] = {"data": led["data"], "compute": led["compute"], "realized_tokens": led["realized_tokens"]}
    budget["calls"] = {k: led[k] for k in ("calls", "calls_failed", "calls_not_debited", "prehistory_calls")}
    budget["prehistory_data"] = led["prehistory_data"]
    budget["by_class"] = led["by_class"]
    budget["summary_stale"] = bool(summary) and (abs(float((summary.get("total") or {}).get("compute", 0.0)) - led["compute"]) > 1e-6
                                                 or abs(float((summary.get("total") or {}).get("data", 0.0)) - led["data"]) > 1e-6)
    budget["deferrals_by_binding"] = led.get("deferrals_by_binding")
    out = {"run_id": run["run_id"], "scenario": run["scenario"], "seed": run["seed"], "mode": run["mode"], "regime": regime,
           "horizon_quarters": H, "population": run["population"], "career_pressure": run.get("career_pressure", "none")}
    out["field"] = field_level(final, first, rollups, core_ids, theta_of, hyps, events, budget, regime, H, rd)
    out["theory"] = theory_level(final, core_ids, fam_of, pred_of, theta_of, truth.get("q_T", {}), hyps, events, regime, rd)
    out["author"] = author_level(hyps, rd, regime)
    out["handbook"] = handbook_scores(rd)
    out["realism"] = realism_checks(hyps, events, regime)
    out["timing"] = timing_stats(rd, H, regime)
    out["careers"] = career_stats(all_events)
    out["call_groups"] = call_group_correlation(rd, regime)
    if regime == "repository":
        out["failure_indices"] = failure_indices(rd, theta_of)
        out["verification_power"] = verification_power_stats(rd)
    # data and compute used per unit of knowledge, beside the paired world's spend
    budget["per_unit"] = budget_per_unit(budget, out["field"], out["handbook"], rd)
    out["budget"] = budget
    (rd / "audit").mkdir(exist_ok=True)
    write_json(rd / "audit" / "audit.json", out)
    write_text(rd / "audit" / "summary.md", summary_md(out))
    if not quiet:
        print(summary_md(out))
    return out


def _load_cfg(rd: Path) -> dict:
    import yaml
    from .util import read_text
    return yaml.safe_load(read_text(rd / "config.resolved.yaml"))


# ---------------------------------------------------------------------- field level
def field_level(final, first, rollups, core_ids, theta_of, hyps, events, budget, regime, H, rd) -> dict:
    qs = {k: v for k, v in final["questions"].items() if k in core_ids}
    covered = [k for k, v in qs.items() if v["n_run_units"] > 0]
    covered3 = [k for k, v in qs.items() if v["n_run_units"] >= 3]
    fringe_cert = [k for k, v in final["questions"].items() if k not in core_ids and v["n_units"] > 0]
    promoted = [e["question_id"] for e in events if e.get("event") == "fringe_promoted"]
    acc, nulls, calib, net, net_cont = {}, {}, {}, {}, {}
    net0 = {}
    for rule in RULES:
        c = [v["scored"][rule]["correct"] for v in qs.values() if v["n_units"] > 0]
        decided = [x for x in c if x is not None]
        acc[rule] = {"covered": len(c), "accuracy": (sum(decided) / len(decided)) if decided else None, "unsettled": c.count(None),
                     "sign_accuracy": _sign_acc(qs, rule), "null_accuracy": _null_acc(qs, rule)}
        nq = [v for v in qs.values() if v["theta"] == 0.0]
        nulls[rule] = {"n_null_questions": len(nq),
                       "believed_null": sum(1 for v in nq if v["rules"][rule]["belief"] == "null") / len(nq) if nq else None,
                       "believed_effect": sum(1 for v in nq if v["rules"][rule]["belief"] in ("positive", "negative")) / len(nq) if nq else None,
                       "contested": sum(1 for v in nq if v["rules"][rule]["belief"] == "contested") / len(nq) if nq else None,
                       "uncovered": sum(1 for v in nq if v["rules"][rule]["belief"] == "uncovered") / len(nq) if nq else None}
        br = [v["scored"][rule]["brier"] for v in qs.values() if v["scored"][rule]["brier"] is not None]
        confs = [(v["scored"][rule]["confidence"], 1.0 if v["scored"][rule]["correct"] else 0.0) for v in qs.values() if v["scored"][rule]["correct"] is not None]
        slope = None
        if len(confs) >= 5 and np.std([c[0] for c in confs]) > 0:
            slope = float(np.polyfit([c[0] for c in confs], [c[1] for c in confs], 1)[0])
        calib[rule] = {"brier": float(np.mean(br)) if br else None, "slope": slope, "n": len(confs),
                       "reliability": _reliability_table(confs)}
        net[rule] = sum(1 if v["scored"][rule]["correct"] else (-1 if v["scored"][rule]["correct"] is False else 0)
                        for v in qs.values() if v["n_units"] > 0 and v["scored"][rule]["confidence"] >= 0.7)
        net_cont[rule] = float(sum(v["scored"][rule]["confidence"] * (1 if v["scored"][rule]["correct"] else (-1 if v["scored"][rule]["correct"] is False else 0))
                                   for v in qs.values() if v["n_units"] > 0))
        if first:
            fq = {k: v for k, v in first["questions"].items() if k in core_ids}
            net0[rule] = sum(1 if v["scored"][rule]["correct"] else (-1 if v["scored"][rule]["correct"] is False else 0)
                             for v in fq.values() if v["n_units"] > 0 and v["scored"][rule]["confidence"] >= 0.7)
    # precision (B2)
    prec = []
    for v in qs.values():
        if v["n_units"] >= 2 and v["rules"]["B2"].get("estimate") is not None:
            prec.append({"abs_err": abs(v["rules"]["B2"]["estimate"] - v["theta"]), "width": v["rules"]["B2"]["width"],
                         "covers": v["rules"]["B2"]["lo"] <= v["theta"] <= v["rules"]["B2"]["hi"]})
    precision = {"n_questions": len(prec), "median_abs_error": float(np.median([p["abs_err"] for p in prec])) if prec else None,
                 "median_width": float(np.median([p["width"] for p in prec])) if prec else None,
                 "coverage": float(np.mean([p["covers"] for p in prec])) if prec else None}
    # certification accuracy and the file drawer: one unit definition for both regimes (UNIT_DEFINITION)
    run_h = [h for h in hyps if not h.get("legacy")]
    cert_rows = [h for h in run_h if h.get("certified")]              # every certified row of the run (all kinds)
    cert = [h for h in cert_rows if _is_focal(h)]                      # the certified units
    tc = lambda h: h.get("truth_check_stated") or h.get("truth_check")
    true_share = (sum(1 for h in cert if tc(h) in TRUE_CHECKS) / len(cert)) if cert else None
    fd_share = (sum(1 for h in cert if tc(h) == "false_discovery") / len(cert)) if cert else None
    analysed_true = [h for h in run_h if tc(h) in ("true_discovery", "true_discovery_negligible") and _is_focal(h)]
    file_drawer = (sum(1 for h in analysed_true if not h.get("certified")) / len(analysed_true)) if analysed_true else None
    # the starting literature, inherited: reported apart, never among the run's units
    origin = certification_by_origin(hyps)
    # self-correction of the run's own false units, from the hypothesis rows (not the roll-up, whose legacy units differed by regime)
    selfc = self_correction(hyps)
    # efficiency denominators
    n_cert_units = len(cert)
    n_analyses = len({h.get("paper_id") or h.get("analysis_id") for h in run_h})
    n_reviews = len([r for r in read_jsonl(rd / "ledger" / ("reviews.jsonl" if regime == "paper" else "verifications.jsonl")) if int(r.get("quarter", 1)) >= 1])
    n_calls = int(budget["calls"]["calls"] + budget["calls"]["calls_not_debited"])   # live calls of the run (the pre-history's rule calls are not counted)
    n_obs = float(budget["total"]["data"])
    eff = {}
    for rule in RULES:
        added = net[rule] - net0.get(rule, 0)
        eff[rule] = {"per_certified_unit": added / n_cert_units if n_cert_units else None, "per_analysis": added / n_analyses if n_analyses else None,
                     "per_review_or_verification": added / n_reviews if n_reviews else None, "per_call": added / n_calls if n_calls else None,
                     "per_10k_obs": added / (n_obs / 10000) if n_obs else None}
    # speed: quarters from the first analysis on a question to its first certified unit
    first_a, first_c = {}, {}
    for h in sorted(run_h, key=lambda h: h["quarter"]):
        first_a.setdefault(h["question_id"], h["quarter"])
        if h.get("certified"):
            first_c.setdefault(h["question_id"], h["certified_quarter"])
    lags = [first_c[q] - first_a[q] for q in first_c if q in first_a]
    return {"coverage": {"core_covered_share": len(covered) / len(core_ids), "core_covered_3plus_share": len(covered3) / len(core_ids),
                         "n_core": len(core_ids), "fringe_certified": len(fringe_cert), "fringe_promoted": len(promoted),
                         "units_per_question_hist": dict(Counter(v["n_run_units"] for v in qs.values()))},
            "accuracy": acc, "knowledge_of_nulls": nulls, "precision": precision, "calibration": calib,
            "self_correction": selfc, "net_knowledge": net, "net_knowledge_continuous": net_cont, "net_knowledge_quarter0": net0,
            "efficiency": eff,
            "certification": {"definition": UNIT_DEFINITION,
                              "true_share_of_certified_units": true_share, "false_discovery_share_of_certified_units": fd_share,
                              "n_certified_units": n_cert_units, "n_certified_rows_all_kinds": len(cert_rows),
                              "median_quarters_to_first_certification": float(np.median(lags)) if lags else None,
                              "true_findings_never_certified_share": file_drawer},
            "certification_by_origin": origin,
            "denominators": {"certified_units": n_cert_units, "analyses": n_analyses, "reviews_or_verifications": n_reviews, "calls": n_calls, "observations": n_obs},
            "fringe": {"true_discovery_share": _share([h for h in cert if not h.get("core", True)]), "n_certified": sum(1 for h in cert if not h.get("core", True))}}


def _share(hs):
    return (sum(1 for h in hs if (h.get("truth_check_stated") or h.get("truth_check")) in TRUE_CHECKS) / len(hs)) if hs else None


def _sign_acc(qs, rule):
    c = [(v["rules"][rule]["belief"], v["theta"]) for v in qs.values() if v["n_units"] > 0 and abs(v["theta"]) >= 0.02]
    d = [(b, t) for b, t in c if b in ("positive", "negative")]
    return (sum(1 for b, t in d if (b == "positive") == (t > 0)) / len(d)) if d else None


def _null_acc(qs, rule):
    c = [(v["rules"][rule]["belief"], v["theta"]) for v in qs.values() if v["n_units"] > 0]
    d = [(b, t) for b, t in c if b in ("positive", "negative", "null")]
    return (sum(1 for b, t in d if (b == "null") == (t == 0.0)) / len(d)) if d else None


def _reliability_table(confs):
    bins = defaultdict(list)
    for c, y in confs:
        bins[min(int(c * 10), 9)].append(y)
    return {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {"n": len(v), "accuracy": float(np.mean(v))} for b, v in sorted(bins.items())}


def _cert_q(h: dict) -> int:
    return int(h.get("certified_quarter") if h.get("certified_quarter") is not None else (h.get("quarter") or 0))


def _stc(h: dict):
    return h.get("truth_check_stated") or h.get("truth_check")


def _corrected_by(fu: dict, units: list[dict]) -> dict | None:
    """A false certified unit counts as corrected when a later certified unit on the same question (certified in the run)
    states a different direction; returns that unit."""
    from .scorer import claim_sign
    later = sorted([u for u in units if u.get("question_id") == fu.get("question_id") and not u.get("legacy") and u.get("certified")
                    and _cert_q(u) > _cert_q(fu) and claim_sign(u) != claim_sign(fu)], key=_cert_q)
    return later[0] if later else None


def self_correction(hyps: list[dict]) -> dict:
    """Self-correction of the run's own false certified units (focal, certified at quarter >= 1); the inherited literature's
    false units are in certification_by_origin."""
    units = [h for h in hyps if h.get("certified") and _is_focal(h)]
    run_units = [h for h in units if not h.get("legacy")]
    false_units = [h for h in run_units if _stc(h) == "false_discovery"]
    corrected, lags = 0, []
    for fu in false_units:
        c = _corrected_by(fu, units)
        if c:
            corrected += 1
            lags.append(_cert_q(c) - _cert_q(fu))
    return {"n_false_certified_units": len(false_units), "share_corrected": (corrected / len(false_units)) if false_units else None,
            "median_lag_quarters": float(np.median(lags)) if lags else None}


def certification_by_origin(hyps: list[dict]) -> dict:
    """Certified units by origin: inherited (starting literature, certified at quarter <= 0) and new-in-run certified units,
    the same unit definition in both regimes, with true and false shares and the share of inherited false units a run
    unit later contradicted."""
    units = [h for h in hyps if h.get("certified") and _is_focal(h)]
    out = {}
    for origin, rows in (("inherited", [h for h in units if h.get("legacy")]), ("new_in_run", [h for h in units if not h.get("legacy")])):
        n = len(rows)
        true_n = sum(1 for h in rows if _stc(h) in TRUE_CHECKS)
        false_n = sum(1 for h in rows if _stc(h) == "false_discovery")
        missed = sum(1 for h in rows if _stc(h) == "missed_effect")
        unknown = sum(1 for h in rows if _stc(h) in (None, "unknown"))
        corrected = sum(1 for h in rows if _stc(h) == "false_discovery" and _corrected_by(h, units))
        out[origin] = {"n_units": n, "n_true": true_n, "n_false": false_n, "n_missed_effect": missed, "n_unknown": unknown,
                       "n_from_prehistory": sum(1 for h in rows if h.get("prehistory")),   # warm-start publications (P); the legacy library is the rest
                       "true_share": (true_n / n) if n else None, "false_share": (false_n / n) if n else None,
                       "false_units_corrected_in_run": corrected, "share_false_corrected_in_run": (corrected / false_n) if false_n else None,
                       "n_rows_all_kinds": sum(1 for h in hyps if h.get("certified") and bool(h.get("legacy")) == (origin == "inherited"))}
    out["definition"] = UNIT_DEFINITION
    return out


def verification_power_stats(rd: Path) -> dict:
    """The verifications' minimum detectable effect beside their verdicts (repository regime)."""
    rows = [v for v in read_jsonl(rd / "ledger" / "verifications.jsonl") if int(v.get("quarter", 1)) >= 1] if (rd / "ledger" / "verifications.jsonl").exists() else []
    with_mde = [v for v in rows if v.get("mde_partial_r") is not None]
    powered = [v for v in with_mde if v.get("powered") is not None]
    ratios = [v["n_ratio"] for v in rows if v.get("n_ratio")]
    by_result = {}
    for res in ("reproduced", "not_reproduced", "inconclusive"):
        rr = [v for v in powered if v.get("result") == res]
        by_result[res] = {"n": len(rr), "powered_share": float(np.mean([bool(v["powered"]) for v in rr])) if rr else None}
    return {"n_verifications": len(rows), "n_with_mde": len(with_mde),
            "median_mde_partial_r": float(np.median([v["mde_partial_r"] for v in with_mde])) if with_mde else None,
            "powered_share": float(np.mean([bool(v["powered"]) for v in powered])) if powered else None,
            "median_n_ratio_verifier_over_claim": float(np.median(ratios)) if ratios else None,
            "share_verifier_n_at_least_claim_n": float(np.mean([r >= 1.0 for r in ratios])) if ratios else None,
            "by_result": by_result,
            "note": "MDE: |partial r| detectable at alpha .05 (two-sided) with power .80 at the verifier's N; runs made with an early engine version carry no MDE"}


def budget_per_unit(budget: dict, field: dict, handbook: dict, rd: Path) -> dict:
    """Data and compute per certified unit and per correct handbook answer, beside the paired world's spend."""
    from .budget import paired_totals
    tot = budget.get("total") or {}
    data, comp = float(tot.get("data", 0.0)), float(tot.get("compute", 0.0))
    n_units = int(field["certification"]["n_certified_units"])
    final = handbook[max(handbook, key=lambda k: int(k[1:]))] if handbook else {}
    correct = final.get("correct_answers")
    if correct is None and final.get("accuracy") is not None and final.get("n_scored"):   # runs scored before this field existed
        correct = int(round(final["accuracy"] * final["n_scored"]))
    paired = paired_totals(rd)
    out = {"data_used": data, "compute_used": comp,
           "data_per_certified_unit": (data / n_units) if n_units else None, "compute_per_certified_unit": (comp / n_units) if n_units else None,
           "correct_handbook_answers_final": correct, "final_handbook_snapshot": max(handbook, key=lambda k: int(k[1:])) if handbook else None,
           "data_per_correct_handbook_answer": (data / correct) if correct else None,
           "compute_per_correct_handbook_answer": (comp / correct) if correct else None,
           "data_ceiling_multiple": budget.get("data_ceiling_multiple")}
    if paired:
        out["paired_world"] = {"data": paired["data"], "compute": paired["compute"], "source": paired["source"],
                               "data_ratio": (data / paired["data"]) if paired["data"] else None,
                               "compute_ratio": (comp / paired["compute"]) if paired["compute"] else None}
    return out


# ---------------------------------------------------------------------- theory level
def theory_level(final, core_ids, fam_of, pred_of, theta_of, q_T, hyps, events, regime, rd) -> dict:
    fams = sorted(set(fam_of[q] for q in core_ids))
    out = {}
    believed_by_rule = {r: {} for r in RULES}
    true_standing = {}
    for f in fams:
        qids = [q for q in core_ids if fam_of[q] == f]
        true_standing[f] = float(np.mean([(np.sign(theta_of[q]) == (1 if pred_of[q] == "positive" else -1)) and abs(theta_of[q]) >= 0.02 for q in qids]))
        row = {"true_standing": true_standing[f], "q_T": q_T.get(f), "n_questions": len(qids), "believed_standing": {}, "standing_gap": {}}
        for r in RULES:
            bel = [final["questions"][q]["rules"][r]["belief"] == pred_of[q] for q in qids if q in final["questions"]]
            bs = float(np.mean(bel)) if bel else None
            row["believed_standing"][r] = bs
            row["standing_gap"][r] = (bs - true_standing[f]) if bs is not None else None
            believed_by_rule[r][f] = bs
        out[f] = row
    rank = {r: _spearman([true_standing[f] for f in fams], [believed_by_rule[r][f] or 0.0 for f in fams]) for r in RULES}
    # rise and fall: share of certified units and attention in the first vs second half
    run_h = [h for h in hyps if not h.get("legacy") and h.get("certified")]
    H = max([h["quarter"] for h in run_h], default=1)
    move = {}
    for f in fams:
        a = [h for h in run_h if fam_of.get(h["question_id"]) == f]
        first = sum(1 for h in a if h["quarter"] <= H / 2) / max(1, sum(1 for h in run_h if h["quarter"] <= H / 2))
        second = sum(1 for h in a if h["quarter"] > H / 2) / max(1, sum(1 for h in run_h if h["quarter"] > H / 2))
        move[f] = {"share_first_half": first, "share_second_half": second, "rose": second > first,
                   "correct_movement": (second > first) == (true_standing[f] > 0.5)}
    # schools: citation/endorsement homophily
    homophily = school_homophily(rd, fam_of, regime)
    return {"families": out, "rank_agreement": rank, "movement": move, "schools": homophily}


def school_homophily(rd, fam_of, regime) -> dict:
    events = [e for e in read_jsonl(rd / "ledger" / "events.jsonl") if int(e.get("quarter", 1)) >= 1]
    if regime == "paper":
        papers = {p["paper_id"]: p for p in read_json(rd / "ledger" / "papers_snapshot.json")} if (rd / "ledger" / "papers_snapshot.json").exists() else {}
        edges = [(papers.get(e["from"], {}).get("question_id"), papers.get(e["to"], {}).get("question_id")) for e in events if e.get("event") == "citation"]
        edges = [(fam_of.get(a), fam_of.get(b)) for a, b in edges if a and b]
    else:
        votes = read_jsonl(rd / "ledger" / "votes.jsonl")
        edges = [(v.get("voter_theory"), fam_of.get(v.get("question_id"))) for v in votes if v.get("vote") == "endorse"]
    if not edges:
        return {}
    tot_by_target = Counter(b for _, b in edges)
    n = len(edges)
    out = {}
    for f in sorted(set(a for a, _ in edges) | set(b for _, b in edges)):
        own = [b for a, b in edges if a == f]
        if not own:
            continue
        share_own = sum(1 for b in own if b == f) / len(own)
        base = tot_by_target[f] / n
        out[f] = {"homophily_ratio": (share_own / base) if base else None, "n_edges": len(own)}
    return out


# ---------------------------------------------------------------------- author level
def author_level(hyps, rd, regime) -> dict:
    run_h = [h for h in hyps if not h.get("legacy")]
    contrib = defaultdict(float)
    units = defaultdict(int)
    for h in run_h:
        if not h.get("certified"):
            continue
        authors = h.get("authors") or [h.get("author_id")]
        w = 1.0 / max(1, len(authors))
        val = {"true_discovery": 1, "true_discovery_negligible": 1, "correct_null": 1 if h.get("core", True) else 0, "correct_null_negligible": 0,
               "false_discovery": -1, "missed_effect": 0}.get(h["truth_check"], 0)
        for a in authors:
            contrib[a] += w * val
            units[a] += 1
    # the evolved population (entrants included) when the run saved one; the seeded one otherwise
    wp = rd / "state" / "world.json"
    profiles = {s["id"]: s for s in (read_json(wp)["scientists"] if wp.exists() else read_json(rd / "population" / "scientists.json"))}
    visible = {}
    if regime == "paper":
        papers = read_json(rd / "ledger" / "papers_snapshot.json") if (rd / "ledger" / "papers_snapshot.json").exists() else []
        for p in papers:
            if p.get("legacy") or p.get("status") != "published":
                continue
            for a in p["authors"]:
                v = visible.setdefault(a, {"publications": 0, "lead_publications": 0, "citations": 0})
                v["publications"] += 1
                v["citations"] += p.get("citations", 0)
            visible.setdefault(p["lead"], {"publications": 0, "lead_publications": 0, "citations": 0})["lead_publications"] += 1
        key = "publications"
    else:
        claims = read_jsonl(rd / "ledger" / "claims.jsonl")
        latest = {}
        for c in claims:
            latest[c["claim_id"]] = c
        for c in latest.values():
            if c.get("status") == "endorsed":
                for a in c.get("leads") or []:
                    v = visible.setdefault(a, {"endorsed_claims": 0, "points": 0.0})
                    v["endorsed_claims"] += 1
        rep = read_json(rd / "ledger" / "reputation.json") if (rd / "ledger" / "reputation.json").exists() else {}
        for a, pts in rep.get("scientists", {}).items():
            visible.setdefault(a, {"endorsed_claims": 0, "points": 0.0})["points"] = pts
        key = "endorsed_claims"
    ids = sorted(set(profiles) | set(contrib))
    vis = [visible.get(a, {}).get(key, 0) for a in ids]
    tc = [contrib.get(a, 0.0) for a in ids]
    by_profile = defaultdict(list)
    for a in ids:
        s = profiles.get(a)
        if s:
            by_profile[f"{s['style']}-{s['throughput']}"].append((visible.get(a, {}).get(key, 0), contrib.get(a, 0.0)))
    top_n = max(1, len(ids) // 10)
    top = sorted(ids, key=lambda a: -visible.get(a, {}).get(key, 0))[:top_n]
    top_false = [h for h in run_h if h.get("certified") and h["truth_check"] == "false_discovery" and any(a in top for a in (h.get("authors") or [h.get("author_id")]))]
    top_units = [h for h in run_h if h.get("certified") and any(a in top for a in (h.get("authors") or [h.get("author_id")]))]
    return {"n_scientists": len(ids), "visible_key": key,
            "success_truth_spearman": _spearman(vis, tc),
            "gini_visible": _gini(vis), "gini_true_contribution": _gini([max(0.0, x) for x in tc]),
            "top10_share_visible": (sum(sorted(vis)[-top_n:]) / sum(vis)) if sum(vis) else None,
            "by_profile": {k: {"n": len(v), "mean_visible": float(np.mean([x[0] for x in v])), "mean_true_contribution": float(np.mean([x[1] for x in v]))} for k, v in by_profile.items()},
            "top_decile": {"n": len(top), "share_verified": float(np.mean([profiles[a]["style"] == "VERIFIED" for a in top if a in profiles])) if top else None,
                           "share_T1": float(np.mean([profiles[a]["throughput"] == "T1" for a in top if a in profiles])) if top else None,
                           "mean_true_contribution": float(np.mean([contrib.get(a, 0.0) for a in top])) if top else None,
                           "false_share_of_certified_units": (len(top_false) / len(top_units)) if top_units else None},
            "certifier_accuracy": certifier_accuracy(rd, regime)}


def certifier_accuracy(rd, regime) -> dict:
    if regime == "paper":
        revs = [r for r in read_jsonl(rd / "ledger" / "reviews.jsonl") if int(r.get("quarter", 1)) >= 1]   # live reviews only
        if not revs:
            return {}
        labels = [r["truth_check_focal"] in TRUE_CHECKS for r in revs]
        auc = _auc([r["validity_probability"] for r in revs], labels)
        by_rev = defaultdict(list)
        for r in revs:
            by_rev[r["reviewer_id"]].append(r)
        noticed = [r for r in revs if r.get("analyst_flags")]
        return {"validity_auc_vs_truth": auc, "n_reviews": len(revs),
                "flaw_noticed_rate": (sum(1 for r in noticed if r.get("noticed_trap_kind")) / len(noticed)) if noticed else None,
                "kappa_recommendation": _kappa_by_paper(revs)}
    votes = read_jsonl(rd / "ledger" / "votes.jsonl")
    end = [v for v in votes if v.get("vote") in ("endorse", "challenge")]
    if not end:
        return {}
    return {"endorse_auc_vs_truth": _auc([1.0 if v["vote"] == "endorse" else 0.0 for v in end], [v.get("claim_true") for v in end]),
            "stated_probability_auc": _auc([v.get("probability", 50) for v in end], [v.get("claim_true") for v in end]), "n_votes": len(end)}


def _kappa_by_paper(revs) -> float | None:
    by_p = defaultdict(list)
    for r in revs:
        by_p[(r["paper_id"], r.get("round", 1))].append(r["recommendation"])
    agree, n = 0, 0
    for recs in by_p.values():
        if len(recs) >= 2:
            n += 1
            agree += int(len(set(recs)) == 1)
    if n == 0:
        return None
    cats = Counter(r["recommendation"] for r in revs)
    pe = sum((c / len(revs)) ** 2 for c in cats.values())
    po = agree / n
    return float((po - pe) / (1 - pe)) if pe < 1 else None


# ---------------------------------------------------------------------- handbook, realism, failure indices
def handbook_scores(rd) -> dict:
    out = {}
    for d in sorted((rd / "handbook").glob("Q*")) if (rd / "handbook").exists() else []:
        if (d / "score.json").exists():
            s = read_json(d / "score.json")
            out[d.name] = {k: v for k, v in s.items() if k != "per_question"}
    return out


def call_group_correlation(rd, regime) -> dict:
    """Batched research/analyst calls: do studies that shared one call come out more alike than studies in different calls
    of the same quarter? Pairwise agreement of the focal false-discovery indicator and of the any-analyst-flag indicator,
    within call groups vs across groups (same quarter). A within-across gap is a within-call correlation the batching knob
    may have introduced (a guard added with the scaling knobs)."""
    if regime == "paper":
        p = rd / "ledger" / "papers_snapshot.json"
        rows = [x for x in (read_json(p) if p.exists() else []) if x.get("call_group") and not x.get("legacy") and not x.get("prehistory")]
        q_of = lambda x: x.get("quarter_started")
    else:
        rows = [x for x in read_jsonl(rd / "ledger" / "papers.jsonl") if x.get("call_group")]
        q_of = lambda x: x.get("quarter")
    if not rows:
        return {"n_batched_units": 0, "note": "no batched calls in this run"}
    def ind(x):
        return (1 if x.get("truth_check_focal") == "false_discovery" else 0, 1 if x.get("flags") else 0)
    within, across = [], []
    by_group = defaultdict(list)
    by_quarter = defaultdict(list)
    for x in rows:
        by_group[x["call_group"]].append(x)
        by_quarter[q_of(x)].append(x)
    for g, xs in by_group.items():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                within.append((ind(xs[i]), ind(xs[j])))
    for q, xs in by_quarter.items():
        xs = sorted(xs, key=lambda x: x.get("paper_id") or x.get("analysis_id"))
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if xs[i]["call_group"] != xs[j]["call_group"]:
                    across.append((ind(xs[i]), ind(xs[j])))
                if len(across) > 20000:
                    break
    def agree(pairs, k):
        return float(np.mean([a[k] == b[k] for a, b in pairs])) if pairs else None
    res = {"n_batched_units": len(rows), "n_groups": len(by_group), "n_within_pairs": len(within), "n_across_pairs": len(across),
           "false_discovery_agreement": {"within": agree(within, 0), "across": agree(across, 0)},
           "any_flag_agreement": {"within": agree(within, 1), "across": agree(across, 1)}}
    for k in ("false_discovery_agreement", "any_flag_agreement"):
        w, a = res[k]["within"], res[k]["across"]
        res[k]["gap"] = (w - a) if (w is not None and a is not None) else None
    return res


def realism_checks(hyps, events, regime) -> dict:
    # the reporting checks (support share, hypotheses per unit, caliper) describe the live authors' papers: studies researched
    # in the warm-start pre-history are rule-made and would only echo the rule author's conventions
    run_h = [h for h in hyps if not h.get("legacy") and not h.get("prehistory")]
    by_unit = defaultdict(list)
    for h in run_h:
        by_unit[h.get("paper_id") or h.get("analysis_id")].append(h)
    n_per = [len(v) for v in by_unit.values()]
    verdicts = Counter(h["verdict"] for h in run_h)
    full = [all(h["verdict"] == "supported" for h in v) for v in by_unit.values()]
    cert = [h for h in run_h if h.get("certified") and h.get("partial_r") is not None and h.get("n")]
    small = [abs(h["partial_r"]) for h in cert if h["n"] < 500]
    big = [abs(h["partial_r"]) for h in cert if h["n"] >= 5000]
    ps = [h["p"] for h in cert if h.get("p") is not None]
    just_below = sum(1 for p in ps if 0.04 <= p < 0.05)
    just_above = sum(1 for p in ps if 0.05 <= p < 0.06)
    wrong_sign = [h for h in cert if h["verdict"] == "supported" and h.get("theta") not in (None, 0.0) and np.sign(h["estimate"] or 0) != np.sign(h["theta"])]
    fm = [e for e in events if e.get("event") == "field_metrics"]
    return {"hypotheses_per_unit_median": float(np.median(n_per)) if n_per else None, "n_units": len(n_per),
            "n_units_prehistory_authored": len({h.get("paper_id") for h in hyps if h.get("prehistory") and not h.get("legacy")}),
            "reference": {"support_share": 0.745, "fully_supported_share": 0.406, "note": "corpus references (SMJ seeding tables); rule-agent runs (mock/fast) are uninformative here"},
            "support_share": (verdicts["supported"] / len(run_h)) if run_h else None, "partial_share": (verdicts["partially"] / len(run_h)) if run_h else None,
            "fully_supported_share": float(np.mean(full)) if full else None,
            "median_abs_r_small_N": float(np.median(small)) if small else None, "median_abs_r_large_N": float(np.median(big)) if big else None,
            "caliper_ratio_05": (just_below / just_above) if just_above else None, "n_certified_with_p": len(ps),
            "wrong_sign_share_certified_supported": (len(wrong_sign) / n_sup) if (n_sup := len([h for h in cert if h["verdict"] == "supported"])) else None,
            "accept_rate_mean": float(np.mean([f["accept_rate"] for f in fm if f.get("accept_rate") is not None])) if fm else None,
            "core_share_mean": float(np.mean([f["core_share"] for f in fm if f.get("core_share") is not None])) if fm else None}


def _quantiles(xs) -> dict:
    if not xs:
        return {"n": 0, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "mean": None}
    a = np.asarray(xs, dtype=float)
    return {"n": len(a), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)), "p25": float(np.quantile(a, 0.25)),
            "p75": float(np.quantile(a, 0.75)), "p90": float(np.quantile(a, 0.9)), "mean": float(a.mean())}


def timing_stats(rd, H: int, regime: str) -> dict:
    """The realized submission-to-acceptance and submission-to-first-decision distributions (months; a quarter is three
    months) of the live run, beside the grounding quantiles (4j_05), and what the warm start contributed."""
    from .timing import GROUNDING_MONTHS, MONTHS_PER_QUARTER
    if regime != "paper":
        return {"note": "paper regime only"}
    p = rd / "ledger" / "papers_snapshot.json"
    papers = [x for x in (read_json(p) if p.exists() else []) if not (x.get("legacy") and not x.get("prehistory"))]
    acc = [x for x in papers if x.get("quarter_accepted") is not None and x["quarter_accepted"] >= 1 and x.get("quarter_submitted") is not None]
    first = [x for x in papers if x.get("quarter_first_decision") is not None and x["quarter_first_decision"] >= 1 and x.get("quarter_submitted") is not None]
    to_acc = [(x["quarter_accepted"] - x["quarter_submitted"]) * MONTHS_PER_QUARTER for x in acc]
    to_first = [(x["quarter_first_decision"] - x["quarter_submitted"]) * MONTHS_PER_QUARTER for x in first]
    no_rev = [x for x in acc if list((x.get("decisions") or {}).keys()) == ["1"]]
    pre = [x for x in papers if x.get("prehistory")]
    final_q = lambda x: x.get("quarter_final_decision")   # accept / reject / desk reject / abandonment; None while in the pipeline
    in_pipeline_q0 = [x for x in pre if final_q(x) is None or final_q(x) >= 1]
    return {"submission_to_acceptance_months": _quantiles(to_acc), "submission_to_first_decision_months": _quantiles(to_first),
            "share_accepted_without_revision": (len(no_rev) / len(acc)) if acc else None,
            "reference_months": GROUNDING_MONTHS,
            "warm_start": {"papers_researched_in_prehistory": len(pre),
                           "published_before_q1": sum(1 for x in pre if (x.get("quarter_published") or 1) <= 0),
                           "in_pipeline_at_q0": len(in_pipeline_q0),
                           "published_in_run": sum(1 for x in pre if (x.get("quarter_published") or 0) >= 1),
                           "decided_in_run": sum(1 for x in pre if (final_q(x) or 0) >= 1)},
            "live_authored": {"started": sum(1 for x in papers if not x.get("prehistory") and (x.get("quarter_started") or 0) >= 1),
                              "published": sum(1 for x in papers if not x.get("prehistory") and (x.get("quarter_published") or 0) >= 1),
                              "unfinished_at_horizon": sum(1 for x in papers if not x.get("prehistory") and x.get("status") == "unfinished")},
            "published_in_run_total": sum(1 for x in papers if (x.get("quarter_published") or 0) >= 1)}


def career_stats(all_events) -> dict:
    """Careers moved by the annual step: counts by outcome, in the run and in the pre-history, and the publication record
    behind tenure outcomes (so that 'promotion depends on publications' is checkable from the ledger)."""
    ev = [e for e in all_events if e.get("event") == "career"]
    def block(rows):
        c = Counter(e.get("what") for e in rows)
        ten = [e for e in rows if e.get("what") == "tenured" and e.get("credit") is not None]
        out_ = [e for e in rows if e.get("what") == "tenure_exit" and e.get("credit") is not None]
        return {"counts": dict(c), "mean_credit_tenured": float(np.mean([e["credit"] for e in ten])) if ten else None,
                "mean_credit_tenure_exit": float(np.mean([e["credit"] for e in out_])) if out_ else None}
    return {"run": block([e for e in ev if int(e.get("quarter", 1)) >= 1]), "prehistory": block([e for e in ev if int(e.get("quarter", 1)) <= 0])}


def failure_indices(rd, theta_of) -> dict:
    votes = read_jsonl(rd / "ledger" / "votes.jsonl")
    claims_rows = read_jsonl(rd / "ledger" / "claims.jsonl")
    latest = {}
    for c in claims_rows:
        latest[c["claim_id"]] = c
    endorsed = [c for c in latest.values() if c.get("status") == "endorsed"]
    end_votes = [v for v in votes if v.get("vote") == "endorse"]
    # herding: cascade slope and late-vote accuracy
    slope = None
    if len(end_votes) + sum(1 for v in votes if v.get("vote") == "abstain") >= 20:
        X = np.array([[1.0, v.get("prior_endorsements", 0), 1.0 if v.get("significant") else 0.0, 1.0 if v.get("claim_true") else 0.0]
                      for v in votes if v.get("vote") in ("endorse", "abstain", "challenge")])
        y = np.array([1.0 if v["vote"] == "endorse" else 0.0 for v in votes if v.get("vote") in ("endorse", "abstain", "challenge")])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            slope = float(beta[1])
        except Exception:  # noqa: BLE001
            slope = None
    late = [v for v in end_votes if v.get("prior_endorsements", 0) >= 3]
    early = [v for v in end_votes if v.get("prior_endorsements", 0) == 0]
    unverified = [v for v in end_votes if not v.get("voter_analysed_question")]
    # dominance
    by_lab_size = defaultdict(list)
    for c in endorsed:
        by_lab_size[c.get("proposer_size", 0)].append(c)
    sizes = sorted(by_lab_size)
    top_q = sizes[int(len(sizes) * 0.75):] if sizes else []
    analyses = read_jsonl(rd / "ledger" / "papers.jsonl")
    an_by_size = Counter(a.get("lab_size", 0) for a in analyses)
    # reciprocity: mutual endorsement pairs vs shuffled null
    pairs = [(v["voter_lab"], v["proposer_lab"]) for v in end_votes if v.get("voter_lab") and v.get("proposer_lab")]
    mutual = sum(1 for a, b in set(pairs) if (b, a) in set(pairs)) / 2
    null = []
    rng = np.random.default_rng(0)
    if pairs:
        tgt = [b for _, b in pairs]
        for _ in range(100):
            perm = list(rng.permutation(tgt))
            sp = set(zip([a for a, _ in pairs], perm))
            null.append(sum(1 for a, b in sp if (b, a) in sp) / 2)
    # lock-in: P(endorsed) for contradicting vs first vs concordant
    rel = defaultdict(list)
    for c in latest.values():
        rel[c.get("relation_to_prior", "first claim")].append(c.get("status") == "endorsed")
    # free-riding: verification debt
    debt = [e for e in read_jsonl(rd / "ledger" / "events.jsonl") if e.get("event") == "verification_debt"]
    offers_by_lab = Counter(v["voter_lab"] for v in votes if v.get("vote") == "offer")
    labs = read_json(rd / "population" / "labs.json") if (rd / "population" / "labs.json").exists() else []
    lapsed_true = [c for c in latest.values() if c.get("status") == "lapsed" and c.get("claim_true")]
    # shared-data error
    single = [c for c in endorsed if len(set(c.get("deposit_ids", []))) <= 1 and c.get("deposit_ids")]
    multi = [c for c in endorsed if len(set(c.get("deposit_ids", []))) > 1]
    silence = [c for c in endorsed if c.get("route") == "silence"]
    return {"herding": {"cascade_slope": slope, "late_vote_accuracy": float(np.mean([v["claim_true"] for v in late])) if late else None,
                        "early_vote_accuracy": float(np.mean([v["claim_true"] for v in early])) if early else None,
                        "unverified_endorsement_share": (len(unverified) / len(end_votes)) if end_votes else None},
            "dominance": {"top_quartile_size_share_of_endorsed": (sum(len(by_lab_size[s]) for s in top_q) / len(endorsed)) if endorsed else None,
                          "top_quartile_size_share_of_analyses": (sum(an_by_size[s] for s in top_q) / sum(an_by_size.values())) if an_by_size else None,
                          "gini_endorsed_by_lab": _gini(list(Counter(c["proposer_lab"] for c in endorsed).values())) if endorsed else None},
            "reciprocity": {"mutual_pairs": mutual, "null_mean": float(np.mean(null)) if null else None, "null_p90": float(np.quantile(null, 0.9)) if null else None,
                            "excess": (mutual - float(np.mean(null))) if null else None},
            "lock_in": {k: {"n": len(v), "p_endorsed": float(np.mean(v))} for k, v in rel.items()},
            "free_riding": {"verification_debt_by_quarter": {str(e["quarter"]): e.get("open_without_offers") for e in debt},
                            "share_labs_never_verifying": (sum(1 for l in labs if offers_by_lab.get(l["id"], 0) == 0) / len(labs)) if labs else None,
                            "lapsed_true_claims": len(lapsed_true)},
            "shared_data": {"single_deposit_share_of_endorsed": (len(single) / len(endorsed)) if endorsed else None,
                            "single_deposit_true_share": float(np.mean([c.get("claim_true") for c in single])) if single else None,
                            "multi_deposit_true_share": float(np.mean([c.get("claim_true") for c in multi])) if multi else None},
            "silence": {"share_endorsed_by_silence": (len(silence) / len(endorsed)) if endorsed else None,
                        "silence_true_share": float(np.mean([c.get("claim_true") for c in silence])) if silence else None},
            "label": "measured on rule agents (mock/fast) or LLM labs as the run's mode says; no real-world reference exists"}


# ---------------------------------------------------------------------- summary page
def summary_md(a: dict) -> str:
    f = a["field"]
    L = [f"# Audit: {a['run_id']} ({a['regime']}, mode {a['mode']}, {a['population']} scientists, {a['horizon_quarters']} quarters)", "",
         f"Coverage of the core: {f['coverage']['core_covered_share']:.2f} (>=3 units: {f['coverage']['core_covered_3plus_share']:.2f}); "
         f"fringe certified {f['coverage']['fringe_certified']}, promoted {f['coverage']['fringe_promoted']}.", "",
         "| Rule | Accuracy | Unsettled | Null acc. | Brier | Net knowledge | Net at Q0 |", "|---|---|---|---|---|---|---|"]
    for r in RULES:
        acc = f["accuracy"][r]
        L.append(f"| {r} | {_fmt(acc['accuracy'])} | {acc['unsettled']} | {_fmt(acc['null_accuracy'])} | {_fmt(f['calibration'][r]['brier'])} | "
                 f"{f['net_knowledge'][r]} | {f['net_knowledge_quarter0'].get(r, '-')} |")
    c = f["certification"]
    L += ["", f"Certified units (new in the run; {c['definition']}): {c['n_certified_units']} (certified rows of all kinds {c.get('n_certified_rows_all_kinds', '-')}); "
          f"true share {_fmt(c['true_share_of_certified_units'])}; false-discovery share {_fmt(c['false_discovery_share_of_certified_units'])}; "
          f"true findings never certified {_fmt(c['true_findings_never_certified_share'])}; median quarters to first certification {_fmt(c['median_quarters_to_first_certification'])}.",
          f"Self-correction (the run's own units): {f['self_correction']['n_false_certified_units']} false certified units, share corrected {_fmt(f['self_correction']['share_corrected'])}, median lag {_fmt(f['self_correction']['median_lag_quarters'])}.",
          f"Precision (B2, >=2 units): n {f['precision']['n_questions']}, median |error| {_fmt(f['precision']['median_abs_error'])}, coverage {_fmt(f['precision']['coverage'])}.", ""]
    if f.get("certification_by_origin"):
        o = f["certification_by_origin"]
        L += ["| Certified units by origin | n | true | false | missed effect | false share | false units corrected in run |", "|---|---|---|---|---|---|---|"]
        for k, label in (("inherited", "Inherited (starting literature, certified at Q<=0)"), ("new_in_run", "New in run (Q>=1)")):
            r_ = o[k]
            L.append(f"| {label} | {r_['n_units']} | {r_['n_true']} | {r_['n_false']} | {r_['n_missed_effect']} | {_fmt(r_['false_share'])} | "
                     f"{r_['false_units_corrected_in_run']} ({_fmt(r_['share_false_corrected_in_run'])}) |")
        L.append("")
    L += [
          "## Theory", "", "| Family | True standing | q_T | Believed (B2) | Gap (B2) |", "|---|---|---|---|---|"]
    for fam, row in a["theory"]["families"].items():
        L.append(f"| {fam} | {row['true_standing']:.2f} | {_fmt(row['q_T'])} | {_fmt(row['believed_standing']['B2'])} | {_fmt(row['standing_gap']['B2'])} |")
    L.append(f"\nRank agreement (Spearman, true vs believed): " + ", ".join(f"{r} {_fmt(v)}" for r, v in a["theory"]["rank_agreement"].items()))
    au = a["author"]
    L += ["", "## Authors", "", f"Success-truth Spearman: {_fmt(au['success_truth_spearman'])}; Gini visible {_fmt(au['gini_visible'])}; top-10% share {_fmt(au['top10_share_visible'])}.",
          "By profile: " + "; ".join(f"{k}: n {v['n']}, visible {v['mean_visible']:.2f}, true {v['mean_true_contribution']:.2f}" for k, v in au["by_profile"].items())]
    if au.get("certifier_accuracy"):
        L.append("Certifier accuracy: " + ", ".join(f"{k} {_fmt(v) if isinstance(v, (int, float)) or v is None else v}" for k, v in au["certifier_accuracy"].items()))
    if a.get("handbook"):
        L += ["", "## Handbook", ""] + [f"- {k}: accuracy {_fmt(v.get('accuracy'))}, Brier {_fmt(v.get('brier'))}, confidently wrong {v.get('confidently_wrong')}, "
                                        f"agreement with rules {', '.join(f'{r} {_fmt(x)}' for r, x in (v.get('agreement_with_rules') or {}).items())}" for k, v in a["handbook"].items()]
    r = a["realism"]
    L += ["", "## Realism checks (outputs; corpus references from the SMJ seeding tables)", "",
          f"Hypotheses per unit (median) {_fmt(r['hypotheses_per_unit_median'])} [ref 5]; support share {_fmt(r['support_share'])} [ref .745]; fully supported {_fmt(r['fully_supported_share'])} [ref .406]; "
          f"median |r| N<500 {_fmt(r['median_abs_r_small_N'])} vs N>=5000 {_fmt(r['median_abs_r_large_N'])} [ref .178 / .022]; caliper ratio {_fmt(r['caliper_ratio_05'])} [ref 1.9-2.3]; "
          f"wrong-sign share {_fmt(r['wrong_sign_share_certified_supported'])} [ref .15]; accept rate {_fmt(r['accept_rate_mean'])} [no reference]."]
    if a.get("failure_indices"):
        fi = a["failure_indices"]
        L += ["", "## Consensus failure indices (repository regime)", "",
              f"Herding: cascade slope {_fmt(fi['herding']['cascade_slope'])}, unverified-endorsement share {_fmt(fi['herding']['unverified_endorsement_share'])}, "
              f"late/early vote accuracy {_fmt(fi['herding']['late_vote_accuracy'])}/{_fmt(fi['herding']['early_vote_accuracy'])}.",
              f"Dominance: top-quartile-size share of endorsed {_fmt(fi['dominance']['top_quartile_size_share_of_endorsed'])} vs of analyses {_fmt(fi['dominance']['top_quartile_size_share_of_analyses'])}.",
              f"Reciprocity: mutual pairs {fi['reciprocity']['mutual_pairs']} vs null {_fmt(fi['reciprocity']['null_mean'])} (p90 {_fmt(fi['reciprocity']['null_p90'])}).",
              f"Lock-in: " + "; ".join(f"{k} n {v['n']} P(endorsed) {v['p_endorsed']:.2f}" for k, v in fi["lock_in"].items()) + ".",
              f"Free-riding: labs never verifying {_fmt(fi['free_riding']['share_labs_never_verifying'])}; lapsed true claims {fi['free_riding']['lapsed_true_claims']}.",
              f"Shared data: single-deposit share of endorsed {_fmt(fi['shared_data']['single_deposit_share_of_endorsed'])} (true share {_fmt(fi['shared_data']['single_deposit_true_share'])} vs multi {_fmt(fi['shared_data']['multi_deposit_true_share'])}).",
              f"Silence: {_fmt(fi['silence']['share_endorsed_by_silence'])} of endorsed claims (true share {_fmt(fi['silence']['silence_true_share'])}).", f"_{fi['label']}_"]
    vp = a.get("verification_power") or {}
    if vp.get("n_verifications"):
        L += ["", f"Verification power: {vp['n_verifications']} verifications ({vp['n_with_mde']} with an MDE); median MDE |partial r| {_fmt(vp['median_mde_partial_r'])}; "
              f"powered for the claimed (or a small) effect {_fmt(vp['powered_share'])}; verifier N / claim N median {_fmt(vp['median_n_ratio_verifier_over_claim'])}, "
              f"share at least the claim's N {_fmt(vp['share_verifier_n_at_least_claim_n'])}."]
    t = a.get("timing") or {}
    if t.get("submission_to_acceptance_months"):
        sa, sf, ws = t["submission_to_acceptance_months"], t["submission_to_first_decision_months"], t["warm_start"]
        L += ["", "## Timing (months; a quarter is three months)", "",
              f"Submission to acceptance: n {sa['n']}, median {_fmt(sa['median'])} [p10 {_fmt(sa['p10'])}, p90 {_fmt(sa['p90'])}] "
              f"[ref SMJ 22.1 (11.1, 43.5), OS 23.9 (12.3, 42.3); 4j_05]; submission to first decision: n {sf['n']}, median {_fmt(sf['median'])}; "
              f"accepted without revision {_fmt(t['share_accepted_without_revision'])}.",
              f"Warm start: {ws['papers_researched_in_prehistory']} studies researched in the pre-history, {ws['published_before_q1']} published before Q1 (earlier work), "
              f"{ws['in_pipeline_at_q0']} in the pipeline at Q0, {ws['decided_in_run']} decided and {ws['published_in_run']} published by the live journal; "
              f"live-authored: {t['live_authored']['started']} started, {t['live_authored']['published']} published, {t['live_authored']['unfinished_at_horizon']} unfinished at the horizon."]
    c = a.get("careers") or {}
    if c.get("run", {}).get("counts") or c.get("prehistory", {}).get("counts"):
        L += ["", f"Careers (pressure shown to agents: {a.get('career_pressure', 'none')}): run " + ", ".join(f"{k} {v}" for k, v in sorted(c["run"]["counts"].items()))
              + f" (mean record tenured {_fmt(c['run']['mean_credit_tenured'])} vs tenure exit {_fmt(c['run']['mean_credit_tenure_exit'])}); pre-history "
              + ", ".join(f"{k} {v}" for k, v in sorted(c["prehistory"]["counts"].items())) + "."]
    b = a.get("budget", {})
    if b:
        tot = b.get("total", {})
        calls = b.get("calls", {})
        L += ["", f"Budget (from the ledger): rule {b.get('rule')}; total data {tot.get('data', 0):,.0f} observations, compute {tot.get('compute', 0):,.0f} tariff tokens "
              f"over {calls.get('calls', 0)} debited calls (+{calls.get('calls_not_debited', 0)} not debited, {calls.get('calls_failed', 0)} failed); "
              f"realized agent tokens {tot.get('realized_tokens', 0):,.0f}; pre-history (rule agents, not charged): {calls.get('prehistory_calls', 0)} calls, "
              f"{b.get('prehistory_data', 0):,.0f} observations"
              + (f"; pairing fallback: {b['pairing_fallback']}" if b.get("pairing_fallback") else "")
              + ("; NOTE: the engine's summary.json disagreed with the ledger (stale tallies)" if b.get("summary_stale") else "") + "."]
        pu = b.get("per_unit") or {}
        if pu:
            line = (f"Data per certified unit {_fmt_big(pu.get('data_per_certified_unit'))} observations, per correct handbook answer "
                    f"{_fmt_big(pu.get('data_per_correct_handbook_answer'))} ({pu.get('correct_handbook_answers_final', '-')} correct at {pu.get('final_handbook_snapshot', '-')}); "
                    f"compute per certified unit {_fmt_big(pu.get('compute_per_certified_unit'))} tariff tokens, per correct handbook answer {_fmt_big(pu.get('compute_per_correct_handbook_answer'))}.")
            if pu.get("paired_world"):
                pw = pu["paired_world"]
                line += (f" Paired world: data {pw['data']:,.0f} (this run {_fmt(pw['data_ratio'])} x; ceiling {pu.get('data_ceiling_multiple')} x), "
                         f"compute {pw['compute']:,.0f} (this run {_fmt(pw['compute_ratio'])} x).")
            dbb = b.get("deferrals_by_binding") or {}
            if any(dbb.values()):
                line += " Deferrals by binding resource: " + ", ".join(f"{k} {v}" for k, v in dbb.items() if v) + "."
            L.append(line)
    return "\n".join(L) + "\n"


def _fmt_big(x) -> str:
    return "-" if x is None else f"{float(x):,.0f}"


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return f"{float(x):.3f}"
    except Exception:  # noqa: BLE001
        return str(x)


# ---------------------------------------------------------------------- Monte Carlo and pairing
def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate(run_dirs: list[Path], out_path: Path) -> dict:
    """Median and 10-90% range of every numeric audit measure across seeds."""
    rows = []
    for rd in run_dirs:
        p = Path(rd) / "audit" / "audit.json"
        if p.exists():
            a = read_json(p)
            rows.append((a["seed"], _flatten({k: a[k] for k in ("field", "theory", "author", "handbook", "realism", "timing", "careers", "failure_indices") if k in a})))
    keys = sorted(set(k for _, r in rows for k in r))
    dist = {}
    for k in keys:
        vals = [r[k] for _, r in rows if k in r]
        if vals:
            dist[k] = {"n": len(vals), "median": float(np.median(vals)), "p10": float(np.quantile(vals, 0.1)), "p90": float(np.quantile(vals, 0.9)),
                       "values": vals if len(vals) <= 10 else None}
    out = {"n_runs": len(rows), "seeds": [s for s, _ in rows], "distributions": dist}
    write_json(out_path, out)
    return out


def pair_regimes(p_dirs: dict[int, Path], r_dirs: dict[int, Path], out_path: Path) -> dict:
    """Paired differences R - P by seed for every numeric measure."""
    diffs = defaultdict(list)
    seeds = sorted(set(p_dirs) & set(r_dirs))
    for s in seeds:
        pa, ra = Path(p_dirs[s]) / "audit" / "audit.json", Path(r_dirs[s]) / "audit" / "audit.json"
        if not (pa.exists() and ra.exists()):
            continue
        fp, fr = _flatten(read_json(pa)["field"]), _flatten(read_json(ra)["field"])
        for k in set(fp) & set(fr):
            diffs[k].append(fr[k] - fp[k])
    out = {"seeds": seeds, "paired_differences": {k: {"n": len(v), "median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)),
                                                      "p90": float(np.quantile(v, 0.9)), "values": v if len(v) <= 10 else None} for k, v in diffs.items()}}
    write_json(out_path, out)
    return out
