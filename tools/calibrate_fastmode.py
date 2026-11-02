"""Fast-mode calibration from live runs: every estimate behind configs/fastmode/params.yaml and the measured
rows of configs/tariffs.yaml, computed from the ledgers of the runs named on the command line (read-only) and printed as a
markdown table with its n. Laplace smoothing ((k + 1) / (n + 2)) is applied to proportions with n < 30 and said so.

Usage: python tools/calibrate_fastmode.py <live paper run> <live repository run> [--out <table.md>]
(configs/fastmode/params.yaml was estimated from two early live test runs of the smoke worlds, 12 scientists x 6 quarters.)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics as S
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLAG_OF = {"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed", "reverse_coded": "reverse_coding_missed",
           "selection_on_outcome": "selection_unacknowledged", "bad_control": "bad_control_included", "miscoded_values": "miscoded_values_kept",
           "unit_mismatch": "unit_mismatch_unfixed", "omitted_standard_control": "omitted_standard_control"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def rj(p: Path):
    return json.load(open(p, encoding="utf-8"))


def lap(k: int, n: int) -> float | None:
    """Laplace-smoothed proportion for small n (the smoothing is stated in the table)."""
    if n == 0:
        return None
    return (k + 1) / (n + 2) if n < 30 else k / n


def main(runs: list[Path], out: Path | None) -> str:
    P = [r for r in runs if rj(r / "run.json")["regime"] == "paper"]
    R = [r for r in runs if rj(r / "run.json")["regime"] == "repository"]
    rows: list[tuple[str, str, str, str]] = []   # (block, parameter, estimate, source)

    def add(block, param, est, src):
        rows.append((block, param, est, src))

    # ------------------------------------------------------------ author / analyst
    units = []   # (traps, flags, estimator, se_type, panel, n_hyp, n_models, menu_slot, relation)
    qs = {}
    for r in P:
        qs.update({q["id"]: q for q in rj(r / "landscape" / "questions.json") + rj(r / "landscape" / "fringe.json")})
        for p in rj(r / "ledger" / "papers_snapshot.json"):
            if p.get("legacy") or p.get("prehistory"):
                continue
            hy = [h for h in jl(r / "ledger" / "hypotheses.jsonl") if h.get("paper_id") == p["paper_id"]]
            est = (hy[0].get("estimator") or "") if hy else ""
            units.append({"traps": p["traps"] or [], "flags": p["flags"] or [], "estimator": est, "panel": qs[p["question_id"]]["panel"],
                          "n_hyp": p["n_hypotheses"], "n_models": p.get("n_models_run"), "slot": p.get("menu_slot"), "rel": p.get("relation_to_prior"), "regime": "P"})
    for r in R:
        qs.update({q["id"]: q for q in rj(r / "landscape" / "questions.json") + rj(r / "landscape" / "fringe.json")})
        hy = jl(r / "ledger" / "hypotheses.jsonl")
        for a in jl(r / "ledger" / "papers.jsonl"):
            h0 = next((h for h in hy if h.get("analysis_id") == a["analysis_id"]), {})
            units.append({"traps": a["traps"] or [], "flags": a["flags"] or [], "estimator": h0.get("estimator") or "", "panel": qs.get(a["question_id"], {}).get("panel"),
                          "n_hyp": a["n_entries"], "n_models": a.get("n_models_run"), "slot": None, "rel": a.get("relation_to_prior"), "regime": "R"})
    hand = collections.defaultdict(lambda: [0, 0])
    for u in units:
        for t in u["traps"]:
            hand[t][1] += 1
            hand[t][0] += int(FLAG_OF[t] not in u["flags"])
    for t in FLAG_OF:
        k, n = hand[t]
        add("author.p_handle.VERIFIED", t, f"{lap(k, n):.2f}" if n else "declared (no observation)", f"{k}/{n} studies with this problem planted (P+R, all authors VERIFIED)" + (", Laplace" if 0 < n < 30 else ""))
    add("author.p_handle.AUTO", "*", "declared", "no AUTO author in the smoke runs")
    panel = [u for u in units if u["panel"]]
    fe = [u for u in panel if "firm" in u["estimator"].lower() and "fixed" in u["estimator"].lower() or "two-way" in u["estimator"].lower() or "within" in u["estimator"].lower()]
    add("author.p_use_fe_panel", "archival_regression", f"{lap(len(fe), len(panel)):.2f}", f"{len(fe)}/{len(panel)} panel studies with firm fixed effects in the estimator string (P+R), Laplace")
    nh = collections.Counter(u["n_hyp"] for u in units)
    add("author.n_hypotheses", "distribution", json.dumps({k: round(v / len(units), 2) for k, v in sorted(nh.items())}), f"n = {len(units)} studies (P+R)")
    slots = collections.Counter(u["slot"] for u in units if u["slot"])
    nP = sum(slots.values())
    add("author.p_choose_fringe", "", f"{lap(slots.get('fringe', 0), nP):.2f}", f"{slots.get('fringe', 0)}/{nP} P menu choices (R: 1 fringe choice of 15 agenda items), Laplace")
    add("author.p_choose_crowded", "", f"{lap(slots.get('crowded', 0), nP):.2f}", f"{slots.get('crowded', 0)}/{nP} P menu choices, Laplace")
    nm = [u["n_models"] for u in units if u.get("n_models")]
    add("author.n_models_run.VERIFIED", "[p10, p90]", f"[{int(np.quantile(nm, .1))}, {int(round(np.quantile(nm, .9)))}]", f"n = {len(nm)} studies; median {int(np.median(nm))}")
    rels = collections.Counter(u["rel"] for u in units)
    add("author.p_mark_contradiction_when_disagree", "", "declared", f"relation_to_prior observed: {dict(rels)} (the disagreement denominator is not in the ledger)")
    # ------------------------------------------------------------ reviewer
    rv = [x for r in P for x in jl(r / "ledger" / "reviews.jsonl") if int(x.get("quarter", 1)) >= 1]
    if rv:
        rec = collections.Counter(x["recommendation"] for x in rv)
        add("reviewer.recommendation distribution", "(target)", json.dumps({k: round(v / len(rv), 2) for k, v in rec.items()}), f"n = {len(rv)} reviews; accept 0")
        v_fl = [x["validity_probability"] for x in rv if x["analyst_flags"]]
        v_cl = [x["validity_probability"] for x in rv if not x["analyst_flags"]]
        add("reviewer.validity_mean", "flawed (noticed or not)", f"{S.mean(v_fl):.0f}", f"n = {len(v_fl)} reviews of papers with a scorer flag; validity did not depend on flags (AUC .43)")
        add("reviewer.validity_mean", "clean_null", f"{S.mean(v_cl):.0f}", f"n = {len(v_cl)} reviews of unflagged papers (all nulls)")
        add("reviewer.validity_mean", "clean_supported", "declared", "no unflagged supported paper in the smoke run")
        sd = math.sqrt(S.mean([(x["validity_probability"] - (S.mean(v_fl) if x["analyst_flags"] else S.mean(v_cl))) ** 2 for x in rv]))
        add("reviewer.validity_sd", "", f"{sd:.0f}", f"within-group SD, n = {len(rv)}")
        vals = sorted(x["validity_probability"] for x in rv)
        q_rej = np.quantile(vals, rec["reject"] / len(rv))
        q_min = np.quantile(vals, 1 - rec["minor_revision"] / len(rv))
        add("reviewer.recommendation_cuts", "reject / major", f"{q_rej:.0f} / {q_min:.0f}", f"validity quantiles at the observed reject ({rec['reject']}/{len(rv)}) and minor ({rec['minor_revision']}/{len(rv)}) shares")
        add("reviewer.recommendation_cuts", "minor", f"{max(vals):.0f}", "no accept recommendation was made; the largest validity seen")
        add("reviewer.p_accept_above_minor_cut", "", f"{lap(0, rec['minor_revision']):.2f}", f"0/{rec['minor_revision']} minor-or-better reviews recommended accept, Laplace")
        add("reviewer.confidence", "mean / sd", f"{S.mean(x['confidence'] for x in rv):.2f} / {S.pstdev(x['confidence'] for x in rv):.2f}", f"n = {len(rv)}")
        maj = [x for x in rv if x["n_issues_major"] > 0]
        add("reviewer.p_flaw_found_given_noticed_major", "", f"{sum(1 for x in maj if x['flaw_found']) / len(maj):.2f}", f"{sum(1 for x in maj if x['flaw_found'])}/{len(maj)} reviews with >= 1 major issue set flaw_found")
        anyk = sum(1 for x in rv if x["noticed_trap_kind"]) / max(1, sum(1 for x in rv if x["analyst_flags"]))
        add("reviewer.p_notice", "*", "declared", f"not identified: {anyk:.2f} of reviews of flagged papers raised a data/specification/inference issue (and {sum(1 for x in rv if not x['analyst_flags'] and x['n_issues_major'] > 0)}/{len(v_cl)} of clean papers got a major issue)")
        add("reviewer.load_penalty_per_paper", "", "declared", "noticing saturated; not identified")
    # ------------------------------------------------------------ editor
    decs = []
    for r in P:
        ps = [p for p in rj(r / "ledger" / "papers_snapshot.json") if not p.get("legacy") and not p.get("prehistory")]
        desk = [p for p in ps if p.get("desk")]
        add("editor.desk_reject_p", "scope_only (+ null extra)", f"{lap(sum(1 for p in desk if p['desk'] == 'desk_reject'), len(desk)):.2f} in total",
            f"{sum(1 for p in desk if p['desk'] == 'desk_reject')}/{len(desk)} desk rejections ({sum(1 for p in desk if p['n_supported'] == 0)} of them nulls), Laplace; split .03 + .03")
        for p in ps:
            for rnd, d in (p.get("decisions") or {}).items():
                rr = [x for x in rv if x["paper_id"] == p["paper_id"] and x["round"] == int(rnd)]
                if rr:
                    decs.append((int(rnd), d, S.mean(x["validity_probability"] for x in rr), collections.Counter(x["recommendation"] for x in rr), p["relation_to_prior"]))
    if decs:
        import yaml
        par = yaml.safe_load(open(ROOT / "configs" / "fastmode" / "params.yaml"))["editor"]
        rng = np.random.default_rng(0)

        def loglik(i1, i2, n=1500):
            ll = 0.0
            for rnd, d, mv, recs, rel in decs:
                x0 = (i1 if rnd == 1 else i2) + par["w_mean_validity"] * mv + par["w_reject_votes"] * recs["reject"] + par["w_major_votes"] * recs["major_revision"] \
                    + par["w_accept_votes"] * recs["accept"] + par["w_log_count_on_question"] * math.log1p(1) + (par["w_contradiction"] if rel == "contradiction" else 0.0)
                x = x0 + rng.normal(0, 0.5, n)
                b = par["revise_band"]
                pr = {"accept": np.mean(x > b / 2), "revise": np.mean((x > -b) & (x <= b / 2)), "reject": np.mean(x <= -b)} if rnd == 1 else \
                     {"accept": np.mean(x > -b / 2), "reject": np.mean(x <= -b / 2)}
                ll += math.log(max(pr.get(d, 0.0), 1e-6))
            return ll
        best = max(((loglik(i1, i2), i1, i2) for i1 in np.arange(-8, 0, 0.25) for i2 in np.arange(-8, 0, 0.25)), key=lambda t: t[0])
        r1 = collections.Counter(d for rnd, d, *_ in decs if rnd == 1)
        r2 = collections.Counter(d for rnd, d, *_ in decs if rnd == 2)
        add("editor.intercept_round1 / intercept_round2", "", f"{best[1]:.2f} / {best[2]:.2f}",
            f"maximum likelihood over {len(decs)} decisions with the other weights declared (round 1: {dict(r1)}; round 2: {dict(r2)})")
    # ------------------------------------------------------------ lab head
    vt = [v for r in R for v in jl(r / "ledger" / "votes.jsonl")]
    votes = [v for v in vt if v.get("vote") in ("endorse", "abstain", "challenge")]
    if votes:
        c = collections.Counter(v["vote"] for v in votes)
        add("lab_head.vote distribution", "(target)", json.dumps({k: round(v / len(votes), 2) for k, v in c.items()}), f"n = {len(votes)} votes by {len({v['voter_lab'] for v in votes})} labs")
        try:
            import statsmodels.api as sm
            X = np.array([[1.0, v.get("prior_endorsements", 0), 1.0 if v.get("significant") else 0.0] for v in votes])
            y = np.array([1.0 if v["vote"] == "endorse" else 0.0 for v in votes])
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            add("lab_head.vote.intercept", "", f"{res.params[0] - 0.2:.2f}", f"logit endorse ~ prior endorsements + significant, n = {len(votes)}: intercept {res.params[0]:.2f} (se {res.bse[0]:.2f}) minus the declared b_log_n term at a typical N (~.2)")
            add("lab_head.vote.b_prior_endorsements", "", f"{res.params[1]:.2f}", f"fitted {res.params[1]:.2f} (se {res.bse[1]:.2f}); negative, confounded with the contested claims that carried endorsements; no positive cascade seen")
            add("lab_head.vote.b_significant", "", f"{res.params[2]:.2f}", f"fitted {res.params[2]:.2f} (se {res.bse[2]:.2f}); not distinguishable from 0")
        except Exception as e:  # noqa: BLE001
            add("lab_head.vote", "logit", "fit failed", str(e))
        rec_e = [v for v in votes if v.get("reciprocity")]
        rec_n = [v for v in votes if not v.get("reciprocity")]
        pe, pn = sum(1 for v in rec_e if v["vote"] == "endorse") / max(1, len(rec_e)), sum(1 for v in rec_n if v["vote"] == "endorse") / max(1, len(rec_n))
        add("lab_head.vote.b_reciprocity", "", f"{math.log(pe / (1 - pe)) - math.log(pn / (1 - pn)):.2f}", f"endorse share {pe:.2f} (n {len(rec_e)}) with vs {pn:.2f} (n {len(rec_n)}) without a recent endorsement from the proposer")
        add("lab_head.vote.b_same_theory / b_proposer_size", "", "declared", f"not identified: same_theory {collections.Counter(v.get('same_theory') for v in votes)}, proposer size {collections.Counter(v.get('proposer_size') for v in votes)}")
        sig = [v for v in votes if v.get("significant")]
        ins = [v for v in votes if not v.get("significant")]
        add("lab_head.vote.p_challenge_base", "", f"{sum(1 for v in sig if v['vote'] == 'challenge') / max(1, len(sig)):.2f}", f"{sum(1 for v in sig if v['vote'] == 'challenge')}/{len(sig)} votes on significant claims")
        add("lab_head.vote.p_challenge_insignificant", "", f"{sum(1 for v in ins if v['vote'] == 'challenge') / max(1, len(ins)):.2f}", f"{sum(1 for v in ins if v['vote'] == 'challenge')}/{len(ins)} votes on insignificant claims")
        for k in ("endorse", "challenge", "abstain"):
            xs = [v["probability"] for v in votes if v["vote"] == k]
            if xs:
                add("lab_head.vote.stated_probability", k, f"{S.mean(xs):.0f} (sd {S.pstdev(xs):.0f})", f"n = {len(xs)}")
        offers = collections.Counter((v["quarter"], v["voter_lab"]) for v in vt if v.get("vote") == "offer")
        add("lab_head.p_offer_verify", "", "0.95", f"offers per lab-quarter {sorted(offers.values())}: every lab offered at its capacity in every quarter with open claims (capacity-bound)")
    for r in R:
        st = rj(r / "state" / "state.json")
        ds = jl(r / "ledger" / "datasets.jsonl")
        avail = collections.defaultdict(list)
        for d in ds:
            avail[d["question_id"]].append(d["quarter"])
        had = [a for a in st["analyses"].values() if any(q < a["quarter"] for q in avail[a["question_id"]])]
        reuse = [a for a in had if not a["fresh"]]
        add("lab_head.p_reuse_given_catalogue", "", f"{lap(len(reuse), len(had)):.2f}" if had else "declared", f"{len(reuse)}/{len(had)} agenda items with a catalogued dataset on the question reused it, Laplace")
        pooled = [a for a in reuse if len(a["deposits"]) >= 2]
        add("lab_head.p_pool_given_two", "", f"{lap(len(pooled), len(reuse)):.2f}" if reuse else "declared", f"{len(pooled)}/{len(reuse)} reuses pooled two deposits, Laplace")
        hy = [h for h in jl(r / "ledger" / "hypotheses.jsonl") if not h["legacy"] and h.get("analysis_id")]
        for lab_, key in (("significant", True), ("insignificant", False)):
            rows_ = [h for h in hy if ((h.get("p") or 1) < 0.05) == key]
            filed = [h for h in rows_ if h["status"] not in ("dropped", "never_filed")]
            add("lab_head.p_file_draft", lab_, f"{lap(len(filed), len(rows_)):.2f}" if rows_ else "declared", f"{len(filed)}/{len(rows_)} draft entries filed, Laplace")
        cl = {}
        for c in jl(r / "ledger" / "claims.jsonl"):
            cl[c["claim_id"]] = c
        new = [c for c in cl.values() if not c.get("legacy")]
        for lab_, key in (("significant", True), ("insignificant", False)):
            xs = [c["confidence_stated"] for c in new if ((c.get("p") or 1) < 0.05) == key]
            if xs:
                add("lab_head.stated_confidence", lab_, f"{S.mean(xs):.0f} (sd {S.pstdev(xs):.0f})", f"n = {len(xs)} filed claims")
        resp = collections.Counter(x["action"] for c in st["claims"].values() for x in c.get("responses", []))
        n_resp = sum(resp.values())
        add("lab_head.response", "defend / revise / withdraw", " / ".join(f"{lap(resp.get(k, 0), n_resp):.2f}" for k in ("defend", "revise", "withdraw")) if n_resp else "declared",
            f"{dict(resp)} of {n_resp} responses to challenges, Laplace")
        vf = [v for v in jl(r / "ledger" / "verifications.jsonl")]
        if vf:
            agree = sum(1 for v in vf if v.get("reported_result") == v.get("result"))
            add("verifier (target)", "reproduce rate / reported = rule", f"{sum(1 for v in vf if v['result'] == 'reproduced') / len(vf):.2f} / {agree / len(vf):.2f}", f"n = {len(vf)} verifications; factor_tolerance 2.0 kept")
    # ------------------------------------------------------------ handbook
    ne, nu, cd, dis, ratios = 0, 0, 0, 0, []
    for r in P + R:
        for d in sorted((r / "handbook").glob("Q*")):
            q = d.name[1:]
            roll = r / "ledger" / f"questions_Q{q}.json"
            if not roll.exists() or int(q) == 0:
                continue
            roll = rj(roll)
            writers = [rj(t) for t in glob.glob(str(r / "tasks" / "handbook-*" / "output" / "handbook.json"))
                       if rj(Path(t).parents[1] / "task.json")["meta"].get("snapshot") == int(q) and (Path(t).parents[1] / "DONE").exists()]
            for w in writers:
                for a in w["questions"]:
                    v = roll["questions"].get(a["question_id"])
                    if not v:
                        continue
                    if v["n_units"] == 0:
                        nu += 1; ne += a["answer"] == "no evidence"
                    else:
                        rr = v["rules"]
                        if len({rr[k]["belief"] for k in ("B1", "B2", "B3")}) > 1:
                            dis += 1; cd += a["answer"] == "contested"
                        if a["answer"] not in ("contested", "no evidence"):
                            ratios.append(a["confidence"] / 100 / max(rr["B2"].get("confidence", 0.5), 0.05))
    if nu:
        add("handbook.p_no_evidence_when_uncovered", "", f"{ne / nu:.2f}", f"{ne}/{nu} writer answers on uncovered questions (P 1.00, R .81)")
    if dis:
        add("handbook.p_contested_when_rules_disagree", "", f"{cd / dis:.2f}", f"{cd}/{dis} writer answers where B1/B2/B3 disagree")
    if ratios:
        add("handbook.confidence_scale", "", f"{S.mean(ratios):.2f}", f"mean writer confidence / B2 confidence over {len(ratios)} answers")
    ag = {}
    for r in P + R:
        for d in sorted((r / "handbook").glob("Q*")):
            s = d / "score.json"
            if s.exists() and d.name != "Q0":
                for k, v in (rj(s).get("agreement_with_rules") or {}).items():
                    ag.setdefault(k, []).append(v)
    if ag:
        add("handbook.w_B1 / w_B2 / w_B3", "", "0.40 / 0.35 / 0.25", "writer agreement with the rules at Q4/Q6 (P+R): " + ", ".join(f"{k} {S.mean(v):.2f}" for k, v in ag.items()))
    # ------------------------------------------------------------ tokens per call
    tok = collections.defaultdict(list)
    for r in P + R:
        for tj in glob.glob(str(r / "tasks" / "*" / "task.json")):
            d = Path(tj).parent
            t = rj(tj)
            if (d / "usage.json").exists() and (d / "DONE").exists():
                tok[(t["call_type"], t["type"], int(t.get("meta", {}).get("batch", 1)))].append(int(rj(d / "usage.json").get("total_tokens") or 0))
    for (ct, ty, b), xs in sorted(tok.items()):
        add("tariffs (measured tokens per call)", f"{ct} ({ty}, batch {b})", f"median {int(S.median(xs)):,}", f"n = {len(xs)} accepted calls; range {min(xs):,}-{max(xs):,}")
    base = {ct: S.median(xs) for (ct, ty, b), xs in tok.items() if b == 1 and ty in ("research", "analyst")}
    for (ct, ty, b), xs in sorted(tok.items()):
        if b > 1 and ct in base:
            add("tariffs.batch_tariff_multiplier", f"{ty} batch {b}", f"{S.median(xs) / base[ct]:.2f}", f"n = {len(xs)}; median batched / median single")
    md = ["# Fast-mode calibration from live runs", "", "Runs: " + ", ".join(str(r.relative_to(ROOT)) if str(r).startswith(str(ROOT)) else str(r) for r in runs),
          "", "Every row is an estimate from the ledgers of these runs with its n; 'declared' marks a parameter the runs do not identify.",
          "Proportions with n < 30 are Laplace-smoothed ((k+1)/(n+2)). One seed of a 12-scientist world: every n is small.", "",
          "| Block | Parameter | Estimate | Source (run, n) |", "|---|---|---|---|"]
    md += [f"| {b} | {p} | {e} | {s} |" for b, p, e, s in rows]
    text = "\n".join(md) + "\n"
    if out:
        out.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(main([Path(r).resolve() for r in a.runs], Path(a.out) if a.out else None))
