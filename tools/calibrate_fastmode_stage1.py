"""Fast-mode recalibration from the eight 8-quarter live runs of the reported worlds.

Estimates every rule-agent parameter from the live runs' ledgers, state and task outputs (read-only), with its n and
source, and writes configs/fastmode/params-stage1.yaml (configs/fastmode/params.yaml, the earlier calibration from the test runs,
is left as it is) plus a provenance table. Only quarters >= 1 are used: the warm-start pre-history is rule-made in every mode.

Pooling: the paper worlds S0-150 s1-s3 and S0P-150 s1-s2 share the author, reviewer and editor roles (career pressure only
adds profile lines); the repository runs R-b-150 s1-s3 share the lab-head, analyst and verifier roles; the author role
(research call in P, analyst call in R) is pooled across P and R where the parameter exists in both.

Small-n convention (as in tools/calibrate_fastmode.py): proportions with n < 30 are Laplace-smoothed ((k+1)/(n+2));
n >= 30 use k/n. `declared` marks a value the runs do not identify; it is carried over from params.yaml.

    python tools/calibrate_fastmode_stage1.py [--out configs/fastmode/params-stage1.yaml] [--table <md>] [--sims 200]

Two-stage editor fit: (1) the decision weights by maximum likelihood on the live editors' decisions given the live
reviews (ordered probit, the rule's own N(0, 0.5) decision noise and revise band); (2) the two intercepts by indirect
inference, so that the fitted rule editor fed by the fitted rule reviewers (simulated on the live papers' observable
conditions) reproduces the live round-1 reject share and round-2 accept share; then the round-1 accept threshold is set so
that the expected number of round-1 acceptances over the live papers is one half (0 of 231 observed).
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import hashlib
import json
import math
import statistics as S
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

P_RUNS = [("S0-150", 1), ("S0-150", 2), ("S0-150", 3), ("S0P-150", 1), ("S0P-150", 2)]
R_RUNS = [("R-b-150", 1), ("R-b-150", 2), ("R-b-150", 3)]
FLAG_OF = {"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed", "reverse_coded": "reverse_coding_missed",
           "selection_on_outcome": "selection_unacknowledged", "bad_control": "bad_control_included", "miscoded_values": "miscoded_values_kept",
           "unit_mismatch": "unit_mismatch_unfixed", "omitted_standard_control": "omitted_standard_control"}
KINDS_FLAW = ("data", "specification", "inference")
REVISE_BAND = 1.2           # declared in params.yaml; kept (the intercepts absorb the level)
NOISE = 0.5                 # the rule editor's decision noise (rules.py)


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rj(p: Path):
    with open(p, encoding="utf-8-sig") as f:   # some agent-written outputs carry a byte-order mark
        return json.load(f)


def lap(k: int, n: int) -> float | None:
    if n == 0:
        return None
    return (k + 1) / (n + 2) if n < 30 else k / n


def run_dir(scen: str, s: int) -> Path:
    return ROOT / "runs" / scen / str(s) / "llm"


class Fit:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str, str]] = []   # block, parameter, estimate, n, source
        self.v: dict = {}

    def add(self, block, param, value, n, src, key=None):
        self.rows.append((block, param, value if isinstance(value, str) else json.dumps(value), str(n), src))
        if key:
            self.v[key] = value


# ======================================================================================== author / analyst
def fit_author(F: Fit, old: dict) -> None:
    units = []
    for scen, s in P_RUNS:
        rd = run_dir(scen, s)
        qs = {q["id"]: q for q in rj(rd / "landscape" / "questions.json") + rj(rd / "landscape" / "fringe.json")}
        pop = {x["id"]: x for x in rj(rd / "population" / "scientists.json")}
        h1 = {h["paper_id"]: h for h in jl(rd / "ledger" / "hypotheses.jsonl") if h.get("hyp_id") == "H1" and h.get("paper_id")}
        for p in rj(rd / "ledger" / "papers_snapshot.json"):
            if p.get("legacy") or p.get("prehistory"):
                continue
            units.append({"w": "P", "style": p.get("style"), "traps": p["traps"] or [], "flags": p["flags"] or [],
                          "est": (h1.get(p["paper_id"], {}).get("estimator") or ""), "panel": qs.get(p["question_id"], {}).get("panel"),
                          "rep": pop.get(p["lead"], {}).get("repertoire", "archival_regression"), "n_hyp": p["n_hypotheses"],
                          "n_models": p.get("n_models_run"), "slot": p.get("menu_slot")})
    for scen, s in R_RUNS:
        rd = run_dir(scen, s)
        qs = {q["id"]: q for q in rj(rd / "landscape" / "questions.json") + rj(rd / "landscape" / "fringe.json")}
        pop = {x["id"]: x for x in rj(rd / "population" / "scientists.json")}
        st = rj(rd / "state" / "state.json")
        for a in jl(rd / "ledger" / "papers.jsonl"):
            dr = st["drafts"].get(st["analyses"].get(a["analysis_id"], {}).get("draft_id"), {})
            e0 = next((e for e in dr.get("entries", []) if e.get("kind") == "focal"), (dr.get("entries") or [{}])[0])
            units.append({"w": "R", "style": a["style"], "traps": a["traps"] or [], "flags": a["flags"] or [], "est": e0.get("estimator") or "",
                          "panel": qs.get(a["question_id"], {}).get("panel"), "rep": pop.get(a["lead"], {}).get("repertoire", "archival_regression"),
                          "n_hyp": a["n_entries"], "n_models": a.get("n_models_run"), "slot": None})
    nP, nR = sum(u["w"] == "P" for u in units), sum(u["w"] == "R" for u in units)
    styles = collections.Counter(u["style"] for u in units)
    hand = {}
    for t in FLAG_OF:
        k = sum(1 for u in units if t in u["traps"] and FLAG_OF[t] not in u["flags"])
        n = sum(1 for u in units if t in u["traps"])
        if n:
            hand[t] = round(lap(k, n), 2)
            F.add("author.p_handle.VERIFIED", t, hand[t], n, f"{k}/{n} live-authored studies with this problem planted handled (no scorer flag); P {nP} papers + R {nR} analyses" + (", Laplace" if n < 30 else ""))
        else:
            hand[t] = old["author"]["p_handle"]["VERIFIED"][t]
            F.add("author.p_handle.VERIFIED", t, hand[t], 0, "declared (kept): not planted in any live-authored study")
    F.v["p_handle_VERIFIED"] = hand
    F.add("author.p_handle.AUTO", "*", "declared (kept)", 0, f"no AUTO author in Stage 1 (styles {dict(styles)})")

    def is_fe(est: str) -> bool:
        e = est.lower()
        return ("fixed" in e and "effect" in e) or "within" in e or "two-way" in e or e.strip() == "fe"
    fe = {}
    for rep in ("archival_regression", "causal_design", "survey"):
        pan = [u for u in units if u["panel"] and u["rep"] == rep and u["est"]]
        k = sum(is_fe(u["est"]) for u in pan)
        if pan:
            fe[rep] = round(lap(k, len(pan)), 2)
            F.add("author.p_use_fe_panel", rep, fe[rep], len(pan), f"{k}/{len(pan)} panel studies by {rep} leads with firm fixed effects in the focal estimator string (P+R)" + (", Laplace" if len(pan) < 30 else ""))
        else:
            fe[rep] = old["author"]["p_use_fe_panel"][rep]
            F.add("author.p_use_fe_panel", rep, fe[rep], 0, "declared (kept)")
    F.v["p_use_fe_panel"] = fe
    nh = collections.Counter(int(u["n_hyp"]) for u in units if u["n_hyp"])
    tot = sum(nh.values())
    dist = {k: round(v / tot, 3) for k, v in sorted(nh.items())}
    F.add("author.n_hypotheses", "distribution", dist, tot, f"hypotheses per paper (P, median {S.median([u['n_hyp'] for u in units if u['w'] == 'P'])}) and entries per analysis (R, median {S.median([u['n_hyp'] for u in units if u['w'] == 'R'])}), pooled")
    F.v["n_hypotheses"] = dist
    slots = collections.Counter(u["slot"] for u in units if u["slot"])
    ns = sum(slots.values())
    F.add("author.p_choose_fringe", "", round(lap(slots.get("fringe", 0), ns), 2), ns, f"{slots.get('fringe', 0)}/{ns} live P menu choices took the fringe slot (slots {dict(slots)})", key="p_choose_fringe")
    F.add("author.p_choose_crowded", "", round(lap(slots.get("crowded", 0), ns), 2), ns, f"{slots.get('crowded', 0)}/{ns} live P menu choices took the crowded slot", key="p_choose_crowded")
    nm = [u["n_models"] for u in units if u.get("n_models")]
    F.add("author.n_models_run.VERIFIED", "[p10, p90]", [int(np.quantile(nm, .1)), int(round(np.quantile(nm, .9)))], len(nm), f"models run per study, median {np.median(nm):.0f} (P+R)", key="n_models_run_VERIFIED")
    F.add("author.p_mark_contradiction_when_disagree", "", "declared (kept)", 0, "the disagreement denominator is not in the ledger")
    F.add("author.p_revise_when_asked", "", "kept 0.9", 0, "an engine lottery applied in live and fast runs alike (paper.py publish phase), so the live runs cannot re-identify it")
    F.add("author.p_limitations_mention_selection", "", "declared (kept)", 0, "not used by the fast path; no selection problem planted in Stage 1")
    # repository analysts: verdict from the found direction
    hy = [h for scen, s in R_RUNS for h in jl(run_dir(scen, s) / "ledger" / "hypotheses.jsonl") if not h.get("legacy") and h.get("analysis_id")]
    n_pred = sum(1 for h in hy if h.get("predicted_sign"))
    sig = [h for h in hy if (h.get("p") or 1) < 0.05]
    sup_sig = sum(1 for h in sig if h["verdict"] in ("supported", "partially"))
    F.add("author.analyst_reports_found_sign", "", True, len(hy), f"{n_pred}/{len(hy)} live analyst entries carried a predicted sign; {sup_sig}/{len(sig)} significant entries were reported supported, "
          f"{sum(1 for h in hy if (h.get('p') or 1) >= 0.05 and h['verdict'] == 'supported')}/{len(hy) - len(sig)} non-significant ones", key="analyst_reports_found_sign")


# ======================================================================================== reviewer
def load_reviews():
    rows = []
    for scen, s in P_RUNS:
        rd = run_dir(scen, s)
        snap = {p["paper_id"]: p for p in rj(rd / "ledger" / "papers_snapshot.json")}
        h1 = {h["paper_id"]: h for h in jl(rd / "ledger" / "hypotheses.jsonl") if h.get("hyp_id") == "H1" and h.get("paper_id")}
        for x in jl(rd / "ledger" / "reviews.jsonl"):
            if int(x.get("quarter", 1)) < 1:
                continue
            p = snap.get(x["paper_id"], {})
            x = dict(x)
            x["run"] = f"{scen}-{s}"
            x["flagged"] = bool(x.get("analyst_flags"))
            x["supported"] = h1.get(x["paper_id"], {}).get("verdict") in ("supported", "partially")
            x["paper_relation"] = p.get("relation_to_prior")
            rows.append(x)
    return rows


def fit_reviewer(F: Fit, rv: list[dict]) -> None:
    r1 = [x for x in rv if x["round"] == 1]
    # cell means on round 1 (unselected papers) with relation adjustments, by OLS
    X, y = [], []
    for x in r1:
        X.append([1.0, float(x["flagged"]), float((not x["flagged"]) and x["supported"]),
                  float(x["paper_relation"] == "contradiction"), float(x["paper_relation"] == "replication")])
        y.append(x["validity_probability"])
    X, y = np.array(X), np.array(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    clean_null, flawed, clean_sup = b[0], b[0] + b[1], b[0] + b[2]
    n_cs = int(X[:, 2].sum()); n_fl = int(X[:, 1].sum()); n_cn = len(r1) - n_cs - n_fl
    F.add("reviewer.validity_mean", "clean_null", round(float(clean_null), 1), n_cn, "round-1 reviews (quarter >= 1) of unflagged papers whose H1 is not supported: OLS intercept (relation adjusted)", key="vm_clean_null")
    F.add("reviewer.validity_mean", "clean_supported", round(float(clean_sup), 1), n_cs, "unflagged, H1 supported or partially", key="vm_clean_supported")
    F.add("reviewer.validity_mean", "flawed_noticed = flawed_unnoticed", round(float(flawed), 1), n_fl, "reviews of papers carrying a scorer flag (noticing is not identified, see p_notice)", key="vm_flawed")
    F.add("reviewer.validity_mean", "contradiction_penalty", round(float(b[3]), 1), int(X[:, 3].sum()), "OLS coefficient, paper relation = contradiction", key="vm_contra")
    F.add("reviewer.validity_mean", "replication_bonus", round(float(b[4]), 1), int(X[:, 4].sum()), "OLS coefficient, paper relation = replication", key="vm_repl")
    # between- and within-paper SD of the residual (one-way ANOVA on paper-rounds, round 1)
    res = y - X @ b
    g = collections.defaultdict(list)
    for x, e in zip(r1, res):
        g[(x["run"], x["paper_id"])].append(e)
    grp = [v for v in g.values() if len(v) >= 2]
    allr = np.concatenate([np.array(v) for v in grp]); gm = allr.mean()
    k = float(np.mean([len(v) for v in grp]))
    msb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in grp) / (len(grp) - 1)
    msw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in grp) / (len(allr) - len(grp))
    sd_b, sd_w = math.sqrt(max(0.0, (msb - msw) / k)), math.sqrt(msw)
    F.add("reviewer.validity_paper_sd", "", round(sd_b, 1), len(grp), f"between-paper SD of the residual validity, one-way ANOVA over {len(grp)} round-1 paper reviews of k = {k:.1f} referees (ICC {sd_b**2 / (sd_b**2 + sd_w**2):.2f})", key="paper_sd")
    F.add("reviewer.validity_sd", "", round(sd_w, 1), len(allr), "within-paper SD of the residual validity (round 1)", key="validity_sd")
    # recommendation cuts at the live validity quantiles of the pooled reviews (both rounds)
    vals = np.sort([x["validity_probability"] for x in rv])
    rc = collections.Counter(x["recommendation"] for x in rv)
    q_rej = float(np.quantile(vals, rc["reject"] / len(rv)))
    q_maj = float(np.quantile(vals, 1 - rc["minor_revision"] / len(rv)))
    F.v["rec_targets"] = (rc["reject"] / len(rv), rc["minor_revision"] / len(rv))
    F.add("reviewer.recommendation_cuts", "reject / major (live quantiles; superseded)", [round(q_rej, 1), round(q_maj, 1)], len(rv), f"validity quantiles at the live reject ({rc['reject']}/{len(rv)}) and minor ({rc['minor_revision']}/{len(rv)}) shares, both rounds", key="cuts")
    F.add("reviewer.recommendation_cuts", "minor", 99.5, len(rv), "no live review recommended accept: everything above the major cut is a minor revision")
    F.add("reviewer.p_accept_above_minor_cut", "", 0.0, rc.get("minor_revision", 0), f"0 accept recommendations in {len(rv)} live reviews ({rc.get('minor_revision', 0)} minor)")
    clean = [x for x in rv if not x["flagged"]]
    flagged = [x for x in rv if x["flagged"]]
    kc = sum(any(k_ in KINDS_FLAW for k_ in x["issue_kinds"]) for x in clean)
    F.add("reviewer.p_unprompted_issue", "", round(lap(kc, len(clean)), 3), len(clean), f"{kc}/{len(clean)} reviews of unflagged papers raised a data, specification or inference objection", key="p_unprompted_issue")
    km = sum(x["n_issues_major"] > 0 for x in clean)
    F.add("reviewer.p_unprompted_major", "", round(lap(km, len(clean)), 3), len(clean), f"{km}/{len(clean)} reviews of unflagged papers raised at least one major issue", key="p_unprompted_major")
    # detection: kind-specific issue rate on flagged papers vs the same kind on clean papers
    base = {k_: np.mean([k_ in x["issue_kinds"] for x in clean]) for k_ in ("data", "specification", "inference", "reporting", "interpretation")}
    from synthsci.agents.rules import TRAP_KIND, _flag_to_trap
    obs = expct = n_tot = 0.0
    per = {}
    for x in flagged:
        for f in x["analyst_flags"]:
            t = _flag_to_trap(f)
            if not t:
                continue
            kd = TRAP_KIND[t]
            hit = kd in x["issue_kinds"]
            obs += hit; expct += base[kd]; n_tot += 1
            per.setdefault(kd, [0, 0, base[kd]]); per[kd][0] += hit; per[kd][1] += 1
    lift = (obs - expct) / (n_tot - expct) if n_tot > expct else 0.0
    F.add("reviewer.p_notice", "* (all problems)", 0.0, int(n_tot),
          f"not identified as detection: the kind-matched objection rate on flagged papers ({obs:.0f}/{n_tot:.0f}) equals the base rate on clean papers ({expct:.1f} expected); pooled lift {lift:.3f}; by kind "
          + "; ".join(f"{k_} {a}/{n} vs base {bb:.2f}" for k_, (a, n, bb) in sorted(per.items())) + ". Set 0: the objections are carried by p_unprompted_issue", key="p_notice")
    F.add("reviewer.p_notice_assist_multiplier / load_penalty_per_paper", "", "declared (kept)", 0, "no prescreen or AI-written review arm in Stage 1; noticing not identified")
    maj = [x for x in rv if x["n_issues_major"] > 0]
    kf = sum(bool(x["flaw_found"]) for x in maj)
    F.add("reviewer.p_flaw_found_given_noticed_major", "", round(kf / len(maj), 3), len(maj), f"{kf}/{len(maj)} reviews with >= 1 major issue set flaw_found (0/{len(rv) - len(maj)} without)", key="p_flaw_found")
    F.add("reviewer.confidence", "mean / sd", [round(float(np.mean([x['confidence'] for x in rv])), 3), round(float(np.std([x['confidence'] for x in rv])), 3)], len(rv), "stated confidence, all live reviews", key="confidence")
    F.add("reviewer.p_prefer_fresh_reviewers_round2", "", "declared (kept)", 0, "not used by the engine")


# ======================================================================================== editor
def load_decisions():
    out = []
    for scen, s in P_RUNS:
        rd = run_dir(scen, s)
        for d in jl(rd / "ledger" / "decisions.jsonl"):
            if int(d.get("quarter", 1)) >= 1:
                d = dict(d); d["run"] = f"{scen}-{s}"
                out.append(d)
    return out


def x0_of(par: dict, rnd: int, mv: float, recs: list[str], count: int, rel: str | None) -> float:
    """The rule editor's linear index without its noise (mirrors RuleEditor.decide)."""
    return (par["intercept_round1"] if rnd == 1 else par["intercept_round2"]) + par["w_mean_validity"] * mv \
        + par["w_reject_votes"] * recs.count("reject") + par["w_major_votes"] * recs.count("major_revision") \
        + par["w_accept_votes"] * recs.count("accept") + par["w_log_count_on_question"] * math.log1p(max(count, 0)) \
        + (par["w_contradiction"] if rel == "contradiction" else 0.0)


def fit_editor_ml(F: Fit, dec: list[dict], old: dict) -> dict:
    from scipy.optimize import minimize
    from scipy.stats import norm
    rows = [d for d in dec if d["stage"] in ("round1", "round2") and d.get("mean_validity") is not None]
    names = ["intercept_round1", "intercept_round2", "w_mean_validity", "w_reject_votes", "w_major_votes", "w_log_count_on_question", "w_contradiction"]

    def nll(th):
        par = dict(zip(names, th)) | {"w_accept_votes": 0.0}
        ll = 0.0
        for d in rows:
            rnd = 1 if d["stage"] == "round1" else 2
            x = x0_of(par, rnd, d["mean_validity"], d["recommendations"], d.get("count_on_question", 0), d.get("relation_to_prior"))
            if rnd == 1:
                pr = {"reject": norm.cdf((-REVISE_BAND - x) / NOISE), "accept": 0.0}
                pr["revise"] = 1 - pr["reject"]
            else:
                pa = 1 - norm.cdf((-REVISE_BAND / 2 - x) / NOISE)
                pr = {"accept": pa, "reject": 1 - pa}
            ll += math.log(max(pr.get(d["decision"], 0.0), 1e-12))
        return -ll
    e = old["editor"]
    th0 = np.array([e["intercept_round1"], e["intercept_round2"], e["w_mean_validity"], e["w_reject_votes"], e["w_major_votes"], e["w_log_count_on_question"], e["w_contradiction"]], dtype=float)
    res = minimize(nll, th0, method="BFGS")
    # standard errors from the BFGS inverse Hessian
    se = np.sqrt(np.clip(np.diag(res.hess_inv), 0, None))
    par = dict(zip(names, res.x))
    c1 = collections.Counter(d["decision"] for d in rows if d["stage"] == "round1")
    c2 = collections.Counter(d["decision"] for d in rows if d["stage"] == "round2")
    for nm, val, s_ in zip(names, res.x, se):
        if nm.startswith("intercept"):
            continue
        F.add("editor." + nm, "", round(float(val), 3), len(rows), f"ML (ordered probit, the rule's N(0, .5) noise, revise band {REVISE_BAND}) over {len(rows)} live decisions (round 1 {dict(c1)}; round 2 {dict(c2)}); se {s_:.3f}")
    F.add("editor.intercept_round1 / round2 (stage 1, ML)", "", [round(par['intercept_round1'], 3), round(par['intercept_round2'], 3)], len(rows), f"ML given the live reviews (se {se[0]:.2f} / {se[1]:.2f}); replaced by the indirect-inference values below")
    F.add("editor.w_accept_votes", "", "declared (kept)", 0, "no live reviewer recommended accept")
    F.v["editor_ml"] = par | {"w_accept_votes": old["editor"]["w_accept_votes"]}
    F.v["editor_rows"] = rows
    F.v["editor_counts"] = (c1, c2)
    return par


def paper_conditions(dec: list[dict], rv: list[dict]) -> list[dict]:
    """The live round-1 papers' observable conditions as the rule reviewer sees them."""
    rv_by = collections.defaultdict(list)
    for x in rv:
        rv_by[(x["run"], x["paper_id"], x["round"])].append(x)
    out = []
    for d in dec:
        if d["stage"] != "round1":
            continue
        rs = rv_by.get((d["run"], d["paper_id"], 1), [])
        if not rs:
            continue
        x = rs[0]
        out.append({"flagged": x["flagged"], "supported": x["supported"], "relation": x["paper_relation"], "count": d.get("count_on_question", 0),
                    "k": len(rs), "flags": list(x["analyst_flags"])})
    return out


def simulate(params: dict, conds: list[dict], sims: int, seed: int = 7) -> dict:
    """Run the fitted rule reviewer and rule editor (rules.py, exactly as the engine calls them) on the live papers'
    conditions: round-1 reviews and decision; a revised paper is re-reviewed (same paper shock) and decided in round 2."""
    from synthsci.agents.rules import RuleEditor, RuleReviewer
    rev, ed = RuleReviewer(params), RuleEditor(params)
    rng = np.random.default_rng(seed)
    n1 = collections.Counter(); n2 = collections.Counter(); recs = collections.Counter(); x1 = []; vals = []
    agree = []
    for i in range(sims):
        for j, c in enumerate(conds):
            item = {"paper_id": f"S{i}-{j}", "hypotheses": [{"iv": "x", "dv": "y", "n": 1000 + j, "se": float(rng.uniform(0.001, 1.0)),
                                                              "verdict": "supported" if c["supported"] else "not_supported"}],
                    "visible_problems": c["flags"], "relation_to_prior": c["relation"]}
            out1 = [rev.review(item, rng, "none", 3) for _ in range(c["k"])]
            recs.update(r["recommendation"] for r in out1)
            vals.extend(r["validity_probability"] for r in out1)
            agree.append(len({r["recommendation"] for r in out1}) == 1)
            d1 = ed.decide(item, out1, 1, c["count"], rng)
            n1[d1["decision"]] += 1
            mv = float(np.mean([r["validity_probability"] for r in out1]))
            x1.append(x0_of(params["editor"], 1, mv, [r["recommendation"] for r in out1], c["count"], c["relation"]))
            if d1["decision"] == "revise":
                out2 = [rev.review(item, rng, "none", 3) for _ in range(c["k"])]
                recs.update(r["recommendation"] for r in out2)
                vals.extend(r["validity_probability"] for r in out2)
                agree.append(len({r["recommendation"] for r in out2}) == 1)
                n2[ed.decide(item, out2, 2, c["count"], rng)["decision"]] += 1
    tot1, tot2, totr = sum(n1.values()), sum(n2.values()), sum(recs.values())
    pe = sum((v / totr) ** 2 for v in recs.values())
    po = float(np.mean(agree))
    return {"r1": {k: v / tot1 for k, v in n1.items()}, "r2_accept": n2.get("accept", 0) / tot2 if tot2 else None, "n_r2": tot2 / sims,
            "recs": {k: v / totr for k, v in recs.items()}, "kappa": (po - pe) / (1 - pe), "x1": np.array(x1), "vals": np.array(vals)}


def fit_editor_ii(F: Fit, params: dict, conds: list[dict], sims: int) -> None:
    """Indirect inference: the intercepts that make the composite (rule reviewers -> rule editor) match the live shares."""
    c1, c2 = F.v["editor_counts"]
    t_rej = c1["reject"] / sum(c1.values())
    t_acc2 = c2["accept"] / sum(c2.values())
    p = copy.deepcopy(params)
    p["editor"]["accept_cut_round1"] = 50.0     # no round-1 acceptances while the intercepts are searched

    def bisect(fun, lo, hi, target, increasing, it=16):
        for _ in range(it):
            mid = (lo + hi) / 2
            v = fun(mid)
            if (v < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    # the recommendation cuts are set at the quantiles of the composite's own validity distribution (both rounds) that
    # reproduce the live reject and minor shares (the live-quantile cuts do not, because the rule's validity distribution is
    # not the live one); cuts and intercepts depend on each other through round-2 selection, so two passes
    rej_share, min_share = F.v["rec_targets"]
    for _ in range(2):
        sim0 = simulate(p, conds, sims)
        cr = float(np.quantile(sim0["vals"], rej_share))
        cm = float(np.quantile(sim0["vals"], 1 - min_share))
        p["reviewer"]["recommendation_cuts"] = {"reject": round(cr, 1), "major": round(cm, 1), "minor": 99.5}
        i1 = bisect(lambda v: simulate(p | {"editor": p["editor"] | {"intercept_round1": v}}, conds, sims)["r1"].get("reject", 0.0),
                    p["editor"]["intercept_round1"] - 6, p["editor"]["intercept_round1"] + 6, t_rej, increasing=False)
        p["editor"]["intercept_round1"] = i1
        i2 = bisect(lambda v: simulate(p | {"editor": p["editor"] | {"intercept_round2": v}}, conds, sims)["r2_accept"] or 0.0,
                    p["editor"]["intercept_round2"] - 6, p["editor"]["intercept_round2"] + 6, t_acc2, increasing=True)
        p["editor"]["intercept_round2"] = i2
    F.add("reviewer.recommendation_cuts", "reject / major (indirect inference; used)", [round(cr, 1), round(cm, 1)], len(conds),
          f"quantiles of the composite simulation's validity (both rounds) at the live reject ({rej_share:.3f}) and minor ({min_share:.3f}) shares; replaces the live-quantile cuts above", key="cuts")
    sim = simulate(p, conds, sims)
    # round-1 accept threshold: expected round-1 acceptances over the live papers = 1/2 (0 of n observed)
    from scipy.optimize import brentq
    from scipy.stats import norm
    x1 = sim["x1"]
    n_live = sum(c1.values())
    target = 0.5 / n_live
    cut = brentq(lambda c: float(np.mean(1 - norm.cdf((c - x1) / NOISE))) - target, -10, 60)
    F.add("editor.intercept_round1", "", round(i1, 3), sum(c1.values()), f"indirect inference: the composite rule reviewer -> rule editor on the {len(conds)} live round-1 papers' conditions ({sims} simulations each) reproduces the live round-1 reject share {t_rej:.3f} ({c1['reject']}/{sum(c1.values())})", key="i1")
    F.add("editor.intercept_round2", "", round(i2, 3), sum(c2.values()), f"indirect inference: reproduces the live round-2 accept share {t_acc2:.3f} ({c2['accept']}/{sum(c2.values())}); the simulated round 2 keeps each paper's shared referee shift", key="i2")
    F.add("editor.accept_cut_round1", "", round(cut, 3), n_live, f"0/{n_live} live round-1 acceptances: threshold set so the expected count over the live papers is 1/2 (mean P(accept) {target:.4f}); old rule: revise_band/2 = {REVISE_BAND / 2}", key="cut1")
    F.v["sim_check"] = sim


# ======================================================================================== desk
def fit_desk(F: Fit, dec: list[dict]) -> None:
    desk = [d for d in dec if d["stage"] == "desk"]
    k = sum(d["decision"] == "desk_reject" for d in desk)
    F.add("editor.desk_reject_p.scope_only", "", round(lap(k, len(desk)), 3), len(desk), f"{k}/{len(desk)} live desk decisions were desk rejections (S0-150 desk strictness scope_only)", key="desk_scope")
    nulls = [d for d in desk if d.get("truth_check_focal") in ("correct_null", "correct_null_negligible", "missed_effect")]
    F.add("editor.desk_reject_null_extra", "", 0.0, len(nulls), f"0 desk rejections among {len(nulls)} desk decisions on papers whose focal result is null", key="desk_null")
    F.add("editor.desk_reject_p.contribution / design", "", "declared (kept)", 0, "not the Stage 1 strictness")
    F.add("editor.revise_band / merit_sd", "", "declared (kept)", 0, "the band is absorbed by the intercepts; merit is not a decision input")


# ======================================================================================== lab head, verifier
def fit_lab(F: Fit, old: dict) -> None:
    votes_all, claims_n = [], {}
    reuse_had = reuse_k = pool_k = 0
    fresh_n = req_n = 0
    mults = []
    resp = collections.Counter()
    filed = collections.Counter(); decided = collections.Counter()
    conf = {True: [], False: []}
    offers = collections.Counter()
    vf = []
    for scen, s in R_RUNS:
        rd = run_dir(scen, s)
        for c in jl(rd / "ledger" / "claims.jsonl"):
            if c.get("n") is not None:
                claims_n[(scen, s, c["claim_id"])] = c["n"]
        vt = [v for v in jl(rd / "ledger" / "votes.jsonl") if int(v.get("quarter", 1)) >= 1]
        for v in vt:
            if v.get("vote") in ("endorse", "abstain", "challenge"):
                v = dict(v); v["n"] = claims_n.get((scen, s, v["claim_id"]))
                votes_all.append(v)
            elif v.get("vote") == "offer":
                offers[(scen, s, v["quarter"], v["voter_lab"])] += 1
        st = rj(rd / "state" / "state.json")
        qs = {q["id"]: q for q in rj(rd / "landscape" / "questions.json") + rj(rd / "landscape" / "fringe.json")}
        ds = jl(rd / "ledger" / "datasets.jsonl")
        avail = collections.defaultdict(list)
        for d in ds:
            avail[d["question_id"]].append(d["quarter"])
        for a in st["analyses"].values():
            if int(a.get("quarter", 1)) < 1:
                continue
            if any(q < a["quarter"] for q in avail[a["question_id"]]):
                reuse_had += 1
                if not a["fresh"]:
                    reuse_k += 1
                    pool_k += len(a["deposits"]) >= 2
            if a["fresh"]:
                fresh_n += 1
                if a.get("n_requested"):
                    req_n += 1
                    p50 = (qs.get(a["question_id"], {}).get("frame_quantiles") or {}).get("p50")
                    if p50:
                        mults.append(a["n_requested"] / p50)
        for c in st["claims"].values():
            for x in c.get("responses", []):
                if int(x.get("quarter", 1) or 1) >= 1:
                    resp[x["action"]] += 1
        for h in jl(rd / "ledger" / "hypotheses.jsonl"):
            if h.get("legacy") or not h.get("analysis_id"):
                continue
            sig = (h.get("p") or 1) < 0.05
            if h.get("status") == "never_filed":
                continue
            decided[sig] += 1
            filed[sig] += h.get("status") != "dropped"
        for c in jl(rd / "ledger" / "claims.jsonl"):
            if c.get("legacy") or c.get("event") != "filed" or c.get("confidence_stated") is None:
                continue
            conf[(c.get("p") or 1) < 0.05].append(c["confidence_stated"])
        vf += [v for v in jl(rd / "ledger" / "verifications.jsonl") if int(v.get("quarter", 1)) >= 1]
    F.add("lab_head.p_reuse_given_catalogue", "", round(lap(reuse_k, reuse_had), 3), reuse_had, f"{reuse_k}/{reuse_had} live agenda items on a question with a catalogued deposit reused one", key="p_reuse")
    F.add("lab_head.p_pool_given_two", "", round(lap(pool_k, reuse_k), 3), reuse_k, f"{pool_k}/{reuse_k} reuses pooled two or more deposits", key="p_pool")
    pf = {}
    for sig, lab_ in ((True, "significant"), (False, "insignificant")):
        pf[lab_] = round(lap(filed[sig], decided[sig]), 3)
        F.add("lab_head.p_file_draft", lab_, pf[lab_], decided[sig], f"{filed[sig]}/{decided[sig]} {lab_} draft entries filed among those the head decided (never-filed entries at the horizon excluded)")
    F.v["p_file_draft"] = pf
    sdc = math.sqrt(np.mean([(x - np.mean(conf[k])) ** 2 for k in conf for x in conf[k]]))
    F.add("lab_head.stated_confidence", "significant / insignificant / sd", [round(float(np.mean(conf[True])), 1), round(float(np.mean(conf[False])), 1), round(sdc, 1)],
          len(conf[True]) + len(conf[False]), f"stated confidence of filed claims (n {len(conf[True])} / {len(conf[False])}); sd pooled within", key="stated_conf")
    # vote: logit of endorse on the rule's covariates
    import statsmodels.api as sm
    V = [v for v in votes_all if v.get("n") is not None]
    X = np.array([[1.0, 1.0 if v.get("significant") else 0.0, math.log1p(v["n"]) / 5, float(v.get("prior_endorsements", 0)),
                   1.0 if v.get("same_theory") else 0.0, 1.0 if v.get("reciprocity") else 0.0, float(v.get("proposer_size", 0))] for v in V])
    y = np.array([1.0 if v["vote"] == "endorse" else 0.0 for v in V])
    m = sm.Logit(y, X).fit(disp=0, maxiter=200)
    nm = ["intercept", "b_significant", "b_log_n", "b_prior_endorsements", "b_same_theory", "b_reciprocity", "b_proposer_size"]
    vote = {}
    for k_, b_, s_ in zip(nm, m.params, m.bse):
        vote[k_] = round(float(b_), 3)
        F.add("lab_head.vote." + k_, "", vote[k_], len(V), f"logit of endorse (vs abstain or challenge) on the rule's covariates (log1p(N)/5, prior endorsements as a count, proposer size in members), {len(V)} live votes; se {s_:.3f}")
    sig = [v for v in votes_all if v.get("significant")]
    ins = [v for v in votes_all if not v.get("significant")]
    vote["p_challenge_base"] = round(sum(v["vote"] == "challenge" for v in sig) / len(sig), 3)
    vote["p_challenge_insignificant"] = round(sum(v["vote"] == "challenge" for v in ins) / len(ins), 3)
    F.add("lab_head.vote.p_challenge_base", "", vote["p_challenge_base"], len(sig), f"{sum(v['vote'] == 'challenge' for v in sig)}/{len(sig)} votes on significant claims were challenges")
    F.add("lab_head.vote.p_challenge_insignificant", "", vote["p_challenge_insignificant"], len(ins), f"{sum(v['vote'] == 'challenge' for v in ins)}/{len(ins)} votes on insignificant claims")
    sp = {}
    for k_ in ("endorse", "challenge", "abstain"):
        xs = [v["probability"] for v in votes_all if v["vote"] == k_ and v.get("probability") is not None]
        sp[k_] = round(float(np.mean(xs)), 1)
    sp["sd"] = round(math.sqrt(np.mean([(v["probability"] - sp[v["vote"]]) ** 2 for v in votes_all if v.get("probability") is not None])), 1)
    F.add("lab_head.vote.stated_probability", "endorse / challenge / abstain / sd", [sp["endorse"], sp["challenge"], sp["abstain"], sp["sd"]], len(votes_all), "stated probability by vote; sd pooled within")
    vote["stated_probability"] = sp
    F.v["vote"] = vote
    F.add("lab_head.p_offer_verify", "", "kept 0.95", sum(offers.values()), f"{sum(offers.values())} live offers over {len(offers)} lab-quarters; capacity per lab-quarter is not in the ledger, so the offer rate is not re-identified")
    n_resp = sum(resp.values())
    rr = {k_: round(lap(resp.get(k_, 0), n_resp), 3) for k_ in ("defend", "revise", "withdraw")}
    F.add("lab_head.response", "defend / revise / withdraw", [rr["defend"], rr["revise"], rr["withdraw"]], n_resp, f"{dict(resp)} responses to challenges", key="response")
    F.add("lab_head.p_request_n", "", round(lap(req_n, fresh_n), 3), fresh_n, f"{req_n}/{fresh_n} live fresh-data agenda items named a dataset size", key="p_request_n")
    qd = np.quantile(mults, [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]) if mults else [1.0]
    dist = {round(float(x), 2): 0.1 for x in qd}
    F.add("lab_head.request_n_multiple", "deciles (equal weight)", dist, len(mults), f"requested N / the question's median frame N, deciles of {len(mults)} requests (median {np.median(mults):.2f}, p90 {np.quantile(mults, .9):.2f})", key="request_n_multiple")
    F.add("lab_head.stake / p_challenge_with_test", "", "declared (kept)", 0, "variants c and d were not run")
    agree = sum(1 for v in vf if v.get("reported_result") == v.get("result"))
    rep = sum(1 for v in vf if v.get("result") == "reproduced")
    F.add("verifier.factor_tolerance", "", "kept 2.0", len(vf), f"the rule's verdict on the live verifiers' own estimates agreed with their reported verdict in {agree}/{len(vf)} verifications; reproduced {rep}/{len(vf)}")
    F.add("verifier.p_handle", "", "VERIFIED (kept)", 0, "verifiers use author.p_handle[style]; every Stage 1 verifier was VERIFIED")


# ======================================================================================== handbook
def fit_handbook(F: Fit) -> None:
    ne = nu = cd = dis = 0
    ratios = []
    picks = []    # for covered answers where B1/B2/B3 do not all agree: the set of rules the answer matches
    for scen, s in P_RUNS + R_RUNS:
        rd = run_dir(scen, s)
        for d in sorted((rd / "handbook").glob("Q*")):
            q = int(d.name[1:])
            if q == 0:
                continue       # the S0P quarter-0 handbooks are reused from S0 (byte-identical inputs under common random numbers): Q0 is skipped for all runs
            roll = rd / "ledger" / f"questions_Q{q}.json"
            if not roll.exists():
                continue
            roll = rj(roll)
            for t in glob.glob(str(rd / "tasks" / "handbook-*" / "output" / "handbook.json")):
                tp = Path(t).parents[1]
                if rj(tp / "task.json")["meta"].get("snapshot") != q or not (tp / "DONE").exists():
                    continue
                for a in rj(t)["questions"]:
                    v = roll["questions"].get(a["question_id"])
                    if not v:
                        continue
                    if v["n_units"] == 0:
                        nu += 1; ne += a["answer"] == "no evidence"
                        continue
                    rr = v["rules"]
                    bel = {k: rr[k]["belief"] for k in ("B1", "B2", "B3")}
                    if len(set(bel.values())) > 1:
                        dis += 1; cd += a["answer"] == "contested"
                        if a["answer"] not in ("contested", "no evidence"):
                            picks.append({k for k, b in bel.items() if b == a["answer"]})
                    if a["answer"] not in ("contested", "no evidence"):
                        ratios.append(a["confidence"] / 100 / max(rr["B2"].get("confidence", 0.5), 0.05))
    F.add("handbook.p_no_evidence_when_uncovered", "", round(ne / nu, 3), nu, f"{ne}/{nu} writer answers (Q4, Q8; 8 runs) on questions with no certified unit said 'no evidence'", key="hb_ne")
    F.add("handbook.p_contested_when_rules_disagree", "", round(cd / dis, 3), dis, f"{cd}/{dis} writer answers where B1, B2 and B3 disagreed said 'contested'", key="hb_cd")
    F.add("handbook.confidence_scale", "", round(float(np.mean(ratios)), 3), len(ratios), "mean writer confidence / B2 confidence over answered covered questions", key="hb_cs")
    # multinomial weights by maximum likelihood when the answer may match several rules
    best = None
    for w1 in np.arange(0.02, 0.97, 0.01):
        for w2 in np.arange(0.02, 0.97 - w1 + 1e-9, 0.01):
            w3 = 1 - w1 - w2
            if w3 < 0.01:
                continue
            w = {"B1": w1, "B2": w2, "B3": w3}
            ll = sum(math.log(max(sum(w[k] for k in m), 1e-9)) for m in picks)
            if best is None or ll > best[0]:
                best = (ll, w1, w2, w3)
    matched = [m for m in picks if m]
    F.add("handbook.w_B1 / w_B2 / w_B3", "", [round(best[1], 2), round(best[2], 2), round(best[3], 2)], len(picks),
          f"ML of the rule's pick weights over {len(picks)} answers where the rules disagreed ({len(picks) - len(matched)} matched none)", key="hb_w")


# ======================================================================================== output
def render_yaml(F: Fit, old: dict) -> str:
    v = F.v
    e = v["editor_ml"]
    vt = v["vote"]
    hp = v["p_handle_VERIFIED"]
    auto = old["author"]["p_handle"]["AUTO"]
    pn = {k: 0.0 for k in old["reviewer"]["p_notice"]}
    fmt = lambda d: "{" + ", ".join(f"{k}: {json.dumps(x)}" for k, x in d.items()) + "}"
    L = [
        "# Rule-agent parameters recalibrated for fast mode from the eight 8-quarter live runs of the reported worlds. Fast mode only:",
        "# the reported live runs' warm start used configs/fastmode/params.yaml.",
        "# STATUS: calibrated-stage1-2026-10-15. Written by tools/calibrate_fastmode_stage1.py (do not edit by hand; rerun it).",
        "# Sources (read-only, quarters >= 1 only; the warm-start pre-history is rule-made in every mode):",
        "#   paper world: runs/S0-150/{1,2,3}/llm and runs/S0P-150/{1,2}/llm pooled (authors, referees, editor);",
        "#   repository world: runs/R-b-150/{1,2,3}/llm pooled (lab heads, analysts, verifiers); author role pooled across P and R.",
        "# The same script writes a table with every value's estimate, n and source. Convention: proportions with n < 30",
        "# Laplace-smoothed ((k+1)/(n+2)); `declared (kept)` = not identified by these runs, carried over from",
        "# configs/fastmode/params.yaml (the 2026-10-14 calibration from the early test runs).",
        "# Keys new in this file (absent or 0 in params.yaml = the earlier rule behaviour): reviewer.validity_paper_sd, p_unprompted_issue,",
        "# p_unprompted_major; editor.accept_cut_round1; author.analyst_reports_found_sign; lab_head.p_request_n, request_n_multiple.",
        "# Institutional settings (timing, tenure, capacity, truth) are NOT here: they live in the scenario configs and are design",
        "# choices grounded in the corpus.",
        "status: calibrated-stage1-2026-10-15",
        "author:",
        "  # VERIFIED: live-authored P papers + R analyses (all VERIFIED); reverse_coded and selection_on_outcome never planted: declared (kept)",
        f"  p_handle:",
        f"    VERIFIED: {fmt(hp)}",
        f"    AUTO:     {fmt(auto)}   # declared (kept): no AUTO author in Stage 1",
        f"  p_use_fe_panel: {fmt(v['p_use_fe_panel'])}   # panel studies with firm fixed effects, by lead repertoire (P+R)",
        f"  n_hypotheses: {fmt(v['n_hypotheses'])}   # hypotheses per P paper and entries per R analysis, pooled",
        f"  p_choose_fringe: {v['p_choose_fringe']}            # live P menu choices",
        f"  p_choose_crowded: {v['p_choose_crowded']}",
        f"  affinity_weight_power: {old['author']['affinity_weight_power']}       # declared (kept)",
        f"  p_mark_contradiction_when_disagree: {old['author']['p_mark_contradiction_when_disagree']}   # declared (kept)",
        f"  n_models_run: {{VERIFIED: {v['n_models_run_VERIFIED']}, AUTO: {old['author']['n_models_run']['AUTO']}}}   # VERIFIED p10-p90 (P+R); AUTO declared",
        f"  p_revise_when_asked: {old['author']['p_revise_when_asked']}         # kept: an engine lottery in live and fast alike, not re-identifiable",
        f"  p_limitations_mention_selection: {fmt(old['author']['p_limitations_mention_selection'])}   # declared (kept)",
        f"  analyst_reports_found_sign: {str(v['analyst_reports_found_sign']).lower()}   # R analysts state the direction found (no predicted sign): supported = significant either way",
        "reviewer:",
        "  # noticing a specific planted problem is not identified: live referees raised data/specification/inference objections on every",
        "  # paper, flagged or not, at the same kind-specific rates (detection lift ~0); the objections are p_unprompted_issue",
        f"  p_notice: {fmt(pn)}",
        f"  p_notice_assist_multiplier: {fmt(old['reviewer']['p_notice_assist_multiplier'])}   # declared (kept)",
        f"  load_penalty_per_paper: {old['reviewer']['load_penalty_per_paper']}     # declared (kept)",
        f"  p_unprompted_issue: {v['p_unprompted_issue']}        # reviews of unflagged papers raising a data/specification/inference objection",
        f"  p_unprompted_major: {v['p_unprompted_major']}        # ... with at least one major issue",
        "  # validity (0-100): round-1 cell means (OLS, relation adjusted), a shift shared by the paper's referees (between-paper SD),",
        "  # and the referee's own draw (within-paper SD)",
        f"  validity_mean: {{clean_supported: {v['vm_clean_supported']}, clean_null: {v['vm_clean_null']}, flawed_noticed: {v['vm_flawed']}, flawed_unnoticed: {v['vm_flawed']}, contradiction_penalty: {v['vm_contra']}, replication_bonus: {v['vm_repl']}}}",
        f"  validity_paper_sd: {v['paper_sd']}",
        f"  validity_sd: {v['validity_sd']}",
        f"  recommendation_cuts: {{reject: {v['cuts'][0]}, major: {v['cuts'][1]}, minor: 99.5}}   # indirect inference: quantiles of the composite simulation's validity at the live reject and minor shares",
        "  p_accept_above_minor_cut: 0.0    # 0 accept recommendations in 909 live reviews",
        f"  confidence: {{mean: {v['confidence'][0]}, sd: {v['confidence'][1]}}}",
        f"  p_flaw_found_given_noticed_major: {v['p_flaw_found']}",
        f"  p_prefer_fresh_reviewers_round2: {old['reviewer']['p_prefer_fresh_reviewers_round2']}   # declared (kept)",
        "editor:",
        f"  desk_reject_p: {{scope_only: {v['desk_scope']}, contribution: {old['editor']['desk_reject_p']['contribution']}, design: {old['editor']['desk_reject_p']['design']}}}   # scope_only: 0/202 live; others declared (kept)",
        f"  desk_reject_null_extra: {v['desk_null']}",
        "  # weights: ML over the live decisions given the live reviews; intercepts: indirect inference through the rule reviewers",
        f"  w_mean_validity: {round(e['w_mean_validity'], 3)}",
        f"  w_reject_votes: {round(e['w_reject_votes'], 3)}",
        f"  w_major_votes: {round(e['w_major_votes'], 3)}",
        f"  w_accept_votes: {e['w_accept_votes']}             # declared (kept): no live accept recommendation",
        f"  w_log_count_on_question: {round(e['w_log_count_on_question'], 3)}",
        f"  w_contradiction: {round(e['w_contradiction'], 3)}",
        f"  intercept_round1: {round(v['i1'], 3)}",
        f"  intercept_round2: {round(v['i2'], 3)}",
        f"  accept_cut_round1: {round(v['cut1'], 3)}       # 0/231 live round-1 acceptances (old rule: revise_band / 2)",
        f"  revise_band: {old['editor']['revise_band']}                 # declared (kept)",
        f"  merit_sd: {old['editor']['merit_sd']}                     # declared (kept)",
        "lab_head:",
        f"  p_reuse_given_catalogue: {v['p_reuse']}",
        f"  p_pool_given_two: {v['p_pool']}",
        f"  p_file_draft: {fmt(v['p_file_draft'])}   # of the entries the head decided",
        f"  stated_confidence: {{significant: {v['stated_conf'][0]}, insignificant: {v['stated_conf'][1]}, sd: {v['stated_conf'][2]}}}",
        "  vote:",
        f"    intercept: {vt['intercept']}",
        f"    b_significant: {vt['b_significant']}",
        f"    b_log_n: {vt['b_log_n']}",
        f"    b_prior_endorsements: {vt['b_prior_endorsements']}",
        f"    b_same_theory: {vt['b_same_theory']}",
        f"    b_reciprocity: {vt['b_reciprocity']}",
        f"    b_proposer_size: {vt['b_proposer_size']}",
        f"    p_challenge_base: {vt['p_challenge_base']}",
        f"    p_challenge_insignificant: {vt['p_challenge_insignificant']}",
        f"    stated_probability: {fmt(vt['stated_probability'])}",
        f"  p_offer_verify: {old['lab_head']['p_offer_verify']}              # kept: capacity-bound, not re-identified",
        f"  response: {{defend: {v['response']['defend']}, revise: {v['response']['revise']}, withdraw: {v['response']['withdraw']}}}",
        f"  p_request_n: {v['p_request_n']}               # fresh-data agenda items that name a dataset size",
        f"  request_n_multiple: {fmt(v['request_n_multiple'])}   # requested N / the question's median frame N (deciles)",
        f"  stake: {json.dumps(old['lab_head']['stake'])}   # declared (kept): variant d not run",
        f"  p_challenge_with_test: {old['lab_head']['p_challenge_with_test']}       # declared (kept): variant c not run",
        "verifier:",
        "  p_handle: VERIFIED",
        f"  factor_tolerance: {old['verifier']['factor_tolerance']}           # kept",
        "handbook:",
        f"  w_B1: {v['hb_w'][0]}",
        f"  w_B2: {v['hb_w'][1]}",
        f"  w_B3: {v['hb_w'][2]}",
        f"  confidence_scale: {v['hb_cs']}",
        f"  p_no_evidence_when_uncovered: {v['hb_ne']}",
        f"  p_contested_when_rules_disagree: {v['hb_cd']}",
        "# Tokens per call by role are tariffs (configs/tariffs.yaml), not rule-agent behaviour; not refitted here.",
    ]

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "configs" / "fastmode" / "params-stage1.yaml"))
    ap.add_argument("--table", default=str(ROOT / "runs" / "fastcal-2026-10-15" / "fit_table.md"))
    ap.add_argument("--sims", type=int, default=120)
    a = ap.parse_args()
    import yaml
    old = yaml.safe_load(open(ROOT / "configs" / "fastmode" / "params.yaml", encoding="utf-8"))
    F = Fit()
    fit_author(F, old)
    rv = load_reviews()
    fit_reviewer(F, rv)
    dec = load_decisions()
    fit_desk(F, dec)
    fit_editor_ml(F, dec, old)
    fit_lab(F, old)
    v = F.v
    v["response"] = dict(zip(("defend", "revise", "withdraw"), v["response"]))
    fit_handbook(F)
    # parameters for the composite simulation: the fitted reviewer + ML editor weights
    params = copy.deepcopy(old)
    params["reviewer"].update({"p_notice": {k: 0.0 for k in old["reviewer"]["p_notice"]}, "p_unprompted_issue": v["p_unprompted_issue"],
                               "p_unprompted_major": v["p_unprompted_major"],
                               "validity_mean": {"clean_supported": v["vm_clean_supported"], "clean_null": v["vm_clean_null"], "flawed_noticed": v["vm_flawed"],
                                                 "flawed_unnoticed": v["vm_flawed"], "contradiction_penalty": v["vm_contra"], "replication_bonus": v["vm_repl"]},
                               "validity_paper_sd": v["paper_sd"], "validity_sd": v["validity_sd"],
                               "recommendation_cuts": {"reject": v["cuts"][0], "major": v["cuts"][1], "minor": 99.5}, "p_accept_above_minor_cut": 0.0,
                               "p_flaw_found_given_noticed_major": v["p_flaw_found"],
                               "confidence": {"mean": v["confidence"][0], "sd": v["confidence"][1]}})
    params["editor"].update({k: float(x) for k, x in v["editor_ml"].items()})
    conds = paper_conditions(dec, rv)
    fit_editor_ii(F, params, conds, a.sims)
    sim = F.v["sim_check"]
    c1, c2 = F.v["editor_counts"]
    F.add("(check) composite simulation", "round-1 shares / round-2 accept / review mix / kappa",
          f"r1 {json.dumps({k: round(x, 3) for k, x in sim['r1'].items()})}; r2 accept {sim['r2_accept']:.3f}; recs {json.dumps({k: round(x, 3) for k, x in sim['recs'].items()})}; kappa {sim['kappa']:.3f}",
          len(conds), f"the fitted rule reviewer and editor on the {len(conds)} live round-1 papers' conditions, {a.sims} simulations each, accept cut 50 (search value); live: r1 reject {c1['reject'] / sum(c1.values()):.3f}, r2 accept {c2['accept'] / sum(c2.values()):.3f}")
    out = Path(a.out)
    out.write_text(render_yaml(F, old), encoding="utf-8")
    yaml.safe_load(out.read_text(encoding="utf-8"))   # must parse
    md = ["| Block | Parameter | Estimate | n | Source |", "|---|---|---|---|---|"]
    md += [f"| {b} | {p} | {e} | {n} | {s} |" for b, p, e, n, s in F.rows]
    tp = Path(a.table)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out} ({hashlib.sha256(out.read_bytes()).hexdigest().upper()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
