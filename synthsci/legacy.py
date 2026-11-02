"""The legacy library: ~100 rule-generated 'already published' studies with known truth.

Each legacy study is an analytic draw from its question's truth (the scripted producer handles traps with a design-default
probability), published if its focal |t| > 1.96. In P they are legacy papers (structured abstracts, citable); in R the
same studies are legacy claims with status endorsed (legacy). The prose abstract is a template here; in LLM mode a
batched Haiku call (legacy_abstracts) rewrites it and the engine stores the model's text beside the template.
"""
from __future__ import annotations

import numpy as np

from .config import rng
from .stats import partial_r

LEGACY_HANDLE_P = 0.8   # design default: scripted producers of the legacy era handle a planted trap with this probability


def build_legacy(cfg: dict, seeds: dict, landscape, gen) -> list[dict]:
    lc = cfg["landscape"]["legacy"]
    r = rng(seeds["world"], "legacy")
    n_total, core_share, t_thr = int(lc["size"]), float(lc["core_share"]), float(lc["t_threshold"])
    n_core = int(round(n_total * core_share))
    core_ids = [q["id"] for q in landscape.questions]
    # about a quarter of core questions start with none: weight questions by a seeded gamma draw
    w = r.gamma(0.7, 1.0, size=len(core_ids))
    w = w / w.sum()
    fringe_ids = [f["id"] for f in landscape.fringe]
    papers: list[dict] = []
    attempts = 0
    target = {"core": n_core, "fringe": n_total - n_core}
    counts = {"core": 0, "fringe": 0}
    while (counts["core"] < target["core"] or counts["fringe"] < target["fringe"]) and attempts < 20 * n_total:
        attempts += 1
        kind = "core" if counts["core"] < target["core"] else "fringe"
        if kind == "core" and counts["fringe"] < target["fringe"] and r.uniform() < 0.3:
            kind = "fringe"
        qid = core_ids[int(r.choice(len(core_ids), p=w))] if kind == "core" else fringe_ids[int(r.integers(len(fringe_ids)))]
        ds_seed = int(r.integers(1, 2 ** 62))
        study = gen.analytic_study(qid, ds_seed, upper_half=False, handled={}, use_fe=True)
        handled = {t: bool(r.uniform() < LEGACY_HANDLE_P) for t in study["traps"]}
        study = gen.analytic_study(qid, ds_seed, upper_half=False, handled=handled, use_fe=True)
        est, se = study["focal"]["estimate"], study["focal"]["se"]
        t = est / se if se > 0 else 0.0
        if abs(t) <= t_thr:
            continue  # the scripted selection rule: the file drawer
        q = landscape.question(qid)
        pid = f"L{len(papers) + 1:04d}"
        hyps = _legacy_hypotheses(q, study, r)
        papers.append({
            "paper_id": pid, "question_id": qid, "core": q["core"], "quarter_published": 0, "legacy": True,
            "title": f"{q['iv_label']} and {q['dv_label'].lower()}: evidence from {'a panel of' if q['panel'] else 'a sample of'} {study['n']:,} {q['unit']}s",
            "authors": [f"LA{len(papers) + 1:03d}"], "theory": q["family"], "n": int(study["n_reported"]),
            "estimator": "fixed effects" if q["panel"] else "OLS", "design": q["design"], "unit": q["unit"],
            "hypotheses": hyps, "abstract_template": _abstract(q, hyps, study), "abstract": None,
            "dataset_seed": ds_seed, "citations": 0,
            "hidden": {"theta": study["hidden"]["theta"], "theta_fe": study["hidden"]["theta_fe"],
                       "theta_naive": study["hidden"]["theta_naive"], "traps": study["traps"], "handled": handled,
                       "focal_t": float(t)},
        })
        counts[kind] += 1
    return papers


def _legacy_hypotheses(q: dict, study: dict, r) -> list[dict]:
    pred = 1 if q["prediction_sign"] == "positive" else -1
    out = []

    def verdict(est, se, predicted):
        t = est / se if se > 0 else 0.0
        p = float(2 * (1 - _norm_cdf(abs(t))))
        if p < 0.05 and np.sign(est) == predicted:
            return "supported", p
        if p < 0.10 and np.sign(est) == predicted:
            return "partially", p
        return "not_supported", p

    v, p = verdict(study["focal"]["estimate"], study["focal"]["se"], pred)
    out.append({"hyp_id": "H1", "iv": q["iv_var"], "dv": q["dv_var"], "moderator": None, "kind": "focal",
                "predicted_sign": "positive" if pred > 0 else "negative", "estimate": study["focal"]["estimate"],
                "se": study["focal"]["se"], "p": p, "n": study["n_reported"], "verdict": v,
                "partial_r": partial_r(study["focal"]["estimate"] / study["focal"]["se"], study["df"])})
    k = 2
    n_extra = int(r.integers(1, 4))
    mods = list(study["moderators"].items())
    for m, mv in mods[:max(0, n_extra - 1)]:
        v, p = verdict(mv["estimate"], mv["se"], 1)
        out.append({"hyp_id": f"H{k}", "iv": q["iv_var"], "dv": q["dv_var"], "moderator": m, "kind": "moderation",
                    "predicted_sign": "positive", "estimate": mv["estimate"], "se": mv["se"], "p": p,
                    "n": study["n_reported"], "verdict": v, "partial_r": partial_r(mv["estimate"] / mv["se"], study["df"])})
        k += 1
    if q.get("secondary_dv_var"):
        s = study["secondary"]
        v, p = verdict(s["estimate"], s["se"], pred)
        out.append({"hyp_id": f"H{k}", "iv": q["iv_var"], "dv": q["secondary_dv_var"], "moderator": None, "kind": "secondary",
                    "predicted_sign": "positive" if pred > 0 else "negative", "estimate": s["estimate"], "se": s["se"],
                    "p": p, "n": study["n_reported"], "verdict": v, "partial_r": partial_r(s["estimate"] / s["se"], study["df"])})
    return out


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(x))


def _abstract(q: dict, hyps: list[dict], study: dict) -> str:
    h1 = hyps[0]
    direction = "positively" if h1["estimate"] > 0 else "negatively"
    sup = "supported" if h1["verdict"] == "supported" else ("partially supported" if h1["verdict"] == "partially" else "not supported")
    n_sup = sum(1 for h in hyps if h["verdict"] == "supported")
    return (f"Drawing on {q['family'].replace('_', ' ')} arguments, this study examines how {q['iv_label'].lower()} relates to "
            f"{q['dv_label'].lower()}. Using {'a firm-year panel' if q['panel'] else 'a cross-section'} of {study['n_reported']:,} "
            f"{q['unit']} observations and {'fixed-effects' if q['panel'] else 'OLS'} regressions with standard controls, we find that "
            f"{q['iv_label'].lower()} is {direction} related to {q['dv_label'].lower()} (estimate {h1['estimate']:.3f}, "
            f"p = {h1['p']:.3f}); the focal hypothesis is {sup}. Of {len(hyps)} hypotheses, {n_sup} receive support. "
            f"The results speak to debates on {q['dv_label'].lower()} and suggest directions for work on the boundary conditions "
            f"of {q['iv_label'].lower()}.")
