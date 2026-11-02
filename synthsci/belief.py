"""Belief rules B1, B2, B3, B5 over certified units, and the truth-class comparison.

A certified unit is a dict with at least: estimate (partial-r scale), se, n, verdict ('supported'|'partially'|'not_supported'),
sign (+1/-1 of the estimate), attention (citations in P; endorsements or stake in R), quarter, and optionally
deposit_id (R: B2 counts one estimate per distinct deposit) and unit_id.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sstats

from .stats import fisher_z, inv_fisher_z

NEGLIGIBLE = 0.02


def truth_class(theta: float, threshold: float = NEGLIGIBLE) -> str:
    if theta == 0.0:
        return "null"
    if abs(theta) < threshold:
        return "negligible"
    return "positive" if theta > 0 else "negative"


def is_correct(belief: str, tclass: str) -> bool | None:
    """True/False, or None for contested (unsettled) and uncovered."""
    if belief in ("contested", "uncovered", "no evidence"):
        return None
    if tclass == "negligible":
        return belief == "null" or belief in ("positive", "negative")
    return belief == tclass


def _unit_sign(u: dict) -> int:
    e = u.get("estimate")
    if e is None or not np.isfinite(e):
        return 0
    return 1 if e > 0 else (-1 if e < 0 else 0)


def _states_effect(u: dict) -> bool:
    return u.get("verdict") in ("supported", "partially")


def b1_vote(units: list[dict], share: float = 2 / 3, weights: list[float] | None = None) -> dict:
    if not units:
        return {"belief": "uncovered", "confidence": 0.0, "n_units": 0}
    w = np.asarray(weights if weights is not None else [1.0] * len(units), dtype=float)
    pos = sum(wi for u, wi in zip(units, w) if _states_effect(u) and _unit_sign(u) > 0)
    neg = sum(wi for u, wi in zip(units, w) if _states_effect(u) and _unit_sign(u) < 0)
    nul = sum(wi for u, wi in zip(units, w) if not _states_effect(u))
    tot = w.sum()
    best, val = max((("positive", pos), ("negative", neg), ("null", nul)), key=lambda x: x[1])
    if tot > 0 and val / tot >= share:
        return {"belief": best, "confidence": float(val / tot), "n_units": len(units)}
    return {"belief": "contested", "confidence": float(val / tot) if tot else 0.0, "n_units": len(units)}


def b2_pooled(units: list[dict], one_per_deposit: bool = False) -> dict:
    """DerSimonian-Laird random-effects pooling on the Fisher-z scale, weights from N (var = 1/(n-3))."""
    us = [u for u in units if u.get("estimate") is not None and np.isfinite(u["estimate"]) and (u.get("n") or 0) > 3]
    if one_per_deposit:
        latest: dict = {}
        for u in sorted(us, key=lambda x: x.get("quarter", 0)):
            latest[u.get("deposit_id") or u.get("unit_id")] = u
        us = list(latest.values())
    if not us:
        return {"belief": "uncovered", "confidence": 0.0, "n_units": 0}
    z = np.array([fisher_z(u["estimate"]) for u in us])
    v = np.array([1.0 / (u["n"] - 3) for u in us])
    w = 1.0 / v
    zbar = float((w * z).sum() / w.sum())
    if len(us) > 1:
        Q = float((w * (z - zbar) ** 2).sum())
        C = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (Q - (len(us) - 1)) / C) if C > 0 else 0.0
        w_re = 1.0 / (v + tau2)
        zbar = float((w_re * z).sum() / w_re.sum())
        se = float(np.sqrt(1.0 / w_re.sum()))
    else:
        se = float(np.sqrt(v[0]))
    lo, hi = inv_fisher_z(zbar - 1.96 * se), inv_fisher_z(zbar + 1.96 * se)
    est = inv_fisher_z(zbar)
    p = float(2 * sstats.norm.sf(abs(zbar / se))) if se > 0 else 0.0
    if lo > 0:
        belief, conf = "positive", 1 - p
    elif hi < 0:
        belief, conf = "negative", 1 - p
    elif -NEGLIGIBLE <= lo and hi <= NEGLIGIBLE:
        belief, conf = "null", max(0.5, 1 - p)
    else:
        belief, conf = "contested", 0.5
    return {"belief": belief, "confidence": float(min(max(conf, 0.0), 1.0)), "estimate": est, "lo": lo, "hi": hi,
            "width": hi - lo, "n_units": len(us), "p": p}


def b3_most_attended(units: list[dict]) -> dict:
    if not units:
        return {"belief": "uncovered", "confidence": 0.0, "n_units": 0}
    tot = sum(float(u.get("attention", 0) or 0) for u in units)
    top = max(units, key=lambda u: (float(u.get("attention", 0) or 0), u.get("quarter", 0), str(u.get("unit_id", ""))))
    if _states_effect(top):
        belief = "positive" if _unit_sign(top) > 0 else "negative"
    else:
        belief = "null"
    conf = (float(top.get("attention", 0) or 0) / tot) if tot > 0 else 1.0 / len(units)
    return {"belief": belief, "confidence": float(conf), "unit_id": top.get("unit_id"), "n_units": len(units)}


def b5_attention_weighted(units: list[dict], share: float = 2 / 3) -> dict:
    w = [1.0 + float(u.get("attention", 0) or 0) for u in units]
    out = b1_vote(units, share=share, weights=w)
    return out


def all_rules(units: list[dict], share: float = 2 / 3, one_per_deposit: bool = False) -> dict:
    out = {"B1": b1_vote(units, share), "B2": b2_pooled(units, one_per_deposit), "B3": b3_most_attended(units),
           "B5": b5_attention_weighted(units, share)}
    if one_per_deposit:
        out["B2_naive"] = b2_pooled(units, False)
    return out


def score_belief(rule_out: dict, theta: float) -> dict:
    tc = truth_class(theta)
    corr = is_correct(rule_out["belief"], tc)
    conf = float(rule_out.get("confidence", 0.0))
    brier = None if corr is None else (1.0 - conf) ** 2 if corr else conf ** 2
    return {"truth_class": tc, "correct": corr, "confidence": conf, "brier": brier,
            "confidently_wrong": bool(corr is False and conf >= 0.7)}
