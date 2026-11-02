"""Realistic review timing in calendar time. A quarter is three calendar months.

Latencies are per-paper draws from a discretised lognormal (config `journal.timing.<kind>`: median_quarters, sigma), so the
simulated submission-to-acceptance distribution can be matched to the grounding quantiles (4j_05: SMJ median 22.1 months,
p10 11.1, p90 43.5; OS 23.9 / 12.3 / 42.3). `sojourn_quantiles` reproduces the two-round path (first round, revision,
re-review) from the config alone, which the calibration test compares with those quantiles.
"""
from __future__ import annotations

import math

import numpy as np

from .config import rng

MONTHS_PER_QUARTER = 3
# Grounding reference (grounding/tables/4j_05_review_durations_by_journal.csv, interval rec_acc, months)
GROUNDING_MONTHS = {"SMJ": {"p10": 11.07, "p25": 16.13, "p50": 22.08, "p75": 30.52, "p90": 43.52, "n": 773},
                    "OS": {"p10": 12.29, "p25": 16.86, "p50": 23.92, "p75": 32.41, "p90": 42.35, "n": 746}}


def spec_of(cfg: dict, kind: str) -> dict:
    t = (cfg.get("journal") or {}).get("timing") or {}
    s = t.get(kind)
    if s is None:
        return {"median_quarters": 0.0, "sigma": 0.0, "min_quarters": 0}
    if isinstance(s, (int, float)):
        return {"median_quarters": float(s), "sigma": 0.0, "min_quarters": 0}
    return {"median_quarters": float(s.get("median_quarters", 0.0)), "sigma": float(s.get("sigma", 0.0)),
            "min_quarters": int(s.get("min_quarters", 1 if kind == "revision" else 0))}


def draw_latency(cfg: dict, kind: str, run_seed: int, *tags) -> int:
    """Quarters of latency for one paper and stage; deterministic in (run seed, kind, tags)."""
    s = spec_of(cfg, kind)
    med, sig, lo = s["median_quarters"], s["sigma"], s["min_quarters"]
    if sig <= 0 or med <= 0:
        return max(lo, int(round(med)))
    z = rng(run_seed, "latency", kind, *tags).normal()
    return max(lo, int(round(med * math.exp(sig * z))))


def expected_latency(cfg: dict, kind: str) -> float:
    """Mean of the (continuous) lognormal: median x exp(sigma^2 / 2)."""
    s = spec_of(cfg, kind)
    return max(float(s["min_quarters"]), s["median_quarters"] * math.exp(0.5 * s["sigma"] ** 2))


def expected_sojourn_quarters(cfg: dict, p_sent_out: float = 0.9, p_revise: float = 0.45) -> float:
    """Planning value: quarters a lead author's slot is held by one study (research, then review rounds), with the design's
    planning shares (10% desk rejection, 45% revise). Used to keep the realized start rate at the productivity target when
    slots are held through review (population.effective_availability)."""
    t = (cfg.get("journal") or {}).get("timing") or {}
    research = float(t.get("research_quarters", 1))
    first = expected_latency(cfg, "review_round")
    second = expected_latency(cfg, "revision") + expected_latency(cfg, "rereview")
    return research + p_sent_out * (first + p_revise * second)


def sojourn_quantiles(cfg: dict, n: int = 20000, seed: int = 0, rounds: int = 2) -> dict:
    """Monte Carlo of the submission-to-acceptance time (months) of a paper accepted after `rounds` rounds, from the config's
    latency specs alone (no engine). Returns quantiles in months beside the grounding reference."""
    r = np.random.default_rng(seed)
    def draws(kind):
        s = spec_of(cfg, kind)
        if s["sigma"] <= 0 or s["median_quarters"] <= 0:
            return np.full(n, max(s["min_quarters"], int(round(s["median_quarters"]))))
        return np.maximum(s["min_quarters"], np.rint(s["median_quarters"] * np.exp(s["sigma"] * r.normal(size=n))))
    q = draws("review_round")
    if rounds >= 2:
        q = q + draws("revision") + draws("rereview")
    months = q * MONTHS_PER_QUARTER
    qs = {f"p{p}": float(np.quantile(months, p / 100)) for p in (10, 25, 50, 75, 90)}
    return {"months": qs, "mean": float(months.mean()), "rounds": rounds, "n": n, "reference": GROUNDING_MONTHS}
