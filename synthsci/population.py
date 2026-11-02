"""Scientists, slot availability draws, teams, labs and turnover.

Profiles come from the population-layer seed; per-quarter availability draws from the run-layer seed, so that the
paper and repository regimes at the same master seed have identical capacity.
"""
from __future__ import annotations

import numpy as np

from .config import rng

STAGES = ["doctoral", "assistant", "associate", "full"]
REPERTOIRES = {"archival_regression": 0.70, "causal_design": 0.18, "survey": 0.12}   # design default, informed by the corpus estimator shares


def _choice(r, items, weights):
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    return items[int(r.choice(len(items), p=w))]


def build_population(cfg: dict, seeds: dict, theory_shares: dict[str, float], n: int | None = None,
                     entry_quarter: int = 0, start_index: int = 0) -> list[dict]:
    """n scientist profiles. Deterministic in (population seed, start_index)."""
    pp = cfg["population_profile"]
    n = n or cfg["population"]
    r = rng(seeds["population"], "profiles", start_index)
    fams = list(theory_shares)
    fw = [theory_shares[f] for f in fams]
    stages, sw = list(pp["stages"]), list(pp["stages"].values())
    auto_share, t1_share = pp["care"]["auto_share"], pp["throughput"]["t1_share"]
    out = []
    for i in range(n):
        idx = start_index + i
        primary = _choice(r, fams, fw)
        secondary = _choice(r, [f for f in fams if f != primary], [theory_shares[f] for f in fams if f != primary])
        stage = _choice(r, stages, sw) if entry_quarter == 0 else "doctoral"
        u_care, u_tp = r.uniform(), r.uniform()
        mode = pp.get("care_throughput_assignment", "independent")
        if mode == "correlated":
            u_tp = u_care
        elif mode == "anticorrelated":
            u_tp = 1 - u_care
        style = "AUTO" if u_care < auto_share else "VERIFIED"
        throughput = "T1" if u_tp < t1_share else "T0"
        rep = _choice(r, list(REPERTOIRES), list(REPERTOIRES.values()))
        years_in_rank = {"doctoral": r.integers(0, 5), "assistant": r.integers(0, 6), "associate": r.integers(0, 10),
                         "full": r.integers(0, 20)}[stage] if entry_quarter == 0 else 0
        productivity = float(np.exp(r.normal(0.0, pp["productivity_log_sd"])))
        s = {
            "id": f"A{idx + 1:03d}", "stage": stage, "years_in_rank": int(years_in_rank),
            "entry_quarter": int(entry_quarter), "active": True,
            "primary_theory": primary, "secondary_theory": secondary, "repertoire": rep,
            "productivity": productivity,
            "style": style, "throughput": throughput, "tier": pp.get("tier", "sonnet"),
            "slots": 2 if throughput == "T1" else 1,
        }
        s["prior_credit"] = prior_credit(cfg, s, r)
        out.append(s)
    return out


def prior_credit_rate(cfg: dict) -> float:
    """Publications per career year before the simulated record began. The scenario's productivity target when set, else the
    grounded median (0.25 top-7-journal papers per career year; author_profiles n = 638, 4j_06), or the explicit knob."""
    lh = cfg["long_horizon"]["turnover"]
    if lh.get("prior_credit_rate") is not None:
        return float(lh["prior_credit_rate"])
    target = (cfg["population_profile"].get("productivity_target") or {}).get("papers_per_author_year")
    return float(target) if target is not None else 0.25


def prior_credit(cfg: dict, s: dict, r) -> int:
    """Seeded record before the run (population layer): a Poisson count at the prior rate x the scientist's productivity
    multiplier over the years spent in rank (plus the full previous ranks for associates and full professors), floored at the
    bar the turnover rule would have applied at their last promotion, so the record is consistent with the rank (a design default)."""
    lh = cfg["long_horizon"]["turnover"]
    rate = prior_credit_rate(cfg) * float(s.get("productivity", 1.0))
    years = {"doctoral": 0.0, "assistant": float(s["years_in_rank"]),
             "associate": float(lh["tenure_years"] + s["years_in_rank"]),
             "full": float(lh["tenure_years"] + lh["assoc_to_full_years"] + s["years_in_rank"])}[s["stage"]]
    n = int(r.poisson(rate * years)) if years > 0 else 0
    floor = {"doctoral": 0, "assistant": 0, "associate": int(lh["tenure_threshold"]), "full": 2 * int(lh["tenure_threshold"])}[s["stage"]]
    return max(n, floor)


def availability(seeds: dict, author_id: str, quarter: int, slot: int, p: float, multiplier: float) -> bool:
    """Run-layer draw: is this slot available this quarter? Identical across regimes at a seed."""
    r = rng(seeds["run"], "avail", author_id, quarter, slot)
    return bool(r.uniform() < min(0.95, p * multiplier))


def effective_availability(cfg: dict) -> dict:
    """Per-slot availability p. Default: `availability_p` (.33, a design default). With `productivity_target.papers_per_author_year`
    set, p is derived so that expected published papers per author-year (any authorship) match the target:
      starts per author-quarter = target / (mean team size * planned acceptance * 4)
      p = starts per author-quarter / (mean slots * mean productivity multiplier)
    Grounding for the target: author_profiles (OpenAlex, high-confidence matches, n = 638): median 0.25 top-7-journal
    papers per career year (mean .34). Cost then scales with papers, not authors."""
    pp = cfg["population_profile"]
    pt = pp.get("productivity_target") or {}
    target = pt.get("papers_per_author_year")
    if target is None:
        return {"availability_p": float(pp["availability_p"]), "derived": False}
    t = pp["teams"]
    team = 1 * t["solo"] + 2 * t["two"] + 3 * t["three"] + 4 * t["four"]
    acc = float(pt.get("planned_acceptance", 0.6))
    slots = 1.0 + float(pp["throughput"]["t1_share"])
    mult = float(np.exp(0.5 * pp["productivity_log_sd"] ** 2))
    starts_pq = float(target) / (team * acc * 4.0)
    p = starts_pq / (slots * mult)
    # with calendar-time review a slot is held while its study is under review, so a free slot is drawn only (1 - busy share) of the time;
    # p is raised by that factor so the realized start rate stays at the target (busy share = starts x expected sojourn per slot)
    from .timing import expected_sojourn_quarters
    sojourn = expected_sojourn_quarters(cfg)
    busy = min(0.9, starts_pq * sojourn / slots)
    p_adj = p / (1.0 - busy)
    return {"availability_p": float(min(0.95, p_adj)), "derived": True, "starts_per_author_quarter": starts_pq, "mean_team_size": team,
            "planned_acceptance": acc, "mean_multiplier": mult, "papers_per_author_year": float(target),
            "expected_sojourn_quarters": sojourn, "busy_share": busy, "availability_p_unadjusted": float(min(0.95, p))}


def draw_team(seeds: dict, lead: dict, pool: list[dict], quarter: int, paper_index: int, teams_cfg: dict,
              coauthor_history: dict[str, set]) -> list[str]:
    """Lead plus 0-3 coauthors, preferring prior coauthors and shared theory."""
    r = rng(seeds["run"], "team", lead["id"], quarter, paper_index)
    sizes = [1, 2, 3, 4]
    w = [teams_cfg["solo"], teams_cfg["two"], teams_cfg["three"], teams_cfg["four"]]
    size = _choice(r, sizes, w)
    cands = [a for a in pool if a["id"] != lead["id"] and a.get("active", True)]
    if not cands or size == 1:
        return [lead["id"]]
    weights = []
    hist = coauthor_history.get(lead["id"], set())
    for a in cands:
        wt = 1.0
        if a["id"] in hist:
            wt += 3.0
        if a["primary_theory"] == lead["primary_theory"]:
            wt += 2.0
        elif a["primary_theory"] == lead["secondary_theory"] or a["secondary_theory"] == lead["primary_theory"]:
            wt += 1.0
        weights.append(wt)
    w = np.asarray(weights); w = w / w.sum()
    k = min(size - 1, len(cands))
    picks = r.choice(len(cands), size=k, replace=False, p=w)
    return [lead["id"]] + [cands[int(i)]["id"] for i in picks]


# ------------------------------------------------------------------ labs (repository regime)
def form_labs(cfg: dict, seeds: dict, scientists: list[dict]) -> list[dict]:
    """Labs: heads are full professors (highest productivity first), size targets from a configured distribution,
    members by weighted theory match; doctoral students last."""
    rc = cfg["repository"]
    r = rng(seeds["population"], "labs")
    active = [s for s in scientists if s.get("active", True)]
    n_labs = max(1, int(round(len(active) / rc["lab_mean_size"])))
    fulls = sorted([s for s in active if s["stage"] == "full"], key=lambda s: -s["productivity"])
    heads = fulls[:n_labs]
    if len(heads) < n_labs:
        assoc = sorted([s for s in active if s["stage"] == "associate"], key=lambda s: -s["productivity"])
        heads += assoc[:n_labs - len(heads)]
    if len(heads) < n_labs:
        rest = sorted([s for s in active if s not in heads], key=lambda s: -s["productivity"])
        heads += rest[:n_labs - len(heads)]
    head_ids = {h["id"] for h in heads}
    sizes_dist = {int(k): float(v) for k, v in rc["lab_size_dist"].items()}
    targets = [int(_choice(r, list(sizes_dist), list(sizes_dist.values()))) for _ in heads]
    # rescale targets to sum to the population
    total = sum(targets)
    scale = len(active) / total
    targets = [max(2, int(round(t * scale))) for t in targets]
    while sum(targets) < len(active):
        targets[int(np.argmin(targets))] += 1
    while sum(targets) > len(active):
        i = int(np.argmax(targets))
        if targets[i] > 2:
            targets[i] -= 1
        else:
            break
    labs = [{"id": f"L{i + 1:02d}", "head": h["id"], "members": [h["id"]], "target": targets[i],
             "primary_theory": h["primary_theory"], "secondary_theory": h["secondary_theory"], "reputation": 0.0,
             "points": float(rc["stakes"]["endowment"])} for i, h in enumerate(heads)]
    others = [s for s in active if s["id"] not in head_ids]
    non_doc = [s for s in others if s["stage"] != "doctoral"]
    docs = [s for s in others if s["stage"] == "doctoral"]
    order = list(r.permutation(len(non_doc)))
    for i in order:
        _assign(non_doc[i], labs, r, scientists, doctoral=False)
    order = list(r.permutation(len(docs)))
    for i in order:
        _assign(docs[i], labs, r, scientists, doctoral=True)
    for lab in labs:
        lab["size"] = len(lab["members"])
        lab.pop("target", None)
    return labs


def _assign(s: dict, labs: list[dict], r, scientists, doctoral: bool):
    by_id = {x["id"]: x for x in scientists}
    open_labs = [l for l in labs if len(l["members"]) < l["target"]] or labs
    w = []
    for l in open_labs:
        wt = 1.0
        if s["primary_theory"] == l["primary_theory"]:
            wt = 3.0
        elif s["primary_theory"] == l["secondary_theory"] or s["secondary_theory"] == l["primary_theory"]:
            wt = 2.0
        if doctoral and any(by_id[m]["stage"] == "assistant" for m in l["members"]):
            wt *= 2.0
        w.append(wt)
    lab = _choice(r, open_labs, w)
    lab["members"].append(s["id"])


# ------------------------------------------------------------------ turnover (long horizons)
def annual_step(cfg: dict, seeds: dict, scientists: list[dict], quarter: int, records: dict, theory_shares: dict,
                regime: str, labs: list[dict] | None = None) -> dict:
    """Careers move every four quarters; the population is kept constant by entry cohorts.

    records[author_id] = {'publications': n (P) | 'endorsed_claims': n, 'verifications': n (R)}; the scientist's seeded
    `prior_credit` (publications before the record began) is added to the credit, so the bar is applied to a whole career.
    Returns an event summary; each event is (id, what, credit_at_the_check) so that the outcome and the record behind it can
    be shown as department news (consequences only). Mutates scientists (stage, years_in_rank, active) and labs (succession)."""
    lh = cfg["long_horizon"]["turnover"]
    if not lh.get("enabled", True):
        return {"events": []}
    r = rng(seeds["population"], "annual", quarter)
    events = []
    exits = []
    for s in scientists:
        if not s.get("active", True):
            continue
        s["years_in_rank"] += 1
        s["age_years"] = s.get("age_years", 0) + 1
        rec = records.get(s["id"], {})
        if regime == "paper":
            credit = rec.get("publications", 0)
        else:
            credit = rec.get("endorsed_claims", 0) + lh["verification_credit"] * rec.get("verifications", 0)
        credit += int(s.get("prior_credit", 0))
        if s["stage"] == "doctoral" and s["years_in_rank"] >= lh["doctoral_years"]:
            s["stage"], s["years_in_rank"] = "assistant", 0
            events.append((s["id"], "promoted_assistant", credit))
        elif s["stage"] == "assistant" and s["years_in_rank"] >= lh["tenure_years"]:
            if credit >= lh["tenure_threshold"]:
                s["stage"], s["years_in_rank"] = "associate", 0
                events.append((s["id"], "tenured", credit))
            else:
                s["active"] = False
                exits.append(s["id"])
                events.append((s["id"], "tenure_exit", credit))
        elif s["stage"] == "associate" and s["years_in_rank"] >= lh["assoc_to_full_years"] and \
                (s["years_in_rank"] - lh["assoc_to_full_years"]) % 2 == 0:
            if credit >= 2 * lh["tenure_threshold"]:
                s["stage"], s["years_in_rank"] = "full", 0
                events.append((s["id"], "promoted_full", credit))
            else:
                events.append((s["id"], "not_promoted_full", credit))
        retire_at = lh["retire_years"] if s.get("entry_stage", s["stage"]) == "doctoral" else lh["retire_years"] - 5
        if s.get("active", True) and s.get("age_years", 0) >= retire_at:
            s["active"] = False
            exits.append(s["id"])
            events.append((s["id"], "retired", credit))
        if s.get("active", True) and lh["exit_hazard"] > 0 and r.uniform() < lh["exit_hazard"]:
            s["active"] = False
            exits.append(s["id"])
            events.append((s["id"], "exit", credit))
    # entry: replace exits (plus growth) with doctoral students; some mid-career entrants
    n_active = sum(1 for s in scientists if s.get("active", True))
    n_target = int(round(cfg["population"] * (1 + lh["growth"]) ** max(quarter // 4, 0)))
    n_new = max(0, n_target - n_active)
    if n_new:
        entrants = build_population(cfg, seeds, theory_shares, n=n_new, entry_quarter=quarter, start_index=len(scientists))
        seniors = [s for s in scientists if s.get("active", True) and s["stage"] in ("associate", "full")]
        for e in entrants:
            if r.uniform() < lh["midcareer_entry_p"]:
                e["stage"] = "assistant" if r.uniform() < 0.7 else "associate"
                e["years_in_rank"] = int(r.integers(0, 4)) if e["stage"] == "assistant" else int(r.integers(0, 6))
            else:
                e["stage"] = "doctoral"
                if seniors:
                    adv = seniors[int(r.integers(len(seniors)))]
                    e["primary_theory"] = adv["primary_theory"]
            e["prior_credit"] = prior_credit(cfg, e, r)   # a mid-career entrant brings a record; a doctoral entrant none
            e["entry_stage"] = e["stage"]
            e["age_years"] = 0
            events.append((e["id"], f"entered_{e['stage']}", int(e["prior_credit"])))
        scientists.extend(entrants)
    if labs is not None:
        _lab_succession(cfg, r, scientists, labs, exits, events)
    return {"events": events, "exits": exits, "n_active": sum(1 for s in scientists if s.get("active", True))}


def _lab_succession(cfg, r, scientists, labs, exits, events):
    by_id = {s["id"]: s for s in scientists}
    rank = {"full": 3, "associate": 2, "assistant": 1, "doctoral": 0}
    for lab in labs:
        lab["members"] = [m for m in lab["members"] if by_id[m].get("active", True)]
        if lab["head"] in exits or lab["head"] not in lab["members"]:
            if lab["members"]:
                lab["head"] = max(lab["members"], key=lambda m: (rank[by_id[m]["stage"]], by_id[m]["years_in_rank"]))
                lab["primary_theory"] = by_id[lab["head"]]["primary_theory"]
                events.append((lab["id"], f"new_head_{lab['head']}"))
    dissolved = [l for l in labs if len(l["members"]) < 2]
    for l in dissolved:
        for m in l["members"]:
            others = [x for x in labs if x is not l and len(x["members"]) >= 2]
            if others:
                tgt = others[int(r.integers(len(others)))]
                tgt["members"].append(m)
        labs.remove(l)
        events.append((l["id"], "dissolved"))
    # unassigned actives (entrants) join labs by theory weight; a new full professor without a lab founds one
    assigned = {m for l in labs for m in l["members"]}
    for s in scientists:
        if s.get("active", True) and s["id"] not in assigned:
            if s["stage"] == "full" and len(labs) < cfg["population"] / cfg["repository"]["lab_mean_size"] + 1:
                labs.append({"id": f"L{len(labs) + 1 + int(r.integers(900)):03d}", "head": s["id"], "members": [s["id"]],
                             "primary_theory": s["primary_theory"], "secondary_theory": s["secondary_theory"],
                             "reputation": 0.0, "points": float(cfg["repository"]["stakes"]["endowment"]), "size": 1})
            else:
                w = [3.0 if s["primary_theory"] == l["primary_theory"] else 1.0 for l in labs]
                lab = _choice(r, labs, w)
                lab["members"].append(s["id"])
    for l in labs:
        l["size"] = len(l["members"])
