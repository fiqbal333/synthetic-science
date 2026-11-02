"""The research landscape: 48 core questions in 8 theory families, the fringe pool, hidden truth, de-identification.

Everything here is driven by the seeding table set and the world-layer seed (truth, questions, fringe) or the
run-layer seed (de-identification permutation). Agents never see truth.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import get, rng
from .seeding import SeedingTables
from .util import write_json

FAMILY_CODES = {"agency": "AG", "rbv": "RB", "tce": "TC", "org_learning": "OL", "institutional": "IN",
                "networks": "NW", "upper_echelons": "UE", "dynamic_capabilities": "DC"}

# The mechanism clause must agree with the predicted sign (menu coherence): one clause per family and direction.
FAMILY_CLAUSE = {
    "agency": {"positive": "because it aligns managerial and owner interests and strengthens monitoring",
               "negative": "because it widens the gap between managerial and owner interests and weakens monitoring"},
    "rbv": {"positive": "because it builds resources that are valuable and hard to imitate",
            "negative": "because it dissipates or stretches the resources that are valuable and hard to imitate"},
    "tce": {"positive": "because it lowers the costs of governing exchange",
            "negative": "because it raises the costs of governing exchange"},
    "org_learning": {"positive": "because accumulated experience and knowledge improve routines",
                     "negative": "because accumulated experience entrenches routines that no longer fit"},
    "institutional": {"positive": "because conformity to prevailing norms confers legitimacy and access to resources",
                      "negative": "because it departs from prevailing norms and costs legitimacy"},
    "networks": {"positive": "because network ties carry information and resources",
                 "negative": "because network ties impose obligations and constrain action"},
    "upper_echelons": {"positive": "because executives' characteristics shape strategic choices toward it",
                       "negative": "because executives' characteristics shape strategic choices away from it"},
    "dynamic_capabilities": {"positive": "because the capacity to reconfigure resources sustains advantage",
                             "negative": "because reconfiguring resources disrupts the routines that sustain advantage"},
}


def prediction_sentence(family_label: str, iv_label: str, dv_label: str, sign: str, fam: str) -> str:
    word = "raises" if sign == "positive" else "lowers"
    return f"{family_label} predicts that higher {iv_label.lower()} {word} {dv_label.lower()}, {FAMILY_CLAUSE[fam][sign]}."

SUBSAMPLE_CONTRAST = {"name": "high_tech", "label": "high- vs low-technology industries",
                      "definition": "1 if the firm's primary industry is classified as high-technology"}

COINED_ADJ = ["relational", "strategic", "coordination", "temporal", "attentional", "reputational", "cognitive",
              "structural", "procedural", "symbolic", "adaptive", "political", "informational", "contractual",
              "architectural", "dialogic", "spatial", "generative", "boundary", "cultural"]
COINED_NOUN = ["slack", "tempo", "density", "breadth", "cadence", "friction", "ambidexterity", "distance", "depth",
               "velocity", "latitude", "orientation", "modularity", "porosity", "bandwidth", "elasticity", "fit",
               "momentum", "redundancy", "granularity"]


def _weighted_choice(r: np.random.Generator, items: list, weights: list[float]):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return items[int(r.choice(len(items), p=w))]


class Landscape:
    """Builds and holds the landscape for one world seed."""

    def __init__(self, cfg: dict, seeds: dict, tables: SeedingTables):
        self.cfg, self.seeds, self.tables = cfg, seeds, tables
        self.lc = cfg["landscape"]
        self.measures = tables.construct_measures()
        self.questions: list[dict] = []
        self.fringe: list[dict] = []
        self.truth: dict = {"questions": {}, "fringe": {}, "q_T": {}}
        self.families = {f: {"label": spec["label"], "code": FAMILY_CODES[f]} for f, spec in tables.families.items()}

    # ------------------------------------------------------------ core build
    def build(self) -> "Landscape":
        self._draw_qT()
        self._build_core()
        self._build_fringe(self.lc["fringe_pool"], start_index=0)
        return self

    def _draw_qT(self):
        r = rng(self.seeds["world"], "qT")
        spec = self.lc["truth"]["q_T"]
        for fam in self.tables.family_ids:
            if spec.get("fixed") is not None:
                self.truth["q_T"][fam] = float(spec["fixed"])
            else:
                self.truth["q_T"][fam] = float(r.uniform(spec["low"], spec["high"]))

    def _candidate_pairs(self) -> list[dict]:
        """Candidate questions: top construct pairs, drop same-construct and performance-IV pairs, assign families (the pair-to-theory table in configs/seeding/smj/pair_theory.csv)."""
        pt = self.tables.pair_theory()
        pt_map = {(a, b): (fam, src) for a, b, fam, src in zip(pt.x_construct, pt.y_construct, pt.family, pt.source)}
        perf = set(self.tables.spec["performance_ivs"])
        allowed = self.tables.spec.get("performance_iv_allowed", {})
        out = []
        top = self.tables.pairs()
        seen = set()
        for _, row in top.iterrows():
            x, y = row.x_construct, row.y_construct
            if x == y or (x, y) in seen:
                continue
            if (x, y) not in pt_map:
                continue  # no family reaches a plurality in the pair-to-theory table; skipped and logged
            fam, src = pt_map[(x, y)]
            if x in perf and y not in allowed.get(fam, []):
                continue
            seen.add((x, y))
            out.append({"x": x, "y": y, "family": fam, "n_papers": int(row.n_papers),
                        "share_positive_pred": float(row.share_positive_pred), "source": f"top50;{src}"})
        # fill candidates from the pair-to-theory table that are not in the top-50
        for (x, y), (fam, src) in pt_map.items():
            if (x, y) in seen or x == y:
                continue
            if x not in self.measures["constructs"] or y not in self.measures["constructs"]:
                continue
            spp = self.tables.predicted_positive_share(x, y)
            out.append({"x": x, "y": y, "family": fam, "n_papers": self.tables.pair_count(x, y),
                        "share_positive_pred": spp if spp is not None else self.tables.predicted_direction_fallback(),
                        "source": f"fill;{src}"})
            seen.add((x, y))
        return out

    def _build_core(self):
        per_family = self.lc["questions_per_family"]
        cands = self._candidate_pairs()
        r_design = rng(self.seeds["world"], "design")
        design_shares = self.tables.design_shares()
        unit_shares = self.tables.unit_shares()
        mod_pool = self.measures.get("moderators_pool", [])
        for fam in self.tables.family_ids:
            rows = [c for c in cands if c["family"] == fam]
            rows.sort(key=lambda c: (-c["n_papers"], c["x"], c["y"]))
            if len(rows) < per_family:
                raise ValueError(f"family {fam}: only {len(rows)} candidate pairs; need {per_family} (extend pair_theory.csv)")
            for i, c in enumerate(rows[:per_family], start=1):
                qid = f"Q-{FAMILY_CODES[fam]}-{i:02d}"
                self.questions.append(self._make_question(qid, fam, c, r_design, design_shares, unit_shares, mod_pool))
        assert len(self.questions) == self.lc["families"] * per_family, len(self.questions)
        r_truth = rng(self.seeds["world"], "truth")
        for q in self.questions:
            self.truth["questions"][q["id"]] = self._draw_truth(q, r_truth, self.truth["q_T"][q["family"]])

    def _make_question(self, qid, fam, c, r, design_shares, unit_shares, mod_pool) -> dict:
        mx, my = self.measures["constructs"][c["x"]], self.measures["constructs"][c["y"]]
        # design and unit are fixed per question; survey-measured constructs force a survey design
        if mx.get("source") == "survey" or my.get("source") == "survey":
            design = _weighted_choice(r, ["survey+archival", "survey_only"], [0.65, 0.35])
        else:
            design = _weighted_choice(r, list(design_shares), list(design_shares.values()))
        if design in ("survey_only",):
            unit = _weighted_choice(r, ["firm", "individual"], [0.6, 0.4])
        else:
            unit = _weighted_choice(r, list(unit_shares), list(unit_shares.values()))
        if c["y"] == "firm_survival" and unit == "firm-year":
            unit = "firm"
        panel = unit == "firm-year"
        costly = "survey" in design or "hand_collected" in design
        sign = "positive" if c["share_positive_pred"] >= 0.5 else "negative"
        mods = [m for m in mod_pool if m not in (c["x"], c["y"])]
        chosen_mods = [mods[int(i)] for i in r.choice(len(mods), 2, replace=False)]
        controls = [k for k in self.measures["controls"] if k["construct"] not in (c["x"], c["y"])]
        instrument = bool(r.uniform() < self.lc["truth"]["instrument_p"])
        direction_word = "raises" if sign == "positive" else "lowers"
        prediction = prediction_sentence(self.families[fam]["label"], mx["label"], my["label"], sign, fam)
        return {
            "id": qid, "family": fam, "iv": c["x"], "dv": c["y"],
            "iv_var": mx["var"], "dv_var": my["var"], "iv_label": mx["label"], "dv_label": my["label"],
            "secondary_dv_var": (my.get("secondary") or {}).get("var"),
            "prediction_sign": sign, "prediction": prediction,
            "moderators": chosen_mods,
            "moderator_vars": [self.measures["constructs"][m]["var"] if m in self.measures["constructs"] else m for m in chosen_mods],
            "subsample_contrast": SUBSAMPLE_CONTRAST,
            "controls_pool": [k["var"] for k in controls] + ["industry", "year"] if panel else [k["var"] for k in controls] + ["industry"],
            "design": design, "unit": unit, "panel": panel, "access_cost": "costly" if costly else "cheap",
            "frame_quantiles": self.tables.frame_quantiles(unit),
            "panel_length_quantiles": self.tables.panel_length_quantiles() if panel else None,
            "identification": {"instrument_variable": "regional_policy_index" if instrument else None,
                               "firm_identifiers": True},
            "corpus_papers": c["n_papers"], "pair_source": c["source"], "core": True,
        }

    def _draw_truth(self, q: dict, r: np.random.Generator, q_T: float) -> dict:
        t = self.lc["truth"]
        if r.uniform() < t["pi0"]:
            theta = 0.0
        else:
            mag = abs(r.normal(0.0, t["slab_scale"]))
            agree = r.uniform() < q_T
            pred = 1.0 if q["prediction_sign"] == "positive" else -1.0
            theta = float(mag * (pred if agree else -pred))
        mods = {}
        for m in q["moderator_vars"]:
            mods[m] = 0.0 if r.uniform() < t["moderator_zero_p"] else float(abs(r.normal(0, t["moderator_scale"])) * r.choice([-1, 1]))
        subsample_offset = float(r.normal(0, t["subsample_offset_sd"]))
        secondary = 0.0 if theta == 0.0 and r.uniform() < 0.7 else float(theta + r.normal(0, t["secondary_dv_sd"]))
        return {
            "theta": theta,
            "truth_class": truth_class(theta),
            "moderators": mods,
            "subsample_offset": subsample_offset,
            "theta_secondary": secondary,
            "confounder": {"r_ux": float(r.uniform(t["confounder_r"]["low"], t["confounder_r"]["high"])),
                           "r_uy": float(r.uniform(t["confounder_r"]["low"], t["confounder_r"]["high"])),
                           "time_invariant_share": float(r.uniform(t["confounder_time_invariant"]["low"],
                                                                   t["confounder_time_invariant"]["high"]))},
            "instrument": {"available": q["identification"]["instrument_variable"] is not None,
                           "F": float(r.uniform(t["instrument_F"]["low"], t["instrument_F"]["high"])),
                           "exclusion_r": float(t["instrument_exclusion_r"])},
            "q_T": q_T,
        }

    # ------------------------------------------------------------ fringe
    def _build_fringe(self, n: int, start_index: int):
        r = rng(self.seeds["world"], "fringe", start_index)
        shares = self.tables.theory_shares()
        fams = list(shares)
        design_shares = self.tables.design_shares()
        unit_shares = self.tables.unit_shares()
        q_mean = float(np.mean(list(self.truth["q_T"].values())))
        dvs = self.tables.spec["standard_dvs"]
        used = {f["coined"] for f in self.fringe}
        mod_pool = self.measures.get("moderators_pool", [])
        i = start_index
        while len(self.fringe) < start_index + n:
            adj, noun = COINED_ADJ[int(r.integers(len(COINED_ADJ)))], COINED_NOUN[int(r.integers(len(COINED_NOUN)))]
            coined = f"{adj} {noun}"
            if coined in used:
                coined = f"{adj} {noun} ({i})"
            used.add(coined)
            fam = _weighted_choice(r, fams, [shares[f] for f in fams])
            dv = dvs[int(r.integers(len(dvs)))]
            my = self.measures["constructs"][dv]
            design = _weighted_choice(r, list(design_shares), list(design_shares.values()))
            unit = _weighted_choice(r, list(unit_shares), list(unit_shares.values()))
            if dv == "firm_survival" and unit == "firm-year":
                unit = "firm"
            panel = unit == "firm-year"
            sign = "positive" if r.uniform() < self.tables.predicted_direction_fallback() else "negative"
            var = coined.replace(" ", "_").replace("(", "").replace(")", "")
            mods = [m for m in mod_pool if m != dv]
            chosen_mods = [mods[int(k)] for k in r.choice(len(mods), 2, replace=False)]
            controls = [k for k in self.measures["controls"] if k["construct"] != dv]
            fid = f"F-{i + 1:04d}"
            card = {
                "id": fid, "family": fam, "iv": var, "dv": dv, "iv_var": var, "dv_var": my["var"],
                "iv_label": coined.capitalize(), "dv_label": my["label"], "coined": coined,
                "iv_definition": f"Index of {coined} constructed from the firm's reported activities (0-1)",
                "secondary_dv_var": (my.get("secondary") or {}).get("var"),
                "prediction_sign": sign,
                "prediction": prediction_sentence(self.families[fam]["label"], coined, my["label"], sign, fam),
                "moderators": chosen_mods,
                "moderator_vars": [self.measures["constructs"][m]["var"] if m in self.measures["constructs"] else m for m in chosen_mods],
                "subsample_contrast": SUBSAMPLE_CONTRAST,
                "controls_pool": [k["var"] for k in controls] + (["industry", "year"] if panel else ["industry"]),
                "design": design, "unit": unit, "panel": panel,
                "access_cost": "costly" if ("survey" in design or "hand_collected" in design) else "cheap",
                "frame_quantiles": self.tables.frame_quantiles(unit),
                "panel_length_quantiles": self.tables.panel_length_quantiles() if panel else None,
                "identification": {"instrument_variable": None, "firm_identifiers": True},
                "corpus_papers": 0, "pair_source": "fringe", "core": False, "promoted_quarter": None,
            }
            self.fringe.append(card)
            self.truth["fringe"][fid] = self._draw_truth(card, r, q_mean)
            i += 1

    def replenish_fringe(self, target: int):
        """Refill the pool to `target` cards (long horizons). Deterministic in the number already drawn."""
        n_have = len(self.fringe)
        if n_have < target:
            self._build_fringe(target - n_have, start_index=n_have)

    # ------------------------------------------------------------ helpers
    def question(self, qid: str) -> dict:
        for q in self.questions:
            if q["id"] == qid:
                return q
        for f in self.fringe:
            if f["id"] == qid:
                return f
        raise KeyError(qid)

    def truth_of(self, qid: str) -> dict:
        return self.truth["questions"].get(qid) or self.truth["fringe"][qid]

    def deid_map(self, run_seed: int) -> dict:
        """Seeded permutation of construct codes and theory letters; shared by both regimes at a seed."""
        r = rng(run_seed, "deid")
        constructs = sorted({q["iv"] for q in self.questions} | {q["dv"] for q in self.questions}
                            | {m for q in self.questions for m in q["moderators"]}
                            | {k["construct"] for k in self.measures["controls"]} | {f["iv"] for f in self.fringe})
        perm = r.permutation(len(constructs))
        cmap = {c: f"C-{int(perm[i]) + 1:02d}" for i, c in enumerate(constructs)}
        letters = [chr(ord("A") + int(i)) for i in r.permutation(len(self.tables.family_ids))]
        fmap = {f: f"Theory {letters[i]}" for i, f in enumerate(self.tables.family_ids)}
        return {"constructs": cmap, "families": fmap}

    def write(self, out_dir: Path, run_seed: int) -> dict:
        out_dir = Path(out_dir)
        write_json(out_dir / "questions.json", self.questions)
        write_json(out_dir / "fringe.json", self.fringe)
        write_json(out_dir / "families.json", self.families)
        write_json(out_dir / "truth.json", self.truth)  # HIDDEN: never copied into any task folder
        deid = self.deid_map(run_seed)
        write_json(out_dir / "deid_map.json", deid)
        return deid


def truth_class(theta: float, threshold: float = 0.02) -> str:
    if theta == 0.0:
        return "null"
    if abs(theta) < threshold:
        return "negligible"
    return "positive" if theta > 0 else "negative"


def build_landscape(cfg: dict, seeds: dict, tables: SeedingTables) -> Landscape:
    return Landscape(cfg, seeds, tables).build()
