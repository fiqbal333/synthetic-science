"""Seeding tables: the landscape generator never reads the corpus; it reads a table set named in config, so another journal's tables are a config change, not a code change."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import WS_ROOT, load_yaml
from .util import sha256_file


class SeedingTables:
    def __init__(self, table_set: str | Path, root: Path = WS_ROOT):
        p = Path(table_set)
        if not p.exists():
            p = root / "configs" / table_set
        self.path = p
        self.spec = load_yaml(p)
        self.root = root
        self._cache: dict[str, pd.DataFrame] = {}
        self.families: dict = self.spec["families"]
        self.family_ids = list(self.families)

    # ------------------------------------------------------------------ access
    def table_path(self, name: str) -> Path:
        rel = self.spec["tables"][name]
        p = Path(rel)
        return p if p.is_absolute() else self.root / rel

    def df(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.table_path(name))
        return self._cache[name]

    def manifest(self) -> dict:
        out = {"name": self.spec["name"], "source": self.spec.get("source"), "tables": {}}
        for k in self.spec["tables"]:
            p = self.table_path(k)
            out["tables"][k] = {"path": str(p.relative_to(self.root)) if p.is_relative_to(self.root) else str(p),
                                "sha256": sha256_file(p)}
        return out

    # ------------------------------------------------------------- derived
    def theory_shares(self) -> dict[str, float]:
        """Share of papers per core family (corpus families folded as the table set says), normalized."""
        t = self.df("theory_shares").set_index("theory_family")["share_of_papers_with_theory"]
        out = {}
        for fam, spec in self.families.items():
            out[fam] = float(sum(t.get(c, 0.0) for c in spec["corpus"][:1]))  # primary corpus family only (no double counting)
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}

    def pairs(self) -> pd.DataFrame:
        return self.df("pairs")

    def pairs_all(self) -> pd.DataFrame:
        return self.df("pairs_all")

    def pair_theory(self) -> pd.DataFrame:
        return self.df("pair_theory")

    def predicted_positive_share(self, x: str, y: str) -> float | None:
        a = self.pairs_all()
        row = a[(a.x_construct == x) & (a.y_construct == y)]
        if len(row):
            return float(row.iloc[0]["share_positive_pred"])
        return None

    def pair_count(self, x: str, y: str) -> int:
        a = self.pairs_all()
        row = a[(a.x_construct == x) & (a.y_construct == y)]
        return int(row.iloc[0]["n_papers"]) if len(row) else 0

    def predicted_direction_fallback(self) -> float:
        d = self.df("predicted_dir").set_index("predicted_direction")["share"]
        pos, neg = float(d.get("positive", 0.6)), float(d.get("negative", 0.31))
        return pos / (pos + neg)

    def moderator_candidates(self, exclude: set[str]) -> list[str]:
        r = self.df("roles")
        r = r[~r.construct.isin(exclude)].copy()
        r["mod_weight"] = r["share_moderator"] * r["n_papers"]
        return list(r.sort_values("mod_weight", ascending=False)["construct"])

    def design_shares(self) -> dict[str, float]:
        d = self.df("design_shares").set_index("paper_data_design")["share"]
        keep = {k: float(v) for k, v in d.items() if k != "experiment" and k != "other" and not str(k).startswith("missing")}
        s = sum(keep.values())
        return {k: v / s for k, v in keep.items()}

    def unit_shares(self) -> dict[str, float]:
        d = self.df("unit_shares").set_index("unit_of_analysis")["share"]
        used = self.spec.get("units_used", ["firm-year", "firm", "individual", "event", "dyad"])
        keep = {u: float(d.get(u, 0.0)) for u in used}
        s = sum(keep.values())
        return {k: v / s for k, v in keep.items()}

    def frame_quantiles(self, unit: str) -> dict[str, float]:
        f = self.df("frames").set_index("group")
        key = f"unit={unit}"
        if key not in f.index:
            key = "all quant+mixed (sample_size_final)"
        row = f.loc[key]
        return {q: float(row[q]) for q in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")}

    def panel_length_quantiles(self) -> dict[str, float]:
        f = self.df("frames").set_index("group").loc["panel length years (sample_period_length)"]
        return {q: float(f[q]) for q in ("p10", "p25", "p50", "p75", "p90")}

    def estimator_shares(self) -> dict[str, float]:
        d = self.df("estimators")
        d = d[~d.primary_estimation_method.isin(["other", "missing"])]
        return {str(k): float(v) for k, v in zip(d.primary_estimation_method, d.share)}

    def controls_median(self) -> float:
        c = self.df("controls_counts").set_index("measure")
        return float(c.loc["controls per paper (median over its models)"]["p50"])

    def corr25(self) -> tuple[list[str], np.ndarray]:
        m = self.df("corr25")
        labels = list(m["construct"])
        mat = m.drop(columns=["construct"]).to_numpy(dtype=float)
        return labels, mat

    def pair_sd_median(self) -> float:
        p = self.df("pair_sd").sort_values("n_papers", ascending=False).head(30)
        return float(np.median(p["sd_r_across_papers"]))

    def measurement(self) -> dict[str, float]:
        m = self.df("measurement").set_index("stat")["value"]
        return {"alpha_median": float(m.get("alpha median", 0.81)), "alpha_p10": float(m.get("alpha p10", 0.70)),
                "alpha_p90": float(m.get("alpha p90", 0.92)),
                "single_item_share": float(m.get("share single-item (where item_count reported)", 0.72))}

    def descriptive(self, family: str) -> tuple[float, float]:
        d = self.df("descriptives").set_index("h_family")
        if family in d.index:
            return float(d.loc[family, "mean_med"]), float(d.loc[family, "sd_med"])
        return 0.0, 1.0

    def construct_measures(self) -> dict:
        return load_yaml(self.table_path("construct_measures"))

    def hypotheses_dist(self) -> dict[int, float]:
        h = self.df("hypotheses")
        return {int(k): float(v) for k, v in zip(h.n_hypotheses, h.share)}

    def effects_anchor(self) -> dict:
        e = self.df("effects")
        out = {}
        for _, r in e.iterrows():
            out[f"{r['subset']} | {r['stat']}"] = {"n": int(r["n"]), "p50": float(r["p50"])}
        return out
