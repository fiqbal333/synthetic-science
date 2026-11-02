"""Dataset generator with the hidden-truth bookkeeping and the analytic (sufficient-statistics) path used by fast mode.

Full path: frame draw -> perturbed nuisance structure -> recursive linear system with firm and year components ->
measurement -> raw scales -> missingness -> traps -> admissibility -> codebook.md / sample_frame.md -> leak scan.

Hidden quantities per dataset:
  theta        the design parameter: population partial r of latent X and Y given the controls AND the latent confounder U
  theta_naive  what pooled OLS of the measured variables with the standard controls converges to (total covariance)
  theta_fe     what firm fixed effects converge to (within-firm covariance; U_firm and year shocks absorbed)
  plus the latent covariance (truth for any other relation among the dataset's variables), traps and their handling.
Measurement error sits on the focal variables (controls are measured exactly; a design default), so the two estimands are exact.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .config import rng
from .leakscan import scan_text
from .stats import ols, partial_corr_from_cov
from .traps import MEDIATOR_SHARE, MEDIATOR_VAR, PRE_FIT_NAIVE, REGISTRY, draw_traps
from .util import sha256_text, write_text

STANDARD_CONTROL_CONSTRUCTS = {"ln_assets": "firm_size", "firm_age": "firm_age", "leverage": "capital_structure",
                               "rnd_intensity": "innovation_input", "prior_roa": "accounting_performance",
                               "ln_experience": "experience_learning"}
BINARY_FAMILIES = {"survival_exit", "ceo_duality", "ipo_public_status"}
PANEL_C_FIRM, PANEL_C_WITHIN = 0.8, 0.6          # firm-level and within shares of the controls' variance (design default)
EY = {"firm": 0.45, "year": 0.15}                 # firm and year components of the outcome disturbance (design default)


# ---------------------------------------------------------------------- frame
def draw_frame(q: dict, r: np.random.Generator, n_cap: int, upper_half: bool = False) -> dict:
    """N from the corpus quantiles for the question's unit (log-linear interpolation); panel length; balance."""
    qs = q["frame_quantiles"]
    probs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    vals = np.log(np.array([qs["p05"], qs["p10"], qs["p25"], qs["p50"], qs["p75"], qs["p90"], qs["p95"]]))
    u = r.uniform(0.5 if upper_half else 0.05, 0.95)
    n = int(round(np.exp(np.interp(u, probs, vals))))
    n = int(min(max(n, 60), n_cap))
    frame = {"n_target": n, "unit": q["unit"], "panel": q["panel"], "upper_half": bool(upper_half)}
    if q["panel"]:
        pq = q["panel_length_quantiles"]
        T = int(round(np.exp(np.interp(r.uniform(0.1, 0.9), [0.1, 0.25, 0.5, 0.75, 0.9],
                                        np.log([max(pq["p10"], 2), max(pq["p25"], 3), pq["p50"], pq["p75"], pq["p90"]])))))
        T = int(min(max(T, 3), 20))
        n_firms = max(20, n // T)
        frame.update({"n_firms": n_firms, "n_years": T, "start_year": int(r.integers(1998, 2012)),
                      "balance": float(r.uniform(0.75, 1.0))})
    return frame


# ---------------------------------------------------------------------- structure
def _nuisance_matrix(labels: list[str], pooled_labels: list[str], pooled: np.ndarray, r: np.random.Generator,
                     z_sd: float, min_eigen: float) -> np.ndarray:
    k = len(labels)
    R = np.eye(k)
    idx = {c: i for i, c in enumerate(pooled_labels)}
    for i in range(k):
        for j in range(i + 1, k):
            a, b = labels[i], labels[j]
            base = pooled[idx[a], idx[b]] if a in idx and b in idx else 0.0
            z = np.arctanh(np.clip(base, -0.95, 0.95)) + r.normal(0, z_sd)
            R[i, j] = R[j, i] = np.tanh(z)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, min_eigen, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / d[:, None] / d[None, :]


class Structure:
    """Linear structural model for one dataset with exact total and within-firm covariances."""

    def __init__(self, q: dict, truth: dict, controls: list[str], pooled_labels, pooled, r: np.random.Generator,
                 z_sd: float, min_eigen: float, traps: list[str], measures: dict, archival_validity: dict, alpha: dict):
        self.q, self.truth, self.controls, self.traps = q, truth, list(controls), list(traps)
        self.panel = bool(q["panel"])
        cons = measures["constructs"]
        constructs = [STANDARD_CONTROL_CONSTRUCTS.get(c, c) for c in controls]
        labels = constructs + [q["iv"], q["dv"]]
        R = _nuisance_matrix(labels, pooled_labels, pooled, r, z_sd, min_eigen)
        k = self.k = len(controls)
        self.R_cc = R[:k, :k]
        r_cx, r_cy = R[:k, k].copy(), R[:k, k + 1].copy()
        if "omitted_standard_control" in traps and "ln_assets" in controls:
            i = controls.index("ln_assets")
            r_cx[i] = 0.45 * np.sign(r_cx[i] if r_cx[i] != 0 else 1.0)
            r_cy[i] = 0.45 * np.sign(r_cy[i] if r_cy[i] != 0 else 1.0)
        Rcc_inv = np.linalg.inv(self.R_cc)
        a = Rcc_inv @ r_cx
        a_u = truth["confounder"]["r_ux"]
        var_x_expl = float(a @ self.R_cc @ a + a_u ** 2)
        if var_x_expl > 0.7:
            s = np.sqrt(0.7 / var_x_expl)
            a, a_u, var_x_expl = a * s, a_u * s, 0.7
        self.a, self.a_u, self.sd_ex = a, a_u, float(np.sqrt(1.0 - var_x_expl))
        d = Rcc_inv @ r_cy
        d = d * min(1.0, np.sqrt(0.5 / max(float(d @ self.R_cc @ d), 1e-9)))
        self.d, self.d_u = d, float(truth["confounder"]["r_uy"])
        theta = truth["theta"]
        self.beta_total = theta / (self.sd_ex * np.sqrt(1.0 - theta ** 2))
        self.has_mediator = "bad_control" in traps
        self.beta_direct = self.beta_total * (1.0 - MEDIATOR_SHARE) if self.has_mediator else self.beta_total
        self.lambda_m = 0.6 if self.has_mediator else 0.0
        self.gamma_m = (self.beta_total * MEDIATOR_SHARE / self.lambda_m) if self.has_mediator else 0.0
        theta2 = truth["theta_secondary"]
        self.beta2 = theta2 / (self.sd_ex * np.sqrt(1.0 - theta2 ** 2))
        self.mod_coefs = {m: truth["moderators"].get(m, 0.0) for m in q["moderator_vars"]}
        self.sub_coef = truth["subsample_offset"]
        s = truth["confounder"]["time_invariant_share"]
        self.u_firm_sd, self.u_it_sd = float(np.sqrt(s)), float(np.sqrt(1 - s))
        self.ey_firm, self.ey_year = (EY["firm"], EY["year"]) if self.panel else (0.0, 0.0)
        self.ey_idio = float(np.sqrt(1 - self.ey_firm ** 2 - self.ey_year ** 2))
        self.c_firm, self.c_within = (PANEL_C_FIRM, PANEL_C_WITHIN) if self.panel else (1.0, 0.0)
        # measurement reliabilities for the focal variables (fresh per dataset)
        def rel(construct):
            c = cons.get(construct, {})
            if not c:
                return 1.0
            if c.get("source") == "survey":
                return float(np.clip(r.normal(alpha["median"], (alpha["median"] - alpha["p10"]) / 1.28), 0.5, 0.98))
            return float(r.uniform(archival_validity["low"], archival_validity["high"]))
        self.rel_x, self.rel_y, self.rel_y2 = rel(q["iv"]), rel(q["dv"]), rel(q["dv"])
        dvk = cons[q["dv"]]["kind"]
        self.dv_binary = dvk == "binary" or cons[q["dv"]].get("family") in BINARY_FAMILIES
        self.dv_p = 0.5
        self._build_loadings()

    # shocks: Cf(k) Ci(k) U_f U_i e_x e_m e_yf e_yy e_yi e_y2 m_x m_y ; within flag per shock
    def _build_loadings(self):
        k = self.k
        nb = 2 * k + 10
        self.shock_within = np.array([False] * k + [True] * k + [False, True, True, True, False, False, True, True, True, True])
        Cov = np.eye(nb)
        Cov[:k, :k] = self.R_cc
        Cov[k:2 * k, k:2 * k] = self.R_cc
        self.Cov = Cov
        i = {"Uf": 2 * k, "Ui": 2 * k + 1, "ex": 2 * k + 2, "em": 2 * k + 3, "eyf": 2 * k + 4, "eyy": 2 * k + 5,
             "eyi": 2 * k + 6, "ey2": 2 * k + 7, "mx": 2 * k + 8, "my": 2 * k + 9}
        self.ix = i
        C = np.zeros((k, nb))
        for j in range(k):
            C[j, j] = self.c_firm
            C[j, k + j] = self.c_within
        U = np.zeros(nb); U[i["Uf"]] = self.u_firm_sd; U[i["Ui"]] = self.u_it_sd
        Uf = np.zeros(nb); Uf[i["Uf"]] = 1.0
        X = self.a @ C + self.a_u * U
        X[i["ex"]] += self.sd_ex
        M = self.lambda_m * X
        M[i["em"]] += np.sqrt(max(1 - self.lambda_m ** 2, 0.1))
        Y = self.beta_direct * X + self.gamma_m * M + self.d @ C + self.d_u * U
        Y[i["eyf"]] += self.ey_firm; Y[i["eyy"]] += self.ey_year; Y[i["eyi"]] += self.ey_idio
        self.sd_y = float(np.sqrt(Y @ Cov @ Y))
        Y = Y / self.sd_y
        Y2 = self.beta2 * X + self.d @ C + self.d_u * U
        Y2[i["ey2"]] += 1.0
        self.sd_y2 = float(np.sqrt(Y2 @ Cov @ Y2))
        Y2 = Y2 / self.sd_y2
        Xo = np.sqrt(self.rel_x) * X; Xo[i["mx"]] += np.sqrt(1 - self.rel_x)
        Yo = np.sqrt(self.rel_y) * Y; Yo[i["my"]] += np.sqrt(1 - self.rel_y)
        rows = [X, Y, Y2] + [C[j] for j in range(k)] + [U, Uf, M, Xo, Yo]
        self.names = ["X", "Y", "Y2"] + self.controls + ["U", "U_firm", "M", "X_obs", "Y_obs"]
        self.A = np.vstack(rows)

    def _cov(self, within: bool) -> np.ndarray:
        if within:
            mask = self.shock_within
            A = self.A[:, mask]
            S = A @ self.Cov[np.ix_(mask, mask)] @ A.T
        else:
            S = self.A @ self.Cov @ self.A.T
        if self.dv_binary:  # biserial attenuation of the measured binary outcome (a design approximation)
            z = sstats.norm.ppf(self.dv_p)
            f = sstats.norm.pdf(z) / np.sqrt(self.dv_p * (1 - self.dv_p))
            iy = self.names.index("Y_obs")
            S = S.copy()
            S[iy, :] *= f; S[:, iy] *= f; S[iy, iy] /= f * f
        return S

    def analytic_truth(self) -> dict:
        ix = {n: i for i, n in enumerate(self.names)}
        ctrl = [ix[c] for c in self.controls]
        St = self._cov(False)
        theta = partial_corr_from_cov(St, ix["X"], ix["Y"], ctrl + [ix["U"]])
        theta_naive = partial_corr_from_cov(St, ix["X_obs"], ix["Y_obs"], ctrl)
        if self.panel:
            Sw = self._cov(True)
            theta_fe = partial_corr_from_cov(Sw, ix["X_obs"], ix["Y_obs"], ctrl)
        else:
            theta_fe = theta_naive
        return {"theta_implied": theta, "theta_naive": theta_naive, "theta_fe": theta_fe, "cov_names": self.names,
                "cov": np.round(St, 6).tolist(), "rel_x": self.rel_x, "rel_y": self.rel_y}

    def relation_truth(self, x: str, y: str, controls: list[str]) -> float | None:
        """Population partial r (latent scale) of any two dataset variables given the standard controls and U."""
        ix = {n: i for i, n in enumerate(self.names)}
        key = {self.q["iv_var"]: "X", self.q["dv_var"]: "Y", self.q.get("secondary_dv_var"): "Y2", MEDIATOR_VAR: "M"}
        xx, yy = key.get(x, x), key.get(y, y)
        if xx not in ix or yy not in ix or xx == yy:
            return None
        S = self._cov(False)
        given = [ix[c] for c in controls if c in ix and c not in (xx, yy)] + [ix["U"]]
        return partial_corr_from_cov(S, ix[xx], ix[yy], given)

    # simulation mirrors the loadings exactly
    def simulate(self, frame: dict, r: np.random.Generator) -> pd.DataFrame:
        q, k = self.q, self.k
        if frame["panel"]:
            nf, T = frame["n_firms"], frame["n_years"]
            firm = np.repeat(np.arange(nf), T)
            year = np.tile(np.arange(T), nf) + frame["start_year"]
            n = nf * T
        else:
            n = frame["n_target"]
            firm, year, nf = np.arange(n), np.full(n, 2015), n
        L = np.linalg.cholesky(self.R_cc)
        Cf = (L @ r.normal(size=(k, nf))).T[firm] * self.c_firm
        Ci = (L @ r.normal(size=(k, n))).T * self.c_within if self.c_within > 0 else 0.0
        C = Cf + Ci
        U = r.normal(size=nf)[firm] * self.u_firm_sd + r.normal(size=n) * self.u_it_sd
        X = C @ self.a + self.a_u * U + self.sd_ex * r.normal(size=n)
        M = self.lambda_m * X + np.sqrt(max(1 - self.lambda_m ** 2, 0.1)) * r.normal(size=n)
        # a moderator that is also a standard control IS that control (no separate draw)
        mods = {m: (C[:, self.controls.index(m)] if m in self.controls else r.normal(size=n)) for m in q["moderator_vars"]}
        industry = r.integers(1, 9, size=nf)[firm]
        high_tech = (industry <= 3).astype(float)
        ey = r.normal(size=nf)[firm] * self.ey_firm + r.normal(size=len(np.unique(year)))[year - year.min()] * self.ey_year \
            + self.ey_idio * r.normal(size=n)
        inter = sum(self.mod_coefs[m] * X * mods[m] for m in mods) + self.sub_coef * X * (high_tech - high_tech.mean())
        Y = (self.beta_direct * X + self.gamma_m * M + C @ self.d + self.d_u * U + inter + ey) / self.sd_y
        Y2 = (self.beta2 * X + C @ self.d + self.d_u * U + r.normal(size=n)) / self.sd_y2
        Xo = np.sqrt(self.rel_x) * X + np.sqrt(1 - self.rel_x) * r.normal(size=n)
        Yo = np.sqrt(self.rel_y) * Y + np.sqrt(1 - self.rel_y) * r.normal(size=n)
        Y2o = np.sqrt(self.rel_y2) * Y2 + np.sqrt(1 - self.rel_y2) * r.normal(size=n)
        df = pd.DataFrame({"firm_id": firm + 1001, "year": year.astype(int), "industry": industry.astype(int),
                           "data_source": np.where(r.uniform(size=nf)[firm] < 0.15, "B", "A")})
        if not frame["panel"]:
            df = df.drop(columns=["year"])
        df["_X"], df["_Y"], df["_Y2"], df["_M"] = Xo, Yo, Y2o, M
        for j, c in enumerate(self.controls):
            df[f"_{c}"] = C[:, j]
        for m in mods:
            if m not in self.controls:
                df[f"_{m}"] = mods[m]
        df["high_tech"] = high_tech.astype(int)
        if q["identification"]["instrument_variable"]:
            F = self.truth["instrument"]["F"]
            rho = np.sqrt(F / (F + n / 10.0))
            Z = rho * (X - C @ self.a) / max(self.sd_ex, 1e-6) + np.sqrt(max(1 - rho ** 2, 0.05)) * r.normal(size=n) \
                + self.truth["instrument"]["exclusion_r"] * ey
            df[q["identification"]["instrument_variable"]] = np.round(Z, 4)
        if frame["panel"] and frame["balance"] < 1.0:
            keep = r.uniform(size=n) < frame["balance"] + (1 - frame["balance"]) * (year < year.max())
            df = df[keep].reset_index(drop=True)
        return df


# ---------------------------------------------------------------------- raw scales
def apply_scales(df: pd.DataFrame, q: dict, st: Structure, controls: list[str], measures: dict, tables,
                 r: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Standardized measured variables -> raw scales from the descriptives table (03g)."""
    cons = measures["constructs"]
    meta = {}

    def scale(z, construct, var, rel):
        c = cons.get(construct, {})
        fam = c.get("family")
        mean, sd = tables.descriptive(fam) if fam else (0.5, 0.2)
        if fam == "roa":
            sd = 0.22
        kind = c.get("kind", "continuous")
        if kind == "binary" or fam in BINARY_FAMILIES:
            p = float(np.clip(mean, 0.05, 0.95)) if 0 < mean < 1 else 0.5
            out = (z > np.quantile(z, 1 - p)).astype(int)
        else:
            out = mean + sd * z
            if kind == "ratio" and fam not in ("roa", "ros", "stock_return_car", "profit_margin", "slack", "cash_flow"):
                out = np.clip(out, 0.0, None)
            out = np.round(out, 4)
        meta[var] = {"reliability": rel, "mean": mean, "sd": sd, "kind": kind}
        return out

    out = pd.DataFrame({k: df[k] for k in df.columns if not k.startswith("_")})
    out[q["iv_var"]] = scale(df["_X"].to_numpy(), q["iv"], q["iv_var"], st.rel_x)
    out[q["dv_var"]] = scale(df["_Y"].to_numpy(), q["dv"], q["dv_var"], st.rel_y)
    if q.get("secondary_dv_var"):
        sec = cons[q["dv"]]["secondary"]
        m, s = tables.descriptive(sec["family"])
        out[q["secondary_dv_var"]] = np.round(m + s * df["_Y2"].to_numpy(), 4)
        meta[q["secondary_dv_var"]] = {"reliability": st.rel_y2, "mean": m, "sd": s, "kind": "continuous"}
    for c in controls:
        out[c] = scale(df[f"_{c}"].to_numpy(), STANDARD_CONTROL_CONSTRUCTS.get(c, c), c, 1.0)
    for m in q["moderator_vars"]:
        if m in out.columns:
            continue
        construct = next((k for k, v in cons.items() if v["var"] == m), None)
        rel = float(r.uniform(0.6, 0.9))
        z = np.sqrt(rel) * df[f"_{m}"].to_numpy() + np.sqrt(1 - rel) * r.normal(size=len(df))
        out[m] = scale(z, construct, m, rel) if construct else np.round(z, 4)
    return out, meta


def apply_missingness(df: pd.DataFrame, controls: list[str], r: np.random.Generator, rate: float) -> pd.DataFrame:
    df = df.copy()
    if rate <= 0 or len(controls) < 2:
        return df
    size = df["ln_assets"].to_numpy(dtype=float) if "ln_assets" in df.columns else np.zeros(len(df))
    z = (size - np.nanmean(size)) / (np.nanstd(size) + 1e-9)
    for c in controls[1:3]:
        p = np.clip(rate * (1 - 0.5 * z), 0.01, 0.5)   # smaller firms more often missing (MAR on size)
        mask = r.uniform(size=len(df)) < p * (rate / p.mean())
        df.loc[mask, c] = np.nan
    return df


# ---------------------------------------------------------------------- codebook and sample frame text
def codebook_md(q: dict, controls: list[str], measures: dict, df: pd.DataFrame) -> str:
    cons = measures["constructs"]
    ctrl_defs = {c["var"]: c for c in measures["controls"]}
    lines = [f"# Codebook: {q['iv_label']} and {q['dv_label']}", "",
             f"Unit of observation: {q['unit']}. Rows: {len(df):,}. Columns: {len(df.columns)}.", "",
             "| Variable | Definition | Unit / coding | Source |", "|---|---|---|---|"]
    src = {"firm_id": "Firm identifier (anonymized)", "year": "Fiscal year",
           "industry": "Primary industry group (1-8; groups 1-3 are high-technology)",
           "data_source": "Data vendor from which the firm's records were assembled (A or B)",
           "high_tech": "1 if the firm's primary industry group is high-technology (groups 1-3)"}
    for v in df.columns:
        if v in src:
            lines.append(f"| {v} | {src[v]} | code | compiled |")
        elif v == q["iv_var"]:
            c = cons.get(q["iv"], {})
            lines.append(f"| {v} | {c.get('definition', q.get('iv_definition', q['iv_label']))} | {c.get('unit', 'index')} | {c.get('source', 'archival')} |")
        elif v == q["dv_var"]:
            c = cons[q["dv"]]
            lines.append(f"| {v} | {c['definition']} | {c['unit']} | {c['source']} |")
        elif v == q.get("secondary_dv_var"):
            s = cons[q["dv"]]["secondary"]
            lines.append(f"| {v} | {s['definition']} | as reported | archival |")
        elif v in ctrl_defs:
            c = ctrl_defs[v]
            lines.append(f"| {v} | {c['definition']} | {c['unit']} | archival |")
        elif v == MEDIATOR_VAR:
            lines.append(f"| {v} | Operating efficiency: operating costs over sales, inverted and standardized, measured at the end of the fiscal year | index | archival |")
        elif v[-6:-1] == "_item":
            lines.append(f"| {v} | Survey item {v[-1]} of the {q['iv_label'].lower()} scale; the scale score is the mean of the three items | 1-7 | survey |")
        elif v == q["identification"].get("instrument_variable"):
            lines.append(f"| {v} | Index of regional policy support for the firm's main activity in the year | index | archival |")
        else:
            construct = next((k for k, c in cons.items() if c["var"] == v), None)
            c = cons.get(construct, {}) if construct else {}
            lines.append(f"| {v} | {c.get('definition', v.replace('_', ' '))} | {c.get('unit', 'index')} | {c.get('source', 'archival')} |")
    lines += ["", "Missing values are coded as empty cells.", ""]
    return "\n".join(lines)


def sample_frame_md(q: dict, frame: dict, df: pd.DataFrame, traps: list[str]) -> str:
    lines = ["# Sample frame", ""]
    if frame["panel"]:
        lines.append(f"The data are an unbalanced firm-year panel of {df['firm_id'].nunique():,} firms observed over "
                     f"{frame['n_years']} fiscal years starting in {frame['start_year']}, {len(df):,} firm-years in all.")
    else:
        lines.append(f"The data are a cross-section of {len(df):,} {q['unit']}-level observations for one fiscal year.")
    lines.append("Firms were drawn from the population of publicly listed firms in the industries covered by the two data vendors "
                 "(coded A and B in `data_source`); vendor records were merged on firm identifiers.")
    if "selection_on_outcome" in traps:
        lines.append("The sample consists of firms that remained in the vendors' coverage lists at the end of the observation window.")
    if "survey" in q["design"]:
        lines.append("Survey measures were collected from one senior executive per firm; archival measures were matched from the vendors' records.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- oracle fits
def fit_focal(df: pd.DataFrame, q: dict, controls: list[str], panel: bool, y: str | None = None, x: str | None = None) -> dict:
    y, x = y or q["dv_var"], x or q["iv_var"]
    terms = [x] + [c for c in controls if c in df.columns and c not in ("industry", "year", x, y)]
    if panel:
        return ols(df, y, terms, cluster="firm_id", fe="firm_id", year="year")
    d = df.copy()
    ind = pd.get_dummies(d["industry"].astype(int), prefix="ind", drop_first=True).astype(float)
    d = pd.concat([d, ind], axis=1)
    return ols(d, y, terms + list(ind.columns), cluster=None)


def oracle_fits(df_planted: pd.DataFrame, q: dict, controls: list[str], traps: list[str]) -> dict:
    """Correct analysis (all traps handled) and the naive analysis per trap (that trap unhandled, others handled)."""
    panel = q["panel"]
    d = df_planted
    for t in traps:
        d = REGISTRY[t].correct(d, q)
    correct = fit_focal(d, q, controls, panel)
    naive = {}
    for t in traps:
        d2 = df_planted
        for t2 in traps:
            if t2 != t:
                d2 = REGISTRY[t2].correct(d2, q)
        if t in PRE_FIT_NAIVE:
            d2 = PRE_FIT_NAIVE[t](d2, q)
        naive[t] = fit_focal(d2, q, REGISTRY[t].naive_controls(controls, q), panel)
    return {"correct": _focal(correct, q), "naive": {t: _focal(v, q) for t, v in naive.items()}}


def _focal(fit: dict, q: dict) -> dict:
    c = fit["coef"].get(q["iv_var"], {})
    return {"estimate": c.get("estimate"), "se": c.get("se"), "t": c.get("t"), "p": c.get("p"),
            "partial_r": c.get("partial_r"), "n": fit["n"], "df": fit.get("df")}


def admissible(theta_target: float, fits: dict, traps: list[str], bias_share: float) -> tuple[bool, list[str]]:
    """Admissibility: the correct analysis recovers the target within 1.96 SE (partial-r scale); each trap biases the
    naive analysis by at least max(bias_share*|target|, 1 SE) or flips the sign (duplicates: inflates N)."""
    fails = []
    c = fits["correct"]
    if c["partial_r"] is None or not np.isfinite(c["partial_r"]):
        return False, ["correct fit failed"]
    se_r = 1.0 / np.sqrt(max(c["df"] or 10, 10))
    if abs(c["partial_r"] - theta_target) > 1.96 * se_r:
        fails.append(f"correct analysis misses target: {c['partial_r']:.4f} vs {theta_target:.4f} (se {se_r:.4f})")
    for t in traps:
        nv = fits["naive"][t]
        if nv["partial_r"] is None or not np.isfinite(nv["partial_r"]):
            fails.append(f"{t}: naive fit failed")
            continue
        bias = abs(nv["partial_r"] - c["partial_r"])
        flipped = np.sign(nv["partial_r"]) != np.sign(c["partial_r"]) and abs(c["partial_r"]) > se_r
        se_cut = 1.0 / np.sqrt(max(nv["df"] or 10, 10))
        need = max(bias_share * abs(theta_target), se_cut)
        ok = nv["n"] > c["n"] * 1.01 if t == "duplicates" else (bias >= need or flipped)
        if not ok:
            fails.append(f"{t}: naive bias {bias:.4f} < required {need:.4f}")
    return (not fails), fails


# ---------------------------------------------------------------------- the generator entry point
class DatasetGenerator:
    def __init__(self, cfg: dict, tables, landscape, run_dir: Path | None = None):
        self.cfg, self.tables, self.landscape = cfg, tables, landscape
        self.dc = cfg["datasets"]
        self.measures = tables.construct_measures()
        self.pooled_labels, self.pooled = tables.corr25()
        m = tables.measurement()
        self.alpha = {"median": m["alpha_median"], "p10": m["alpha_p10"]}
        self.run_dir = Path(run_dir) if run_dir else None

    def spec_for(self, q: dict) -> dict:
        cons = self.measures["constructs"]
        controls = [c for c in q["controls_pool"] if c not in ("industry", "year")]
        return {"iv_var": q["iv_var"], "dv_var": q["dv_var"], "panel": q["panel"], "controls": controls,
                "dv_kind": cons[q["dv"]]["kind"],
                "iv_source": cons.get(q["iv"], {}).get("source", "archival") if q["iv"] in cons else "archival",
                "design": q["design"]}

    def _structure(self, q, truth, spec, traps, r) -> Structure:
        return Structure(q, truth, spec["controls"], self.pooled_labels, self.pooled, r, self.dc["perturb_z_sd"],
                         self.dc["min_eigen"], traps, self.measures, self.dc["archival_validity"], self.alpha)

    def generate(self, qid: str, dataset_seed: int, dataset_id: str, upper_half: bool = False,
                 write_dir: Path | None = None, n_override: int | None = None) -> dict:
        """Full path: returns {'meta', 'hidden', 'df'} and writes data.csv/codebook.md/sample_frame.md if write_dir."""
        q = self.landscape.question(qid)
        truth = self.landscape.truth_of(qid)
        spec = self.spec_for(q)
        frame = draw_frame(q, rng(dataset_seed, "frame"), self.dc["n_cap"], upper_half=upper_half)
        if n_override:
            frame["n_target"] = int(n_override)
            if frame["panel"]:
                frame["n_firms"] = max(20, frame["n_target"] // frame["n_years"])
        traps = draw_traps(spec, rng(dataset_seed, "traps"), self.dc["traps"]["density"], self.dc["traps"]["registry"])
        attempts, ok, result = [], False, None
        max_redraws = self.dc["traps"]["admissibility"]["max_redraws"]
        attempt = 0
        while True:
            r = rng(dataset_seed, "draw", attempt)
            result = self._draw_once(q, truth, spec, frame, traps, r)
            ok, fails = admissible(result["target"], result["fits"], traps, self.dc["traps"]["admissibility"]["bias_share"])
            attempts.append({"attempt": attempt, "traps": list(traps), "ok": ok, "fails": fails})
            if ok or attempt >= max_redraws + 4:
                break
            attempt += 1
            if attempt > max_redraws and traps and attempt % 2 == 1:
                traps = traps[:-1]  # after repeated inadmissible draws, drop the last trap and keep drawing
        df = result["df"]
        codebook = codebook_md(q, spec["controls"], self.measures, df)
        sframe = sample_frame_md(q, frame, df, traps)
        leaks = scan_text(codebook, "strict") + scan_text(sframe, "strict")
        if leaks:
            raise RuntimeError(f"leak scan failed for {dataset_id}: {leaks}")
        csv_text = to_csv_text(df)
        data_hash = sha256_text(csv_text)
        if write_dir is not None:
            write_dir = Path(write_dir)
            write_text(write_dir / "data.csv", csv_text)
            write_text(write_dir / "codebook.md", codebook)
            write_text(write_dir / "sample_frame.md", sframe)
        meta = {"dataset_id": dataset_id, "question_id": qid, "dataset_seed": int(dataset_seed), "n_rows": int(len(df)),
                "n_correct": int(result["fits"]["correct"]["n"]), "frame": frame, "data_sha256": data_hash,
                "columns": list(df.columns), "admissible": bool(ok),
                "variables": {"iv": q["iv_var"], "dv": q["dv_var"], "secondary_dv": q.get("secondary_dv_var"),
                              "moderators": q["moderator_vars"], "controls": spec["controls"],
                              "instrument": q["identification"]["instrument_variable"], "subsample": "high_tech"}}
        an = result["analytic"]
        hidden = {"traps": traps, "theta": truth["theta"], "theta_fe": an["theta_fe"], "theta_naive": an["theta_naive"],
                  "theta_implied": an["theta_implied"], "target": result["target"], "oracle": result["fits"],
                  "admissibility": attempts, "mediator_var": MEDIATOR_VAR if "bad_control" in traps else None,
                  "moderators": truth["moderators"], "subsample_offset": truth["subsample_offset"],
                  "theta_secondary": truth["theta_secondary"], "cov_names": an["cov_names"], "cov": an["cov"],
                  "reliability": {"x": an["rel_x"], "y": an["rel_y"]}, "measurement": result["meas_meta"]}
        return {"meta": meta, "hidden": hidden, "df": df, "structure": result["structure"]}

    def _draw_once(self, q, truth, spec, frame, traps, r):
        st = self._structure(q, truth, spec, traps, r)
        analytic = st.analytic_truth()
        latent = st.simulate(frame, r)
        df, meas_meta = apply_scales(latent, q, st, spec["controls"], self.measures, self.tables, r)
        if "bad_control" in traps:
            df[MEDIATOR_VAR] = np.round(latent["_M"].to_numpy(), 4)
        df = apply_missingness(df, spec["controls"], r, self.dc["missing_rate"])
        for t in traps:
            df = REGISTRY[t].plant(df, q, r)
        fits = oracle_fits(df, q, spec["controls"], traps)
        target = analytic["theta_fe"] if q["panel"] else analytic["theta_naive"]
        return {"df": df, "fits": fits, "analytic": analytic, "target": target, "meas_meta": meas_meta, "structure": st}

    # ------------------------------------------------------------ analytic (fast) path
    def analytic_study(self, qid: str, dataset_seed: int, upper_half: bool, handled: dict[str, bool],
                       use_fe: bool, n_override: int | None = None) -> dict:
        """Sufficient-statistics draw (fast mode): estimate ~ N(target + unhandled bias, se^2) with no CSV."""
        q = self.landscape.question(qid)
        truth = self.landscape.truth_of(qid)
        spec = self.spec_for(q)
        frame = draw_frame(q, rng(dataset_seed, "frame"), self.dc["n_cap"], upper_half=upper_half)
        if n_override:
            frame["n_target"] = int(n_override)
            if frame["panel"]:
                frame["n_firms"] = max(20, frame["n_target"] // frame["n_years"])
        traps = draw_traps(spec, rng(dataset_seed, "traps"), self.dc["traps"]["density"], self.dc["traps"]["registry"])
        r = rng(dataset_seed, "draw", 0)
        st = self._structure(q, truth, spec, traps, r)
        an = st.analytic_truth()
        n = frame["n_firms"] * frame["n_years"] if frame["panel"] else frame["n_target"]
        if frame["panel"]:
            n = int(n * (frame["balance"] + (1 - frame["balance"]) * (frame["n_years"] - 1) / frame["n_years"]))
        n_eff = int(n * (1 - self.dc["missing_rate"] * 0.6))
        k = len(spec["controls"]) + 8
        fe = bool(q["panel"] and use_fe)
        target = an["theta_fe"] if fe else an["theta_naive"]
        df_resid = max(n_eff - k - (frame["n_firms"] if fe else 0), 10)
        se = 1.0 / np.sqrt(df_resid)
        est = target + r.normal(0, se)
        for t in traps:
            if not handled.get(t, False):
                est, se = REGISTRY[t].analytic(est, se, r, q)
        n_reported = n if not ("duplicates" in traps and not handled.get("duplicates", False)) else int(n * 1.055)
        atten = np.sqrt(an["rel_x"] * an["rel_y"])
        mods = {m: {"estimate": float(truth["moderators"].get(m, 0.0) * atten + r.normal(0, se * 1.1)), "se": float(se * 1.1)}
                for m in q["moderator_vars"]}
        sec = float(truth["theta_secondary"] * atten + r.normal(0, se))
        sub = float(truth["subsample_offset"] * atten + r.normal(0, se * 1.6))
        return {"question_id": qid, "frame": frame, "traps": traps, "n": int(n), "n_reported": int(n_reported), "df": int(df_resid),
                "focal": {"estimate": float(est), "se": float(se)}, "moderators": mods,
                "secondary": {"estimate": sec, "se": float(se)}, "subsample": {"estimate": sub, "se": float(se * 1.6)},
                "hidden": {"theta": truth["theta"], "theta_fe": an["theta_fe"], "theta_naive": an["theta_naive"],
                           "target": target, "traps": traps, "theta_secondary": truth["theta_secondary"],
                           "moderators": truth["moderators"], "subsample_offset": truth["subsample_offset"],
                           "mediator_var": MEDIATOR_VAR if "bad_control" in traps else None}}


def to_csv_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.4f", lineterminator="\n")
    return buf.getvalue()
