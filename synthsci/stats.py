"""Small, fast numpy estimators used by the oracle, the admissibility check and the rule agents.

OLS with HC1 or firm-clustered standard errors; firm fixed effects by within-transformation plus year dummies.
Partial r = t / sqrt(t^2 + df). Not used by agents' own analysis.py (which must be self-contained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def partial_r(t: float, df: int) -> float:
    if df <= 0 or not np.isfinite(t):
        return float("nan")
    return float(t / np.sqrt(t * t + df))


def _ols_core(y: np.ndarray, X: np.ndarray, cluster: np.ndarray | None):
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    if cluster is None:
        meat = (X * resid[:, None]).T @ (X * resid[:, None])
        V = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    else:
        codes, inv = np.unique(cluster, return_inverse=True)
        G = len(codes)
        scores = X * resid[:, None]
        S = np.zeros((G, k))
        np.add.at(S, inv, scores)
        meat = S.T @ S
        adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        V = XtX_inv @ meat @ XtX_inv * adj
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se, resid


def ols(df: pd.DataFrame, y: str, terms: list[str], cluster: str | None = None, fe: str | None = None,
        year: str | None = None) -> dict:
    """Fit y on terms (+ intercept). fe: column to demean by (firm effects). year: add year dummies.

    Returns coefficient table for `terms` with estimate, se, t, p, partial_r and the model N.
    """
    cols = [y] + list(terms) + ([cluster] if cluster else []) + ([fe] if fe else []) + ([year] if year else [])
    d = df[list(dict.fromkeys(cols))].dropna()
    n = len(d)
    if n < len(terms) + 5:
        return {"n": n, "coef": {t: {"estimate": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"),
                                     "partial_r": float("nan")} for t in terms}}
    Y = d[y].to_numpy(dtype=float)
    X = d[list(terms)].to_numpy(dtype=float)
    extra = 0
    if year:
        yrs = pd.get_dummies(d[year].astype(int), drop_first=True).to_numpy(dtype=float)
        X = np.hstack([X, yrs])
        extra += yrs.shape[1]
    if fe:
        g = d[fe].to_numpy()
        codes, inv = np.unique(g, return_inverse=True)
        counts = np.bincount(inv).astype(float)
        def demean(M):
            M = np.asarray(M, dtype=float)
            if M.ndim == 1:
                s = np.bincount(inv, weights=M) / counts
                return M - s[inv]
            out = np.empty_like(M)
            for j in range(M.shape[1]):
                s = np.bincount(inv, weights=M[:, j]) / counts
                out[:, j] = M[:, j] - s[inv]
            return out
        Y = demean(Y)
        X = demean(X)
        k_fe = len(codes)
        cl = d[cluster].to_numpy() if cluster else g
    else:
        X = np.hstack([X, np.ones((n, 1))])
        k_fe = 0
        cl = d[cluster].to_numpy() if cluster else None
    beta, se, _ = _ols_core(Y, X, cl)
    df_resid = max(n - X.shape[1] - k_fe, 1)
    from scipy import stats as st
    out = {"n": n, "df": df_resid, "coef": {}}
    for j, t in enumerate(terms):
        tval = beta[j] / se[j] if se[j] > 0 else float("nan")
        p = float(2 * st.t.sf(abs(tval), df_resid)) if np.isfinite(tval) else float("nan")
        out["coef"][t] = {"estimate": float(beta[j]), "se": float(se[j]), "t": float(tval), "p": p,
                          "partial_r": partial_r(tval, df_resid)}
    return out


def partial_corr_from_cov(S: np.ndarray, i: int, j: int, given: list[int]) -> float:
    """Population partial correlation of variables i and j given `given`, from a covariance matrix."""
    idx = [i, j] + list(given)
    sub = S[np.ix_(idx, idx)]
    try:
        P = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(sub)
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))
