"""Deterministic truth scorer and analyst-flag checker. No agent calls; re-runnable from the ledger.

The flag groups are scored from the structured claims block.
"""
from __future__ import annotations

import os
import re
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .belief import NEGLIGIBLE, truth_class
from .traps import MEDIATOR_VAR, REGISTRY, handled_by_report
from .util import read_text

CAUSAL_RE = re.compile(r"\b(causes?|caused|causal(?:ly)?|leads? to|led to|drives?|drove|increases?|decreases?|raises?|lowers?|"
                       r"improves?|reduces?|boosts?|results? in|effect of .{0,40} on|impact of .{0,40} on)\b", re.I)
HEDGE_RE = re.compile(r"\b(may|might|could|suggests?|associated|correlat|consistent with|appear|cannot|not|no evidence|"
                      r"we do not claim|caution)\b", re.I)


def truth_check(verdict: str, estimate: float | None, theta: float) -> str:
    """Primary rule: false_discovery / true_discovery / missed_effect / correct_null / unknown."""
    if estimate is None or not np.isfinite(estimate):
        return "unknown"
    states_effect = verdict in ("supported", "partially")
    if states_effect:
        if theta == 0.0 or np.sign(theta) != np.sign(estimate):
            return "false_discovery"
        return "true_discovery" if abs(theta) >= NEGLIGIBLE else "true_discovery_negligible"
    if theta == 0.0:
        return "correct_null"
    if abs(theta) >= NEGLIGIBLE:
        return "missed_effect"
    return "correct_null_negligible"


def claim_sign(h: dict) -> str:
    """The direction a claim states, from its estimate and p (never from support of the prediction): a significant
    estimate whose sign is opposite to the predicted one is a claim of that sign, not 'no relation' (2026-10-14 fix of
    the repository smoke run; regression test in tests/test_pre_stage1.py)."""
    p, est = h.get("p"), h.get("estimate") or 0.0
    if p is None or not np.isfinite(p) or p >= 0.05 or est == 0:
        return "none"
    return "positive" if est > 0 else "negative"


MDE_Z = 1.959964 + 0.841621    # two-sided alpha .05, power .80


def verification_power(n: int | None, k_controls: int, claim_partial_r: float | None, claim_sign_: str | None,
                       claim_n: int | None = None) -> dict:
    """Minimum detectable partial r at the verifier's N (alpha .05 two-sided, power .80: |r| >= 2.80 / sqrt(df)), recorded
    with every verdict. `powered` says whether the verification could detect the claimed effect (effect
    claims) or a small effect of |r| .05 (null claims); n_ratio is the verifier's N over the claim's."""
    if not n or n <= 0:
        return {"mde_partial_r": None, "powered": None, "powered_for": None, "n_ratio": None}
    df = max(int(n) - int(k_controls or 0) - 2, 1)
    mde = float(MDE_Z / np.sqrt(MDE_Z ** 2 + df))
    if claim_sign_ in (None, "none"):
        target, label = 0.05, "small effect (|r| .05)"
    else:
        target = abs(float(claim_partial_r)) if claim_partial_r is not None and np.isfinite(claim_partial_r) else None
        label = "the claimed effect"
    return {"mde_partial_r": mde, "powered": (target >= mde) if target is not None else None, "powered_for": label,
            "n_ratio": (float(n) / float(claim_n)) if claim_n else None}


def magnitude_ratio(estimate: float | None, theta: float) -> float | None:
    if estimate is None or theta == 0.0 or not np.isfinite(estimate):
        return None
    return float(estimate / theta)


def ci_covers(estimate: float | None, se: float | None, theta: float) -> bool | None:
    if estimate is None or se is None or not (np.isfinite(estimate) and np.isfinite(se)):
        return None
    return bool(estimate - 1.96 * se <= theta <= estimate + 1.96 * se)


def relation_theta(hidden: dict, iv: str, dv: str, moderator: str | None, kind: str, q: dict, structure=None,
                   controls: list[str] | None = None) -> float | None:
    """Truth for any hypothesis among the dataset's focal variables."""
    if moderator:
        if iv == q["iv_var"] and dv == q["dv_var"]:
            if moderator == "high_tech":
                return float(hidden.get("subsample_offset", 0.0))
            return float(hidden.get("moderators", {}).get(moderator, 0.0))
        return 0.0
    if iv == q["iv_var"] and dv == q["dv_var"]:
        return float(hidden["theta"])
    if iv == q["iv_var"] and dv == q.get("secondary_dv_var"):
        return float(hidden.get("theta_secondary", 0.0))
    if structure is not None:
        v = structure.relation_truth(iv, dv, controls or [])
        if v is not None:
            return float(v)
    cov = hidden.get("cov")
    names = hidden.get("cov_names")
    if cov and names:
        from .stats import partial_corr_from_cov
        key = {q["iv_var"]: "X", q["dv_var"]: "Y", q.get("secondary_dv_var"): "Y2", MEDIATOR_VAR: "M"}
        xx, yy = key.get(iv, iv), key.get(dv, dv)
        if xx in names and yy in names and xx != yy:
            ix = {n: i for i, n in enumerate(names)}
            given = [ix[c] for c in (controls or []) if c in ix and c not in (xx, yy)] + [ix["U"]]
            return float(partial_corr_from_cov(np.array(cov), ix[xx], ix[yy], given))
    return None


def score_hypothesis(h: dict, theta: float | None, theta_fe: float | None) -> dict:
    est = h.get("estimate")
    if theta is None:
        return {"theta": None, "truth_class": "unknown", "truth_check": "unknown", "truth_check_lenient": "unknown",
                "magnitude_ratio": None, "ci_covers": None}
    return {"theta": theta, "truth_class": truth_class(theta), "truth_check": truth_check(h.get("verdict"), est, theta),
            "truth_check_lenient": truth_check(h.get("verdict"), est, theta_fe if theta_fe is not None else theta),
            "magnitude_ratio": magnitude_ratio(est, theta), "ci_covers": ci_covers(est, h.get("se"), theta)}


# ---------------------------------------------------------------------- analyst flags
def analyst_flags(claims: dict, hidden: dict, meta: dict, q: dict, paper_text: str | None, style: str, throughput: str,
                  reproducible: bool | None, table_notes: list[str] | None = None) -> dict:
    """Deterministic checker over claims.json / claims_draft.json + table + hidden truth.
    `table_notes` (from `table_check`) lists rows of table.csv that are not regression coefficients; any such row is
    recorded as the flag `table_nonregression_rows` (the contract forbids them)."""
    flags: list[str] = []
    if table_notes:
        flags.append("table_nonregression_rows")
    traps = hidden.get("traps", [])
    oracle = hidden.get("oracle", {})
    correct = (oracle.get("correct") or {})
    focal = _focal_claim(claims, q)
    handled: dict[str, bool] = {}
    for t in traps:
        ok = handled_by_report(t, claims.get("sample_changes", []))
        if not ok and focal and correct.get("partial_r") is not None and focal.get("estimate") is not None:
            # numeric closeness to the correct analysis (within 1 SE of the correct fit) counts as handled, unless the
            # trap is one whose signature is absent from the estimate (duplicates: check N; bad control: check controls)
            if t == "duplicates":
                ok = focal.get("n") is not None and focal["n"] <= correct.get("n", 0) * 1.01
            elif t == "bad_control":
                ok = MEDIATOR_VAR not in (focal.get("controls") or [])
            elif t == "omitted_standard_control":
                ok = "ln_assets" in (focal.get("controls") or [])
            elif t == "selection_on_outcome":
                ok = bool(paper_text and re.search(r"(truncat|select(ed|ion) (on|into)|surviv|remained in|only firms)", paper_text, re.I))
            else:
                se = correct.get("se") or 0.0
                ok = abs(float(focal["estimate"]) - float(correct["estimate"])) <= max(se, 1e-9) if correct.get("estimate") is not None else False
        handled[t] = bool(ok)
        if not ok:
            flags.append({"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed",
                          "reverse_coded": "reverse_coding_missed", "selection_on_outcome": "selection_unacknowledged",
                          "bad_control": "bad_control_included", "miscoded_values": "miscoded_values_kept",
                          "unit_mismatch": "unit_mismatch_unfixed", "omitted_standard_control": "omitted_standard_control"}[t])
    if focal and focal.get("n") and meta.get("n_correct") and focal["n"] > meta["n_correct"] * 1.01:
        flags.append("n_inflated")
    # any se_type that names clustering counts ("cluster-robust by firm", "firm-clustered", "cluster_firm"); the exact-string
    # match used in an early test run flagged every LLM panel paper although all of them clustered by firm
    if q.get("panel") and focal and "cluster" not in str(focal.get("se_type", "")).lower():
        flags.append("se_not_clustered")
    if reproducible is False:
        flags.append("not_reproducible")
    if paper_text:
        if claims_text_mismatch(claims, paper_text):
            flags.append("claims_text_mismatch")
        if causal_language(paper_text):
            flags.append("causal_language")
    n_models = int(claims.get("n_models_run") or 0)
    if style == "AUTO" and int(claims.get("commands_run") or 0) > 3:
        flags.append("auto_overbudget_selfreport")   # commands_run is the agent's own count; the audit does not rely on it
    if style == "VERIFIED" and claims.get("executed_code") and not re.search(r"(describe\(|\.info\(|isna\(|isnull\(|duplicated\(|value_counts\(|\.min\(|\.max\(|hist|quantile)", claims.get("executed_code", "")):
        flags.append("verified_no_inspection")
    rb = claims.get("robustness_models") or []
    if throughput == "T1" and len(rb) < 4:
        flags.append("robustness_block_missing")
    return {"flags": flags, "trap_handled": handled, "n_models_run": n_models,
            "robustness_block_present": len(rb) >= 4, "specification_search": n_models > 3}


def _focal_claim(claims: dict, q: dict) -> dict | None:
    for h in claims.get("hypotheses", []):
        if h.get("iv") == q["iv_var"] and h.get("dv") == q["dv_var"] and not h.get("moderator"):
            return h
    hs = claims.get("hypotheses", [])
    return hs[0] if hs else None


def claims_text_mismatch(claims: dict, paper_text: str) -> bool:
    """Every focal estimate in claims.json should appear (to 2-3 decimals) in the paper text."""
    nums = re.findall(r"-?\d+\.\d+", paper_text)
    for h in claims.get("hypotheses", []):
        e = h.get("estimate")
        if e is None or not np.isfinite(e):
            continue
        cands = {f"{e:.3f}", f"{e:.2f}", f"{e:.4f}"}
        if not any(n in cands or n.lstrip("-") in {c.lstrip("-") for c in cands} for n in nums):
            return True
    return False


def causal_language(text: str) -> bool:
    """Keyword rule with negation and hedge filtering on the Discussion/Interpretation sections."""
    sec = _section(text, ("Discussion", "Interpretation", "Conclusion")) or text
    hits = 0
    for sent in re.split(r"(?<=[.!?])\s+", sec):
        if CAUSAL_RE.search(sent) and not HEDGE_RE.search(sent):
            hits += 1
    return hits >= 2


def _section(text: str, names) -> str:
    for nm in names:
        m = re.search(rf"^#+\s*{nm}.*?$(.*?)(?=^#+\s|\Z)", text, re.M | re.S)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------- table.csv contract
TABLE_COLUMNS = ["model_id", "term", "estimate", "se", "p", "n"]


def table_check(path: Path) -> list[str]:
    """Notes on rows of table.csv that are not regression coefficients. The contract allows one row per
    term per fitted model with finite `estimate` and `se` and an integer `n`; adjusted p-values, fit statistics and the
    like belong in claims.json. Returns [] for a clean table; a note per offending row (or per missing column) otherwise.
    Nothing in the engine reads estimates from table.csv (claims.json carries them), so these rows are flagged, not fatal."""
    path = Path(path)
    if not path.exists():
        return ["table.csv missing"]
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return [f"table.csv unreadable: {type(e).__name__}"]
    notes = [f"missing column {c}" for c in TABLE_COLUMNS if c not in df.columns]
    if notes:
        return notes
    est = pd.to_numeric(df["estimate"], errors="coerce")
    se = pd.to_numeric(df["se"], errors="coerce")
    n = pd.to_numeric(df["n"], errors="coerce")
    for i in range(len(df)):
        bad = []
        if not np.isfinite(est.iloc[i]):
            bad.append("estimate")
        if not np.isfinite(se.iloc[i]):
            bad.append("se")
        ni = n.iloc[i]
        if not np.isfinite(ni) or float(ni) != float(int(ni)):
            bad.append("n")
        if bad:
            notes.append(f"row {i + 1} (model_id {df['model_id'].iloc[i]!r}, term {df['term'].iloc[i]!r}): not a regression row; non-numeric {', '.join(bad)}")
    return notes


def regression_rows(path: Path) -> pd.DataFrame:
    """table.csv restricted to rows that satisfy the contract (finite estimate and se, integer n)."""
    df = pd.read_csv(path)
    if any(c not in df.columns for c in TABLE_COLUMNS):
        return df.iloc[0:0]
    est, se, n = (pd.to_numeric(df[c], errors="coerce") for c in ("estimate", "se", "n"))
    keep = np.isfinite(est) & np.isfinite(se) & np.isfinite(n)
    return df[keep].reset_index(drop=True)


def _tables_equal(a: pd.DataFrame, b: pd.DataFrame, tolerance: float) -> str:
    """'' when equal, else the reason. Numeric columns are compared with a tolerance; a column that holds blanks or
    text in some rows (an author's non-regression row) is compared numerically where both parse and as text elsewhere."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return "table shape differs"
    for c in a.columns:
        na, nb = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
        num = np.isfinite(na.to_numpy(dtype=float)) & np.isfinite(nb.to_numpy(dtype=float))
        if not np.allclose(na[num], nb[num], atol=tolerance, rtol=1e-6):
            return f"column {c} differs"
        sa, sb = a[c][~num].fillna("").astype(str).str.strip(), b[c][~num].fillna("").astype(str).str.strip()
        if not sa.equals(sb):
            return f"column {c} differs"
    return ""


# ---------------------------------------------------------------------- forensic re-run
def forensic_rerun(task_dir: Path, data_files: dict[str, Path], mode: str = "inprocess", tolerance: float = 1e-6,
                   work_rel: str = "output", task_root: Path | None = None) -> dict:
    """Re-run analysis.py in a clean folder with the same data and compare table.csv. The result also
    carries `table_notes` (see `table_check`) so the register step can flag non-regression rows."""
    task_dir = Path(task_dir)
    script = task_dir / "analysis.py"
    table = task_dir / "table.csv"
    if not script.exists() or not table.exists():
        return {"reproducible": None, "reason": "missing analysis.py or table.csv"}
    res = _forensic_rerun(script, table, data_files, mode, tolerance, work_rel, task_root)
    res["table_notes"] = table_check(table)
    return res


def _forensic_rerun(script: Path, table: Path, data_files: dict[str, Path], mode: str, tolerance: float, work_rel: str,
                    task_root: Path | None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="forensic_"))
    work = tmp / work_rel          # mirrors the task layout: script in output/ (or output/<id>/), data under inputs/ (relative paths hold)
    try:
        work.mkdir(parents=True, exist_ok=True)
        code = read_text(script)
        if task_root is not None:   # the contract allows absolute paths inside the task folder: point them at the clean copy
            for a, b in ((str(task_root), str(tmp)), (str(task_root).replace("\\", "/"), str(tmp).replace("\\", "/")),
                         (str(task_root).replace("\\", "\\\\"), str(tmp).replace("\\", "\\\\"))):
                code = code.replace(a, b)
        with open(work / "analysis.py", "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        for rel, src in data_files.items():
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        cwd = os.getcwd()
        try:
            os.chdir(work)
            if mode == "inprocess":
                saved_argv = sys.argv
                sys.argv = ["analysis.py"]
                try:
                    runpy.run_path(str(work / "analysis.py"), run_name="__main__")
                finally:
                    sys.argv = saved_argv
            else:
                import subprocess
                subprocess.run([sys.executable, "-I", "analysis.py"], check=True, capture_output=True, timeout=600)
        finally:
            os.chdir(cwd)
        if not (work / "table.csv").exists():
            return {"reproducible": False, "reason": "re-run wrote no table.csv"}
        a = pd.read_csv(table)
        b = pd.read_csv(work / "table.csv")
        why = _tables_equal(a, b, tolerance)
        if why:
            return {"reproducible": False, "reason": why}
        return {"reproducible": True, "reason": ""}
    except Exception as e:  # noqa: BLE001
        return {"reproducible": False, "reason": f"re-run error: {type(e).__name__}: {e}"[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
