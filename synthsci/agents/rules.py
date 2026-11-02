"""Rule agents: a calibrated stand-in for every LLM call type.

Used by (a) the mock backend, where the rule author really writes and executes analysis.py on data.csv, and (b) fast
mode, where studies come from the analytic path. Parameters live in configs/fastmode/params.yaml (design-default placeholders
until estimated from the live runs). Rule agents never see hidden truth except where the engine passes the planted trap list
to the rule *author* to decide, with the parameter file's probability, whether its code handles each trap (that is the
behavioural parameter being mocked); reviewers, editors, lab heads and verifiers see only what an LLM would see.
"""
from __future__ import annotations

import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CONFIG_DIR, load_yaml
from ..stats import partial_r
from ..traps import MEDIATOR_VAR, REGISTRY
from ..util import read_text, write_text

HANDLE_CODE = {"duplicates": "deduplicated", "scale_mixture": "rescaled_source_b", "reverse_coded": "item_reversed",
               "selection_on_outcome": "selection_acknowledged", "bad_control": "excluded_post_treatment_control",
               "miscoded_values": "dropped_impossible_values", "unit_mismatch": "unit_corrected",
               "omitted_standard_control": "size_included"}
TRAP_KIND = {"duplicates": "data", "scale_mixture": "data", "reverse_coded": "data", "selection_on_outcome": "inference",
             "bad_control": "specification", "miscoded_values": "data", "unit_mismatch": "data",
             "omitted_standard_control": "specification", "se_not_clustered": "inference", "n_inflated": "reporting",
             "causal_language": "interpretation", "claims_text_mismatch": "reporting"}
TRAP_TEXT = {"duplicates": "The panel structure should be verified; some firm-years appear to be repeated.",
             "scale_mixture": "The outcome variable looks bimodal across data sources; the scale should be checked.",
             "reverse_coded": "One scale item appears to be reverse-scored and should be reversed before scoring.",
             "selection_on_outcome": "The sample frame conditions on the outcome; the limitation should be discussed and bounded.",
             "bad_control": "A control measured at the end of the period may lie on the path from the focal variable to the outcome.",
             "miscoded_values": "Some values are outside the plausible range of the variable and should be dropped or corrected.",
             "unit_mismatch": "The outcome shows a discontinuity in one year that suggests a change of units.",
             "omitted_standard_control": "Firm size should be included as a control.",
             "se_not_clustered": "Standard errors should be clustered by firm.",
             "n_inflated": "The reported N does not match the sample described.",
             "causal_language": "The discussion uses causal language the design cannot support.",
             "claims_text_mismatch": "Numbers in the text do not match the table."}


def load_params(cfg: dict, prehistory: bool = False) -> dict:
    """The rule-agent parameter file; with prehistory=True the warm-start file `fastmode.prehistory_params`,
    which defaults to `fastmode.params`."""
    fm = cfg.get("fastmode", {})
    p = Path((fm.get("prehistory_params") if prehistory else None) or fm.get("params", "fastmode/params.yaml"))
    return load_yaml(p if p.exists() else CONFIG_DIR / p)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _verdict(est, se, predicted: str) -> tuple[str, float]:
    if est is None or se is None or not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return "not_supported", float("nan")
    from scipy.stats import norm
    t = est / se
    p = float(2 * norm.sf(abs(t)))
    sign_ok = (est > 0) == (predicted == "positive")
    if p < 0.05 and sign_ok:
        return "supported", p
    if p < 0.10 and sign_ok:
        return "partially", p
    return "not_supported", p


# ============================================================================ author / analyst
class RuleAuthor:
    def __init__(self, params: dict):
        self.p = params["author"]

    # -------------------------------------------------- choice of research question
    def choose(self, menu: list[dict], author: dict, r: np.random.Generator) -> dict:
        """menu entries: {question_id, slot, family, count, ...}. Weighted by slot type and 1/(1+count)."""
        weights = []
        for e in menu:
            if e["slot"] == "fringe":
                w = self.p["p_choose_fringe"]
            elif e["slot"] == "crowded":
                w = self.p["p_choose_crowded"]
            else:
                w = (1.0 - self.p["p_choose_fringe"] - self.p["p_choose_crowded"]) / max(1, sum(1 for x in menu if x["slot"] in ("affinity", "random")))
                w *= (1.0 / (1.0 + e.get("count", 0))) ** self.p["affinity_weight_power"] * 3.0
                if e.get("family") == author["primary_theory"]:
                    w *= 1.5
            weights.append(w)
        w = np.asarray(weights); w = w / w.sum()
        return menu[int(r.choice(len(menu), p=w))]

    def n_hypotheses(self, r) -> int:
        d = self.p["n_hypotheses"]
        ks, ws = list(d), np.asarray(list(d.values()), dtype=float)
        return int(ks[int(r.choice(len(ks), p=ws / ws.sum()))])

    def decide_handling(self, traps: list[str], style: str, r) -> dict[str, bool]:
        return {t: bool(r.uniform() < self.p["p_handle"][style].get(t, 0.0)) for t in traps}

    def use_fe(self, q: dict, author: dict, r) -> bool:
        if not q["panel"]:
            return False
        return bool(r.uniform() < self.p["p_use_fe_panel"].get(author.get("repertoire", "archival_regression"), 0.8))

    def plan(self, q: dict, n_hyp: int, variables: dict) -> list[dict]:
        """Hypothesis plan among the focal variables: focal, moderations, subsample, secondary."""
        pred = q["prediction_sign"]
        plan = [{"iv": q["iv_var"], "dv": q["dv_var"], "moderator": None, "kind": "focal", "predicted_sign": pred}]
        extras = []
        for m in variables.get("moderators", []):
            extras.append({"iv": q["iv_var"], "dv": q["dv_var"], "moderator": m, "kind": "moderation", "predicted_sign": "positive"})
        extras.append({"iv": q["iv_var"], "dv": q["dv_var"], "moderator": "high_tech", "kind": "subsample", "predicted_sign": "positive"})
        if variables.get("secondary_dv"):
            extras.append({"iv": q["iv_var"], "dv": variables["secondary_dv"], "moderator": None, "kind": "secondary", "predicted_sign": pred})
        for c in variables.get("controls", [])[:3]:
            extras.append({"iv": c, "dv": q["dv_var"], "moderator": None, "kind": "secondary", "predicted_sign": "positive"})
        return plan + extras[:max(0, n_hyp - 1)]

    def relation_to_prior(self, focal_est: float, view: list[dict], r, first_word: str = "first study") -> str:
        if not view:
            return first_word
        prior_signs = [np.sign(v.get("focal_estimate", 0.0)) for v in view if v.get("focal_verdict") in ("supported", "partially")]
        if not prior_signs:
            return "extension"
        maj = np.sign(sum(prior_signs))
        if maj != 0 and np.sign(focal_est) != maj:
            return "contradiction" if r.uniform() < self.p["p_mark_contradiction_when_disagree"] else "extension"
        return "replication" if r.uniform() < 0.6 else "extension"

    # -------------------------------------------------- full path: write and execute analysis.py
    def analyse_full(self, q: dict, data_rel_paths: list[str], task_dir: Path, author: dict, handled: dict[str, bool],
                     variables: dict, use_fe: bool, n_hyp: int, r, throughput: str, traps_present: list[str],
                     out_subdir: str = "output") -> dict:
        """Emit analysis.py implementing the chosen handling, run it in task_dir/out_subdir, read table.csv, build hypotheses."""
        plan = self.plan(q, n_hyp, variables)
        controls = [c for c in variables.get("controls", [])]
        if "omitted_standard_control" in traps_present and not handled.get("omitted_standard_control", True):
            controls = [c for c in controls if c != "ln_assets"]
        if "bad_control" in traps_present and not handled.get("bad_control", True):
            controls = controls + [MEDIATOR_VAR]
        reverse_handled = handled.get("reverse_coded", True) if "reverse_coded" in traps_present else None
        code = emit_analysis_code(q, data_rel_paths, controls, plan, handled, traps_present, use_fe, throughput == "T1",
                                  reverse_handled, inspect=author.get("style") == "VERIFIED")
        out = Path(task_dir) / out_subdir
        out.mkdir(parents=True, exist_ok=True)
        write_text(out / "analysis.py", code)
        n_runs = 1
        _run_script(out)
        table = pd.read_csv(out / "table.csv")
        hyps = hypotheses_from_table(table, plan, q, controls, use_fe)
        sample_changes = [HANDLE_CODE[t] for t in traps_present if handled.get(t, False) and t != "selection_on_outcome"]
        if handled.get("selection_on_outcome", False):
            sample_changes.append("selection_acknowledged")
        n_models = int(table["model_id"].nunique())
        style = author["style"]
        n_models_run = int(r.integers(self.p["n_models_run"][style][0], self.p["n_models_run"][style][1] + 1)) if style == "VERIFIED" else 1
        rb = [m for m in table["model_id"].unique() if str(m).startswith("R")]
        return {"hypotheses": hyps, "controls": controls, "sample_changes": sample_changes,
                "n_models_run": max(n_models_run, n_models), "robustness_models": list(map(str, rb)),
                "executed_code": code, "commands_run": n_runs, "use_fe": use_fe, "handled": handled, "table": table,
                "n_reported": int(table[table.model_id == "M1"]["n"].iloc[0]) if (table.model_id == "M1").any() else None}

    # -------------------------------------------------- analytic path
    def analyse_analytic(self, gen, q: dict, dataset_seed: int, author: dict, handled: dict, use_fe: bool, n_hyp: int,
                         variables: dict, upper_half: bool, n_override: int | None = None, predict_sign: bool = True) -> dict:
        """predict_sign=False: the analyst states the direction it found instead of a prior prediction, so an
        entry is supported when its estimate is significant in either direction (the live repository analysts' practice);
        the default keeps the predicted-sign verdict."""
        s = gen.analytic_study(q["id"], dataset_seed, upper_half, handled, use_fe, n_override=n_override)
        plan = self.plan(q, n_hyp, variables)
        hyps = []
        df_ = s["df"]
        for i, h in enumerate(plan, start=1):
            if h["kind"] == "focal":
                est, se = s["focal"]["estimate"], s["focal"]["se"]
            elif h["kind"] == "moderation":
                m = s["moderators"].get(h["moderator"], {"estimate": 0.0, "se": s["focal"]["se"]})
                est, se = m["estimate"], m["se"]
            elif h["kind"] == "subsample":
                est, se = s["subsample"]["estimate"], s["subsample"]["se"]
            elif h["dv"] == variables.get("secondary_dv"):
                est, se = s["secondary"]["estimate"], s["secondary"]["se"]
            else:  # control -> dv relation: drawn around an assumed null-ish prior
                est, se = float(np.random.Generator(np.random.PCG64(dataset_seed + i)).normal(0, s["focal"]["se"])), s["focal"]["se"]
            if predict_sign:
                v, p = _verdict(est, se, h["predicted_sign"])
            else:
                v, p = _verdict(est, se, "positive" if est > 0 else "negative")
                h = h | {"predicted_sign": None}
            hyps.append({"hyp_id": f"H{i}", **{k: h[k] for k in ("iv", "dv", "moderator", "kind", "predicted_sign")},
                         "model_id": f"M{i}", "estimate": float(est), "se": float(se), "p": p, "n": int(s["n_reported"]),
                         "estimator": "fixed effects" if use_fe else "OLS", "se_type": "cluster" if use_fe else "HC1",
                         "controls": list(variables.get("controls", [])), "verdict": v,
                         "partial_r": partial_r(est / se, df_) if se > 0 else float("nan")})
        return {"hypotheses": hyps, "study": s, "handled": handled, "use_fe": use_fe,
                "sample_changes": [HANDLE_CODE[t] for t in s["traps"] if handled.get(t, False)],
                "n_models_run": len(hyps), "robustness_models": ["R1", "R2", "R3", "R4"] if author.get("throughput") == "T1" else [],
                "n_reported": int(s["n_reported"]), "n": int(s["n"])}

    # -------------------------------------------------- texts
    def claims_json(self, q: dict, res: dict, relation: str, builds_on: list[dict], nonce: str) -> dict:
        hyps = [{k: v for k, v in h.items() if k not in ("partial_r", "kind")} | {"kind": h.get("kind")} for h in res["hypotheses"]]
        return {"nonce": nonce, "question_id": q["id"], "relation_to_prior": relation, "hypotheses": hyps,
                "builds_on": builds_on, "sample_changes": res["sample_changes"], "n_models_run": int(res["n_models_run"]),
                "robustness_models": res["robustness_models"], "executed_code": res.get("executed_code", ""),
                "commands_run": int(res.get("commands_run", 1))}

    def paper_md(self, q: dict, res: dict, relation: str, author: dict, view: list[dict], handled: dict) -> str:
        hyps = res["hypotheses"]
        h1 = hyps[0]
        fam = q["family"].replace("_", " ")
        cites = ", ".join(v["paper_id"] for v in view[:3]) or "none"
        lines = [f"# {q['iv_label']} and {q['dv_label']}: {'a replication' if relation == 'replication' else 'evidence'} from {h1['n']:,} {q['unit']} observations", "",
                 "## Abstract", "",
                 f"We examine the relation between {q['iv_label'].lower()} and {q['dv_label'].lower()} from the standpoint of {fam}. "
                 f"Using {'a firm-year panel' if q['panel'] else 'a cross-section'} of {h1['n']:,} observations we estimate "
                 f"{'fixed-effects' if res['use_fe'] else 'OLS'} models with standard controls. The focal estimate is {h1['estimate']:.3f} "
                 f"(SE {h1['se']:.3f}, p = {h1['p']:.3f}); hypothesis 1 is {h1['verdict'].replace('_', ' ')}. "
                 f"{sum(1 for h in hyps if h['verdict'] == 'supported')} of {len(hyps)} hypotheses are supported. "
                 f"This study is {'the first on the question' if relation == 'first study' else 'a ' + relation + ' relative to prior work'}.", "",
                 "## Theory and Hypotheses", "",
                 f"{q['prediction']} Prior work on this question ({cites}) {'is absent' if not view else 'informs our expectations'}. "
                 f"We state {len(hyps)} hypotheses about the focal variables."]
        for h in hyps:
            rel = f"{h['iv']} moderated by {h['moderator']} on {h['dv']}" if h["moderator"] else f"{h['iv']} on {h['dv']}"
            lines.append(f"- {h['hyp_id']}: {rel} is {h['predicted_sign']}.")
        lines += ["", "## Data and Sample", "",
                  f"The dataset is described in the codebook: {q['unit']} observations assembled from two vendors. "
                  f"Sample decisions made in code: {', '.join(res['sample_changes']) or 'none beyond listwise deletion'}.", "",
                  "## Method", "",
                  f"{'Firm and year fixed effects with firm-clustered standard errors' if res['use_fe'] else 'OLS with industry indicators and robust standard errors'}; "
                  f"controls: {', '.join(res['controls']) if res.get('controls') else 'standard controls'}.", "",
                  "## Results", "", "| Hypothesis | Model | Estimate | SE | p | N | Verdict |", "|---|---|---|---|---|---|---|"]
        for h in hyps:
            lines.append(f"| {h['hyp_id']} | {h['model_id']} | {h['estimate']:.3f} | {h['se']:.3f} | {h['p']:.3f} | {h['n']} | {h['verdict']} |")
        if res.get("robustness_models"):
            lines += ["", "## Robustness", "", f"Additional specifications ({', '.join(res['robustness_models'])}) are reported in table.csv; "
                      f"the focal sign is {'stable' if res.get('robust_stable', True) else 'not stable'} across them."]
        lines += ["", "## Discussion", "",
                  f"The results are {'consistent' if h1['verdict'] != 'not_supported' else 'not consistent'} with the {fam} account. "
                  f"The association between {q['iv_label'].lower()} and {q['dv_label'].lower()} is "
                  f"{'positive' if h1['estimate'] > 0 else 'negative'} and {'statistically reliable' if h1['p'] < 0.05 else 'not statistically reliable'} in this sample, "
                  f"which suggests that the relation may depend on context.", "",
                  "## Limitations", ""]
        lim = ["Measures are proxies; unobserved heterogeneity may remain despite the controls."]
        if handled.get("selection_on_outcome", False):
            lim.append("The sample frame conditions on firms that remained in coverage, which truncates the outcome; estimates may be attenuated by this selection and should be read as bounds.")
        lines += lim + ["", "## References", ""] + [f"- {v['paper_id']}" for v in view[:3]] + [""]
        return "\n".join(lines)


def hypotheses_from_table(table: pd.DataFrame, plan: list[dict], q: dict, controls: list[str], use_fe: bool) -> list[dict]:
    hyps = []
    for i, h in enumerate(plan, start=1):
        mid = f"M{i}"
        term = f"{h['iv']}:{h['moderator']}" if h["moderator"] else h["iv"]
        row = table[(table.model_id == mid) & (table.term == term)]
        if len(row) == 0:
            est, se, p, n = None, None, None, None
        else:
            rr = row.iloc[0]
            est, se, p, n = float(rr["estimate"]), float(rr["se"]), float(rr["p"]), int(rr["n"])
        v, _ = _verdict(est, se, h["predicted_sign"])
        k = len(controls) + (2 if h["moderator"] else 0) + 2
        pr = partial_r(est / se, max((n or 0) - k, 1)) if (est is not None and se) else float("nan")
        hyps.append({"hyp_id": f"H{i}", "iv": h["iv"], "dv": h["dv"], "moderator": h["moderator"], "kind": h["kind"],
                     "predicted_sign": h["predicted_sign"], "model_id": mid, "estimate": est, "se": se, "p": p, "n": n,
                     "estimator": "fixed effects" if use_fe else "OLS", "se_type": "cluster" if use_fe else "HC1",
                     "controls": list(controls), "verdict": v, "partial_r": pr})
    return hyps


def emit_analysis_code(q: dict, data_rel_paths: list[str], controls: list[str], plan: list[dict], handled: dict,
                       traps: list[str], use_fe: bool, t1: bool, reverse_handled: bool | None, inspect: bool = False) -> str:
    """Self-contained pandas/statsmodels script implementing the chosen handling and models."""
    iv, dv = q["iv_var"], q["dv_var"]
    L = ["import os", "import pandas as pd, numpy as np", "import statsmodels.api as sm", "",
         # abspath removes the '..' segments: a long task-folder path plus '..\\..' can exceed Windows' 260-character limit
         "frames = [" + ", ".join(f'pd.read_csv(os.path.abspath(r"{p}"))' for p in data_rel_paths) + "]",
         "df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]", "rows = []", ""]
    if inspect:  # the careful protocol: inspect before modelling (dimensions, ranges, missingness, duplicates)
        L += ["_desc = df.describe(include='all').T", "_missing = df.isna().sum()", "_dups = int(df.duplicated().sum())",
              "_ranges = {c: (float(df[c].min()), float(df[c].max())) for c in df.columns if df[c].dtype.kind in 'if'}", ""]
    if reverse_handled is True:
        L.append(f'df["{iv}"] = (df["{iv}_item1"] + df["{iv}_item2"] + (8 - df["{iv}_item3"])) / 3.0')
    elif reverse_handled is False:
        L.append(f'df["{iv}"] = (df["{iv}_item1"] + df["{iv}_item2"] + df["{iv}_item3"]) / 3.0')
    if handled.get("duplicates"):
        L.append("df = df.drop_duplicates().copy()")
    if handled.get("miscoded_values"):
        L.append(f'mask = (df["{iv}"] >= 900)')
        L.append('if "leverage" in df.columns: mask |= (df["leverage"] >= 900)')
        L.append('if "firm_age" in df.columns: mask |= (df["firm_age"] < 0)')
        L.append("df = df[~mask].copy()")
    if handled.get("scale_mixture"):
        L.append(f'm = df["data_source"] == "B"')
        L.append(f'df.loc[m, "{dv}"] = df.loc[m, "{dv}"] / 100.0')
    if handled.get("unit_mismatch"):
        L.append(f'med = df.groupby("year")["{dv}"].median(); overall = med.median()')
        L.append(f'for yr, mv in med.items():')
        L.append(f'    if abs(mv) > 50 * (abs(overall) + 0.05): df.loc[df["year"] == yr, "{dv}"] = df.loc[df["year"] == yr, "{dv}"] / 1000.0')
    L += ["", "def fit(data, y, terms, model_id):",
          "    terms = [t for t in dict.fromkeys(terms) if t != y]",
          "    cols = [y] + terms + ['firm_id'] + (['year'] if 'year' in data.columns else []) + (['industry'] if 'industry' in data.columns else [])",
          "    d = data[list(dict.fromkeys(cols))].dropna().copy()",
          "    Y = d[y].astype(float); X = d[terms].astype(float)"]
    if use_fe:
        L += ["    g = d['firm_id']",
              "    Y = Y - Y.groupby(g).transform('mean'); X = X - X.groupby(g).transform('mean')",
              "    X = pd.concat([X, pd.get_dummies(d['year'].astype(int), prefix='yr', drop_first=True).astype(float)], axis=1)",
              "    res = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': g.to_numpy()})"]
    else:
        L += ["    X = pd.concat([X, pd.get_dummies(d['industry'].astype(int), prefix='ind', drop_first=True).astype(float)], axis=1)",
              "    X = sm.add_constant(X)",
              "    res = sm.OLS(Y, X).fit(cov_type='HC1')"]
    L += ["    for t in terms:",
          "        rows.append({'model_id': model_id, 'term': t, 'estimate': float(res.params[t]), 'se': float(res.bse[t]), 'p': float(res.pvalues[t]), 'n': int(res.nobs)})",
          "    return res", ""]
    ctrl = "[" + ", ".join(f'"{c}"' for c in controls) + "]"
    L.append(f"controls = {ctrl}")
    for i, h in enumerate(plan, start=1):
        mid = f"M{i}"
        if h["moderator"]:
            m = h["moderator"]
            L.append(f'df["{h["iv"]}:{m}"] = df["{h["iv"]}"] * df["{m}"]')
            terms = f'["{h["iv"]}", "{m}", "{h["iv"]}:{m}"] + [c for c in controls if c not in ("{h["iv"]}", "{m}")]'
        else:
            terms = f'["{h["iv"]}"] + [c for c in controls if c not in ("{h["iv"]}", "{h["dv"]}")]'
        L.append(f'fit(df, "{h["dv"]}", {terms}, "{mid}")')
    if t1:
        L += [f'fit(df, "{dv}", ["{iv}"] + [c for c in controls if c != "{iv}"][:2], "R1")  # alternative, sparser specification',
              f'fit(df, "{dv}", ["{iv}"] + [c for c in controls if c != "{iv}"], "R2") if False else None']
        if q.get("secondary_dv_var"):
            L.append(f'fit(df, "{q["secondary_dv_var"]}", ["{iv}"] + [c for c in controls if c != "{iv}"], "R2")  # alternative outcome measure')
        else:
            L.append(f'fit(df[df["data_source"] == "A"], "{dv}", ["{iv}"] + [c for c in controls if c != "{iv}"], "R2")  # alternative sample')
        L.append(f'fit(df[df["high_tech"] == 1], "{dv}", ["{iv}"] + [c for c in controls if c != "{iv}"], "R3")  # subsample split')
        extra = "ln_experience" if iv != "ln_experience" else "firm_age"
        L.append(f'fit(df, "{dv}", ["{iv}", "{extra}"] + [c for c in controls if c not in ("{iv}", "{extra}")], "R4")  # alternative specification')
    L += ["", "pd.DataFrame(rows).to_csv('table.csv', index=False)", ""]
    return "\n".join(L)


def _run_script(out_dir: Path) -> None:
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        saved = sys.argv
        sys.argv = ["analysis.py"]
        try:
            runpy.run_path(str(out_dir / "analysis.py"), run_name="__main__")
        finally:
            sys.argv = saved
    finally:
        os.chdir(cwd)


# ============================================================================ reviewer
class RuleReviewer:
    def __init__(self, params: dict):
        self.p = params["reviewer"]

    def review(self, item: dict, r: np.random.Generator, assistance: str = "none", batch_size: int = 1) -> dict:
        """item: {paper_id, hypotheses, visible_problems: [flag codes], relation_to_prior, prior_view, prescreen_flags}.
        visible_problems are the deterministic analyst flags; the rule notices each with p_notice (not the truth)."""
        p = self.p
        mult = p["p_notice_assist_multiplier"].get(assistance, 1.0) * max(0.5, 1.0 - p["load_penalty_per_paper"] * (batch_size - 1))
        noticed = []
        for f in item.get("visible_problems", []):
            key = _flag_to_trap(f)
            if key and r.uniform() < min(0.98, p["p_notice"].get(key, 0.3) * mult):
                noticed.append(key)
        h1 = item["hypotheses"][0] if item.get("hypotheses") else {}
        supported = h1.get("verdict") in ("supported", "partially")
        major_noticed = [k for k in noticed if k not in ("causal_language",)]
        if major_noticed:
            mean = p["validity_mean"]["flawed_noticed"]
        elif item.get("visible_problems"):
            mean = p["validity_mean"]["flawed_unnoticed"]
        else:
            mean = p["validity_mean"]["clean_supported"] if supported else p["validity_mean"]["clean_null"]
        if item.get("relation_to_prior") == "contradiction":
            mean += p["validity_mean"]["contradiction_penalty"]
        elif item.get("relation_to_prior") == "replication":
            mean += p["validity_mean"]["replication_bonus"]
        # a shift shared by every referee of the paper (the live referees agreed: ICC of validity .77); 0 = the
        # old independent draws, and then validity_sd is the whole spread
        paper_sd = float(p.get("validity_paper_sd", 0.0) or 0.0)
        if paper_sd > 0:
            mean += paper_sd * _paper_shock(item)
        validity = float(np.clip(r.normal(mean, p["validity_sd"]), 1, 99))
        cuts = p["recommendation_cuts"]
        if validity < cuts["reject"]:
            rec = "reject"
        elif validity < cuts["major"] or major_noticed:
            rec = "major_revision"
        elif validity < cuts["minor"]:
            rec = "minor_revision"
        else:
            rec = "accept" if r.uniform() < p["p_accept_above_minor_cut"] else "minor_revision"
        issues = []
        for k in noticed:
            sev = "major" if (k in major_noticed and r.uniform() < 0.7) else "minor"
            issues.append({"location": "Method" if TRAP_KIND[k] in ("data", "specification", "inference") else "Discussion",
                           "description": TRAP_TEXT[k], "severity": sev, "kind": TRAP_KIND[k]})
        # a methods objection raised whatever the paper's flags (the live referees raised data, specification or
        # inference objections on every paper, flagged or not); 0 = the old behaviour (no draw is made)
        pu = float(p.get("p_unprompted_issue", 0.0) or 0.0)
        if pu > 0 and r.uniform() < pu:
            kind = UNPROMPTED_KINDS[int(r.integers(len(UNPROMPTED_KINDS)))]
            sev = "major" if r.uniform() < float(p.get("p_unprompted_major", 1.0)) else "minor"
            issues.append({"location": "Method", "description": UNPROMPTED_TEXT[kind], "severity": sev, "kind": kind})
        if not issues and r.uniform() < 0.6:
            issues.append({"location": "Theory", "description": "The theoretical motivation for the secondary hypotheses could be sharper.",
                           "severity": "minor", "kind": "framing"})
        flaw_found = bool(any(i["severity"] == "major" for i in issues) and r.uniform() < p["p_flaw_found_given_noticed_major"])
        rel = item.get("relation_to_prior", "unclear")
        if r.uniform() < 0.15:
            rel = "unclear"
        return {"paper_id": item["paper_id"], "recommendation": rec, "flaw_found": flaw_found, "issues": issues,
                "validity_probability": round(validity, 1), "relation_to_prior": rel,
                "confidence": float(np.clip(r.normal(p["confidence"]["mean"], p["confidence"]["sd"]), 0.05, 0.99)),
                "summary": f"{'A competent' if validity >= 55 else 'A problematic'} study of the question; "
                           f"{len(issues)} issue(s) noted; recommendation {rec.replace('_', ' ')}."}


UNPROMPTED_KINDS = ("data", "specification", "inference")
UNPROMPTED_TEXT = {"data": "The construction of the sample and the measures needs more detail and checking before the estimates can be read.",
                   "specification": "The specification may omit relevant controls; alternative specifications should be reported.",
                   "inference": "The standard errors and the strength of the evidence for the focal estimate need more support."}


def _paper_shock(item: dict) -> float:
    """A standard normal draw fixed by the paper: keyed on the paper id and its focal row (variables, N, SE),
    so every referee of the paper shares it, it persists into round 2 unless the revision changed the sample, and papers
    with the same id in different runs or seeds get different draws."""
    import hashlib
    h1 = (item.get("hypotheses") or [{}])[0]
    se = h1.get("se")
    key = f"{item.get('paper_id')}|{h1.get('iv')}|{h1.get('dv')}|{h1.get('n')}|{float(se):.12g}" if se is not None else \
        f"{item.get('paper_id')}|{h1.get('iv')}|{h1.get('dv')}|{h1.get('n')}|{h1.get('estimate')}"
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return float(np.random.default_rng(seed).standard_normal())


def _flag_to_trap(flag: str) -> str | None:
    m = {"duplicates_not_removed": "duplicates", "scale_mixture_unfixed": "scale_mixture", "reverse_coding_missed": "reverse_coded",
         "selection_unacknowledged": "selection_on_outcome", "bad_control_included": "bad_control",
         "miscoded_values_kept": "miscoded_values", "unit_mismatch_unfixed": "unit_mismatch",
         "omitted_standard_control": "omitted_standard_control", "se_not_clustered": "se_not_clustered",
         "n_inflated": "n_inflated", "causal_language": "causal_language", "claims_text_mismatch": "claims_text_mismatch"}
    return m.get(flag)


# ============================================================================ editor
class RuleEditor:
    def __init__(self, params: dict):
        self.p = params["editor"]

    def desk(self, item: dict, strictness: str, r) -> dict:
        p = self.p["desk_reject_p"].get(strictness, 0.05)
        h1 = item["hypotheses"][0] if item.get("hypotheses") else {}
        if h1.get("verdict") == "not_supported":
            p += self.p["desk_reject_null_extra"]
        rej = r.uniform() < p
        return {"paper_id": item["paper_id"], "decision": "desk_reject" if rej else "send_out",
                "reason": "Insufficient contribution for the Journal." if rej else "Within scope; sent to referees."}

    def decide(self, item: dict, reviews: list[dict], round_no: int, count_on_question: int, r) -> dict:
        p = self.p
        recs = [rv["recommendation"] for rv in reviews]
        mv = float(np.mean([rv["validity_probability"] for rv in reviews])) if reviews else 50.0
        x = (p["intercept_round1"] if round_no == 1 else p["intercept_round2"]) + p["w_mean_validity"] * mv \
            + p["w_reject_votes"] * recs.count("reject") + p["w_major_votes"] * recs.count("major_revision") \
            + p["w_accept_votes"] * recs.count("accept") + p["w_log_count_on_question"] * math.log1p(max(count_on_question, 0)) \
            + (p["w_contradiction"] if item.get("relation_to_prior") == "contradiction" else 0.0)
        x += r.normal(0, 0.5)
        # an accept threshold for round 1 apart from the revise band (the live editors never accepted in round 1);
        # absent or null = the old threshold revise_band / 2
        cut1 = p.get("accept_cut_round1")
        if round_no == 1:
            if x > (p["revise_band"] / 2 if cut1 is None else float(cut1)):
                dec = "accept"
            elif x > -p["revise_band"]:
                dec = "revise"
            else:
                dec = "reject"
        else:
            dec = "accept" if x > -p["revise_band"] / 2 else "reject"
        merit = float(np.clip(mv + 8 * x / p["revise_band"] + r.normal(0, p["merit_sd"]), 0, 100))
        changes = [i["description"] for rv in reviews for i in rv["issues"] if i["severity"] == "major"][:4] if dec == "revise" else []
        return {"paper_id": item["paper_id"], "decision": dec, "merit": round(merit, 1), "required_changes": changes,
                "letter": f"Thank you for your submission. On the basis of the referee reports the decision is {dec}."}


# ============================================================================ lab head
class RuleLabHead:
    def __init__(self, params: dict):
        self.p = params["lab_head"]

    def decide(self, ctx: dict, r: np.random.Generator) -> dict:
        """ctx: lab, members_available [(member_id, style, throughput)], verification_capacity, menu [entries with
        catalogue], drafts [{draft_id, entries:[{entry_id, p, estimate, kind}]}], open_claims [{claim_id, question_id, p, n,
        endorsements, proposer_lab, proposer_size, same_theory, reciprocity, significant}], challenges [{claim_id, kind}],
        variant, remaining_budget."""
        p = self.p
        author_rule = RuleAuthor({"author": ctx["author_params"]})
        agenda = []
        used_q = set()
        for member in ctx["members_available"]:
            entry = author_rule.choose(ctx["menu"], {"primary_theory": ctx["lab"]["primary_theory"]}, r)
            if entry["question_id"] in used_q and len(ctx["menu"]) > 1:
                entry = author_rule.choose([e for e in ctx["menu"] if e["question_id"] not in used_q] or ctx["menu"],
                                           {"primary_theory": ctx["lab"]["primary_theory"]}, r)
            used_q.add(entry["question_id"])
            catalogue = [d for d in entry.get("catalogue", []) if d.get("available", True)]
            data = "new"
            if catalogue and r.uniform() < p["p_reuse_given_catalogue"]:
                picks = sorted(catalogue, key=lambda d: -d["n"])[:2]
                if len(picks) == 2 and ctx.get("pooling", True) and r.uniform() < p["p_pool_given_two"]:
                    data = [picks[0]["deposit_id"], picks[1]["deposit_id"]]
                else:
                    data = picks[0]["deposit_id"]
            item = {"question_id": entry["question_id"], "data": data, "lead": member[0],
                    "brief": f"Test the relation described for {entry['question_id']} among the focal variables; "
                             f"use {'firm and year effects' if entry.get('panel') else 'industry indicators'} and the standard controls; "
                             f"check the sample before modelling."}
            # the head names the size of a fresh dataset (the live heads did for most fresh draws), as a multiple of
            # the question's median frame size; p_request_n absent or 0 = the old behaviour (no draw is made)
            pn = float(p.get("p_request_n", 0.0) or 0.0)
            if data == "new" and pn > 0 and r.uniform() < pn:
                p50 = ((entry.get("question") or {}).get("frame_quantiles") or {}).get("p50")
                mults = p.get("request_n_multiple") or {1.0: 1.0}
                ks, ws = [float(k) for k in mults], np.asarray(list(mults.values()), dtype=float)
                m = ks[int(r.choice(len(ks), p=ws / ws.sum()))]
                if p50:
                    item["n"] = int(max(60, round(m * float(p50))))
            agenda.append(item)
        filings = []
        for d in ctx.get("drafts", []):
            for e in d["entries"]:
                sig = (e.get("p") is not None) and e["p"] < 0.05
                pf = p["p_file_draft"]["significant" if sig else "insignificant"]
                if ctx.get("autonomous_filing"):
                    pf = 1.0
                action = "file" if r.uniform() < pf else "drop"
                conf = float(np.clip(r.normal(p["stated_confidence"]["significant" if sig else "insignificant"], p["stated_confidence"]["sd"]), 5, 95))
                filings.append({"draft_id": d["draft_id"], "entry": e["entry_id"], "action": action, "probability": round(conf, 1),
                                "note": "" if action == "file" else "not stable enough to file"})
        votes = []
        n_stakes = 0
        for c in ctx.get("open_claims", []):
            v = p["vote"]
            x = v["intercept"] + v["b_significant"] * (1 if c.get("significant") else 0) + v["b_log_n"] * math.log1p(c.get("n") or 0) / 5 \
                + v["b_prior_endorsements"] * c.get("endorsements", 0) + v["b_same_theory"] * (1 if c.get("same_theory") else 0) \
                + v["b_reciprocity"] * (1 if c.get("reciprocity") else 0) + v["b_proposer_size"] * c.get("proposer_size", 0)
            pe = _sigmoid(x)
            pc = v["p_challenge_insignificant"] if not c.get("significant") else v["p_challenge_base"]
            u = r.uniform()
            if u < pc:
                vote, kind = "challenge", ["specification", "data", "inference"][int(r.integers(3))]
            elif u < pc + pe:
                vote, kind = "endorse", None
            else:
                vote, kind = "abstain", None
            sp = p.get("stated_probability", {"endorse": 72, "challenge": 35, "abstain": 50, "sd": 10})   # calibrated in params.yaml
            prob = float(np.clip(r.normal(sp.get(vote, 50), sp.get("sd", 10)), 1, 99))
            rec = {"claim_id": c["claim_id"], "vote": vote, "probability": round(prob, 1),
                   "reason": {"endorse": "consistent with the evidence shown", "challenge": "the specification does not support the claim",
                              "abstain": "would want a fresh sample"}[vote], "objection_kind": kind}
            if ctx.get("variant") == "c" and vote == "challenge":
                rec["test"] = {"specification": "fixed effects with standard controls, firm-clustered errors",
                               "n_option": ["median", "2x_median", "p75"][int(r.integers(3))],
                               "decision_rule": "the claim holds if the focal estimate has the claimed sign with p < .05"}
            if ctx.get("variant") == "d" and n_stakes < 6 and r.uniform() < p["stake"]["p_stake"]:
                amts = p["stake"]["amount_weights"]
                ks, ws = list(amts), np.asarray(list(amts.values()), dtype=float)
                amount = int(ks[int(r.choice(len(ks), p=ws / ws.sum()))])
                side = "for" if (vote == "endorse" or (vote == "abstain" and r.uniform() < 0.5)) else "against"
                rec["stake"] = {"side": side, "amount": amount}
                n_stakes += 1
            votes.append(rec)
        verifications = []
        cap = int(ctx.get("verification_capacity", 0))
        for c in sorted(ctx.get("open_claims", []), key=lambda c: (-c.get("endorsements", 0), c["claim_id"])):
            if len(verifications) >= cap:
                break
            if c.get("verifiable", True) and r.uniform() < p["p_offer_verify"]:
                verifications.append(c["claim_id"])
        responses = []
        for ch in ctx.get("challenges", []):
            opts, ws = list(p["response"]), np.asarray(list(p["response"].values()), dtype=float)
            action = opts[int(r.choice(len(opts), p=ws / ws.sum()))]
            if ctx.get("variant") == "c" and ch.get("test"):
                action = "accept_test" if r.uniform() < 0.7 else ("amend_test" if r.uniform() < 0.6 else "withdraw")
            responses.append({"claim_id": ch["claim_id"], "action": action, "text": "We have re-examined the analysis in light of the objection."})
        return {"nonce": ctx["nonce"], "agenda": agenda, "filings": filings, "votes": votes, "verifications": verifications,
                "responses": responses}


# ============================================================================ verifier (rule on structured numbers)
def reproduction_result(claim_est: float, claim_p: float | None, own_est: float, own_se: float, factor: float = 2.0,
                        claim_sign: str | None = None) -> str:
    """Repository variant b: reproduced when the verifier's own analysis gives the same sign with p < .05 and an estimate within
    a factor of `factor` of the claimed estimate. For a claim that states NO effect (sign 'none', or a non-significant claimed
    estimate when the sign is unknown) the rule is: reproduced when the verifier's estimate is not significant at .05 (a design choice: a null claim is reproduced by a null result)."""
    if own_se is None or not np.isfinite(own_est) or own_se <= 0:
        return "inconclusive"
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(own_est / own_se)))
    null_claim = claim_sign == "none" or (claim_sign is None and claim_p is not None and claim_p >= 0.05)
    if null_claim:
        return "reproduced" if p >= 0.05 else "not_reproduced"
    same_sign = np.sign(own_est) == np.sign(claim_est)
    within = (abs(claim_est) / factor) <= abs(own_est) <= abs(claim_est) * factor if claim_est != 0 else abs(own_est) < 2 * own_se
    return "reproduced" if (same_sign and p < 0.05 and within) else "not_reproduced"


# ============================================================================ handbook (pseudo-B4)
class RuleHandbook:
    def __init__(self, params: dict):
        self.p = params["handbook"]

    def write(self, question_rules: dict[str, dict], theories: dict[str, list[str]], r, nonce: str,
              predictions: dict[str, str] | None = None) -> dict:
        """question_rules[qid] = {'B1':..., 'B2':..., 'B3':..., 'B5':..., 'n_units': int}; predictions[qid] = the theory's
        predicted direction, which defines the `agree` field (agree = the answer is the predicted direction)."""
        p = self.p
        predictions = predictions or {}
        qs = []
        for qid, rules in question_rules.items():
            if rules.get("n_units", 0) == 0 or rules["B1"]["belief"] == "uncovered":
                ans, conf = ("no evidence", 10.0) if r.uniform() < p["p_no_evidence_when_uncovered"] else ("contested", 30.0)
            else:
                answers = {"B1": rules["B1"]["belief"], "B2": rules["B2"]["belief"], "B3": rules["B3"]["belief"]}
                ws = {"B1": p["w_B1"], "B2": p["w_B2"], "B3": p["w_B3"]}
                if len(set(answers.values())) > 1 and r.uniform() < p["p_contested_when_rules_disagree"]:
                    ans = "contested"
                else:
                    ks = list(ws); w = np.asarray([ws[k] for k in ks]); pick = ks[int(r.choice(len(ks), p=w / w.sum()))]
                    ans = answers[pick]
                    if ans == "uncovered":
                        ans = "no evidence"
                conf = float(np.clip(100 * p["confidence_scale"] * rules["B2"].get("confidence", 0.5) + r.normal(0, 8), 5, 95))
                if ans == "contested":
                    conf = min(conf, 55.0)
            qs.append({"question_id": qid, "answer": ans, "confidence": round(conf, 1), "n_read": int(rules.get("n_units", 0)),
                       "key_units": [], "agree": bool(predictions.get(qid)) and ans == predictions.get(qid),
                       "reasoning": "Based on the certified evidence read."})
        th = []
        by_q = {x["question_id"]: x for x in qs}
        for t, qids in theories.items():
            sup = [by_q[q]["answer"] == theories_pred.get(q) for q in qids if q in by_q] if (theories_pred := {}) else []
            share = float(np.mean([1.0 if by_q[q]["answer"] in ("positive", "negative") else 0.0 for q in qids if q in by_q])) if qids else 0.0
            standing = "strong" if share >= 0.67 else ("moderate" if share >= 0.4 else ("weak" if share > 0 else "untested"))
            th.append({"theory": t, "standing": standing, "share_supported": round(share, 2), "reasoning": "From the answers above."})
        open_qs = [{"question_id": q["question_id"], "reason": "conflicting_results", "text": "The certified evidence points in different directions."}
                   for q in qs if q["answer"] == "contested"][:10]
        return {"nonce": nonce, "questions": qs, "theories": th, "open_questions": open_qs}
