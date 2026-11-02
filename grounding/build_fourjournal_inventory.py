# -*- coding: utf-8 -*-
"""build_fourjournal_inventory.py -- four-journal (AMJ/ASQ/OS/SMJ) seeding tables.

Reads a keyword-level extraction of four journals (AMJ, ASQ, OS, SMJ; not distributed, see DATA.md) STRICTLY READ-ONLY,
plus the SMJ deep extraction's rep_study_clean.csv (read-only) for a cross-validation crosswalk.
Writes only to ./tables/4j_*.csv next to this script.

Inputs:
  the four-journal extraction folder: --fourjournal-dir <dir> or the environment variable SYNTHSCI_FOURJOURNAL_DIR
  rep_study_clean.csv: --smj-study <file>, SYNTHSCI_SMJ_STUDY, or by default the frozen copy
    corpus/smj-deep-frozen-2026-10-13/rep_study_clean.csv under the repository root

Run (isolated mode so nothing in the input folder can be imported):
  python -I grounding/build_fourjournal_inventory.py --fourjournal-dir <dir>
"""
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _arg(flag, env, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return os.environ.get(env) or default


E = Path(_arg("--fourjournal-dir", "SYNTHSCI_FOURJOURNAL_DIR", "") or ".")
if not _arg("--fourjournal-dir", "SYNTHSCI_FOURJOURNAL_DIR") or not (E / "field_review_complete.csv").exists():
    sys.exit("STOP: give the four-journal extraction folder with --fourjournal-dir <dir> or SYNTHSCI_FOURJOURNAL_DIR "
             "(it must hold field_review_complete.csv; the extraction is not distributed, see DATA.md)")
REP = Path(_arg("--smj-study", "SYNTHSCI_SMJ_STUDY", str(ROOT / "corpus" / "smj-deep-frozen-2026-10-13" / "rep_study_clean.csv")))
if not REP.exists():
    sys.exit(f"STOP: {REP} is missing (the SMJ extraction is not distributed, see DATA.md); give it with --smj-study <file>")
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)
S = {}
save = lambda df, n: df.to_csv(OUT / f"4j_{n}.csv", index=False, encoding="utf-8")


def qtab(x, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    x = pd.Series(x).dropna()
    d = {"n": len(x), "mean": x.mean(), "sd": x.std()}
    d.update({f"p{int(q*100):02d}": x.quantile(q) for q in qs})
    return d


def gini(x):
    x = np.sort(np.asarray(pd.Series(x).dropna(), float))
    if len(x) == 0 or x.sum() == 0: return np.nan
    n = len(x); return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


cp = pd.read_csv(E / "field_review_complete.csv", low_memory=False, encoding_errors="replace")
J = "meta_journal"

# ---------------- year (meta_year is unreliable: 1,308 rows coded 2007, 714 coded 1996)
fp = cp["file_path"].astype(str)
y_folder = pd.to_numeric(fp.str.extract(r"\\((?:19|20)\d\d)-\d+\\")[0], errors="coerce")
y_smjname = pd.to_numeric(fp.str.extract(r"Strategic Management Journal - ((?:19|20)\d\d) - ")[0], errors="coerce")

# ---------------- WoS match validation (1,560 rows share a duplicated WoS id -> bad merges)
tok = lambda s: set(t for t in re.findall(r"[a-z]{3,}", str(s).lower()) if t not in {"the", "and", "for", "from", "with", "pdf"})
stem = fp.str.replace(r"^.*\\", "", regex=True).str.replace(r"\.pdf$", "", regex=True)
def sim(a, b):
    A, Bt = tok(a), tok(b)
    return len(A & Bt) / max(1, min(len(A), len(Bt))) if A and Bt else 0.0
cp["wos_sim"] = [max(sim(t, w), sim(s_, w)) for t, s_, w in zip(cp["meta_title"], stem, cp["wos_title"])]
dupid = cp["wos_wos_id"].notna() & cp["wos_wos_id"].duplicated(keep=False)
cp["wos_ok"] = cp["wos_wos_id"].notna() & ~dupid & (cp["wos_sim"] >= 0.5)
# EBSCO-numbered AMJ/OS files carry no slug; accept unique-id matches when meta_title is junk but WoS year agrees with folder year
weak = cp["wos_wos_id"].notna() & ~dupid & ~cp["wos_ok"] & (pd.to_numeric(cp["wos_year"], errors="coerce") - y_folder).abs().le(1)
cp["wos_ok_lenient"] = cp["wos_ok"] | weak
cp["yr"] = y_folder.fillna(y_smjname).fillna(pd.to_numeric(cp["wos_year"], errors="coerce").where(cp["wos_ok"]))
cp["yr_src"] = np.where(y_folder.notna(), "folder", np.where(y_smjname.notna(), "smj_filename", np.where(cp["yr"].notna(), "wos", "none")))
cp["yr"] = cp["yr"].fillna(pd.to_numeric(cp["meta_year"], errors="coerce").where(~cp["meta_year"].isin([2007, 1996])))
wv = cp.groupby(J).agg(n=("file_path", "size"), wos_any=("wos_wos_id", lambda x: x.notna().mean()),
                       wos_dup_id=("wos_wos_id", lambda x: (x.notna() & x.duplicated(keep=False)).mean()),
                       wos_ok_strict=("wos_ok", "mean"), wos_ok_lenient=("wos_ok_lenient", "mean"),
                       year_resolved=("yr", lambda x: x.notna().mean())).reset_index()
save(wv, "00_wos_match_quality")
S["wos_quality"] = wv.round(3).to_dict("records")
dt = cp["wos_documenttype"].astype(str)
cp["is_article"] = ~dt.str.contains("Editorial|Correction|Review|Book", na=False)

# ---------------- 1. coverage journal x year
cov = pd.crosstab(cp["yr"], cp[J]).reset_index(); save(cov, "01_coverage_journal_year")
cov5 = pd.crosstab(pd.cut(cp["yr"], [1979, 1999, 2004, 2009, 2014, 2019, 2026], labels=["<=1999", "2000-04", "2005-09", "2010-14", "2015-19", "2020-25"]), cp[J]).reset_index()
save(cov5, "01_coverage_journal_5yr")
S["coverage"] = cp[J].value_counts().to_dict(); S["yr_range"] = (cp["yr"].min(), cp["yr"].max())
S["yr_by_j"] = cp.groupby(J)["yr"].agg(["min", "max", lambda x: x.isna().sum()]).to_dict("index")

# ---------------- 2. theory families by journal
tcols = [c for c in cp.columns if re.fullmatch(r"tc_theory_[a-z_]+", c) and not c.endswith(("_count", "_density", "_role")) and c not in ("tc_theory_counts_json", "tc_theory_count", "tc_theory_diversity")]
th = cp.groupby(J)[tcols].apply(lambda d: (pd.to_numeric(d.stack(), errors="coerce").unstack() > 0).mean()).T
th.index = th.index.str.replace("tc_theory_", ""); th["all4"] = (cp[tcols].apply(pd.to_numeric, errors="coerce") > 0).mean().values
th = th.sort_values("all4", ascending=False).reset_index().rename(columns={"index": "theory"})
save(th, "02_theory_share_by_journal")
save(pd.crosstab(cp["tc_dominant_theory"], cp[J], normalize="columns").reset_index(), "02_dominant_theory_by_journal")
save(cp.groupby(J)["tc_theory_count"].apply(lambda x: pd.Series(qtab(x))).unstack().reset_index(), "02_theories_per_paper_by_journal")
save(pd.crosstab(cp["topic_primary_subfield"], cp[J], normalize="columns").reset_index(), "02_primary_subfield_by_journal")
save(pd.crosstab(cp["tc_primary_level"], cp[J], normalize="columns").reset_index(), "02_level_of_theory_by_journal")
S["theory_top"] = th.head(12).set_index("theory").round(3).to_dict("index")

# ---------------- 3. designs by journal
save(pd.crosstab(cp["meth_research_design"], cp[J], normalize="columns").reset_index(), "03_research_design_by_journal")
save(pd.crosstab(cp["hyp_paper_type"], cp[J], normalize="columns").reset_index(), "03_hypothesis_style_by_journal")
mflags = ["meth_is_quantitative", "meth_is_qualitative", "meth_is_mixed_methods", "meth_is_conceptual", "meth_is_simulation",
          "meth_is_meta_analysis", "meth_is_survey", "meth_ols", "meth_logit", "meth_fixed_effects", "meth_did", "meth_iv", "meth_psm",
          "meth_heckman", "meth_gmm", "meth_sem", "meth_hlm", "meth_survival", "meth_event_study", "meth_panel", "meth_is_longitudinal",
          "meth_endogeneity_addressed", "meth_causal_identification", "meth_cmv_addressed", "meth_preregistered", "meth_power_analysis",
          "meth_reports_effect_sizes", "meth_multilevel", "hyp_has_correlation_table", "hyp_has_regression_table",
          "meth_datasource_compustat", "meth_datasource_crsp", "meth_datasource_execucomp", "meth_datasource_sdc", "meth_datasource_patent",
          "meth_datasource_survey_primary", "meth_datasource_interview", "meth_datasource_case_study_data"]
mflags = [m for m in mflags if m in cp]
mf = cp.groupby(J)[mflags].apply(lambda d: (d.apply(pd.to_numeric, errors="coerce") > 0).mean()).T.reset_index().rename(columns={"index": "flag"})
save(mf, "03_method_flags_by_journal")
hy = cp[cp["hyp_hypothesis_count"] > 0]
save(hy.groupby(J)["hyp_hypothesis_count"].apply(lambda x: pd.Series(qtab(x))).unstack().reset_index(), "03_hypotheses_per_paper_by_journal")
save(cp.groupby(J)["meth_sample_size"].apply(lambda x: pd.Series(qtab(x))).unstack().reset_index(), "03_sample_size_by_journal_keyword")
S["design"] = mf.set_index("flag").round(3).to_dict("index")

# ---------------- 4. citations (WoS times cited, validated matches only)
cw = cp[cp["wos_ok_lenient"] & cp["is_article"]].copy()
cw["cites"] = pd.to_numeric(cw["wos_times_cited"], errors="coerce")
cw["wos_y"] = pd.to_numeric(cw["wos_year"], errors="coerce")
EXPORT_YEAR = 2025  # WoS export dated 12-20-2025
cw["age"] = (EXPORT_YEAR - cw["wos_y"]).clip(lower=0) + 0.5
cw["cpy"] = cw["cites"] / cw["age"]
rows = []
for (j, coh), d in cw.groupby([J, pd.cut(cw["wos_y"], [1979, 1999, 2004, 2009, 2014, 2019, 2022, 2026], labels=["<=1999", "2000-04", "2005-09", "2010-14", "2015-19", "2020-22", "2023-25"])], observed=True):
    if len(d) < 10: continue
    lc = np.log1p(d["cites"])
    rows.append({"journal": j, "cohort": coh, **qtab(d["cites"]), "share_uncited": (d["cites"] == 0).mean(),
                 "gini": gini(d["cites"]), "top10pct_share_of_cites": d["cites"].nlargest(max(1, len(d) // 10)).sum() / max(1, d["cites"].sum()),
                 "log1p_mean": lc.mean(), "log1p_sd": lc.std(), "cpy_median": d["cpy"].median()})
cit = pd.DataFrame(rows); save(cit, "04_citations_by_journal_cohort")
cj = cw.groupby(J).apply(lambda d: pd.Series({**qtab(d["cpy"]), "gini_cites": gini(d["cites"]), "n": len(d)})).reset_index()
save(cj, "04_citations_per_year_by_journal")
S["cit"] = cj.set_index(J)[["n", "p25", "p50", "p75", "p90", "p99", "gini_cites"]].round(2).to_dict("index")
S["cit_hc"] = int((cp["wos_highlycitedstatus"] == "Y").sum())

# ---------------- 5. review durations
def pdate(s):
    if not isinstance(s, str): return pd.NaT
    s = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", s).replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not re.search(r"[A-Za-z]", s) or not re.search(r"(19|20)\d\d", s): return pd.NaT
    return pd.to_datetime(s, errors="coerce")
for c in ["meta_date_received", "meta_date_revised", "meta_date_accepted"]:
    cp[c + "_dt"] = cp[c].map(pdate)
cp["rev_months_rec_acc"] = (cp["meta_date_accepted_dt"] - cp["meta_date_received_dt"]).dt.days / 30.44
cp["rev_months_rec_rev"] = (cp["meta_date_revised_dt"] - cp["meta_date_received_dt"]).dt.days / 30.44
cp["rev_months_rev_acc"] = (cp["meta_date_accepted_dt"] - cp["meta_date_revised_dt"]).dt.days / 30.44
ok = lambda x: x.where(x.between(0.5, 120))
rv = []
for j, d in cp.groupby(J):
    for c in ["rec_acc", "rec_rev", "rev_acc"]:
        col = "rev_months_" + c
        rv.append({"journal": j, "interval": col.replace("rev_months_", ""), "n_received_dates": d["meta_date_received_dt"].notna().sum(), **qtab(ok(d[col]))})
rv = pd.DataFrame(rv); save(rv, "05_review_durations_by_journal")
cp["rec_year"] = cp["meta_date_received_dt"].dt.year
rvy = cp[cp[J].isin(["SMJ", "OS"])].groupby([J, pd.cut(cp["rec_year"], [1990, 2004, 2009, 2014, 2019, 2026])], observed=True)["rev_months_rec_acc"].apply(lambda x: pd.Series(qtab(ok(x)))).unstack().reset_index()
rvy = rvy.astype({rvy.columns[1]: str}); save(rvy, "05_review_durations_by_era_SMJ_OS")
S["review"] = rv[rv["interval"] == "rec_acc"][["journal", "n", "p25", "p50", "p75", "p90"]].round(1).to_dict("records")
# publication lag: accepted -> WoS publication year (coarse)
cp["acc_to_pub_years"] = pd.to_numeric(cp["wos_year"], errors="coerce").where(cp["wos_ok_lenient"]) - cp["meta_date_accepted_dt"].dt.year
save(cp.groupby(J)["acc_to_pub_years"].apply(lambda x: pd.Series(qtab(x.where(x.between(-1, 6))))).unstack().reset_index(), "05_accept_to_pub_year_lag")

# ---------------- 6. authors: team size, productivity concentration, careers
au = cp[cp["wos_ok_lenient"] & cp["is_article"]].copy()
au["authors"] = au["wos_authorfullnames"].fillna(au["wos_author"]).astype(str).str.split(";").map(lambda L: [a.strip() for a in L if a.strip() and a.strip() != "nan"])
au["team"] = au["authors"].map(len).replace(0, np.nan)
au["era"] = pd.cut(pd.to_numeric(au["wos_year"], errors="coerce"), [1979, 1999, 2009, 2019, 2026], labels=["<=1999", "2000-09", "2010-19", "2020-25"])
ts = au.groupby([J, "era"], observed=True)["team"].agg(n="size", mean="mean", median="median", share_solo=lambda x: (x == 1).mean(),
                                                        share_2=lambda x: (x == 2).mean(), share_3=lambda x: (x == 3).mean(), share_4plus=lambda x: (x >= 4).mean()).reset_index()
ts = ts.astype({"era": str}); save(ts, "06_team_size_by_journal_era")
S["team"] = au.groupby(J)["team"].agg(["mean", lambda x: (x == 1).mean()]).round(3).to_dict("index")
ex = au.explode("authors")
ppa = ex.groupby("authors").size()
lot = ppa.value_counts().sort_index().rename_axis("papers_in_corpus").reset_index(name="n_authors")
lot["share_authors"] = lot["n_authors"] / lot["n_authors"].sum()
save(lot, "06_lotka_papers_per_author_in_corpus")
S["lotka"] = {"n_author_names": int(len(ppa)), "share_1paper": float((ppa == 1).mean()), "share_5plus": float((ppa >= 5).mean()),
              "top1pct_share_of_authorships": float(ppa.nlargest(max(1, len(ppa) // 100)).sum() / ppa.sum()), "gini": float(gini(ppa))}
# cross-journal mobility: authors publishing in >1 of the four journals
nj = ex.groupby("authors")[J].nunique()
S["authors_multi_journal_share_among_2plus"] = float((nj[ppa >= 2] > 1).mean())
# careers (OpenAlex profiles)
ap = pd.read_csv(E / "author_profiles.csv")
apm = ap[ap["auth_match_confidence"] == "high"]
car = pd.DataFrame([{"stat": k, **qtab(pd.to_numeric(apm[k], errors="coerce"))} for k in
                    ["auth_works_count_total", "auth_works_count_top7", "auth_h_index", "auth_i10_index", "auth_total_citations", "auth_first_pub_year", "auth_dataset_pub_count"]])
save(car, "06_author_career_profiles_highmatch")
aw = pd.read_csv(E / "author_works.csv", low_memory=False)
aw = aw[aw["auth_match_confidence"] == "high"]
aw["work_year"] = pd.to_numeric(aw["work_year"], errors="coerce"); aw["c"] = pd.to_numeric(aw["work_cited_by_count"], errors="coerce")
aw["career_yr"] = aw["work_year"] - pd.to_numeric(aw["auth_first_pub_year"], errors="coerce")
per = aw.groupby("auth_openalex_id").agg(n=("c", "size"), cites=("c", "sum"), gini_within=("c", gini),
                                          top7=("work_is_top7", lambda x: x.astype(str).eq("True").sum()))
traj = aw[aw["career_yr"].between(0, 40)].groupby("career_yr").agg(works=("c", "size"), med_cites=("c", "median"), share_top7=("work_is_top7", lambda x: x.astype(str).eq("True").mean())).reset_index()
traj["works_per_author"] = traj["works"] / aw["auth_openalex_id"].nunique()
save(traj, "06_career_trajectory_by_career_year")
# Matthew signature: early (first 5 yrs) citations -> later productivity/citations
early = aw[aw["career_yr"].between(0, 5)].groupby("auth_openalex_id")["c"].sum()
late = aw[aw["career_yr"].between(6, 15)].groupby("auth_openalex_id").agg(late_n=("c", "size"), late_c=("c", "sum"))
mt = late.join(early.rename("early_c"), how="inner")
S["matthew"] = {"n_authors": int(len(mt)),
                "spearman_early_cites_late_cites": float(mt["early_c"].corr(mt["late_c"], method="spearman")),
                "spearman_early_cites_late_n": float(mt["early_c"].corr(mt["late_n"], method="spearman")),
                "median_within_author_gini": float(per["gini_within"].median()),
                "gini_total_cites_across_authors": float(gini(per["cites"])), "n_high_match_authors": int(len(apm))}
# article-level: prior productivity of first author vs citations (credible rows)
art = cw.copy()
for c in ["auth_first_prior_pubs_total", "auth_first_career_years", "auth_team_top7_total"]:
    art[c] = pd.to_numeric(art[c], errors="coerce")
S["matthew_article"] = {c: {"n": int(art[[c, "cpy"]].dropna().shape[0]), "spearman_with_cites_per_year": float(art[c].corr(art["cpy"], method="spearman"))}
                        for c in ["auth_first_prior_pubs_total", "auth_first_career_years", "auth_team_top7_total"]}
S["author_coverage_by_j"] = cp.groupby(J)["auth_first_career_years"].apply(lambda x: pd.to_numeric(x, errors="coerce").notna().mean()).round(3).to_dict()

# ---------------- 7. SSRN pipeline
sp = pd.read_csv(E / "ssrn_published_pairs.csv")
sm = pd.read_csv(E / "ssrn_metadata_enriched.csv", low_memory=False)
smt = sm.groupby("relevance_tier").agg(n=("openalex_id", "size"), share_published_elsewhere=("is_published_elsewhere", "mean"),
                                       median_cites=("cited_by_count", "median"), median_authors=("author_count", "median")).reset_index()
save(smt, "07_ssrn_by_relevance_tier")
core = sm[sm["relevance_tier"] == "core"]
save(core["pub_journal"].value_counts().head(30).rename_axis("pub_journal").reset_index(name="n"), "07_ssrn_core_published_venues")
save(sm.groupby("year").agg(n=("openalex_id", "size"), share_pub=("is_published_elsewhere", "mean")).reset_index(), "07_ssrn_by_year")
S["ssrn"] = {"n_records": len(sm), "pairs_matched_to_corpus": len(sp), "pairs_journals": sp["pub_journal"].value_counts().to_dict(),
             "tiers": smt.round(3).to_dict("records")}

# ---------------- 8. topic hotness
th2 = pd.read_csv(E / "topic_hotness_openalex.csv").merge(cp[["file_path", J]], on="file_path", how="left")
save(th2.groupby(J)[["topic_hotness_oa_narrow_3yr", "topic_hotness_oa_narrow_trend"]].describe().T.reset_index(), "08_topic_hotness_by_journal")
S["hot_nonzero"] = th2.groupby(J)["topic_hotness_oa_narrow_3yr"].apply(lambda x: (x > 0).mean()).round(3).to_dict()

# ---------------- 9. relationship / correlation availability by journal
rc = os.listdir(E / "relationships_cache")
S["relcache"] = {"files": len(rc), "extraction_json": sum(f.endswith("_extraction.json") for f in rc), "candidates_json": sum(f.endswith("_candidates.json") for f in rc),
                 "rel_relationships_csv_exists": (E / "rel_relationships.csv").exists(), "rel_variables_csv_exists": (E / "rel_variables.csv").exists()}
ct = [f for f in os.listdir(E / "correlation_tables") if f.endswith(".xlsx")]
recs, cells = [], []
for f in ct:
    j = f.split("_")[0]
    try:
        x = pd.read_excel(E / "correlation_tables" / f, sheet_name=None, header=None)
    except Exception:
        recs.append({"file": f, "journal": j, "ok": False}); continue
    meta = {}
    if "Metadata" in x:
        m = x["Metadata"]; meta = dict(zip(m.iloc[1:, 0].astype(str), m.iloc[1:, 1]))
    C = x.get("Correlations")
    if C is None or C.shape[0] < 3:
        recs.append({"file": f, "journal": j, "ok": False, "method": meta.get("extraction_method")}); continue
    M = C.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").values
    k = M.shape[0]
    low = M[np.tril_indices(min(k, M.shape[1]), -1)] if M.shape[1] >= 2 else np.array([])
    vals = low[~np.isnan(low)]
    diag = np.diag(M[:min(k, M.shape[1]), :min(k, M.shape[1])])
    recs.append({"file": f, "journal": j, "ok": True, "k": k, "n_lower": len(low), "fill": len(vals) / max(1, len(low)),
                 "share_out_of_range": float((np.abs(vals) > 1).mean()) if len(vals) else np.nan,
                 "diag_is_one": float(np.nanmean(np.isclose(diag, 1))) if len(diag) else np.nan,
                 "N": pd.to_numeric(meta.get("sample_size"), errors="coerce"), "method": meta.get("extraction_method"),
                 "paper": meta.get("file_path")})
    cells.append(pd.DataFrame({"journal": j, "r": vals}))
ctab = pd.DataFrame(recs)
cells = pd.concat(cells, ignore_index=True)
cs = ctab.groupby("journal").agg(tables=("file", "size"), parsed=("ok", "sum"), papers=("paper", "nunique"), median_k=("k", "median"),
                                 median_fill=("fill", "median"), share_out_of_range=("share_out_of_range", "mean"),
                                 diag_is_one=("diag_is_one", "mean"), median_N=("N", "median")).reset_index()
cr = cells[cells["r"].abs() <= 1].groupby("journal")["r"].apply(lambda x: pd.Series({"cells": len(x), "abs_r_p25": x.abs().quantile(.25), "abs_r_med": x.abs().median(),
                                                                                    "abs_r_p75": x.abs().quantile(.75), "abs_r_p90": x.abs().quantile(.9), "share_abs_gt_.5": (x.abs() > .5).mean()})).unstack().reset_index()
cs = cs.merge(cr, on="journal", how="left"); save(cs, "09_correlation_tables_by_journal")
S["corr4j"] = cs.round(3).to_dict("records")
S["corr4j_methods"] = ctab["method"].value_counts(dropna=False).to_dict()

# ---------------- 10. SMJ crosswalk: four-journal keyword fields vs deep extraction
rep = pd.read_csv(REP, low_memory=False)
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
rep["k"] = rep["filename"].map(norm)
smj = cp[cp[J] == "SMJ"].copy(); smj["k"] = stem[smj.index].map(norm)
keys = {}
for i, kk in smj["k"].items():
    keys.setdefault(kk[:45], i)
rep["fr_idx"] = rep["k"].map(lambda k: next((keys[p] for p in [k[:45]] if p in keys), np.nan))
mrep = rep.dropna(subset=["fr_idx"]).copy()
fr = cp.loc[mrep["fr_idx"].astype(int)].reset_index(drop=True); mrep = mrep.reset_index(drop=True)
def kappa(a, b):
    a, b = pd.Series(a).astype(int), pd.Series(b).astype(int)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan
th_rx = {"rbv": r"resource[- ]based|\brbv\b", "agency": r"agency", "tce": r"transaction cost", "institutional": r"institutional",
         "upper_echelons": r"upper echelon", "social_network": r"network|embeddedness|structural hole", "rdt": r"resource dependen"}
xw = []
for t, rx in th_rx.items():
    a = mrep["theories_invoked"].astype(str).str.lower().str.contains(rx)
    b = pd.to_numeric(fr[f"tc_theory_{t}"], errors="coerce").fillna(0) > 0
    xw.append({"field": f"theory:{t}", "n": len(a), "deep_share": a.mean(), "keyword_share": b.mean(), "agreement": (a == b).mean(), "kappa": kappa(a, b)})
a = mrep["paper_type"].isin(["quantitative_empirical", "mixed_methods"]); b = pd.to_numeric(fr["meth_is_quantitative"], errors="coerce").fillna(0) > 0
xw.append({"field": "quantitative", "n": len(a), "deep_share": a.mean(), "keyword_share": b.mean(), "agreement": (a == b).mean(), "kappa": kappa(a, b)})
for deep, kw in [("hypotheses_total", "hyp_hypothesis_count"), ("sample_size_final", "meth_sample_size")]:
    x1, x2 = pd.to_numeric(mrep[deep], errors="coerce"), pd.to_numeric(fr[kw], errors="coerce")
    xw.append({"field": f"{deep}~{kw}", "n": int((x1.notna() & x2.notna()).sum()), "spearman": x1.corr(x2, method="spearman"),
               "exact_match": float((x1 == x2)[x1.notna() & x2.notna()].mean())})
xw = pd.DataFrame(xw); save(xw, "10_smj_crosswalk_keyword_vs_deep")
S["crosswalk_n"] = len(mrep); S["crosswalk"] = xw.round(3).to_dict("records")

json.dump(S, open(OUT / "4j_summary.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str)[:20000])
