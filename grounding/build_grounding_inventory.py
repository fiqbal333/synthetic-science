# -*- coding: utf-8 -*-
"""build_grounding_inventory.py -- grounding inventory for the synthetic strategy-research world.

Reads the frozen copy of the SMJ article extraction (STRICTLY READ-ONLY) and writes compact
sampling tables under ./tables/ (next to this script), or under --out <dir> for a check rebuild.

Source: the project's own frozen copy of the extraction (not distributed; see DATA.md), expected under the repository root:
  corpus/smj-deep-frozen-2026-10-13/: rep_study_clean.csv, rep_relationships_clean_v3.csv,
    rep_correlations_clean_v2.csv, rep_descriptives_clean.csv, rep_measurement_clean.csv,
    rep_variables_clean.csv, construct-dictionary.json
  corpus/smj-deep-frozen-supplement-2026-10-19/: name_map_v1.csv, rulebook_v1.json, harmonize.py
    (imported without bytecode)
Every input is checked against its folder's SHA256SUMS.txt before it is read; a mismatch stops the build.
(Originally built 2026-10-13 from the extraction's working files, which were byte-identical to the frozen copy on
2026-10-19.)

Run: python grounding/build_grounding_inventory.py [--out <dir>]
"""
import sys
sys.dont_write_bytecode = True          # never write __pycache__ into the read-only corpus
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "corpus" / "smj-deep-frozen-2026-10-13"
SUPP = ROOT / "corpus" / "smj-deep-frozen-supplement-2026-10-19"


def _verify(folder: Path, names) -> None:
    sums = dict(reversed(ln.split(None, 1)) for ln in (folder / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines() if ln.strip())
    for n in names:
        if not (folder / n).exists():
            sys.exit(f"STOP: {folder / n} is missing. The SMJ extraction is not distributed with this repository (see DATA.md); "
                     f"place the frozen copy under corpus/{folder.name}/")
        got = hashlib.sha256((folder / n).read_bytes()).hexdigest().upper()
        if sums.get(n, "").upper() != got:
            sys.exit(f"STOP: {folder / n} does not match {folder.name}/SHA256SUMS.txt")


_verify(FROZEN, ["rep_study_clean.csv", "rep_relationships_clean_v3.csv", "rep_correlations_clean_v2.csv",
                 "rep_descriptives_clean.csv", "rep_measurement_clean.csv", "rep_variables_clean.csv",
                 "construct-dictionary.json"])
_verify(SUPP, ["name_map_v1.csv", "rulebook_v1.json", "harmonize.py"])
DATA = FROZEN
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SUPP))
import harmonize as hz  # noqa: E402  (read-only import)
_load_rulebook, _load_alias_map = hz.load_rulebook, hz.load_alias_map
hz.load_rulebook = lambda path=SUPP / "rulebook_v1.json": _load_rulebook(path)                 # frozen inputs, not the
hz.load_alias_map = lambda path=FROZEN / "construct-dictionary.json": _load_alias_map(path)    # module's own folder

pd.options.mode.copy_on_write = True
SUMMARY = {}


def save(df, name, index=False):
    df.to_csv(OUT / f"{name}.csv", index=index, encoding="utf-8")


def qtab(x, qs=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    x = pd.Series(x).dropna()
    d = {"n": len(x), "mean": x.mean(), "sd": x.std()}
    d.update({f"p{int(q*100):02d}": x.quantile(q) for q in qs})
    d["min"], d["max"] = x.min(), x.max()
    return d


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
s = pd.read_csv(DATA / "rep_study_clean.csv", low_memory=False)
r = pd.read_csv(DATA / "rep_relationships_clean_v3.csv", low_memory=False)
c = pd.read_csv(DATA / "rep_correlations_clean_v2.csv", low_memory=False)
desc = pd.read_csv(DATA / "rep_descriptives_clean.csv", low_memory=False)
meas = pd.read_csv(DATA / "rep_measurement_clean.csv", low_memory=False)
v = pd.read_csv(DATA / "rep_variables_clean.csv", low_memory=False)
nmap = pd.read_csv(SUPP / "name_map_v1.csv", low_memory=False)
cdict = json.load(open(FROZEN / "construct-dictionary.json", encoding="utf-8"))

# year from filename (meta 'year' is populated for only 898/2358 rows)
yr = lambda f: pd.to_numeric(pd.Series(f).str.extract(r"_-_((?:19|20)\d\d)_-_")[0], errors="coerce")
s["yr"] = yr(s["filename"]).values
s["era"] = pd.cut(s["yr"], [1979, 1999, 2009, 2019, 2030], labels=["<=1999", "2000-09", "2010-19", "2020-25"])
TYPEGRP = {"quantitative_empirical": "quant", "mixed_methods": "mixed", "qualitative_empirical": "qual",
           "conceptual": "conceptual", "review": "conceptual", "methodology": "methods/sim/meta",
           "simulation": "methods/sim/meta", "meta_analysis": "methods/sim/meta", "unknown": "unknown"}
s["type_grp"] = s["paper_type"].map(TYPEGRP).fillna("unknown")
emp = s[s["paper_type"].isin(["quantitative_empirical", "mixed_methods"])].copy()

# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------
cov_year = pd.crosstab(s["yr"], s["type_grp"]).assign(total=lambda d: d.sum(axis=1))
save(cov_year.reset_index(), "01a_articles_by_year_type")
save(s["paper_type"].value_counts().rename_axis("paper_type").reset_index(name="n_articles"), "01b_paper_types")

c_ok = c[(c["verification_status"] != "phantom_article_suspect") & c["final_risk"].notna() & (c["final_risk"] < 3)
         & c["correlation_r"].between(-1, 1)]
c_ver = c_ok[c_ok["verification_status"].str.contains("verified|confirmed|patch|adjudicated", na=False)]
rel_hyp = r[r["rel_result"].isin(["supported", "not_supported", "partially_supported"])]
usable = pd.DataFrame([
    ("all articles", s["filename"].nunique()),
    ("quant + mixed empirical", len(emp)),
    ("papers with >=1 stated hypothesis (hypotheses_total>0)", (s["hypotheses_total"] > 0).sum()),
    ("papers with paper-level support_rate", s["support_rate"].notna().sum()),
    ("hypothesis-level rows with a support verdict", len(rel_hyp)),
    ("papers with hypothesis-level verdicts", rel_hyp["filename"].nunique()),
    ("focal coefficient rows, any tier", r["coefficient_value"].notna().sum()),
    ("focal coefficient rows, tier high", (r["confidence_tier_v3"] == "high").sum()),
    ("focal coefficient rows, tier high+medium", r["confidence_tier_v3"].isin(["high", "medium"]).sum()),
    ("papers with >=1 high-tier coefficient", r.loc[r["confidence_tier_v3"] == "high", "filename"].nunique()),
    ("papers with >=1 high/medium-tier coefficient", r.loc[r["confidence_tier_v3"].isin(["high", "medium"]), "filename"].nunique()),
    ("rows with verified coef AND SE (t computable)", (r["coef_verified"].notna() & r["se_verified"].notna()).sum()),
    ("correlation cells, all", len(c)),
    ("papers with correlation tables, all", c["file_name"].nunique()),
    ("correlation cells, final_risk<3", len(c_ok)),
    ("papers with correlation tables, final_risk<3", c_ok["file_name"].nunique()),
    ("correlation cells verified (>=2 channels/adjudicated), final_risk<3", len(c_ver)),
    ("descriptives rows (means/SDs), all", len(desc)),
    ("descriptives rows, extraction_confidence high", (desc["extraction_confidence"] == "high").sum()),
    ("papers with descriptives, high confidence", desc.loc[desc["extraction_confidence"] == "high", "file_name"].nunique()),
    ("variable rows (rep_variables_clean)", len(v)),
    ("measurement rows (rep_measurement_clean)", len(meas)),
], columns=["item", "n"])
save(usable, "01c_usable_coverage")
SUMMARY["coverage"] = usable.set_index("item")["n"].to_dict()

# ---------------------------------------------------------------------------
# Construct mapping helper (harmonization v1 ladder: alias -> curated family regex)
# ---------------------------------------------------------------------------
RB = hz.load_rulebook()
AM = hz.load_alias_map()
_map_cache = {}


def map_series(names):
    nn = pd.Series(names).map(hz.norm_name)
    # second try with parentheticals stripped (relationship names carry glosses)
    nn2 = pd.Series(names).astype(str).str.replace(r"\(.*?\)", " ", regex=True).map(hz.norm_name)
    uniq = sorted(set(nn.dropna()) | set(nn2.dropna()))
    todo = [u for u in uniq if u not in _map_cache]
    if todo:
        mp = hz.map_names(todo, RB, AM)
        for k, row in mp.iterrows():
            _map_cache[k] = (row["construct"], row["family"], row["category"])
    get = lambda k: _map_cache.get(k, ("", "", ""))
    out = []
    for a, b in zip(nn, nn2):
        t = get(a)
        if not t[0] or t[1] in ("AMBIGUOUS",):
            t2 = get(b)
            if t2[0] and t2[1] != "AMBIGUOUS":
                t = t2
        out.append(t)
    return pd.DataFrame(out, columns=["construct", "family", "category"], index=pd.Series(names).index)


BAD_CATS = {"design_variable", "interaction", ""}

# ---------------------------------------------------------------------------
# 2a. Topics: theory families, construct pairs, relationship types
# ---------------------------------------------------------------------------
THEORY = [
    ("resource_based_view", r"resource[- ]based|\brbv\b|vrin"),
    ("knowledge_based_view", r"knowledge[- ]based"),
    ("dynamic_capabilities", r"dynamic capabilit"),
    ("agency_theory", r"agency"),
    ("stewardship", r"stewardship"),
    ("transaction_cost_economics", r"transaction cost|\btce\b|williamson"),
    ("property_rights_incomplete_contracts", r"property rights|incomplete contract"),
    ("institutional_theory", r"institutional (theory|isomorph)|neo-?institutional|isomorphism|institution-based"),
    ("legitimacy", r"legitima"),
    ("resource_dependence", r"resource dependen"),
    ("upper_echelons", r"upper echelon"),
    ("behavioral_theory_of_firm", r"behavio(u)?ral theory|aspiration|performance feedback|carnegie"),
    ("organizational_learning", r"organi[sz]ational learning|learning theory|learning curve|experiential learning"),
    ("absorptive_capacity", r"absorptive"),
    ("social_network_theory", r"network|structural hole|embeddedness|brokerage|centrality"),
    ("social_capital", r"social capital"),
    ("stakeholder_theory", r"stakeholder"),
    ("signaling_theory", r"signal(l)?ing"),
    ("real_options", r"real option"),
    ("industrial_organization", r"industrial organi[sz]ation|\bi/?o economics|structure[- ]conduct|five forces|market power"),
    ("competitive_dynamics", r"competitive dynamic|awareness[- ]motivation|multimarket|mutual forbearance|rivalry"),
    ("game_theory", r"game[- ]theor"),
    ("evolutionary_economics", r"evolutionary"),
    ("attention_based_view", r"attention[- ]based|attention"),
    ("contingency_theory", r"contingency"),
    ("population_ecology", r"ecolog|liability of newness"),
    ("imprinting", r"imprint"),
    ("status_reputation", r"\bstatus\b|reputation"),
    ("social_identity_categorization", r"social identity|categori[sz]ation|self-categori"),
    ("prospect_theory_behavioral_decision", r"prospect theory|behavio(u)?ral (decision|agency)|cognitive bias|heuristic"),
    ("managerial_cognition_sensemaking", r"cognition|sensemaking|mental model"),
    ("corporate_governance_general", r"corporate governance|board"),
    ("internationalization_mne", r"internationali[sz]ation|eclectic|\boli\b|liability of foreignness|uppsala|multinational"),
    ("innovation_search_nk", r"\bsearch\b|exploration|exploitation|ambidext|\bnk\b|rugged landscape"),
    ("diversification_corporate_strategy", r"diversification|corporate strategy|relatedness|synerg"),
    ("information_economics_asymmetry", r"information asymmetr|adverse selection|moral hazard"),
    ("tournament_compensation", r"tournament|equity theory|social comparison"),
    ("upper_tier_power_politics", r"\bpower\b|political"),
    ("entrepreneurship_theory", r"entrepreneur"),
]
th_rows = []
for _, row in s.iterrows():
    try:
        items = json.loads(row["theories_invoked"]) if isinstance(row["theories_invoked"], str) else []
    except Exception:
        items = []
    fams = set()
    for it in items:
        low = str(it).lower()
        hit = [f for f, p in THEORY if re.search(p, low)]
        fams.update(hit if hit else ["other_unclassified"])
    for f in fams:
        th_rows.append((row["filename"], row["type_grp"], row["era"], f))
th = pd.DataFrame(th_rows, columns=["filename", "type_grp", "era", "theory_family"])
n_with_theory = th["filename"].nunique()
th_tab = (th.groupby("theory_family")["filename"].nunique().rename("n_papers").reset_index()
          .assign(share_of_papers_with_theory=lambda d: d["n_papers"] / n_with_theory)
          .sort_values("n_papers", ascending=False))
save(th_tab, "02a_theory_families")
th_n = s.loc[s["theories_count"].notna(), "theories_count"]
save(pd.DataFrame([qtab(th_n)]), "02a_theories_per_paper")
SUMMARY["theory_top"] = th_tab.head(10).set_index("theory_family")["n_papers"].to_dict()
SUMMARY["n_with_theory"] = n_with_theory

# construct pairs for hypothesis-level relationships
rx = map_series(r["x_variable"]).add_prefix("x_")
ry = map_series(r["y_variable"]).add_prefix("y_")
r = pd.concat([r, rx, ry], axis=1)
r["pair_mapped"] = (~r["x_category"].isin(BAD_CATS)) & (~r["y_category"].isin(BAD_CATS)) \
    & (r["x_construct"] != "") & (r["y_construct"] != "")
SUMMARY["rel_x_mapped"] = float((~r["x_category"].isin(BAD_CATS)).mean())
SUMMARY["rel_y_mapped"] = float((~r["y_category"].isin(BAD_CATS)).mean())
SUMMARY["rel_pair_mapped"] = float(r["pair_mapped"].mean())
pm = r[r["pair_mapped"]]
pairs = (pm.groupby(["x_construct", "y_construct"])
         .agg(n_rel=("filename", "size"), n_papers=("filename", "nunique"),
              share_supported=("rel_result", lambda x: (x == "supported").mean()),
              share_positive_pred=("rel_direction", lambda x: (x == "positive").mean()),
              share_moderation=("rel_type", lambda x: x.isin(["moderation", "interaction", "moderated", "three_way"]).mean()))
         .reset_index().sort_values(["n_papers", "n_rel"], ascending=False))
save(pairs.head(50), "02b_top50_construct_pairs")
save(pairs, "02b_all_construct_pairs")
famp = (pm.groupby(["x_family", "y_family"]).agg(n_rel=("filename", "size"), n_papers=("filename", "nunique"))
        .reset_index().sort_values(["n_papers", "n_rel"], ascending=False))
save(famp.head(50), "02b_top50_family_pairs")
# top DVs and IVs (construct level)
dvs = r[~r["y_category"].isin(BAD_CATS)].groupby("y_construct")["filename"].nunique().sort_values(ascending=False)
ivs = r[~r["x_category"].isin(BAD_CATS)].groupby("x_construct")["filename"].nunique().sort_values(ascending=False)
save(dvs.head(40).rename("n_papers").reset_index(), "02c_top_dv_constructs")
save(ivs.head(40).rename("n_papers").reset_index(), "02c_top_iv_constructs")
cat_pair = pd.crosstab(r.loc[r["pair_mapped"], "x_category"], r.loc[r["pair_mapped"], "y_category"])
save(cat_pair.reset_index(), "02d_category_pair_matrix")
rt = r["rel_type"].replace({"main_effect": "direct", "interaction": "moderation", "moderated": "moderation"})
rel_types = pd.DataFrame({"n": rt.value_counts(), "share": rt.value_counts(normalize=True)}).rename_axis("rel_type").reset_index()
save(rel_types, "02e_relationship_types")
rel_dir = r["rel_direction"].value_counts(normalize=True).rename("share").rename_axis("predicted_direction").reset_index()
save(rel_dir, "02e_predicted_direction")
SUMMARY["top_pairs"] = pairs.head(10)[["x_construct", "y_construct", "n_papers"]].values.tolist()

# ---------------------------------------------------------------------------
# 2b. Constructs and measures
# ---------------------------------------------------------------------------
dict_tab = pd.DataFrame([(k, val.get("domain"), len(val.get("aliases", []))) for k, val in cdict.items() if k != "_meta"],
                        columns=["construct", "domain", "n_aliases"])
vm = map_series(v["var_name"])
v = pd.concat([v, vm.add_prefix("h_")], axis=1)
use = (v[~v["h_category"].isin(BAD_CATS)].groupby("h_construct")
       .agg(n_papers=("filename", "nunique"), n_vars=("filename", "size"),
            share_dv=("var_role", lambda x: (x == "dv").mean()),
            share_focal_iv=("var_role", lambda x: (x == "focal_iv").mean()),
            share_moderator=("var_role", lambda x: (x == "moderator").mean()),
            share_control=("var_role", lambda x: (x == "control").mean()),
            share_binary=("measure_type", lambda x: (x == "binary").mean()),
            share_continuous=("measure_type", lambda x: (x == "continuous").mean()),
            share_count=("measure_type", lambda x: (x == "count").mean()),
            share_survey=("data_source_type", lambda x: (x == "survey").mean()),
            share_archival=("data_source_type", lambda x: (x == "archival").mean()))
       .reset_index().rename(columns={"h_construct": "construct"}).sort_values("n_papers", ascending=False))
cat_of = nmap.dropna(subset=["construct"]).drop_duplicates("construct").set_index("construct")["category"]
use["category"] = use["construct"].map(cat_of)
save(use, "03a_construct_usage_harmonized")
save(dict_tab.sort_values(["domain", "construct"]), "03b_construct_dictionary_entries")
SUMMARY["var_mapped_share"] = float((~v["h_category"].isin(BAD_CATS)).mean())
SUMMARY["dict_entries"] = len(dict_tab)
roles = v["var_role"].value_counts().head(7)
save(pd.crosstab(v["var_role"].where(v["var_role"].isin(roles.index)), v["measure_type"].where(v["measure_type"].isin(
    ["continuous", "binary", "count", "ratio", "index", "ordinal", "categorical"])), normalize="index").reset_index(),
    "03c_measure_type_by_role")
save(v["data_source_type"].value_counts(normalize=True).head(10).rename("share").rename_axis("data_source_type").reset_index(),
     "03d_variable_data_source_types")
mm = pd.DataFrame([
    ("share validated scale (all measured vars)", meas["validated_scale"].astype(str).eq("True").mean()),
    ("share adapted from existing", meas["adapted_from_existing"].astype(str).eq("True").mean()),
    ("item_count median (where reported)", meas["item_count"].median()),
    ("share single-item (where item_count reported)", (meas["item_count"] == 1).mean() / meas["item_count"].notna().mean()),
    ("n vars with alpha", meas["reliability_alpha"].notna().sum()),
    ("alpha p10", meas["reliability_alpha"].quantile(.1)),
    ("alpha median", meas["reliability_alpha"].median()),
    ("alpha p90", meas["reliability_alpha"].quantile(.9)),
    ("share alpha < .70", (meas["reliability_alpha"] < .70).sum() / meas["reliability_alpha"].notna().sum()),
], columns=["stat", "value"])
save(mm, "03e_measurement_quality")
save(meas["construct_category"].value_counts(normalize=True).head(14).rename("share").rename_axis("pass5_category").reset_index(),
     "03f_pass5_construct_categories")
# descriptives by family: medians of means/SDs (raw-scale generator seeds)
dm = map_series(desc["var_name"])
desc = pd.concat([desc, dm.add_prefix("h_")], axis=1)
dd = desc[(~desc["h_category"].isin(BAD_CATS)) & desc["mean"].notna()]
dfam = (dd.groupby("h_family").agg(n_papers=("file_name", "nunique"), n=("mean", "size"),
                                   mean_p25=("mean", lambda x: x.quantile(.25)), mean_med=("mean", "median"),
                                   mean_p75=("mean", lambda x: x.quantile(.75)), sd_med=("sd", "median"),
                                   cv_med=("coef_variation", "median"))
        .reset_index().sort_values("n_papers", ascending=False))
save(dfam.head(80), "03g_descriptives_by_family_top80")
SUMMARY["SUMMARY_alpha_median"] = float(meas["reliability_alpha"].median())

# ---------------------------------------------------------------------------
# 2c. Designs
# ---------------------------------------------------------------------------
E = emp
srcp = v.groupby("filename")["data_source_type"].agg(lambda x: set(x.dropna()))
src_tab = pd.DataFrame({k: srcp.map(lambda st, k=k: k in st) for k in
                        ["archival", "survey", "hand_collected", "experimental", "text_analysis", "administrative"]})
def src_class(st):
    if "experimental" in st: return "experiment"
    if "survey" in st and ("archival" in st or "hand_collected" in st): return "survey+archival"
    if "survey" in st: return "survey_only"
    if "hand_collected" in st: return "archival+hand_collected"
    if "archival" in st: return "archival_only"
    return "other/unknown"
srcc = srcp.map(src_class).value_counts(normalize=True).rename("share").rename_axis("paper_data_design").reset_index()
save(srcc, "04a_paper_data_source_design")
save(src_tab.mean().rename("share_papers_using").rename_axis("data_source_type").reset_index(), "04a_paper_data_source_any")
SUMMARY["data_design"] = srcc.set_index("paper_data_design")["share"].round(3).to_dict()

panel = E["sample_panel_structure"].where(E["sample_panel_structure"].isin(
    ["unbalanced_panel", "cross_section", "pooled_cross_section", "balanced_panel", "time_series", "not_specified"]), "other")
save(panel.value_counts(normalize=True).rename("share").rename_axis("panel_structure").reset_index(), "04b_panel_structure")
unit = E["sample_unit_of_analysis"].where(E["sample_unit_of_analysis"].isin(
    ["firm-year", "firm", "individual", "dyad", "event", "team", "industry", "country"]), "other")
save(unit.value_counts(normalize=True).rename("share").rename_axis("unit_of_analysis").reset_index(), "04b_unit_of_analysis")
ss = []
ss.append({"group": "all quant+mixed (sample_size_final)", **qtab(E["sample_size_final"])})
for u in ["firm-year", "firm", "individual", "dyad", "event"]:
    ss.append({"group": f"unit={u}", **qtab(E.loc[unit == u, "sample_size_final"])})
for p in ["unbalanced_panel", "cross_section", "pooled_cross_section"]:
    ss.append({"group": f"panel={p}", **qtab(E.loc[panel == p, "sample_size_final"])})
ss.append({"group": "firms (sample_size_firms)", **qtab(E["sample_size_firms"])})
ss.append({"group": "panel length years (sample_period_length)", **qtab(E["sample_period_length"])})
ss.append({"group": "relationship-level N (observations_in_model)", **qtab(r["observations_in_model"])})
ssq = pd.DataFrame(ss)
save(ssq, "04c_sample_size_quantiles")
SUMMARY["N_quant"] = ssq.iloc[0].to_dict()

est = E["primary_estimation_method"].fillna("missing")
top_est = est.value_counts()
est_tab = pd.DataFrame({"n_papers": top_est, "share": top_est / len(E)}).rename_axis("primary_estimation_method").reset_index()
save(est_tab, "04d_estimator_shares")
save(pd.crosstab(E["era"], E["method_family"], normalize="index").reset_index(), "04d_method_family_by_era")
SUMMARY["estimators"] = est_tab.head(8).set_index("primary_estimation_method")["share"].round(3).to_dict()

def fe_class(x):
    try: L = [t.lower() for t in json.loads(x)]
    except Exception: L = []
    if not L: return "none_reported"
    has = lambda k: any(k in t for t in L)
    if has("firm") and (has("year") or has("time")): return "firm+year (two-way)"
    if has("firm"): return "firm only"
    if has("industry") and has("year"): return "industry+year"
    if has("year") or has("time"): return "year only"
    if has("industry"): return "industry only"
    return "other"
save(E["fixed_effects_type"].map(fe_class).value_counts(normalize=True).rename("share").rename_axis("fixed_effects").reset_index(), "04e_fixed_effects")

def cl_class(x):
    x = str(x).lower()
    if x in ("nan", "") or x.startswith("not specified"):
        return "robust_not_clustered" if "robust" in x else "not_specified"
    for k in ["firm", "industry", "state", "country", "individual", "dyad", "year", "team"]:
        if k in x: return k
    return "other"
save(E["standard_errors_clustering"].map(cl_class).value_counts(normalize=True).rename("share").rename_axis("clustering").reset_index(), "04e_se_clustering")

ctrl_rel = r.groupby("filename")["control_variable_count"].median()
ctrl_var = v[v["var_role"] == "control"].groupby("filename").size()
ctrl = pd.DataFrame([{"measure": "controls per focal model (relationship rows)", **qtab(r["control_variable_count"])},
                     {"measure": "controls per paper (median over its models)", **qtab(ctrl_rel)},
                     {"measure": "control variables listed per paper (rep_variables)", **qtab(ctrl_var)}])
save(ctrl, "04f_controls_counts")
SUMMARY["controls"] = ctrl.iloc[1][["p25", "p50", "p75"]].to_dict()
top_controls = (v[(v["var_role"] == "control") & ~v["h_category"].isin(["interaction", ""])]
                .groupby("h_family")["filename"].nunique().sort_values(ascending=False).head(30))
save((top_controls / v.loc[v["var_role"] == "control", "filename"].nunique()).rename("share_of_papers_with_controls")
     .rename_axis("control_family").reset_index(), "04f_top_control_families")

rob = pd.DataFrame([{"practice": k, "share_papers": E[k].astype(str).eq("True").mean()} for k in
                    ["placebo_test", "subsample_analysis", "alternative_dv", "alternative_iv", "alternative_estimation",
                     "alternative_sample", "robustness_any_conclusion_changed", "robustness_all_confirmed",
                     "lagged_iv", "lagged_dv_included", "winsorization_used", "multicollinearity_check", "cmv_test",
                     "exact_p_values_reported", "effect_sizes_reported", "marginal_significance_invoked",
                     "preregistered", "endogeneity_discussed"]])
save(rob, "04g_robustness_and_reporting_practices")
save(pd.DataFrame([qtab(E["robustness_checks_count"]), qtab(E["model_specifications_count"]), qtab(E["interaction_terms_count"])],
                  index=["robustness_checks_count", "model_specifications_count", "interaction_terms_count"]).reset_index(),
     "04g_robustness_counts")
endo = E["endogeneity_strategy"].fillna("missing").value_counts(normalize=True).head(16)
save(endo.rename("share").rename_axis("endogeneity_strategy").reset_index(), "04h_causal_inference_strategy")
save(E["design_supports_causation"].value_counts(normalize=True).rename("share").reset_index(), "04h_design_supports_causation")
save(pd.crosstab(E["era"], E["endogeneity_strategy"].fillna("missing"), normalize="index").reset_index(), "04h_causal_strategy_by_era")
SUMMARY["endo"] = endo.head(8).round(3).to_dict()

# ---------------------------------------------------------------------------
# 2d. Effect sizes (partial r from t)
# ---------------------------------------------------------------------------
num = lambda x: pd.to_numeric(x, errors="coerce")
r["t_ver"] = r["coef_verified"] / r["se_verified"]
r["t_rep"] = num(r["t_stat_or_z_stat"])
r["t_llm"] = num(r["coefficient_value"]) / num(r["standard_error_value"])
r["t_use"] = r["t_ver"].where(r["t_ver"].notna() & r["confidence_tier_v3"].isin(["high", "medium"]))
r["t_src"] = np.where(r["t_use"].notna(), "verified_coef_se", "")
fill = r["t_use"].isna() & r["t_rep"].notna()
r.loc[fill, "t_use"] = r.loc[fill, "t_rep"]; r.loc[fill, "t_src"] = "reported_t_or_z"
r.loc[np.isinf(r["t_use"]), "t_use"] = np.nan
k = r["control_variable_count"].fillna(r["control_variable_count"].median())
r["df"] = (r["observations_in_model"] - k - 2).clip(lower=3)
r["partial_r"] = r["t_use"] / np.sqrt(r["t_use"] ** 2 + r["df"])
sgn = np.where(r["rel_direction"] == "negative", -1, np.where(r["rel_direction"] == "positive", 1, np.nan))
r["partial_r_pred_dir"] = r["partial_r"] * sgn            # + = in hypothesized direction
r["abs_t"] = r["t_use"].abs()
foc = r[r["partial_r"].notna() & r["observations_in_model"].notna() & (r["abs_t"] < 100)]
foc["rt"] = foc["rel_type"].replace({"main_effect": "direct", "interaction": "moderation", "moderated": "moderation"})
es = []
for lab, d in [("all focal rows with t and N", foc),
               ("direct effects", foc[foc["rt"] == "direct"]),
               ("moderation effects", foc[foc["rt"] == "moderation"]),
               ("curvilinear", foc[foc["rt"] == "curvilinear"]),
               ("direct, tier high only (verified coef/SE)", foc[(foc["rt"] == "direct") & (foc["confidence_tier_v3"] == "high") & (foc["t_src"] == "verified_coef_se")]),
               ("direct, N<500", foc[(foc["rt"] == "direct") & (foc["observations_in_model"] < 500)]),
               ("direct, 500<=N<5000", foc[(foc["rt"] == "direct") & foc["observations_in_model"].between(500, 4999)]),
               ("direct, N>=5000", foc[(foc["rt"] == "direct") & (foc["observations_in_model"] >= 5000)])]:
    es.append({"subset": lab, "stat": "|partial r|", **qtab(d["partial_r"].abs())})
    es.append({"subset": lab, "stat": "partial r in predicted direction", **qtab(d["partial_r_pred_dir"])})
    es.append({"subset": lab, "stat": "|t|", **qtab(d["abs_t"])})
est_tab2 = pd.DataFrame(es)
save(est_tab2, "05a_partial_r_distributions")
# histogram (sampling-ready) of signed-in-predicted-direction partial r for direct effects
dd_ = foc.loc[foc["rt"] == "direct", "partial_r_pred_dir"].dropna()
bins = [-1, -.3, -.2, -.1, -.05, -.02, 0, .02, .05, .1, .15, .2, .3, .5, 1]
h = pd.cut(dd_, bins).value_counts(normalize=True).sort_index()
save(h.rename("share").rename_axis("partial_r_bin").reset_index().astype({"partial_r_bin": str}), "05b_partial_r_pred_dir_hist_direct")
# small-study / inflation signature: |r| vs 1/sqrt(N)
dsub = foc[foc["rt"] == "direct"].dropna(subset=["partial_r_pred_dir"])
dsub["inv_sqrt_n"] = 1 / np.sqrt(dsub["observations_in_model"])
slope = np.polyfit(dsub["inv_sqrt_n"], dsub["partial_r_pred_dir"], 1)
corr_ss = np.corrcoef(dsub["inv_sqrt_n"], dsub["partial_r_pred_dir"])[0, 1]
# |t| ~ independent of N would indicate N-driven targeting of significance
corr_tN = np.corrcoef(np.log(dsub["observations_in_model"]), dsub["t_use"].abs())[0, 1]
sup = foc[foc["rel_result"].isin(["supported", "not_supported", "partially_supported"])]
by_sup = sup.groupby("rel_result")["partial_r_pred_dir"].describe()
save(by_sup.reset_index(), "05c_partial_r_by_support_verdict")
SUMMARY["es"] = {"n_focal_t": int(len(foc)), "n_direct": int((foc["rt"] == "direct").sum()),
                 "abs_r_direct_med": float(foc.loc[foc["rt"] == "direct", "partial_r"].abs().median()),
                 "abs_r_direct_p25": float(foc.loc[foc["rt"] == "direct", "partial_r"].abs().quantile(.25)),
                 "abs_r_direct_p75": float(foc.loc[foc["rt"] == "direct", "partial_r"].abs().quantile(.75)),
                 "abs_r_mod_med": float(foc.loc[foc["rt"] == "moderation", "partial_r"].abs().median()),
                 "share_wrong_sign_direct": float((dd_ < 0).mean()),
                 "egger_slope": float(slope[0]), "egger_intercept": float(slope[1]), "corr_r_invsqrtN": float(corr_ss),
                 "corr_abs_t_logN": float(corr_tN), "t_src": foc["t_src"].value_counts().to_dict()}
# effect sizes per top construct pair
fp = foc[foc["pair_mapped"] & (foc["rt"] == "direct")]
pes = (fp.groupby(["x_construct", "y_construct"])
       .agg(n_rel=("partial_r", "size"), n_papers=("filename", "nunique"),
            med_partial_r_pred=("partial_r_pred_dir", "median"), med_abs_partial_r=("partial_r", lambda x: x.abs().median()),
            share_supported=("rel_result", lambda x: (x == "supported").mean()))
       .reset_index().query("n_papers>=3").sort_values("n_papers", ascending=False))
save(pes, "05d_partial_r_by_construct_pair_direct")

# ---------------------------------------------------------------------------
# 2e. Correlation structures
# ---------------------------------------------------------------------------
cm1 = map_series(c_ok["var1_name"]).add_prefix("a_")
cm2 = map_series(c_ok["var2_name"]).add_prefix("b_")
cc = pd.concat([c_ok, cm1, cm2], axis=1)
cc = cc[(~cc["a_category"].isin(BAD_CATS)) & (~cc["b_category"].isin(BAD_CATS))
        & (cc["a_construct"] != "") & (cc["b_construct"] != "")]
SUMMARY["corr_cells_mapped_share"] = len(cc) / len(c_ok)
SUMMARY["corr_abs_r"] = qtab(c_ok["correlation_r"].abs())
save(pd.DataFrame([{"set": "all cells final_risk<3", **qtab(c_ok["correlation_r"].abs())},
                   {"set": "signed r, all cells final_risk<3", **qtab(c_ok["correlation_r"])}]), "06a_abs_r_distribution_all_cells")
nvars = c_ok.groupby(["file_name", "table_label"]).apply(lambda g: len(set(g["var1_name"]) | set(g["var2_name"])))
save(pd.DataFrame([qtab(nvars)]), "06a_variables_per_correlation_table")
# paper-level average within construct pair (a paper may have several measures of one construct)
cc["lo"] = np.where(cc["a_construct"] <= cc["b_construct"], cc["a_construct"], cc["b_construct"])
cc["hi"] = np.where(cc["a_construct"] <= cc["b_construct"], cc["b_construct"], cc["a_construct"])
cc = cc[cc["lo"] != cc["hi"]]
cc["z"] = np.arctanh(cc["correlation_r"].clip(-.999, .999))
pp = cc.groupby(["lo", "hi", "file_name"]).agg(z=("z", "mean"), N=("sample_size", "median")).reset_index()
pp["w"] = (pp["N"].fillna(pp["N"].median()).clip(lower=10, upper=1e5) - 3)
def agg_pair(g):
    w = g["w"]; z = g["z"]
    zbar = np.average(z, weights=w)
    return pd.Series({"n_papers": len(g), "r_median": np.tanh(z.median()), "r_wmean": np.tanh(zbar),
                      "r_p25": np.tanh(z.quantile(.25)), "r_p75": np.tanh(z.quantile(.75)),
                      "sd_r_across_papers": np.tanh(z).std(), "median_N": g["N"].median()})
pair_r = pp.groupby(["lo", "hi"]).apply(agg_pair).reset_index().rename(columns={"lo": "construct_a", "hi": "construct_b"})
pair_r = pair_r.sort_values("n_papers", ascending=False)
save(pair_r[pair_r["n_papers"] >= 5], "06b_construct_pair_correlations")
# pooled matrix among the 25 constructs most present in correlation tables
pres = pd.concat([cc[["file_name", "a_construct"]].rename(columns={"a_construct": "k"}),
                  cc[["file_name", "b_construct"]].rename(columns={"b_construct": "k"})]).drop_duplicates()
topk = pres["k"].value_counts().head(25).index.tolist()
M = pd.DataFrame(np.eye(len(topk)), index=topk, columns=topk)
Nn = pd.DataFrame(0, index=topk, columns=topk)
lk = pair_r.set_index(["construct_a", "construct_b"])
for i, a in enumerate(topk):
    for b in topk[i + 1:]:
        key = (a, b) if a <= b else (b, a)
        if key in lk.index:
            M.loc[a, b] = M.loc[b, a] = lk.loc[key, "r_wmean"]
            Nn.loc[a, b] = Nn.loc[b, a] = lk.loc[key, "n_papers"]
        else:
            M.loc[a, b] = M.loc[b, a] = np.nan
missing = int(M.isna().sum().sum() / 2)
Mf = M.fillna(0.0)
ev = np.linalg.eigvalsh(Mf.values)
# nearest-PSD by eigenvalue clipping + rescale to unit diagonal
w_, V = np.linalg.eigh(Mf.values)
Mp = V @ np.diag(np.clip(w_, 1e-3, None)) @ V.T
dg = np.sqrt(np.diag(Mp)); Mp = Mp / np.outer(dg, dg)
save(M.round(3).reset_index().rename(columns={"index": "construct"}), "06c_pooled_corr_matrix_top25_raw")
save(pd.DataFrame(Mp, index=topk, columns=topk).round(3).reset_index().rename(columns={"index": "construct"}), "06c_pooled_corr_matrix_top25_psd")
save(Nn.reset_index().rename(columns={"index": "construct"}), "06c_pooled_corr_matrix_top25_npapers")
SUMMARY["corr_matrix"] = {"k": len(topk), "missing_pairs": missing, "min_eig_raw": float(ev.min()),
                          "max_abs_change_psd": float(np.nanmax(np.abs(Mp - Mf.values))), "topk": topk}
# heterogeneity: across-paper SD of r for the 30 most-replicated pairs
het = pair_r.head(30)
SUMMARY["corr_het_median_sd"] = float(het["sd_r_across_papers"].median())
# controls-vs-focal correlation: focal IV-DV pairs vs others
SUMMARY["corr_pairs_n5"] = int((pair_r["n_papers"] >= 5).sum())

# ---------------------------------------------------------------------------
# 2f. Reporting practice
# ---------------------------------------------------------------------------
hyp = E["hypotheses_total"]
hdist = hyp.value_counts().sort_index()
save(pd.DataFrame({"n_hypotheses": hdist.index, "n_papers": hdist.values, "share": hdist.values / hdist.sum()}), "07a_hypotheses_per_paper")
save(pd.DataFrame([qtab(hyp[hyp > 0])]), "07a_hypotheses_per_paper_quantiles")
sr = s.loc[(s["hypotheses_total"] > 0) & s["support_rate"].notna(), "support_rate"]
srb = pd.cut(sr, [-0.001, 0, .25, .5, .75, .999, 1.0], labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1.0"]).value_counts(normalize=True).sort_index()
save(srb.rename("share").rename_axis("support_rate_bin").reset_index(), "07b_support_rate_distribution")
SUMMARY["support"] = {"n": int(len(sr)), "share_100": float((sr == 1).mean()), "mean": float(sr.mean()), "median": float(sr.median()),
                      "share_0": float((sr == 0).mean()), "hyp_median": float(hyp[hyp > 0].median()),
                      "hyp_mean": float(hyp[hyp > 0].mean()),
                      "rel_supported": float((rel_hyp["rel_result"] == "supported").mean()),
                      "rel_partial": float((rel_hyp["rel_result"] == "partially_supported").mean()),
                      "rel_not": float((rel_hyp["rel_result"] == "not_supported").mean())}
save(pd.crosstab(s.loc[sr.index, "era"], (sr == 1).rename("all_supported"), normalize="index").reset_index(), "07b_full_support_by_era")
# support by relationship type
save(pd.crosstab(rt[rel_hyp.index], rel_hyp["rel_result"], normalize="index").reset_index(), "07c_support_by_relationship_type")
# p-value bins and caliper
pb = r["p_value_bin_recomputed"].value_counts(normalize=True)
save(pb.rename("share").rename_axis("p_bin").reset_index(), "07d_p_value_bins_recomputed")
def caliper(p, lo, mid, hi):
    a = ((p >= lo) & (p < mid)).sum(); b = ((p >= mid) & (p < hi)).sum()
    return a, b, (a / b if b else np.nan)
cal = []
for lab, p in [("p_verified, tier high", r.loc[r["confidence_tier_v3"] == "high", "p_verified"]),
               ("p_verified, tier high+medium", r.loc[r["confidence_tier_v3"].isin(["high", "medium"]), "p_verified"]),
               ("p_value_recomputed (all, incl. unverified)", r["p_value_recomputed"])]:
    for (lo, mid, hi) in [(.04, .05, .06), (.045, .05, .055), (.03, .05, .07), (.005, .01, .015), (.09, .10, .11)]:
        a, b, ratio = caliper(p.dropna(), lo, mid, hi)
        cal.append({"p_source": lab, "window": f"[{lo},{mid}) vs [{mid},{hi})", "n_below": a, "n_above": b, "ratio": ratio})
for lab, d in [("|t| verified coef/SE, tier high", r.loc[(r["confidence_tier_v3"] == "high") & (r["t_src"] == "verified_coef_se"), "t_use"]),
               ("|t| verified coef/SE, tier high+medium", r.loc[r["t_src"] == "verified_coef_se", "t_use"])]:
    z = d.abs().dropna()
    a = ((z >= 1.96) & (z < 2.16)).sum(); b = ((z >= 1.76) & (z < 1.96)).sum()
    cal.append({"p_source": lab, "window": "z-caliper [1.96,2.16) vs [1.76,1.96)", "n_below": a, "n_above": b, "ratio": a / b if b else np.nan})
cal = pd.DataFrame(cal)
save(cal, "07e_caliper_tests")
SUMMARY["caliper"] = cal.head(5).to_dict("records")
# p-value histogram (verified, high+medium) for generator validation
pv = r.loc[r["confidence_tier_v3"].isin(["high", "medium"]), "p_verified"].dropna()
ph = pd.cut(pv, [0, .001, .005, .01, .02, .03, .04, .05, .06, .07, .08, .1, .2, .5, 1.0], include_lowest=True).value_counts(normalize=True).sort_index()
save(ph.rename("share").rename_axis("p_bin").reset_index().astype({"p_bin": str}), "07f_p_verified_histogram")
zt = foc["abs_t"]
zh = pd.cut(zt, [0, 1, 1.645, 1.96, 2.33, 2.58, 3.29, 4, 6, 10, 100], include_lowest=True).value_counts(normalize=True).sort_index()
save(zh.rename("share").rename_axis("abs_t_bin").reset_index().astype({"abs_t_bin": str}), "07g_abs_t_histogram_focal")
SUMMARY["share_p_lt05_focal"] = float((zt > 1.96).mean())
SUMMARY["practices"] = rob.set_index("practice")["share_papers"].round(3).to_dict()

json.dump(SUMMARY, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(SUMMARY, indent=1, default=str)[:12000])
