# Grounding Inventory for the Synthetic Strategy-Research World

*Built 2026-10-13. Scripts: `build_grounding_inventory.py` (SMJ deep extraction) and `build_fourjournal_inventory.py` (four journals). All sampling tables are in `tables/`. SMJ tables are named `NN_*.csv` and four-journal tables `4j_NN_*.csv`. Headline numbers are in `tables/summary.json` and `tables/4j_summary.json`.*

**Sources (read-only; nothing was written to either source; neither is distributed with this repository, see DATA.md):**

- **SMJ deep extraction (S1).** The SMJ article extraction's analysis-grade files, frozen for this project under `corpus/smj-deep-frozen-2026-10-13/` (hashes in its `SHA256SUMS.txt`): `rep_study_clean`, `rep_relationships_clean_v3` (filtered on `confidence_tier_v3`), `rep_correlations_clean_v2` (filtered on `final_risk < 3`), `rep_descriptives_clean`, `rep_measurement_clean` and `rep_variables_clean`. Variable names were mapped to constructs with the extraction's own harmonization layer: `name_map_v1.csv` plus the `harmonize.py` rulebook (frozen under `corpus/smj-deep-frozen-supplement-2026-10-19/`), imported without writing bytecode. That layer works as a ladder: exact dictionary alias, then curated family regex.
- **Four-journal extraction (S2).** A keyword-level extraction of AMJ, ASQ, OS and SMJ articles. Section 8 lists its input files; `build_fourjournal_inventory.py` takes the folder as `--fourjournal-dir`.

---

## 1. Coverage

### 1.1 SMJ deep extraction (S1)

S1 holds **2,358 unique SMJ articles from 1997 to 2025** (no 1999 issues). Use the filename year: the `year` column is filled for only 898 rows. By year and type, see `01a_articles_by_year_type.csv`.

| Paper type | n |
|---|---:|
| Quantitative empirical | 1,709 |
| Mixed methods | 154 |
| Qualitative | 98 |
| Conceptual + review | 196 |
| Methodology, simulation, meta-analysis | 176 |
| Unknown | 25 |

Usable layers (`01c_usable_coverage.csv`):

| Layer | Volume | Papers |
|---|---:|---:|
| Papers with ≥1 stated hypothesis | — | 1,773 |
| Hypothesis rows with a support verdict | 10,675 | 1,870 |
| Focal coefficients, **high** tier | 3,816 | 903 |
| Focal coefficients, high + medium tier | 7,390 | 1,517 |
| Rows with verified coefficient and SE (t computable) | 4,475 | — |
| Correlation cells with `final_risk < 3` | 179,407 | 1,451 |
| … of which verified by ≥2 channels or adjudicated | 125,941 | — |
| Descriptives (mean/SD), high confidence | 16,341 | 1,030 |
| Variable rows | 29,519 | — |
| Measurement rows | 22,603 | — |

### 1.2 Four journals (S2)

S2 holds **6,732 articles**: SMJ 2,597, OS 1,905, AMJ 1,760 and ASQ 470.

The year comes from the issue folder (`<journal>\YYYY-I\`) or from the SMJ filename. **Do not use `meta_year`**: 1,308 rows default to 2007 and 714 to 1996. The resolved range is 1998–2025, with 61 SMJ rows unresolved.

| Period | AMJ | ASQ | OS | SMJ |
|---|---:|---:|---:|---:|
| ≤1999 | 44 | 23 | 45 | 192 |
| 2000–04 | 270 | 96 | 218 | 348 |
| 2005–09 | 306 | 84 | 254 | 368 |
| 2010–14 | 329 | 66 | 451 | 533 |
| 2015–19 | 419 | 69 | 371 | 465 |
| 2020–25 | 392 | 132 | 566 | 630 |

`4j_01_coverage_journal_year.csv` has the per-year counts. S1 and S2 overlap on SMJ: 2,333 of the 2,358 S1 articles matched an S2 row by normalised filename. That overlap is the basis for the validity crosswalk in section 3.6.

---

## 2. What S1 (SMJ deep) can seed

### (a) Research topics and agendas

**Theory families.** I regex-mapped `theories_invoked` into 39 families (`02a_theory_families.csv`). Of the 1,801 papers with theories listed:

| Theory family | Share of papers |
|---|---:|
| Resource-based view | 37% |
| Agency | 18% |
| Transaction cost economics | 18% |
| Organizational learning | 12% |
| Institutional | 12% |
| Social networks | 11% |
| Upper echelons | 10% |
| Knowledge-based view | 10% |
| Dynamic capabilities | 8% |
| Behavioral theory of the firm | 8% |

Papers invoke a median of 5 theories (IQR 4–6).

**Construct pairs.** I mapped relationship X and Y variables to the 189 harmonized constructs.

- The most common DVs are accounting performance (275 papers), market performance (196), innovation output (170), survival/exit (116) and M&A/corporate transactions (108).
- The most common IVs are experience/learning, innovation output, diversification, M&A, accounting performance and alliances.

The top-50 pairs, with support rates, are in `02b_top50_construct_pairs.csv` (full list in `02b_all_*`; family level in `02b_top50_family_pairs.csv`; category × category in `02d`). Examples:

- M&A → market performance (19 papers)
- diversification → ROA (16)
- R&D → innovation output (14)
- diversification → Tobin's q (13)
- internationalization → ROA (10)
- alliances → patents (10)
- executive pay ↔ performance (9)

Same-construct pairs such as patents → citations are real within-construct family links.

**Relationship types** (`02e_*`):

- 62% direct, 29% moderation, 3.3% curvilinear, 1.7% three-way, 1.5% mediation.
- Predicted direction: 60% positive, 31% negative, about 3% inverted-U.

**Caveat on mapping:** only 26% of relationship rows map on both sides (39% of X, 56% of Y). Focal IVs are idiosyncratic and coined, while DVs are standard. The construct-pair tables are therefore a *sample of the agenda*, not a census.

### (b) Construct families and measures

- **Dictionary.** `construct-dictionary.json` has 233 entries, although its `_meta` says 192, across 16 domains (`03b_*`). The harmonization layer adds 256 name-families and 15 categories.
- **Usage by construct** (`03a_construct_usage_harmonized.csv`) gives papers, role shares (DV, focal IV, moderator, control), measure-type shares and source shares. It is the per-construct "how is this usually measured and used" seed. Firm size, for example, appears in 883 papers and is 86% control, 82% continuous and 80% archival.
- **Measure type by role** (`03c`): about 45% continuous, 20–25% binary, about 12% count, 6–10% ratio, and indexes are common among focal IVs (11%).
- **Data source of variables** (`03d`): 70% archival, 14% survey, 10% hand-collected.
- **Measurement quality** (`03e`, Pass 5):
  - Validated scales cover only 5% of variables.
  - 72% are single-item where item counts are known.
  - Cronbach's α (n = 903) has median .81, p10 .70 and 9% below .70.
- **Raw-scale seeds** (`03g_descriptives_by_family_top80.csv`) give the median mean, IQR of means, median SD and CV per variable family. Examples: ln(size) median mean 7.3 with SD 1.6; ROA median mean .07 with SD .22. **Watch the scale mixture:** ROA is recorded both as a fraction and as a percent.

### (c) Designs

Base: quant + mixed papers, n = 1,863. Tables `04*`.

- **Data design:**

  | Data design | Share |
  |---|---:|
  | Archival only | 44% |
  | Archival + hand-collected | 28% |
  | Survey + archival | 13% |
  | Survey only | 7% |
  | Experiment | 5% |

  84% use some archival data; 23% use a survey.
- **Structure:**
  - Panel: unbalanced 48%, cross-section 31%, pooled 7%, balanced 2%.
  - Unit: firm-year 28%, firm 21%, individual 10%, event 8%, dyad 8%.
- **Sample size** (`04c_sample_size_quantiles.csv`):

  | Group | Median N | p10 | p25 | p75 | p90 |
  |---|---:|---:|---:|---:|---:|
  | All papers | 845 | 105 | 227 | 7,340 | 45,841 |
  | Firm-year | 2,023 | — | — | — | — |
  | Cross-section | 230 | — | — | — | — |
  | Firm | 235 | — | — | — | — |
  | Event | 543 | — | — | — | — |

  Panel length has a median of 9 years. Model-level N has a median of 851.
- **Estimators** (`04d_estimator_shares.csv`; trend by era in `04d_method_family_by_era.csv`):

  | Estimator | Share |
  |---|---:|
  | OLS | 20% |
  | Fixed effects | 9% |
  | Logistic | 7% |
  | Negative binomial | 4% |
  | Diff-in-diff | 3.5% |
  | Parametric survival | 3.3% |
  | SEM | 3.2% |
  | Random effects | 3% |
  | IV/2SLS | 2.8% |
  | Cox | 2.7% |
  | HLM | 2.3% |

  DID rises from 0% before 2010 to 14% of method families in 2020–25.
- **Fixed effects:** 41% report none, 16% year only, 14% firm + year, 9% industry + year.
- **SE clustering:** 57% do not specify; 16% cluster by firm.
- **Controls** (`04f_controls_counts.csv`): a median of **8 per model** (IQR 4–11; p90 15). The most common control families are size (40% of papers), age (16%), R&D intensity (16%), experience (15%), leverage (15%) and ROA (12%). The top 30 are in `04f_top_control_families.csv`.
- **Robustness** (`04g_*`): a median of 4 checks per paper and 6 model specifications.
  - Subsample analysis 56%, alternative estimator 54%, alternative IV 47%, alternative sample 42%, alternative DV 41%, placebo 7%.
  - "All robustness checks confirm": 84%. "A conclusion changed": 6%.
- **Causal inference** (`04h_*`):
  - Endogeneity: not discussed 31%, multiple strategies 15%, acknowledged but not addressed 10%, fixed effects 7%, IV 7%, Heckman 7%, natural experiment 4.5%.
  - Design: observational with correction 45%, purely observational 36%, quasi-experimental 15%, experimental 3%.

### (d) Effect-size distributions

Method: partial r = t / √(t² + df), with df = N − k − 2. t comes from the verified coefficient/SE in the high or medium tier (4,127 rows), with the reported t/z as fallback (881 rows). Focal rows with t and N: n = 5,008 (`05a_partial_r_distributions.csv`).

| Subset | n | Median \|partial r\| | IQR | Median r in predicted direction |
|---|---:|---:|---|---:|
| Direct effects | 3,086 | **.084** | .030–.193 | .068 |
| Moderation effects | 1,519 | .053 | .017–.134 | .030 |
| Direct, high tier only | 1,460 | .071 | .020–.171 | .059 |
| Direct, N < 500 | 1,320 | **.178** | .101–.292 | .168 |
| Direct, 500 ≤ N < 5,000 | 982 | .074 | .042–.120 | .064 |
| Direct, N ≥ 5,000 | 784 | **.022** | .008–.044 | .021 |

- **Wrong sign:** 15% of direct effects have the opposite sign to the hypothesis. The sampling histogram is `05b_partial_r_pred_dir_hist_direct.csv`.
- **Small-study signature:** regressing signed r on 1/√N gives slope 1.35 and intercept .046 (r = .29). |t| barely rises with N (ρ(|t|, log N) = .18). Effect sizes are therefore N-dependent in the way that selection or targeting of significance produces.
- **By verdict** (`05c`): the median r is .069 for "supported", .059 for "partial" and .001 for "not supported".
- **By construct pair** (pairs in ≥3 papers): `05d_partial_r_by_construct_pair_direct.csv`.
- **Caution:** partial r from logit/count z is an approximation, and df ignores clustering.

### (e) Correlation structures

- **All cells** (`06a`; 179,407 cells with `final_risk < 3`): median |r| .08, p75 .18, p90 .34, p95 .48. Tables have a median of 12 variables (IQR 7–17).
- **Construct-pair meta-correlations** (`06b_construct_pair_correlations.csv`): 652 pairs observed in ≥5 papers. Each paper-level average is Fisher-z pooled, weighted by N. The table gives the median, weighted mean, IQR, **between-paper SD** and median N. Examples:

  | Pair | Papers | Weighted r | Median r |
  |---|---:|---:|---:|
  | Size–age | 183 | .23 | .23 |
  | Size–ROA | 281 | .14 | .05 |
  | Size–diversification | 129 | .28 | .14 |
  | Leverage–ROA | 150 | −.12 | −.08 |
  | R&D–patents | 104 | .20 | .08 |

  The median between-paper SD of r across the 30 most-replicated pairs is **.18**, so heterogeneity is large. A generator should draw a paper-specific matrix around the pooled mean rather than reuse the pooled mean itself.
- **Pooled 25 × 25 construct matrix** (`06c_pooled_corr_matrix_top25_raw/psd/npapers.csv`) covers firm size, accounting and market performance, R&D, patents, experience, CEO characteristics, age, leverage, diversification, environment, ownership, M&A, alliances, board, internationalization, compensation, executive demographics, investment, human capital, network position, competition, industry structure and reputation.
  - Every pair is observed.
  - The raw matrix is already positive definite (minimum eigenvalue .30), so the PSD version is identical.
- **Caveats:**
  - Signs are only meaningful within an operationalization family. Leverage and equity ratio, for example, flip sign.
  - Only 26% of cells map to constructs on both sides.

### (f) Reporting practice

- **Hypotheses per paper** (`07a`): median 5, mean 5.7. The modes are 4 (18%), 6 (14%), 3 (15%) and 5 (13%); 9% of papers state ≥10.
- **Support rate per paper** (`07b`; n = 1,772): mean .78, median .80.
  - **40.6% of papers support 100% of their hypotheses** (the extraction's own reported figure is 40.1%).
  - 1.6% support none.
  - Full support rises by era: 19% (≤1999), 30% (2000s), 43% (2010s), **51% (2020–25)**.
- **Hypothesis-level verdicts:** 74.5% supported, 8.0% partially, 17.5% not supported. This is flat across direct, moderation and curvilinear relationships (`07c`).
- **p-values** (`07d`, `07f`; verified high + medium tier): 36% of p ≤ .001, and **65.8% of focal |t| exceed 1.96** (`07g` histogram).
- **Caliper bunching at .05** (`07e_caliper_tests.csv`):

  | p source | Caliper | Ratio |
  |---|---|---:|
  | Verified, high tier | Fine, [.04,.05) / [.05,.06) | **3.35** (104/31) |
  | Verified, high + medium | Fine | **2.23** (183/82) |
  | Verified, high + medium | z-caliper (1.96 ± .2) | 1.89 |
  | Verified, high tier | z-caliper (1.96 ± .2) | 2.15 |
  | Recomputed, all rows | Fine | 1.85 |
  | Placebo threshold | Ratio at .01 | ≈1.0–1.17 |

  The extraction's own headline estimate is ~1.9 on verified p-values versus 4.30 under reported stars. So about half of the raw bunching was extraction or reporting artefact, but a genuine ~1.9–2.3× excess remains.
- **Other practices** (`04g`): exact p reported 38%, effect sizes reported 46%, "marginal significance" invoked 41%, preregistration 0.6%.

---

## 3. What the four-journal data adds (S2)

**Validity warning first.** S2 is a *keyword/dictionary* extraction (Phase 1/2), not the multi-channel verified S1 pipeline. The SMJ crosswalk (section 3.6) shows which fields are trustworthy. Use S2 for **journal-level marginals and metadata**, not for paper-level design truth.

### 3.1 Theory families by journal

Source: `4j_02_theory_share_by_journal.csv`, plus dominant theory, theories per paper, primary subfield and level of theory.

| Theory | AMJ | ASQ | OS | SMJ |
|---|---:|---:|---:|---:|
| RBV | .08 | .04 | .08 | **.24** |
| Organizational learning | .10 | .08 | **.16** | .12 |
| Institutional | .12 | **.17** | .12 | .08 |
| TCE | .05 | .03 | .07 | **.13** |
| Agency | .07 | .04 | .06 | **.12** |
| Dynamic capabilities | .03 | .02 | .07 | **.13** |
| Sensemaking | **.11** | **.11** | .09 | .03 |
| Social identity | .09 | **.11** | .08 | .02 |
| Social networks | .05 | **.09** | .07 | .05 |

These are the journal "dialects" that the synthetic field's journals should reproduce. SMJ is economics- and strategy-theoretic; ASQ and AMJ are sociological and micro. Agreement of these keyword flags with S1 is moderate: κ .34–.61 (section 3.6).

### 3.2 Designs by journal (low validity; use with care)

Source: `4j_03_*`.

- The keyword flags are noisy. For example, `meth_did` fires on 65–89% of papers, almost certainly matching the word "did". The quantitative flag has κ = .13 against S1.
- More usable proxies:
  - Primary-survey data source: AMJ .60, ASQ .47, OS .50, SMJ .38.
  - Interview data: ASQ .34, OS .28, AMJ .27, SMJ .13.
  - Compustat: SMJ .24 against ≈.10 elsewhere.
  - SEM: AMJ .16 against ≤.07 elsewhere.
  - HLM: AMJ .12.
  - CMV addressed: AMJ .20.
- These are adequate for **relative** journal differences (AMJ more micro, survey and multilevel; SMJ more archival and Compustat) but not for absolute shares.
- For absolute design shares, use S1 for SMJ and apply S2 journal *ratios* to the other journals.

### 3.3 Citations (Web of Science)

**Merge-quality problem.** 1,560 S2 rows share a duplicated WoS ID (wrong merges). One example: the ASQ Haleblian & Finkelstein (1999) paper is matched to a pharmaceutical-rivalry paper with match score 1.0.

I kept two sets of matches (`4j_00_wos_match_quality.csv`):

- **Strict:** a unique ID and title-token overlap ≥ .5.
- **Lenient:** a unique ID and WoS year within ±1 of the folder year. Lenient is needed for EBSCO-numbered AMJ/OS files, whose `meta_title` is a journal-name placeholder. Spot checks look right, but this set is *unverified*.

Usable share of rows:

| | AMJ | ASQ | OS | SMJ |
|---|---:|---:|---:|---:|
| Strict | 0.1% | 53% | 2% | 85% |
| Lenient | 56% | 53% | 67% | 85% |

Citations per year since publication (articles only; `4j_04_citations_per_year_by_journal.csv`, export date 2025-12):

| Journal | n | p25 | Median | p75 | p90 | p99 | Gini (raw cites) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMJ | 970 | 6.1 | **10.3** | 16.2 | 26.6 | 67.9 | .54 |
| ASQ | 246 | 4.9 | 8.5 | 14.0 | 23.9 | 59.8 | .57 |
| SMJ | 2,047 | 4.0 | 7.4 | 13.8 | 25.1 | 71.1 | .62 |
| OS | 1,275 | 2.9 | 5.3 | 8.9 | 14.0 | 51.4 | .62 |

- **By cohort** (`4j_04_citations_by_journal_cohort.csv`): log(1+cites) mean and SD, uncited share, Gini and top-10% share. Within a cohort, Gini is .36–.56, the top 10% of papers take 28–42% of citations, and log1p SD is about 0.7–1.0. A lognormal by journal × age is a good generator family.
- 120 papers carry the WoS "highly cited" flag.
- These fields are what you need to calibrate the synthetic citation process: preferential attachment plus journal prestige plus age.

### 3.4 Review durations (received → accepted)

Source: `4j_05_*`. **Only SMJ (773 usable) and OS (746 usable)** print dates. AMJ and ASQ have essentially none.

| Journal | Interval | p25 | Median | p75 | p90 |
|---|---|---:|---:|---:|---:|
| SMJ | Received → accepted | 16.1 mo | **22.1 mo** | 30.5 | 43.5 |
| OS | Received → accepted | 16.9 | **23.9** | 32.4 | 42.3 |
| OS | Received → first revision | — | 10.6 | — | — |
| OS | Revision → accepted | — | 11.8 | — | — |

- SMJ's "revised" date is the *final* revision (median 1.2 months before acceptance), so it does not mark the first round.
- Durations shrank over time: SMJ submissions received 2010–14 took a median of 47 months, against 21 months for 2015–25; OS fell from 40 to 21 months. The 2010–14 cells are thin (35–42 papers) and probably biased towards long-lived manuscripts.
- Acceptance → publication lag: `4j_05_accept_to_pub_year_lag.csv`.
- There is **no information on the number of rounds, desk rejections or rejections**.

### 3.5 Authors, teams and careers

Sources: `4j_06_*`. WoS author lists come from the lenient matches.

**Team size:**

| Journal and era | Mean team size | Solo-authored |
|---|---:|---:|
| AMJ | 2.95 | 7% |
| OS | 2.5 | 14% |
| SMJ | 2.5 | 15% |
| ASQ | 2.2 | **28%** |
| SMJ 2000–09 | 2.13 | 25% |
| SMJ 2020–25 | 2.72 | 11% (4+ authors: 22%) |

**Productivity concentration (Lotka)** (`4j_06_lotka_papers_per_author_in_corpus.csv`): of 6,058 author names, 65% have one paper in the corpus and 7.8% have ≥5. Among authors with ≥2 papers, 64.5% publish in more than one of the four journals, which supports a single field with porous journals. Names are abbreviated WoS forms, so some homonym collapse is possible.

**Careers (OpenAlex; `author_profiles`, `author_works`):**

- Only 948 authors were profiled, 666 with a high-confidence match (`4j_06_author_career_profiles_highmatch.csv`).
  - Median 67 works and h-index 21.
  - Median of 6 papers in top-7 journals.
  - Median first publication 1998.
- Coverage is mostly ASQ (56% of articles) and OS (16%); **AMJ and SMJ are about 0%**.
- The career trajectory by career year is in `4j_06_career_trajectory_by_career_year.csv`.
- **Matthew-effect signatures** (532 authors):
  - Citations in years 0–5 predict citations in years 6–15 (Spearman .58) and output in years 6–15 (.48).
  - Within-author Gini of citations across works is .82; across authors it is .70.
  - At the article level, first-author prior publications barely predict citations per year (ρ = .04, n = 452). Team top-7 count does slightly better (ρ = .10).

### 3.6 SMJ crosswalk: how much to trust the keyword fields

Source: `4j_10_smj_crosswalk_keyword_vs_deep.csv`, n = 2,333 matched SMJ papers.

| Field | Agreement with S1 | Verdict |
|---|---|---|
| Theory flags | Agreement .75–.96; κ RBV .35, agency .55, TCE .44, institutional .40, upper echelons .61, networks .34, RDT .55 | Moderate; usable for marginals |
| Quantitative flag | κ .13 | **Not usable** |
| Hypothesis count | ρ = .14; exact match 15% | **Not usable** |
| Sample size | ρ = .72; exact match 48% | Usable with noise |

### 3.7 SSRN pipeline

Source: `4j_07_*`.

- `ssrn_published_pairs.csv` has only **8 pairs, all SMJ**. That is not enough to calibrate anything.
- `ssrn_metadata_enriched.csv` has 55,276 OpenAlex SSRN records (2005–2025), with relevance tiers: core 1,157, adjacent 38,605, peripheral 15,514.
- OpenAlex's `is_published_elsewhere` is 1.3% for the core tier. The venues it lists are mostly off-field (`4j_07_ssrn_core_published_venues.csv`). This is a gross undercount, because OpenAlex rarely links the SSRN and journal versions.
- **Usable for:** the volume of strategy preprints over time and their topics and team sizes.
- **Not usable for:** rejection rates or migration between journals, unless the DOI/title linkage is rebuilt (see section 5).

### 3.8 Topic hotness

`topic_hotness_openalex.csv` is **all zeros** for every journal. It is unusable as shipped.

---

## 4. Relationship- and correlation-level data by journal

| Layer | SMJ | AMJ / ASQ / OS |
|---|---|---|
| Hypothesis-level relationships, coefficients, SEs, p, support | **Yes**: S1, 10,908 rows, verified tiers | **No.** `relationships_cache/` holds only 2,773 Stage-0 `*_candidates.json` files (noisy phrase lists). There are **0 Stage-1 `*_extraction.json` files**, and `rel_relationships.csv` / `rel_variables.csv` do not exist. |
| Correlation matrices | **Yes**: S1 verified (`rep_correlations_clean_v2`), plus S2 tables | **Yes, but unverified.** `correlation_tables/` has 3,253 `.xlsx` (Descriptives, Correlations and Metadata sheets): AMJ 1,231, ASQ 288, OS 826, SMJ 908. 1,943 have `extraction_method = llm`; the rest have no method recorded. |
| Means/SDs | Yes (S1) | Yes, in the same `.xlsx` files, unverified |
| Paper-level hypothesis counts and support | Yes (S1) | Keyword only (`hyp_hypothesis_count`, `res_hypotheses_supported`); **invalid** per section 3.6 |

The S2 correlation tables parse cleanly (`4j_09_correlation_tables_by_journal.csv`):

| | AMJ | ASQ | OS | SMJ (S2) |
|---|---:|---:|---:|---:|
| Tables parsed | 1,158 | 245 | 649 | 864 |
| Median variables (k) | 13 | 16 | 14 | 13 |
| Median \|r\| | .08 | .07 | .03 | .09 |
| Share with \|r\| > .5 | 4.5% | 4.6% | 3.5% | 7.4% |

- Diagonals equal 1 in ≥99.7% of tables, and almost no values fall outside [−1, 1].
- The S2 SMJ distribution (median |r| .09) is close to the verified S1 figure (.08), which is encouraging.
- OS's low median (.03) needs a look before use, as does ASQ's fill rate of .67.
- Spot checks found visible misalignment in some Descriptives sheets: correlations sitting in the Mean column.

**So:** effect sizes and hypothesis-level reporting statistics can only be seeded from SMJ. Correlation structures could be extended to AMJ, ASQ and OS after a verification pass (section 5).

---

## 5. Gaps, caveats and validation targets

### 5.1 What cannot be seeded from either corpus

- **Peer review content:** reviewer reports, number of rounds, desk-reject and reject rates, editor decisions, reviewer assignment. S2 has only received/revised/accepted dates for SMJ and OS.
- **Rejected or abandoned papers and the file drawer.** Both corpora contain only published work, so selection must be modelled, not observed.
- **Citation networks** (who cites whom). S2 has times-cited counts only; `wos_citedreferences` is empty.
- **Full author careers** for AMJ and SMJ authors, plus hiring, tenure and mobility.
- **True effects.** Every estimate in the corpora is post-selection.
- **Within-paper specification search:** unreported models and their counts.
- **Non-published datasets and the cost of data access.**

### 5.2 External data that would fill the gaps

- **OpenAlex** (free, open) for all four journals 1998–2025:
  - Works, the full reference lists, so **who-cites-whom networks**.
  - Author IDs, giving careers, productivity and coauthor networks for SMJ and AMJ authors as well.
  - Institutions and yearly citation counts, which allow citation-*trajectory* calibration rather than end-state counts.
  - Re-linking the SSRN/working-paper versions to the journal versions by DOI or title and authors gives the preprint → publication lag and **migration across venues**: a preprint whose later version appears in a lower-tier journal is a proxy for rejection.
- **Crossref:** "received", "accepted" and "published" dates in the `assertion` metadata for Wiley (SMJ) and INFORMS (OS) journals, to fill and verify the review-duration fields.
- **Journal-reported statistics** (editorials, annual reports): acceptance and desk-rejection rates, and reviewer turnaround. These anchor the review module.
- **Published reviewer-report data:** Publons / Web of Science Reviewer Recognition, and open-review journals such as eLife or Nature Communications, as a stand-in for review text style and length.
- **Meta-analytic correlation matrices** from management (e.g., the metaBUS database) to cross-check the section 2(e) construct correlations.

### 5.3 Validation targets: stylized facts the synthetic field should reproduce

| Target | Value from the corpus | Table |
|---|---|---|
| Hypotheses per paper | Median 5, mode 4, 9% ≥10 | `07a` |
| Papers with 100% support | **~40%**, rising from 19% (≤1999) to 51% (2020s) | `07b` |
| Hypothesis-level support | 74.5% / 8% partial / 17.5% not supported | `07c` |
| Share of focal \|t\| > 1.96 | ~66% | `07g` |
| Caliper at .05 (verified) | ~1.9–2.3 (fine and z-caliper), ~1.0 at the .01 placebo | `07e` |
| Small-study effect | Median \|r\| .18 (N < 500) vs .02 (N ≥ 5,000); slope of r on 1/√N ≈ 1.35 | `05a` |
| Weak \|t\|–N relation | ρ(\|t\|, log N) ≈ .18 | `summary.json` |
| Wrong-sign "focal" effects | ~15% of direct effects | `05b` |
| Median \|r\| in correlation tables | .08 (p90 .34) | `06a` |
| Between-paper SD of a construct-pair r | ~.18 | `06b` |
| Robustness checks that "all confirm" | 84% | `04g` |
| Journal theory dialects | e.g., RBV share SMJ .24 vs ASQ .04 | `4j_02` |
| Citation inequality | Gini .36–.56 within cohort; top 10% get 28–42% | `4j_04` |
| Review time (SMJ/OS) | Median ~22–24 months, p90 ~43 | `4j_05` |
| Team size | Mean 2.2 (ASQ) to 2.95 (AMJ); solo share falling over time | `4j_06` |
| Matthew effect | Early citations → later citations, Spearman ~.58 | `4j_summary.json` |
| Effect-size inflation | A replication-style redo of a published effect at large N should shrink it, consistent with the N gradient above | — |

---

## 6. DGP template: one synthetic research opportunity

### 6.1 Schema

Each opportunity is a hidden ground-truth object. Agents see only the observable layer: data access, the codebook and the sample frame.

```yaml
opportunity_id: OPP-000123
journal_affinity: {SMJ: .6, OS: .25, AMJ: .1, ASQ: .05}   # from 4j_02 theory dialects x topic
topic:
  theory_family: resource_based_view          # sampled from 02a (SMJ) or 4j_02 (by journal)
  subfield: diversification                   # 4j_02_primary_subfield_by_journal
constructs:                                   # harmonized ids (03a); roles from 02b/02c
  X: {construct: diversification, family: diversification_product, measure_type: continuous,
      source: archival, reliability: 0.85}    # alpha ~ 03e for surveys; for archival use proxy validity 0.6-0.9
  Y: {construct: accounting_performance, family: roa, scale: fraction, mean: .07, sd: .22}   # 03g
  W: {construct: environmental_conditions, role: moderator}   # optional; 29% of hypotheses are moderation
  controls_pool: [firm_size, firm_age, leverage, rnd_intensity, industry_dummies, year_dummies]  # 04f
truth:
  effect_X_on_Y: {type: linear, partial_r: 0.03}     # drawn from the TRUE-effect prior (6.2), not from 05a
  moderation_W: {partial_r: 0.00}
  nonlinearity: none                                  # 3% curvilinear in the literature
  heterogeneity_sd: 0.02                              # across subsamples or contexts
  confounders:                                        # latent U affecting X and Y
    - {name: U_managerial_quality, r_UX: .20, r_UY: .15, observable_proxy: none}
    - {name: firm_size, r_UX: .28, r_UY: .14, observable_proxy: firm_size}   # 06b/06c pooled values
  covariance_base: 06c_pooled_corr_matrix_top25_psd  # perturbed per dataset by N(0, .18) on Fisher z, then PSD-projected
measurement:
  error_model: classical                              # reliability -> attenuation; plus scale-mixture flags (fraction/percent)
  missingness: MAR, 10%                               # missing_data_treatment from S1
sample_frame:
  unit: firm-year                                     # 04b
  panel: unbalanced                                   # 04b
  N_firms: 420                                        # 04c (firms) conditional on unit
  T_years: 9                                          # 04c sample_period_length
  N_obs: 2900                                         # implied
  access_cost: medium                                 # archival-only cheap; survey/hand-collected costly (04a)
identification_options:                               # what a careful agent COULD do
  instrument: {available: true, strength_F: 14, exclusion_violation_r: .05}
  natural_experiment: {available: false}
  fixed_effects_absorb_U: 0.6                         # share of U that is time-invariant
reporting_defaults:                                   # priors for agent behaviour, used for validation not truth
  n_hypotheses: 5                                     # 07a
  controls_k: 8                                       # 04f
  estimator: OLS|FE                                   # 04d by era
```

### 6.2 Setting the true effect (ground truth)

Published estimates are selected, so they cannot be read as true effects. The corpus shows strong signs of selection:

- the N gradient in |r|;
- the residual .05 bunching;
- 74.5% support;
- a weak |t|–N relation.

The recommended procedure has four steps.

1. **Prior family.** Use a *spike-and-slab* on the Fisher-z partial-r scale: θ = 0 with probability π₀, otherwise θ ~ Normal(μ_c, τ_c²). The direction is set to agree with the theoretical prediction with probability q, so that some literatures are wrong in sign. Allow construct-pair-specific μ_c, partially pooled towards a field-level μ by a hierarchical model.

2. **Fit the prior by modelling selection, not by reading the published values.** Fit a step-weight **selection model** to the verified focal t-statistics (4,127 rows with verified coefficient and SE, plus N).
   - Method: Hedges 1992; Vevea & Hedges 1995; identified as in Andrews & Kasy 2019 (*AER*). Publication probability steps at p = .01, .05 and .10, plus a "supported vs. not" verdict channel.
   - Robustness: estimate the same quantities with
     - **z-curve 2.0** (Bartoš & Schimmack 2022), for the expected discovery and replication rates;
     - **PET-PEESE** (Stanley & Doucouliagos 2014), to use the 1/√N gradient;
     - **RoBMA** (Maier, Bartoš & Wagenmakers 2023), which averages over selection and PET models.
   - The construct-pair-level μ_c can be shrunk with the **van Zwet–Gelman** approach: shrink z by a mixture prior estimated from many studies (van Zwet & Cator 2021; van Zwet, Schwab & Senn 2021). This corrects the "winner's curse" (Ioannidis 2008) directly.

3. **Anchor checks** from the corpus. The large-N stratum (N ≥ 5,000; median |r| ≈ .02) is the least distorted by selection and gives a plausible ceiling for the typical true |θ|. The not-supported verdicts (median r ≈ 0) bound π₀ from below.
   - Strategy-specific metascience to cite for plausibility: Goldfarb & King 2016 (*SMJ*, a non-trivial share of significant results likely false); Bettis 2012 and Bettis et al. 2016 (*SMJ*, replication and p-hacking); Harrison et al. 2017 (*J. Management*, publication bias in strategy); Bergh et al. 2017 (*SMJ*, reproducibility of reported results).
   - A **provisional default** until the selection model is fitted: π₀ ≈ .4–.5; non-null |θ| ~ half-normal with scale ≈ .05 on the r scale; q ≈ .8.

4. **Close the loop with simulation-based calibration (ABC).**
   - Run the synthetic field with candidate (π₀, τ, q), the agents' own search and reporting behaviour, and the review filter.
   - Accept parameter values whose *published* output reproduces the section 5.3 targets jointly: the partial-r histogram (`05b`), the |t| histogram (`07g`), the caliper, 40% full support, 74.5% hypothesis support, and the N gradient.
   - This decouples truth from publication: the world's true effects are set, and published inflation becomes an *emergent outcome* that can be validated.

**Ground-truth bookkeeping.** For each opportunity, store the following in a hidden ledger that agents never see:

- θ;
- confounder paths;
- reliabilities;
- the "correct" identification strategy;
- the population estimand.

Every synthetic paper is then scored on sign correctness, coverage of θ by its confidence interval, magnitude error (published / true), and whether the paper's claim survives a large-N replication drawn from the same DGP.

### 6.3 Revised seeding architecture across the four journals

| World module | Primary source | Secondary or override |
|---|---|---|
| Effect-size priors, focal coefficient and SE behaviour, p-value and caliper targets, support rates, hypotheses per paper, controls, robustness, estimators, FE/clustering | **S1 (SMJ deep, verified)**, applied field-wide | Journal-specific shifts from S2 ratios (e.g., more SEM and HLM in AMJ) |
| Correlation structures and raw scales | **S1** pooled matrices (`06b`, `06c`) and descriptives (`03g`) | **S2 `correlation_tables/`** for AMJ, ASQ and OS *after* a verification pass (vision or second-channel, as for S1); until then, use S2 only to set journal-specific \|r\| distributions |
| Topics and theory mix per journal | **S2** (`4j_02`), calibrated by the S1 crosswalk (κ .35–.6, so smooth journal shares rather than use paper-level flags) | S1 for the fine-grained SMJ construct agenda (`02b`) |
| Designs per journal | S1 absolute shares for SMJ × S2 relative journal ratios | — |
| Citation process | **S2 WoS counts** (`4j_04`); per-journal lognormal by age, Gini and top-10% share | OpenAlex for citation networks and yearly trajectories |
| Review cycle timing | **S2 dates for SMJ and OS** (`4j_05`) | Crossref dates for AMJ/ASQ; journal reports for acceptance and desk-reject rates |
| Authors, teams and careers | **S2 team sizes and Lotka** (`4j_06`, all four journals); OpenAlex profiles (ASQ/OS-heavy) | Rebuild OpenAlex careers for all authors |
| Preprint pipeline | S2 SSRN volume and topics only | Re-link via OpenAlex/Crossref before use |

---

## 7. Caveats summary

- The S1 construct mapping covers 26–56% of names. The unmapped names are idiosyncratic focal constructs, which a generator can treat as a "novel construct" class with the same effect-size prior.
- Partial r from non-linear models is approximate, df ignores clustering, and moderation effects are the interaction-term partial r.
- S2 fields are keyword-extracted:
  - `meth_did`, `meth_is_quantitative` and `hyp_hypothesis_count` are invalid;
  - `meta_year` is invalid (use the folder year);
  - `meta_num_authors` is mostly 0 (use WoS author lists);
  - the WoS merge is wrong for about 23% of rows (duplicate IDs).
- The lenient AMJ/OS citation matches are plausible but unverified.
- S2 correlation tables are single-channel and unverified, with some visibly misaligned descriptives.
- `topic_hotness_openalex.csv` is all zeros. `ssrn_published_pairs.csv` has only 8 rows.

---

## 8. Input files of the four-journal extraction

`build_fourjournal_inventory.py` reads these files (read-only) from the folder given with `--fourjournal-dir`:

| File or folder | Size | Needed for |
|---|---:|---|
| `field_review_complete.csv` | 46 MB | Master: Phase 1+2, WoS, author features (all section 3 tables) |
| `field_review_extracted_v2.csv` | 29 MB | Phase-2 variables (complete supersedes most) |
| `field_review_extracted.csv` | 23 MB | Phase 1, with `meta_*` dates and `tc_*` theory flags |
| `field_review_wos_merged.csv` | 39 MB | WoS fields, if re-merging |
| `author_profiles.csv`, `author_works.csv` | 1 + 28 MB | Careers, Matthew effect |
| `ssrn_metadata_enriched.csv`, `ssrn_published_pairs.csv`, `ssrn_metadata.csv` | 41 + 0 + 32 MB | Preprint pipeline |
| `correlation_tables/` (3,253 `.xlsx` + `_index.csv`) | — | AMJ/ASQ/OS correlation structures |
| `codebook.json` | 0.07 MB | Variable definitions |
| `empirical_filter.csv` | 5 MB | Quantitative gate (useful for subsetting) |
| *Optional:* `topic_hotness_openalex.csv` (all zeros), `relationships_cache/` (Stage-0 only), `text_cache.zip` (634 MB; needed only to re-extract relationships for AMJ/ASQ/OS) | — | — |

Backup copies, checkpoint `.pkl` files and logs in that folder are not used.

The S1 SMJ files are read from the frozen copy under `corpus/` and checked against its hashes, so a later rebuild of the extraction (which can drop the correlation patches) cannot change the seeding tables.

---

## 9. Table index (`tables/`)

**S1 (SMJ):**

- `01a`–`01c`: coverage
- `02a`–`02e`: topics, theory families, construct and family pairs, DV/IV tops, relationship types
- `03a`–`03g`: construct usage, dictionary, measure types, sources, measurement quality, Pass-5 categories, descriptives by family
- `04a`–`04h`: data design, panel and unit, sample-size quantiles, estimators and era trends, FE and clustering, controls, robustness and reporting practices, causal strategies
- `05a`–`05d`: partial-r distributions, histogram, by verdict, by construct pair
- `06a`–`06c`: |r| distribution, table size, construct-pair correlations, pooled 25 × 25 matrix (raw, PSD, n)
- `07a`–`07g`: hypotheses per paper, support-rate distribution and era trend, support by relationship type, p bins, caliper tests, p and |t| histograms
- `summary.json`

**S2 (four journals):**

- `4j_00`: WoS match quality
- `4j_01`: coverage
- `4j_02`: theories, subfields, levels
- `4j_03`: designs and method flags, hypothesis style, keyword sample size
- `4j_04`: citations
- `4j_05`: review durations
- `4j_06`: teams, Lotka, careers, trajectory
- `4j_07`: SSRN
- `4j_08`: topic hotness
- `4j_09`: correlation tables by journal
- `4j_10`: SMJ crosswalk
- `4j_summary.json`
