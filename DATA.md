# Data statement

This project simulates a research field whose settings are grounded in published Strategic Management Journal (SMJ)
articles. This file explains which data are in this repository, which are in the Zenodo deposit, which are not shared,
and why.

## In this repository

| What | Where | Contents |
|---|---|---|
| SMJ seeding tables | `grounding/tables/01a_*.csv` to `07g_*.csv`, `summary.json` | Aggregate tables derived from the verified extraction of 2,358 SMJ articles (2,293 published 2000–2025 and 65 published 1997–98; none from 1999; see `01a_articles_by_year_type.csv`): counts, shares, quantiles, construct-pair summaries and pooled correlation matrices. No article text, titles, authors or article-level rows. |
| Four-journal tables | `grounding/tables/4j_*.csv`, `4j_summary.json` | Aggregate tables across SMJ, AMJ, OS and ASQ (coverage, theories, designs, citations, review durations, author productivity, team sizes). The engine uses two of them: SMJ review durations and author productivity. |
| Table builders | `grounding/build_grounding_inventory.py`, `grounding/build_fourjournal_inventory.py`, `grounding/GROUNDING-INVENTORY.md` | The code that produced the tables from the article-level extraction, and a description of every table. |
| Seeding manifest | `configs/seeding/smj.yaml` | The 22 inputs the engine reads: 20 tables from `grounding/tables/` and two declared files, set by the designers with no corpus source: `configs/seeding/smj/pair_theory.csv` (the theory family of each construct pair) and `configs/seeding/smj/construct_measures.yaml` (variable names and definitions per construct). |
| Corpus checksums | `corpus/smj-deep-frozen-2026-10-13/SHA256SUMS.txt`, `corpus/smj-deep-frozen-supplement-2026-10-19/SHA256SUMS.txt` | SHA-256 hashes of the frozen article-level extraction files and of the three harmonisation inputs the table builder reads. |

Every run writes the SHA-256 hash of each seeding input into its `run.json`, so a reader can confirm that the tables in
this repository are the ones the reported runs used.

## Not shared, and why

- **Publisher PDFs.** The SMJ articles are copyrighted and are not redistributed.
- **The article-level extraction.** The per-article files (study design, hypotheses, coefficients, correlation and
  descriptive tables, measurement details) are derived from the copyrighted full texts and are keyed to article titles
  and authors, so they are not shared either. Only the aggregate tables above leave the extraction.

The table builder checks every extraction file against the checksum lists before it reads anything, and stops on a
mismatch. Anyone with the same SMJ articles and extraction can therefore confirm that their copy is byte-identical to
ours and rebuild the seeding tables. How the corpus was built, extracted and verified is described in
`docs/Stage 0 - SMJ corpus construction.docx`.

## On Zenodo

The simulation runs are too large for git (several GB) and are deposited on Zenodo: **DOI: [10.5281/zenodo.23873387](https://doi.org/10.5281/zenodo.23873387)**.

- **One zip per world and run length.** The 8-quarter runs and their 20-quarter extensions for the journal world
  (`S0-150`, `S0-150-x20`), the career-pressure journal world (`S0P-150`, `S0P-150-x20`) and the repository world
  (`R-b-150`, `R-b-150-x20`), plus the warm-start datasets (`_warmstart-datasets`). Each run folder holds the hidden
  world, the population, every agent's task folder (instructions, inputs, code and outputs), the ledger, the handbooks
  and the audit.
- **Local paths replaced.** Local file paths and the user name are replaced with placeholders such as `<WORKSPACE>` and
  `<HOME>`. Agent-written files are otherwise unchanged. A few run-infrastructure notes (the notes in `_void/` folders,
  `REUSED.md`, `agent_report.md`) were rewritten in plain language; the deposit's `README.md` lists them.
- **Voided attempts are kept.** Work that was stopped before it counted (for example, an agent cut off by a provider
  rate limit and then re-run on a clean folder) is kept in `_void/` folders inside the run it belongs to, each with a
  short note saying why it was stopped. Voided work is not used in any reported result.
- **Reused outputs are marked.** Only quarter-0 handbook outputs were reused, and only where the task's instructions
  and inputs were byte-identical to those of an earlier, discontinued attempt of the same run or, in the career-pressure
  world, of the paired journal-world run with the same seed (quarter 0 precedes the treatment and both runs share the
  same random numbers). A `REUSED.md` file in the run folder lists every such output with its hash.
- **Checksums.** `MANIFEST.csv` records, for every file, its SHA-256 before and after the path replacement or
  rewrite, so the released copy can be checked file by file.

To reproduce the reported results, unzip the run archives into `runs/` at the repository root (see `README.md`).

## Licences

- Code: [license TBD]
- Data (seeding tables and Zenodo run archives): [license TBD]
