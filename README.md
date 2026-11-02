# Synthetic Science

Code and grounding data for *AI as a Maker or Destroyer of Science? Exploring the Science of Synthetic Science*, a
proposal and technical appendix for Organization Science.

The project builds a simulated research field in strategic management whose true answers are known, so everything it
publishes or certifies can be scored. Language-model agents act as the scientists (authors, referees, editors, lab heads,
analysts and verifiers); a deterministic engine sets the rules, draws the data and keeps the record. The world is
grounded in a verified extraction of 2,358 Strategic Management Journal articles, and the pilot compares a journal
world (blind peer review) with a repository world (certification by independent reproduction), plus a career-pressure
variant of the journal.

## Repository layout

| Folder | Contents |
|---|---|
| `synthsci/` | The engine: world and dataset generation, the two environments, agent prompts, orchestration, budget, scoring and the audit. Command reference in `synthsci/README.md`. |
| `configs/` | Every setting: `base.yaml`, one file per environment in `scenarios/`, the seeding manifest in `seeding/`, and fast-mode, tariff and leak-scan settings. |
| `grounding/` | The aggregate SMJ and four-journal tables that seed the worlds (`tables/`), the scripts that built them and a description of each table. |
| `analysis/` | Result scripts and their outputs: the 20-quarter results (`x20/`), knowledge lineage (`lineage/`) and the catalogue of agent instructions (`instructions/`). |
| `tools/` | Utility scripts: figures, calibration of fast mode, run manifests and checks. |
| `tests/` | The pytest suite. |
| `docs/` | Supplementary documents and figures. |

## Quick start

Python 3.12 was used. There is no pinned requirements file; the code imports:

- engine and tests: `numpy`, `pandas`, `scipy`, `pyyaml`, `jsonschema`;
- analysis and figures: `matplotlib`, `statsmodels`;
- optional: `anthropic` (the API backend, imported only when used).

```
pip install numpy pandas scipy pyyaml jsonschema matplotlib statsmodels
```

Run every command from the repository root.

Run the tests (about 15 minutes; every mock run includes a rule-agent warm start):

```
python -m pytest tests -q
```

Run a world offline with rule agents (no model calls; real data and real analysis code), then re-score it from its ledger:

```
python -m synthsci run --scenario S0 --seed 1 --mode mock
python -m synthsci run --scenario R-b --seed 1 --mode mock     # repository world, paired with S0 seed 1
python -m synthsci audit --run runs/S0/1/mock
```

A live run with language-model agents writes each agent's task folder and stops at the dispatch boundary; the agents are
then run, their outputs ingested and the run resumed:

```
python -m synthsci run --scenario S0-150 --seed 1 --mode llm --backend subagent
python -m synthsci ingest --run runs/S0-150/1/llm
python -m synthsci resume --run runs/S0-150/1/llm
```

Fast mode (calibrated rule agents, no files) supports many seeds and long horizons:

```
python -m synthsci montecarlo --scenario S0 --seeds 1-100 --mode fast
```

The scenarios used in the pilot are `S0-150` (journal), `S0P-150` (journal with career pressure) and `R-b-150`
(repository), 150 scientists each. See `synthsci/README.md` for every command.

## Data

The repository holds aggregate grounding tables only. The publisher PDFs and the article-level extraction are not
shared; the simulation runs are on Zenodo (DOI: [10.5281/zenodo.23873387](https://doi.org/10.5281/zenodo.23873387)). `DATA.md` describes what is where and why.

## Reproducing the results and figures

1. Download the run archives from Zenodo and unzip them into `runs/`, so that folders such as
   `runs/S0-150-x20/1/llm/` and `runs/S0-150/1/llm/` exist.
2. Compute the 20-quarter results (writes `analysis/x20/*.csv` and `analysis/x20/figures/`):
   `python analysis/x20/compute_results.py`
3. Build the knowledge-lineage datasets (writes `analysis/lineage/`):
   `python analysis/lineage/build_lineage.py`
4. Draw the appendix figures from the results (writes `docs/figures/`):
   `python tools/appendix_figures.py`

All three scripts only read the run folders. Any run can also be re-scored from its ledger with
`python -m synthsci audit --run <run folder>`.

## Citation

[Citation to be added: authors, title, year, and the Zenodo DOI.]

## Licence

Code: [license TBD]. Data: [license TBD]. See `DATA.md`.
