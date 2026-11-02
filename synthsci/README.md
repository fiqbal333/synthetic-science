# synthsci — how to run

Python 3 with numpy, pandas, scipy, statsmodels (run by the rule agents' `analysis.py`), PyYAML and jsonschema (matplotlib for the figure scripts; the `anthropic` package only for
the API backend). Run every
command from the repository root; set `PYTHONDONTWRITEBYTECODE=1` to keep the tree clean.

| Goal | Command |
|---|---|
| Offline end-to-end test, paper regime (rule agents, real `analysis.py`) | `python -m synthsci run --scenario S0 --seed 1 --mode mock` |
| Paired repository run (reads `runs/S0/1/mock/budget/spend_by_quarter.json`) | `python -m synthsci run --scenario R-b --seed 1 --mode mock` |
| Live (LLM) run of the journal world: write task packets, stop at the dispatch boundary | `python -m synthsci run --scenario S0-150 --seed 1 --mode llm --backend subagent` |
| The same world with career pressure visible to the agents | `python -m synthsci run --scenario S0P-150 --seed 1 --mode llm --backend subagent` |
| After the agents have run: validate outputs, write DONE/FAILED (and `dispatch/ingest_log.jsonl`: `reported_but_missing`, `pending_after_batch`, `table_notes`) | `python -m synthsci ingest --run runs/S0/1/llm` |
| Continue a paused run (the budget tallies are rebuilt from the ledger) | `python -m synthsci resume --run runs/S0/1/llm` |
| Extend a finished run to a longer horizon (copy to `runs/<scenario>-x20/<seed>/<mode>`, reopen the close-out, keep the original horizon's rules). A repository run pairs with the EXTENDED paper run and waits (`WAITING FOR PAIRED RUN`) until that run has closed the quarter | `python -m synthsci extend --run runs/S0-150/1/llm --to 20`, then `python -m synthsci resume --run runs/S0-150-x20/1/llm` |
| Fast-mode Monte Carlo (no model calls) with distributions | `python -m synthsci montecarlo --scenario S0 --seeds 1-100 --mode fast` |
| Thirty-year fast run | `python -m synthsci run --scenario S0 --seed 1 --mode fast --set horizon_quarters=120 population=120` |
| Paired regime differences by seed | `python -m synthsci pair --paper S0 --repo R-b --seeds 1-3 --mode mock` |
| Re-score a run from its ledger (no agent calls; budget, timing and career sections included) | `python -m synthsci audit --run runs/S0/1/mock` |
| Regenerate a dataset from its seed and check the hash | `python -m synthsci regen-dataset --run runs/S0/1/mock --dataset D-xxxxxxxx --dest tmp/D` |
| Freeze / verify the hashes of the seeding tables, configs and prompt templates | `python -m synthsci freeze` / `python -m synthsci freeze --verify` |
| Planning cost per run (tariffs measured in early test runs; budget pairing rule stated) | `python -m synthsci cost --scenario S0-150` |
| Re-estimate the fast-mode parameters from live runs (provenance table with n) | `python tools/calibrate_fastmode.py <live paper run> <live repository run> --out <table.md>` |
| Tests | `python -m pytest tests -q` (a few minutes; the warm start adds rule-agent quarters to every mock run) |

The dispatch protocol for a live run is written per run into `runs/<scenario>/<seed>/llm/dispatch/dispatch_instructions.md`
with one `batch_<n>.md` per boundary (verbatim stub per task, model tier, batch sizes). Every prompt is a template in `synthsci/prompts/`;
every agent output stays in `tasks/<task_id>/output/`; the ledger, audit, handbook scoring and budget re-run deterministically from files.

Config knobs live in `configs/base.yaml` (every knob with its baseline value) and scenario files in `configs/scenarios/`; `--set key=value`
overrides any of them. Seeding tables are named in `configs/seeding/smj.yaml`; a set built from other journals is a new manifest, not a code change.

The reported worlds are `S0-150` (journal), `S0P-150` (journal with career pressure visible) and `R-b-150` (repository), run live for
8 quarters and then extended to 20. The other scenarios (`S0`, `S1`, `S2`, `R-a`, `R-b`, `smoke-P`, `smoke-R`) are smaller or
alternative worlds used by the tests and for development.

## Calendar time, review latency and the warm start

A quarter is three calendar months (8 / 20 / 40 / 120 quarters = 2 / 5 / 10 / 30 years). `journal.timing` holds the latencies in
quarters, each a per-paper draw from a discretised lognormal: `research_quarters` (start to submission), `review_round` (submission
to first decision), `revision` (revise-and-resubmit to resubmission), `rereview` (resubmission to final decision) and
`accept_to_publish_quarters`. The defaults put the two-round path at months p25 18 / p50 24 / p75 31 / p90 42 against the SMJ
receipt-to-acceptance quantiles 16 / 22 / 31 / 44 (grounding table `4j_05`). Agent calls are made when a stage comes due, so a
longer latency spreads calls instead of adding them; a slot stays held through review and the derived availability is raised by
the expected busy share so the start rate stays at the productivity target.

`journal.timing.warm_start` (default 16 quarters) opens the world with a full pipeline: quarters 1-K..0 run the same loop with the
rule agents (the mock backend) in every mode, with no model call and no budget debit (ledger rows `admitted: prehistory`), so that
an 8-quarter live run inherits papers under review and in revision, which the live agents then review, revise and decide. Studies
published before quarter 1 join the starting literature as earlier work; the quarter-0 handbook is written after the pre-history.
The audit's `timing` section reports the realized submission-to-acceptance distribution beside the grounding quantiles and what the
warm start contributed; `careers` reports promotion and tenure outcomes (the annual step runs at any horizon when
`long_horizon.turnover.min_horizon_quarters` allows, as in the 150-author scenarios).

## Career pressure (consequences only)

`incentives.career_pressure: visible` (scenario `S0P-150`, paired with `S0-150`) adds three lines to every author's and reviewer's
`profile.md`, all computed from the run's own record: a career line (stage, tenure clock, publications before the record began and
since, the publication record promotion has required under the turnover rule), a department-news line (colleagues' promotion and
tenure outcomes of the last 8 quarters) and the Journal's public statistics (submissions, decisions, acceptance rate, acceptance by
the authors' reported main result). No line tells an agent what to prefer; `tests/test_timing_pressure.py` scans them for
instruction verbs and design tokens.

## Budget accounting

`ledger/budget.jsonl` is the source of truth: every call row carries the tariff debit, the batch/lean scale and the realized tokens
the backend reported (`usage.json`), and every dataset row its observations. A resumed run rebuilds its tallies from the ledger
(`Budget.restore`), and `synthsci audit` recomputes the totals from the ledger regardless of `budget/summary.json`.

**Pairing:** the repository world's compute allotment per quarter is the paired paper world's realized compute; its data allotment
is `budget.data_ceiling_multiple` times the paper world's realized data, a ceiling, not a balance. The default in `configs/base.yaml`
is 2.0 (used by `R-b` and the test worlds); the reported repository world `R-b-150` sets 4.0, because at 150 scientists a 2x ceiling
bound often enough to leave compute unspent. Every deferred request names the resource that bound it (`binding` in the ledger row
and in the `budget_admission` event, with the remaining allotment before and after admission). The audit reports data used, data
and compute per certified unit and per correct handbook answer, and this run's ratio to the paired world (`budget.per_unit`).

## Certified units and the inherited literature

One definition in both regimes: a certified unit is a certified estimate of a question's focal relation, certified in the run
(P: the focal hypothesis of a published paper; R: an endorsed claim on the focal relation); a unit states the direction of a
significant estimate and no effect otherwise (not the author's support verdict). The starting literature (legacy library and
warm-start publications) is inherited and reported in its own table (`field.certification_by_origin`), never among the run's
units, so that the run's certification record is not mixed with what it started from. `tools/calibrate_fastmode.py <runs>`
re-estimates `configs/fastmode/params.yaml` from live runs with provenance.

## Extending a finished run

`synthsci extend --run <finished run> --to <H>` copies the run to `<out-root>/<scenario>-x<H>/<seed>/<mode>` (the source is
hashed before and after and never changes), undoes the horizon close-out in the copy (papers set to `unfinished` get their
prior status back; the close-out's hypothesis rows, papers snapshot and citations file move to `extension/removed_at_reopen/`;
repository lab sizes go back to the quarter-H registry phase's), and writes `extension.json` (lineage, source config and
manifest hashes, git commit). The copy's config adds `long_horizon.extended_from: [<old H>]` and
`long_horizon.pin_short_rules: true`: every horizon-gated rule follows the original horizon (no handbook answers in menus,
turnover gate, handbook schedule = configured snapshots + every `handbook_every_quarters` after the old horizon + the end),
the no-start window before the old horizon stays part of the record, and tasks keep the source's namespace. An extended run
therefore matches, row for row, a run configured for the longer horizon from the start with the same pinned rules. Both keys
are absent from every scenario, so existing configs and hashes are unchanged. `tools/runs_manifest.py` writes and verifies
SHA-256 manifests of run folders; `tools/extend_equivalence_check.py` repeats the equivalence test in mock mode.
