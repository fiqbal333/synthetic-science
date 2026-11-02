# Pre-registered test

Your working folder is `{{TASK_DIR}}`. It contains `test.md` (a specification agreed in advance: estimator, controls, fixed effects, sample restrictions, and the decision rule) and a dataset under `{{TASK_DIR}}\inputs\data\` (`data.csv`, `codebook.md`, `sample_frame.md`).

Run the specification in `test.md` exactly as written, nothing more and nothing less. {{PROTOCOL_BLOCK}}

Write into `{{OUTPUT_DIR}}`: `analysis.py` (self-contained; writes `table.csv` with `model_id, term, estimate, se, p, n`), `table.csv`, and `test_result.json`: `{"nonce": "{{NONCE}}", "claim_id": "...", "focal": {"estimate": <number>, "se": <number>, "p": <number>, "n": <integer>}, "notes": "<at most 120 words on anything that did not go as specified>", "executed_code": "<verbatim>", "commands_run": <integer>}`. Every row of `table.csv` is a coefficient from a fitted model (numeric estimate and standard error, integer N). Write the JSON file with Python's `json` module, never with a shell tool. Helper files go in `{{TASK_DIR}}\scratch\` only.