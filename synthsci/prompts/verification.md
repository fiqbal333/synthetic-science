# Verification task

You are asked to check whether {{N_ITEMS}} finding(s) reproduce. Your working folder is `{{TASK_DIR}}`. For each finding, in its own folder under `{{TASK_DIR}}\inputs\`, you have the claim (`claim.md`), the code that produced it (`analysis.py`), the codebook of the dataset the code was written for (`original_codebook.md`), and a new dataset drawn from the same population (`data.csv`, `codebook.md`, `sample_frame.md`){{SAME_DEPOSIT_SENTENCE}}.

For each finding: (i) run the original code unchanged on the new dataset and report what it gives; (ii) {{PROTOCOL_BLOCK}} Report your own estimate for the claimed relation. (iii) Report the discrepancies between the claim, the re-run and your analysis, a `result` (reproduced, not_reproduced, inconclusive) with your reasons, and your `confidence` (0-1).

Write `{{OUTPUT_DIR}}\verifications.json`:
`{"nonce": "{{NONCE}}", "verifications": [ {"claim_id": "...", "rerun": {"estimate": <number>, "se": <number>, "p": <number>, "n": <integer>} | null, "own": {"estimate": <number>, "se": <number>, "p": <number>, "n": <integer>, "estimator": "...", "controls": ["..."]}, "discrepancies": "...", "result": "reproduced|not_reproduced|inconclusive", "confidence": 0.0, "sample_changes": ["..."], "executed_code": "<verbatim>", "commands_run": <integer>} ]}`
in the order shown. Also write each finding's `own_analysis.py` and `own_table.csv` into `{{OUTPUT_DIR}}\<claim id>\`. Write the JSON file with Python's `json` module, never with a shell tool. Helper files go in `{{TASK_DIR}}\scratch\` only.

Findings: {{ITEM_LIST}}