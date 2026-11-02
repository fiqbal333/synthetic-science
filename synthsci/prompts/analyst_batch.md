# Analysis tasks ({{N_ITEMS}} independent analyses)

This packet holds {{N_ITEMS}} unrelated analyses commissioned by different research groups. Each has its own folder under `{{TASK_DIR}}\analyses\`: `brief.md` (what the group wants to know), one folder per dataset under `inputs\` (`data.csv`, `codebook.md`, `sample_frame.md`; if there are two, pool them in code), and `registry.md` (the claims already filed on that question). Do the analyses one at a time, strictly inside each folder; never read, copy or mention another analysis's folder, data or results while working on one.

{{ANALYSES_BLOCK}}

For each analysis, in order:

1. Read its `brief.md`, codebook(s) and `registry.md`.
2. Analyse the relations the brief asks about among the focal variables in the codebook; you may add up to four further relations among those variables.
3. Follow the working protocol given for that analysis above, then write exactly these files into its own output folder (`{{OUTPUT_DIR}}\<analysis id>\`):
   - `analysis.py`: self-contained; reads each dataset as `..\..\analyses\<analysis id>\inputs\<deposit id>\data.csv` relative to its own folder (or that file's absolute path inside the task folder); never copies data into the output folder; helper files go in `{{TASK_DIR}}\scratch\` only; makes every sample decision in code; writes `table.csv` (`model_id, term, estimate, se, p, n`; one row per term per fitted model, each with a numeric estimate and standard error and an integer N; nothing else, such as adjusted p-values or fit statistics, goes into the table).
   - `table.csv`: produced by the script.
   - `claims_draft.json`: `{"nonce": "{{NONCE}}", "question_id": "...", "deposits": ["<deposit id>:<version>"], "relation_to_prior": "first claim|replication|extension|contradiction", "entries": [ {"entry_id": "E1", "iv": "...", "dv": "...", "moderator": null, "kind": "focal|moderation|subsample|secondary", "statement": "<one sentence>", "sign": "positive|negative|none", "magnitude_band": "negligible|small|moderate|large", "confidence": <0-100>, "model_id": "...", "estimate": <number>, "se": <number>, "p": <number>, "n": <integer>, "estimator": "...", "se_type": "...", "controls": ["..."], "verdict": "supported|partially|not_supported", "predicted_sign": "positive|negative"} ], "sample_changes": ["..."], "n_models_run": <integer>, "robustness_models": ["..."], "executed_code": "<verbatim>", "commands_run": <integer>, "correction": null | {"script": "correction.py", "changelog": "..."}}`.
   - `note.md`: 250-400 words of prose (any table you include does not count toward the limit): what was analysed, on which dataset(s), what was found, what you checked or did not check, limitations.

Write every JSON file with Python's `json` module (`json.dump`), never with a shell tool or by hand: the file must parse as plain JSON with exactly the fields above.