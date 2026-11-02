# Analysis task

You are the analyst for a research group. Your working folder is `{{TASK_DIR}}`. It contains `brief.md` (what the group wants to know, written by its head), one folder per dataset under `{{TASK_DIR}}\inputs\` (each with `data.csv`, `codebook.md` and `sample_frame.md`; if there are two, pool them in code), `registry.md` (the claims already filed on this question), and an empty `scratch\` folder for any helper files. Use only these files; do not use any folder outside `{{TASK_DIR}}`.

1. Read `brief.md`, the codebook(s) and `registry.md`.
2. Analyse the relations the brief asks about among the focal variables in the codebook; you may add up to four further relations among those variables.
3. {{PROTOCOL_BLOCK}}
4. {{DEPTH_BLOCK}}
5. Write exactly these files into `{{OUTPUT_DIR}}`:
   - `analysis.py`: self-contained; reads each dataset as `..\inputs\<deposit id>\data.csv` relative to its own folder (or that file's absolute path inside the task folder); never copies data into the output folder; helper scripts go in `{{TASK_DIR}}\scratch\` only; makes every sample decision in code; writes `table.csv` (`model_id, term, estimate, se, p, n`; one row per term per fitted model, each with a numeric estimate and standard error and an integer N; nothing else, such as adjusted p-values or fit statistics, goes into the table).
   - `table.csv`: produced by the script.
   - `claims_draft.json`: `{"nonce": "{{NONCE}}", "question_id": "...", "deposits": ["<deposit id>:<version>"], "relation_to_prior": "first claim|replication|extension|contradiction", "entries": [ {"entry_id": "E1", "iv": "...", "dv": "...", "moderator": null, "kind": "focal|moderation|subsample|secondary", "statement": "<one sentence>", "sign": "positive|negative|none", "magnitude_band": "negligible|small|moderate|large", "confidence": <0-100>, "model_id": "...", "estimate": <number>, "se": <number>, "p": <number>, "n": <integer>, "estimator": "...", "se_type": "...", "controls": ["..."], "verdict": "supported|partially|not_supported", "predicted_sign": "positive|negative"} ], "sample_changes": ["..."], "n_models_run": <integer>, "robustness_models": ["..."], "executed_code": "<verbatim>", "commands_run": <integer: how many times you executed analysis.py or any other Python; your own count>, "correction": null | {"script": "correction.py", "changelog": "..."}}`.
   - `note.md`: 250-400 words of prose (any table you include does not count toward the limit): what was analysed, on which dataset(s), what was found, what you checked or did not check, limitations. No theory section, no literature review.
   - If you changed the data and want the change kept for other groups, also write `correction.py` (a deterministic transform of `data.csv`) with a changelog line in `claims_draft.json`.

Write every JSON file with Python's `json` module (`json.dump`), never with a shell tool or by hand: the file must parse as plain JSON with exactly the fields above.