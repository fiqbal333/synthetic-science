# Example instruction: repository, analyst

Source: `runs\R-b-150\2\llm\tasks\analyst-acd294d520\prompt.md`

Material handed to the agent in its folder: brief.md, forensic.json, inputs\D0004\codebook.md, inputs\D0004\data.csv, inputs\D0004\sample_frame.md, registry.md

---

# Analysis task

You are the analyst for a research group. Your working folder is `<repo>\runs\R-b-150\2\llm\tasks\analyst-acd294d520`. It contains `brief.md` (what the group wants to know, written by its head), one folder per dataset under `<repo>\runs\R-b-150\2\llm\tasks\analyst-acd294d520\inputs\` (each with `data.csv`, `codebook.md` and `sample_frame.md`; if there are two, pool them in code), `registry.md` (the claims already filed on this question), and an empty `scratch\` folder for any helper files. Use only these files; do not use any folder outside `<repo>\runs\R-b-150\2\llm\tasks\analyst-acd294d520`.

1. Read `brief.md`, the codebook(s) and `registry.md`.
2. Analyse the relations the brief asks about among the focal variables in the codebook; you may add up to four further relations among those variables.
3. Work like a careful analyst who checks everything before writing. Inspect the data by running code: dimensions, ranges, distributions, missing values, duplicated rows or identifiers, and anything inconsistent with the codebook. Fix problems you find and record what you changed and why (list each change as a short code in `sample_changes`, e.g. `deduplicated`, `rescaled`, `dropped_impossible_values`, `item_reversed`, `unit_corrected`, `excluded_post_treatment_control`). Decide the sample and the specification. Write analysis.py, run it, and inspect the signs and magnitudes of every estimate before you write.
4. (No additional specifications are required.)
5. Write exactly these files into `<repo>\runs\R-b-150\2\llm\tasks\analyst-acd294d520\output`:
   - `analysis.py`: self-contained; reads each dataset as `..\inputs\<deposit id>\data.csv` relative to its own folder (or that file's absolute path inside the task folder); never copies data into the output folder; helper scripts go in `<repo>\runs\R-b-150\2\llm\tasks\analyst-acd294d520\scratch\` only; makes every sample decision in code; writes `table.csv` (`model_id, term, estimate, se, p, n`; one row per term per fitted model, each with a numeric estimate and standard error and an integer N; nothing else, such as adjusted p-values or fit statistics, goes into the table).
   - `table.csv`: produced by the script.
   - `claims_draft.json`: `{"nonce": "42ea5c8067f8", "question_id": "...", "deposits": ["<deposit id>:<version>"], "relation_to_prior": "first claim|replication|extension|contradiction", "entries": [ {"entry_id": "E1", "iv": "...", "dv": "...", "moderator": null, "kind": "focal|moderation|subsample|secondary", "statement": "<one sentence>", "sign": "positive|negative|none", "magnitude_band": "negligible|small|moderate|large", "confidence": <0-100>, "model_id": "...", "estimate": <number>, "se": <number>, "p": <number>, "n": <integer>, "estimator": "...", "se_type": "...", "controls": ["..."], "verdict": "supported|partially|not_supported", "predicted_sign": "positive|negative"} ], "sample_changes": ["..."], "n_models_run": <integer>, "robustness_models": ["..."], "executed_code": "<verbatim>", "commands_run": <integer: how many times you executed analysis.py or any other Python; your own count>, "correction": null | {"script": "correction.py", "changelog": "..."}}`.
   - `note.md`: 250-400 words of prose (any table you include does not count toward the limit): what was analysed, on which dataset(s), what was found, what you checked or did not check, limitations. No theory section, no literature review.
   - If you changed the data and want the change kept for other groups, also write `correction.py` (a deterministic transform of `data.csv`) with a changelog line in `claims_draft.json`.

Write every JSON file with Python's `json` module (`json.dump`), never with a shell tool or by hand: the file must parse as plain JSON with exactly the fields above.