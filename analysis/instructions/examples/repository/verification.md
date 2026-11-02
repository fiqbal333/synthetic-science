# Example instruction: repository, verification

Source: `runs\R-b-150\1\llm\tasks\verification-00b62ff10a\prompt.md`

Material handed to the agent in its folder: inputs\C0292\analysis.py, inputs\C0292\claim.md, inputs\C0292\codebook.md, inputs\C0292\data.csv, inputs\C0292\original_codebook.md, inputs\C0292\sample_frame.md, inputs\C0293\analysis.py, inputs\C0293\claim.md, inputs\C0293\codebook.md, inputs\C0293\data.csv, inputs\C0293\original_codebook.md, inputs\C0293\sample_frame.md

---

# Verification task

You are asked to check whether 2 finding(s) reproduce. Your working folder is `<repo>\runs\R-b-150\1\llm\tasks\verification-00b62ff10a`. For each finding, in its own folder under `<repo>\runs\R-b-150\1\llm\tasks\verification-00b62ff10a\inputs\`, you have the claim (`claim.md`), the code that produced it (`analysis.py`), the codebook of the dataset the code was written for (`original_codebook.md`), and a new dataset drawn from the same population (`data.csv`, `codebook.md`, `sample_frame.md`).

For each finding: (i) run the original code unchanged on the new dataset and report what it gives; (ii) Work like a careful analyst who checks everything before writing. Inspect the data by running code: dimensions, ranges, distributions, missing values, duplicated rows or identifiers, and anything inconsistent with the codebook. Fix problems you find and record what you changed and why (list each change as a short code in `sample_changes`, e.g. `deduplicated`, `rescaled`, `dropped_impossible_values`, `item_reversed`, `unit_corrected`, `excluded_post_treatment_control`). Decide the sample and the specification. Write analysis.py, run it, and inspect the signs and magnitudes of every estimate before you write. Report your own estimate for the claimed relation. (iii) Report the discrepancies between the claim, the re-run and your analysis, a `result` (reproduced, not_reproduced, inconclusive) with your reasons, and your `confidence` (0-1).

Write `<repo>\runs\R-b-150\1\llm\tasks\verification-00b62ff10a\output\verifications.json`:
`{"nonce": "6dec3738036b", "verifications": [ {"claim_id": "...", "rerun": {"estimate": <number>, "se": <number>, "p": <number>, "n": <integer>} | null, "own": {"estimate": <number>, "se": <number>, "p": <number>, "n": <integer>, "estimator": "...", "controls": ["..."]}, "discrepancies": "...", "result": "reproduced|not_reproduced|inconclusive", "confidence": 0.0, "sample_changes": ["..."], "executed_code": "<verbatim>", "commands_run": <integer>} ]}`
in the order shown. Also write each finding's `own_analysis.py` and `own_table.csv` into `<repo>\runs\R-b-150\1\llm\tasks\verification-00b62ff10a\output\<claim id>\`. Write the JSON file with Python's `json` module, never with a shell tool. Helper files go in `<repo>\runs\R-b-150\1\llm\tasks\verification-00b62ff10a\scratch\` only.

Findings: C0293, C0292