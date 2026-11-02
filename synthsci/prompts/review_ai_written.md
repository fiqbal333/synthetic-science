# Review task

Below are {{N_ITEMS}} submissions to a general management journal, each in its own folder under `{{TASK_DIR}}\inputs\` (`paper.md`, `table.csv`, `analysis.py`, `codebook.md`, `prior.md`). Produce a structured review of each quickly. Do not run code.

For each: `recommendation` (accept, minor_revision, major_revision, reject); `flaw_found`; `issues` [{location, description, severity, kind}]; `validity_probability` (0-100); `relation_to_prior`; `confidence` (0-1); `summary` (at most 80 words).

Write one JSON file to `{{OUTPUT_DIR}}\reviews.json`:
`{"nonce": "{{NONCE}}", "reviews": [ ... ]}` with one review per submission, in the order shown.

Submissions: {{ITEM_LIST}}