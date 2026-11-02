# Editorial decision

You are the editor of a leading general management journal. Below are {{N_ITEMS}} manuscripts that have returned from review, each in its own folder under `{{TASK_DIR}}\inputs\`. For each you have the abstract and the hypotheses with the authors' verdicts (`summary.md`), the results table (`table.csv`){{COUNT_CLAUSE}}, and the referee reports (`reviews.md`){{AUTHOR_STATUS_SENTENCE}}.

For each manuscript decide: `accept`, `revise`, or `reject`. Give a `merit` score from 0 to 100 for how publishable the manuscript is relative to this journal's standard, a list of `required_changes` if you ask for a revision, and a decision `letter` of at most 120 words addressed to the authors. Decide each manuscript on its own merits.{{ROUND_SENTENCE}}

Write one JSON file to `{{OUTPUT_DIR}}\decisions.json`:
`{"nonce": "{{NONCE}}", "decisions": [ {"paper_id": "...", "decision": "accept|revise|reject", "merit": 0, "required_changes": ["..."], "letter": "..."} ]}`
in the order shown.

Manuscripts: {{ITEM_LIST}}