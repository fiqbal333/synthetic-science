# Desk screen

You are the editor of a leading general management journal. Below are up to ten new submissions. For each you have the title, the abstract, the hypothesis list with the authors' verdicts, the main results row, the sample size and design{{COUNT_CLAUSE}}. {{STRICTNESS_SENTENCE}}

For each submission decide `send_out` or `desk_reject`, with one sentence of reason.

Write one JSON file to `{{OUTPUT_DIR}}\desk.json`:
`{"nonce": "{{NONCE}}", "decisions": [ {"paper_id": "...", "decision": "send_out|desk_reject", "reason": "..."} ]}`
in the order shown. Decide each submission on its own merits.

{{SUBMISSIONS}}