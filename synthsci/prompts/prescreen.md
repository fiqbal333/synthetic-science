# Pre-screen

Read each submission under `{{TASK_DIR}}\inputs\` (`paper.md`, `table.csv`, `analysis.py`, `codebook.md`). For each, write a flag list of at most 100 words: anything in the code, the table or the text that a referee should look at. Do not run code.

Write `{{OUTPUT_DIR}}\prescreen.json`: `{"nonce": "{{NONCE}}", "flags": [ {"paper_id": "...", "flags": "..."} ]}` in the order shown.

Submissions: {{ITEM_LIST}}