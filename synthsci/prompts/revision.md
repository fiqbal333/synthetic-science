# Revision task

You are the lead author of a manuscript that received a revise-and-resubmit decision. Your working folder is `{{TASK_DIR}}`. It contains `profile.md`, your original submission in `{{TASK_DIR}}\inputs\original\` (`analysis.py`, `table.csv`, `claims.json`, `paper.md`), the dataset in `{{TASK_DIR}}\inputs\data\` (`data.csv`, `codebook.md`, `sample_frame.md`), the three referee reports in `reviews.md`, the editor's letter in `editor_letter.md`, `menu.md` (the menu card you chose your question from, with the papers it listed then), `prior.md` (the papers now published on your question; you may cite them by id, as you may cite the papers in `menu.md` and those already in your reference list), and an empty `scratch\` folder for helper files. Use only these files; do not use any folder outside `{{TASK_DIR}}`.

1. Read the reports and the letter. Decide what to change.
2. {{PROTOCOL_BLOCK}} You may re-run the analysis within this protocol.
3. {{DEPTH_BLOCK}}
4. Write the revised files following the contract below, and `response.md`: a point-by-point response to the referees and the editor. Put `"nonce": "{{NONCE}}"` in `claims.json`.

{{> _contract_paper.md}}

5. `response.md`: for each referee point, what you changed or why you did not.