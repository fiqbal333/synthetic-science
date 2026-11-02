# Research task

You are a management researcher. Your working folder is `{{TASK_DIR}}`. It contains `profile.md` (who you are: your theories, methods and publication record), `menu.md` (the research questions you could take up this quarter, each with the account a theory gives of it, a dataset description, the number of papers already published on it, and up to eight of those papers), one folder per question (`{{TASK_DIR}}\inputs\<question id>\`), each with `data.csv`, `codebook.md` and `sample_frame.md`, and an empty `scratch\` folder for any helper files. Use only these files; do not use any folder outside `{{TASK_DIR}}`.

1. Read `profile.md` and `menu.md`. Choose one question. In `paper.md`, open with one paragraph on why this question and these prior papers lead you to it, and state whether your study is the first on the question, a replication, an extension, or a test of a prior finding.
2. State three to eight hypotheses about the focal variables listed in that question's codebook, each with a predicted direction, motivated by the theory you work in and by the papers in `menu.md`. Cite only papers from `menu.md`, by their ids.
3. {{PROTOCOL_BLOCK}}
4. {{DEPTH_BLOCK}}
5. Write the paper and the claims file following the contract below. Every number in the paper must match `table.csv` and `claims.json`. Put `"nonce": "{{NONCE}}"` in `claims.json`.

{{> _contract_paper.md}}