# Research tasks ({{N_ITEMS}} independent studies)

This packet holds {{N_ITEMS}} unrelated studies by different researchers. Each study has its own folder under `{{TASK_DIR}}\authors\` and you must do them one at a time, strictly inside that folder: `profile.md` (who the researcher is), `menu.md` (six questions they could take up, with the account a theory gives of each, a dataset description, the number of papers already published on it, and up to eight of those papers), and one folder per question under `inputs\` with `data.csv`, `codebook.md` and `sample_frame.md`. Never read, copy or mention another study's folder, data, menu or results while working on a study; the studies are independent and nothing from one may inform another.

{{AUTHORS_BLOCK}}

For each study, in order:

1. Read that researcher's `profile.md` and `menu.md`. Choose one question. In `paper.md`, open with one paragraph on why this question and these prior papers lead to it, and state whether the study is the first on the question, a replication, an extension, or a test of a prior finding.
2. State three to eight hypotheses about the focal variables listed in that question's codebook, each with a predicted direction, motivated by the researcher's theory and by the papers in that `menu.md`. Cite only papers from that `menu.md`, by their ids.
3. Follow the working protocol given for that researcher above, then write analysis.py, run it, and write the paper and the claims file following the contract below, into that researcher's own output folder. Put `"nonce": "{{NONCE}}"` in every `claims.json`.

{{> _contract_paper.md}}