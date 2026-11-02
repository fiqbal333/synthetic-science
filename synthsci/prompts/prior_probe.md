# Questions

Your folder is `{{TASK_DIR}}`. It contains `questions.md`: a list of research questions, each a relation between coded constructs (C-01, C-02, ...) under a lettered theory, with the theory's prediction. There is no literature in the folder.

For each question write your answer: `positive`, `negative`, `null`, `contested` or `no evidence`, with a confidence from 0 to 100 and one sentence of reasoning. Write `{{OUTPUT_DIR}}\handbook.json`:
`{"nonce": "{{NONCE}}", "questions": [ {"question_id": "...", "answer": "...", "confidence": 0, "n_read": 0, "key_units": [], "agree": true, "reasoning": "..."} ], "theories": [], "open_questions": []}`.