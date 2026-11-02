# Example instruction: repository, handbook

Source: `runs\R-b-150\1\llm\tasks\handbook-9ebc9fd771\prompt.md`

Material handed to the agent in its folder: questions.md, registry.md

---

# State of the field

You are writing the state of a research field from its claim registry alone. Your folder is `<repo>\runs\R-b-150\1\llm\tasks\handbook-9ebc9fd771`. It contains `questions.md` (the field's core questions, each a relation between coded constructs under a lettered theory, with the theory's prediction), `registry.md` (every claim the field has certified so far, grouped by question, with its evidence, status, endorsements, challenges and verifications). Constructs are coded (C-01, C-02, ...) and theories are lettered; work only from what the claim registry says about them. Use only these files.

Write `<repo>\runs\R-b-150\1\llm\tasks\handbook-9ebc9fd771\output\handbook.json`:
```
{"nonce": "9bd982ba8296",
 "questions": [ {"question_id": "...", "answer": "positive|negative|null|contested|no evidence", "confidence": <0-100>, "n_read": <integer>, "key_units": ["<up to three ids>"], "agree": true|false, "reasoning": "<one sentence>"} ],   // one entry per core question, including uncovered ones; "agree" is true only when your answer is the direction the theory predicts for that question in questions.md (false for the opposite direction, null, contested or no evidence)
 "theories": [ {"theory": "Theory A", "standing": "strong|moderate|weak|contested|untested", "share_supported": <0-1>, "reasoning": "<one sentence>"} ],
 "open_questions": [ {"question_id": "...", "reason": "few_studies|conflicting_results|small_samples|single_dominant_unit", "text": "<one sentence>"} ]   // up to ten, ranked
}
```
and `<repo>\runs\R-b-150\1\llm\tasks\handbook-9ebc9fd771\output\state_of_the_field.md`: 600-900 words of prose on what the field has learned, what it disagrees about, what it has not studied.