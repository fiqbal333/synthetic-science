# Research group decisions for this quarter

You head a research group. Your folder is `{{TASK_DIR}}`. It contains `lab.md` (who you and your members are, your group's record, and the capacity your members have this quarter: analysis slots and verifications), `drafts\` (analyses your members completed last quarter, each with its claims draft and note), `registry\` (for each of six questions you could work on: the question, the state of the claims filed on it so far, and the catalogue of datasets available on it), `open_claims.md` (claims by other groups that you can endorse, challenge or offer to verify, with their evidence and code), and `challenges.md` (objections other groups have raised to your claims). Use only these files.

Decide what your group does this quarter and write `{{OUTPUT_DIR}}\decisions.json` with these parts.

- `agenda`: for each analysis you start (at most your available slots), `{"question_id": "...", "data": "new" | "<deposit id>" | ["<deposit id>", "<deposit id>"], "lead": "<member code>", "brief": "<one paragraph for the analyst>", "n": null | <integer>}`. A new dataset costs its own number of observations from your group's data budget (the typical sample sizes for each question are stated in its registry file); you may name the size you want with `n` (at least 60; it is honoured within the data budget stated in `lab.md`, up to the largest sample the field has seen), else the size is drawn from the question's typical range. A catalogued dataset is free to reuse.
- `filings`: for each draft entry, `{"draft_id": "...", "entry": "<entry id>", "action": "file|revise|drop", "probability": <0-100 that the finding would hold in a large fresh sample>, "note": "..."}`.
- `votes`: for each open claim, `{"claim_id": "...", "vote": "endorse|challenge|abstain", "probability": <0-100>, "reason": "<one sentence>", "objection_kind": null | "data|specification|inference|reporting|interpretation"}`. A challenge must name what is wrong.
- `verifications`: the open claim ids your members will reproduce this quarter, within your capacity.
- `responses`: for each challenge to your claims, `{"claim_id": "...", "action": "defend|revise|withdraw", "text": "<one paragraph>"}`.
{{VARIANT_BLOCK}}
Put `"nonce": "{{NONCE}}"` at the top level. Decide each matter on the evidence in these files and your group's judgement.