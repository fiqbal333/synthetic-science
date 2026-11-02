# Example instruction: repository, lab_decision

Source: `runs\R-b-150\1\llm\tasks\lab_decision-2b35847ec2\prompt.md`

Material handed to the agent in its folder: challenges.md, lab.md, open_claims.md, registry\F-0108.md, registry\Q-DC-04.md, registry\Q-IN-02.md, registry\Q-IN-04.md, registry\Q-IN-05.md, registry\Q-IN-06.md

---

# Research group decisions for this quarter

You head a research group. Your folder is `<repo>\runs\R-b-150\1\llm\tasks\lab_decision-2b35847ec2`. It contains `lab.md` (who you and your members are, your group's record, and the capacity your members have this quarter: analysis slots and verifications), `drafts\` (analyses your members completed last quarter, each with its claims draft and note), `registry\` (for each of six questions you could work on: the question, the state of the claims filed on it so far, and the catalogue of datasets available on it), `open_claims.md` (claims by other groups that you can endorse, challenge or offer to verify, with their evidence and code), and `challenges.md` (objections other groups have raised to your claims). Use only these files.

Decide what your group does this quarter and write `<repo>\runs\R-b-150\1\llm\tasks\lab_decision-2b35847ec2\output\decisions.json` with these parts.

- `agenda`: for each analysis you start (at most your available slots), `{"question_id": "...", "data": "new" | "<deposit id>" | ["<deposit id>", "<deposit id>"], "lead": "<member code>", "brief": "<one paragraph for the analyst>", "n": null | <integer>}`. A new dataset costs its own number of observations from your group's data budget (the typical sample sizes for each question are stated in its registry file); you may name the size you want with `n` (at least 60; it is honoured within the data budget stated in `lab.md`, up to the largest sample the field has seen), else the size is drawn from the question's typical range. A catalogued dataset is free to reuse.
- `filings`: for each draft entry, `{"draft_id": "...", "entry": "<entry id>", "action": "file|revise|drop", "probability": <0-100 that the finding would hold in a large fresh sample>, "note": "..."}`.
- `votes`: for each open claim, `{"claim_id": "...", "vote": "endorse|challenge|abstain", "probability": <0-100>, "reason": "<one sentence>", "objection_kind": null | "data|specification|inference|reporting|interpretation"}`. A challenge must name what is wrong.
- `verifications`: the open claim ids your members will reproduce this quarter, within your capacity.
- `responses`: for each challenge to your claims, `{"claim_id": "...", "action": "defend|revise|withdraw", "text": "<one paragraph>"}`.

Put `"nonce": "f79ffa047b7d"` at the top level. Decide each matter on the evidence in these files and your group's judgement.