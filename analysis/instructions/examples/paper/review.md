# Example instruction: paper, review

Source: `runs\S0-150\1\llm\tasks\review-0f4f97318f\prompt.md`

Material handed to the agent in its folder: inputs\P0074\analysis.py, inputs\P0074\codebook.md, inputs\P0074\paper.md, inputs\P0074\prior.md, inputs\P0074\table.csv, inputs\P0082\analysis.py, inputs\P0082\codebook.md, inputs\P0082\paper.md, inputs\P0082\prior.md, inputs\P0082\table.csv, profile.md

---

# Referee task

You are a referee for a general management journal. You are an active researcher; your profile is in `<repo>\runs\S0-150\1\llm\tasks\review-0f4f97318f\profile.md`. Below are 2 submissions, each in its own folder under `<repo>\runs\S0-150\1\llm\tasks\review-0f4f97318f\inputs\`. For each you have the paper with its results table (`paper.md`, `table.csv`), the analysis code (`analysis.py`), the codebook of the dataset it used (`codebook.md`), and the abstracts of the papers already published on the same question (`prior.md`). You do not have the data and must not run any code. Judge each submission on its own merits: are its claims supported by its code, its table and the codebook, does it meet the standard of a leading journal, and how does it stand in relation to the prior work on its question?



For each submission write: `recommendation` (accept, minor_revision, major_revision, reject); `flaw_found` (true if at least one error invalidates or seriously undermines a main claim); `issues`, each with a `location`, a one- or two-sentence `description`, a `severity` (minor or major) and a `kind` (data, specification, inference, reporting, interpretation, framing); `validity_probability` (0-100: the probability that the main finding would hold in a large fresh sample from the same population); `relation_to_prior` (first study, replication, extension, contradiction, unclear); `confidence` (0-1); and a `summary` for the editor of at most 80 words.

Write one JSON file to `<repo>\runs\S0-150\1\llm\tasks\review-0f4f97318f\output\reviews.json`:
`{"nonce": "af9b9c3fd93b", "reviews": [ {"paper_id": "...", "recommendation": "...", "flaw_found": false, "issues": [...], "validity_probability": 0, "relation_to_prior": "...", "confidence": 0.0, "summary": "..."} ]}`
with one review per submission, in the order shown.

Submissions: P0074, P0082