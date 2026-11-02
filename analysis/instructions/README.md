# Instructions given to the agents

4,509 live agent prompts across the eight runs (quarters 1-20, plus handbooks). Built by `tools/instruction_catalog.py` (read-only on runs/).

- `index.csv`: every prompt, with world, run, quarter, role, model, template and sha256 fingerprints, and the path to the exact text in its task folder.
- `templates.md`: the role templates and whether every prompt came from the current version.
- `examples/<world>/<role>.md`: one full prompt per world and role, and the material handed with it.

Not in prompt.md: (1) the short dispatch stub each worker was launched with, which only points it to its folder and tells it where to write (configs/backends; the printed stub is in each run's dispatch/batch_N.md); (2) the material files (profile, menu, data, reviews), which sit beside prompt.md in the same task folder.

| World | Role | Prompts |
|---|---|---|
| paper | decision | 82 |
| paper | desk | 54 |
| paper | handbook | 45 |
| paper | research | 52 |
| paper | research_batch | 105 |
| paper | review | 1,307 |
| paper | revision | 139 |
| paper + career pressure | decision | 58 |
| paper + career pressure | desk | 35 |
| paper + career pressure | handbook | 30 |
| paper + career pressure | research | 38 |
| paper + career pressure | research_batch | 69 |
| paper + career pressure | review | 886 |
| paper + career pressure | revision | 91 |
| repository | analyst | 37 |
| repository | analyst_batch | 75 |
| repository | handbook | 45 |
| repository | lab_decision | 820 |
| repository | verification | 541 |
