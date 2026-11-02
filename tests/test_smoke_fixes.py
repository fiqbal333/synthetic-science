"""Fixes from the first live smoke test (coordinator, 2026-10-13): data-path rule, scratch folder and stray files,
commands_run as self-report, menu coherence and size, revision prior.md."""
import re
from pathlib import Path

import pytest

from synthsci import config as C
from synthsci.landscape import FAMILY_CLAUSE, build_landscape
from synthsci.orchestration.tasks import PROMPTS, Task, TaskStore, render
from synthsci.util import read_json, read_text, write_json, write_text


def test_menu_coherence_all_questions(landscape):
    """The direction word and the mechanism clause agree with the predicted sign for every core question and fringe card."""
    for q in landscape.questions + landscape.fringe:
        s = q["prediction"]
        word = "raises" if q["prediction_sign"] == "positive" else "lowers"
        head, _, clause = s.partition(", because")
        assert f" {word} " in head and (" lowers " if word == "raises" else " raises ") not in head, s
        assert s.endswith(FAMILY_CLAUSE[q["family"]][q["prediction_sign"]] + "."), s


def test_clauses_differ_by_sign():
    for fam, c in FAMILY_CLAUSE.items():
        assert c["positive"] != c["negative"]


def test_menu_has_configured_size(tmp_path):
    from synthsci.engine import World
    from synthsci.regimes.paper import PaperRegime
    cfg = C.resolve("smoke-P")
    w = World(cfg, 12, "fast", "mock", tmp_path / "m", None)
    reg = PaperRegime(w)
    mc = cfg["landscape"]["menu"]
    size = mc["affinity"] + mc["crowded"] + mc["random"] + mc["fringe"]
    for s in w.scientists:
        for q in (1, 2, 3):
            entries, _ = reg._menu_for(s, q, 0, with_data=False)
            assert len(entries) == size, (s["id"], q, len(entries))
            assert len({e["question_id"] for e in entries}) == size
    from synthsci.regimes import texts
    md = texts.menu_md(entries)
    assert md.startswith(f"# Menu: {size} research questions") and f"This menu lists {size} questions" in md


def test_templates_state_data_path_rule_and_scratch():
    base = {"TASK_DIR": "C:\\t", "OUTPUT_DIR": "C:\\t\\output", "NONCE": "n", "PROTOCOL_BLOCK": "P", "DEPTH_BLOCK": "D", "ROBUSTNESS_HEADING": "", "PAPER_SPEC": "x"}
    txt, _ = render("research.md", base | {"DATA_PATH_RULE": "..\\inputs\\<question id>\\data.csv"})
    assert "..\\inputs\\<question id>\\data.csv" in txt and "scratch" in txt and "never copies or moves `data.csv`" in txt
    assert "commands_run" in txt and "your own count" in txt
    txt, _ = render("revision.md", base | {"DATA_PATH_RULE": "..\\inputs\\data\\data.csv"})
    assert "prior.md" in txt and "..\\inputs\\data\\data.csv" in txt
    txt, _ = render("analyst.md", {"TASK_DIR": "C:\\t", "OUTPUT_DIR": "C:\\t\\output", "NONCE": "n", "PROTOCOL_BLOCK": "P", "DEPTH_BLOCK": "D"})
    assert "..\\inputs\\<deposit id>\\data.csv" in txt and "scratch" in txt
    stub = read_text(PROMPTS / "dispatch_stub.md")
    assert "scratch" in stub and "temp folder" in stub


def test_scratch_folder_and_stray_file_flag(tmp_path):
    from synthsci.backends.subagent import _stray_files
    store = TaskStore(tmp_path / "run")
    t = Task(type="desk", quarter=1, identity="b0", tier="sonnet", run_id="r", call_type="desk_screen", template="desk_screen.md",
             variables={"SUBMISSIONS": "### P0001\n\nfine\n", "STRICTNESS_SENTENCE": "s", "COUNT_CLAUSE": ""}, outputs=["desk.json"], meta={})
    d = store.materialize(t)
    assert (d / "scratch").is_dir()
    write_text(d / "scratch" / "helper.py", "x = 1\n")
    write_text(d / "mk.py", "x = 1\n")                 # a helper written outside output/ and scratch/
    write_text(d / "output" / "desk.json", "{}")
    assert _stray_files(d) == ["mk.py"]


def test_forensic_rerun_accepts_absolute_task_path(tmp_path):
    """An analysis.py that reads its data by an absolute path inside the task folder still reproduces in the clean re-run."""
    import pandas as pd
    from synthsci.scorer import forensic_rerun
    d = tmp_path / "task"
    (d / "inputs" / "Q1").mkdir(parents=True)
    (d / "output").mkdir()
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.1, 6.2, 7.9]}).to_csv(d / "inputs" / "Q1" / "data.csv", index=False)
    code = f"""import pandas as pd
df = pd.read_csv(r"{d / 'inputs' / 'Q1' / 'data.csv'}")
pd.DataFrame([{{"model_id": "M1", "term": "x", "estimate": float(df.x.mean()), "se": 0.1, "p": 0.01, "n": len(df)}}]).to_csv("table.csv", index=False)
"""
    write_text(d / "output" / "analysis.py", code)
    pd.DataFrame([{"model_id": "M1", "term": "x", "estimate": 2.5, "se": 0.1, "p": 0.01, "n": 4}]).to_csv(d / "output" / "table.csv", index=False)
    res = forensic_rerun(d / "output", {"inputs/Q1/data.csv": d / "inputs" / "Q1" / "data.csv"}, task_root=d)
    assert res["reproducible"] is True, res


def test_miscoded_and_reverse_coded_are_exclusive_and_guarded(cfg, tables, landscape):
    """Regression: on a survey-IV question the reverse-coded trap replaces the IV column by its items; mis-coding that column
    crashed the 150-author mock run (KeyError 'workforce_education'). The two traps are now exclusive and the correction
    tolerates a missing column."""
    import pandas as pd
    from synthsci.datasets import DatasetGenerator
    from synthsci.traps import REGISTRY, applicable_traps, draw_traps
    from synthsci.util import seed_from
    gen = DatasetGenerator(cfg, tables, landscape)
    q = next(q for q in landscape.questions if gen.spec_for(q)["iv_source"] == "survey")
    spec = gen.spec_for(q)
    assert {"reverse_coded", "miscoded_values"} <= set(applicable_traps(spec))
    for i in range(300):
        t = draw_traps(spec, C.rng(i, "x"), {"one": 0.0, "two": 1.0})
        assert not ({"reverse_coded", "miscoded_values"} <= set(t))
    df = pd.DataFrame({"firm_id": [1, 2, 3], "leverage": [0.1, 999.0, 0.3], "firm_age": [5, 6, -5], f"{q['iv_var']}_item1": [1, 2, 3]})
    out = REGISTRY["miscoded_values"].correct(df, q)   # no IV column: must not raise
    assert len(out) == 1
    for i in range(6):   # the generator completes on the survey question
        gen.generate(q["id"], seed_from("surv", i), f"S{i}")


def test_pairing_guard_is_a_hard_error(tmp_path):
    """A pairing scenario without the paired run's spend file fails unless allow_unpaired is set (then logged)."""
    from synthsci.engine import run
    from synthsci.util import read_jsonl
    with pytest.raises(RuntimeError, match="declares budget.rule pairing"):
        run("smoke-R", 51, "fast", out_root=tmp_path, quiet=True)
    rd = run("smoke-R", 51, "fast", out_root=tmp_path / "allowed", quiet=True, allow_unpaired=True)
    meta = read_json(rd / "run.json")
    assert meta["pairing_fallback"] and "allow-unpaired" in meta["pairing_fallback"]
    assert any(e.get("event") == "pairing_fallback" for e in read_jsonl(rd / "ledger" / "events.jsonl"))


def test_revision_packet_has_prior(tmp_path):
    """A revision packet carries prior.md with the papers now published on the question."""
    from synthsci.engine import run
    rd = run("smoke-P", 14, "mock", out_root=tmp_path, quiet=True)
    revs = list((rd / "tasks").glob("revision-*/prior.md"))
    assert revs, "no revision packet in this run (seed-dependent); choose another seed if this fails"
    assert read_text(revs[0]).startswith("# Prior work on ")
    menu = revs[0].parent / "menu.md"   # the question's menu card travels with the revision packet
    assert menu.exists() and read_text(menu).startswith("# Menu: ")
    assert "menu.md" in read_text(revs[0].parent / "prompt.md")
