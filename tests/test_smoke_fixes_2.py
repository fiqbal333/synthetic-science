"""Regression tests for problems found in the live smoke test: leak scan on word boundaries, the table.csv contract, the
contract and prompt text, the batched-paper folder, a hard error on missing paper files, skipping a placeholder title,
the agent allow-list with a design re-scan of quoted agent text, and the ingest log of reported-but-missing files."""
import json
from pathlib import Path

import pandas as pd
import pytest

from synthsci import config as C
from synthsci.config import CONFIG_DIR, load_yaml
from synthsci.leakscan import agent_allow, is_phrase, scan_text
from synthsci.orchestration.tasks import PROMPTS, Task, TaskStore, render
from synthsci.util import read_json, read_jsonl, read_text, write_json, write_text


# ------------------------------------------------------------------ 1. leak scan on word boundaries
@pytest.mark.parametrize("text, hit", [
    ("the bootstrap standard errors", False),      # 'trap' inside 'bootstrap' (3 reviews failed in the smoke test)
    ("we fell into a trap", True),
    ("the traps were", True),                       # plural inflection
    ("a trapdoor", False),
    ("entrapment", False),
    ("competency trap", False),                     # org-theory allow-list
    ("the learning trap", True),
    ("theta_fe", True),                             # underscore is a separator: engine keys still trip
    ("thetas", True),
    ("a regimen of tests", False),
    ("the two regimes", True),
    ("simulated firms", True),                      # declared prefix
    ("a simulation", True),
    ("admissibility", True),                        # declared prefix
    ("ground truth", True),                         # phrase
    ("the ground-truth", True),
    ("Rows: 1,200. Missing values are coded as empty cells.", False),
    ("the llms", True),
    ("dropped_bad_control", True),                  # engine id inside a coined code (agent_allow handles agent text)
])
def test_leak_tokens_match_on_word_boundaries(text, hit):
    assert bool(scan_text(text, "strict")) is hit, (text, scan_text(text, "strict"))


def test_prefix_tokens_declared_and_listed():
    cfg = load_yaml(str(CONFIG_DIR / "leakscan.yaml"))
    tokens = set(cfg["design"]) | set(cfg["strict"])
    assert {"simulat", "admissib", "leakscan"} <= set(cfg["prefix"])
    assert set(cfg["prefix"]) <= tokens
    for t in cfg["prefix"]:
        assert not is_phrase(t)
    # every non-phrase, non-prefix token is a whole word: a bare prefix of an English word would silently stop matching
    for t in ("simulat", "admissib"):
        assert t in cfg["prefix"]


def test_leak_scan_prefix_must_be_listed(tmp_path):
    write_text(tmp_path / "bad.yaml", "design:\n  - trap\nstrict: []\nprefix:\n  - simulat\n")
    with pytest.raises(ValueError, match="prefix"):
        scan_text("x", "strict", config=tmp_path / "bad.yaml")


def test_agent_allow_only_in_design_rescan():
    assert scan_text("dropped_bad_control", "design")
    assert scan_text("dropped_bad_control", "design", extra_allow=agent_allow()) == []
    assert scan_text("a planted trap", "design", extra_allow=agent_allow())   # the allow-list never covers real design words


# ------------------------------------------------------------------ 2. table.csv contract
HOLM_TABLE = """model_id,term,estimate,se,p,n
M1,Intercept,1.358,0.295,4.2e-06,603
M1,tmt_heterogeneity,-0.5496,0.2704,0.0421,603
M2,tmt_heterogeneity,-0.2972,0.0802,0.0002,603
HOLM,H1_M1_tmt_heterogeneity,0.0841,,0.0421,603
HOLM,H2_M2_tmt_heterogeneity,0.0006,,0.0002,603
"""


def test_table_check_flags_nonregression_rows(tmp_path):
    from synthsci.scorer import regression_rows, table_check
    p = tmp_path / "table.csv"
    write_text(p, HOLM_TABLE)
    notes = table_check(p)
    assert len(notes) == 2 and all("HOLM" in n and "se" in n for n in notes)
    assert len(regression_rows(p)) == 3
    write_text(p, "model_id,term,estimate,se,p,n\nM1,x,0.5,0.1,0.01,100\n")
    assert table_check(p) == []
    write_text(p, "model,term\nM1,x\n")
    assert table_check(p) and "missing column" in table_check(p)[0]
    assert table_check(tmp_path / "none.csv") == ["table.csv missing"]


def test_forensic_rerun_robust_to_holm_rows(tmp_path):
    """The smoke-test pattern (revision-34c5722344): a table with adjusted-p rows whose se is blank still reproduces, and
    the notes travel with the forensic result so the scorer flags them."""
    from synthsci.scorer import analyst_flags, forensic_rerun
    d = tmp_path / "task"
    (d / "inputs" / "Q1").mkdir(parents=True)
    (d / "output").mkdir()
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.1, 6.2, 7.9]}).to_csv(d / "inputs" / "Q1" / "data.csv", index=False)
    code = """import pandas as pd
df = pd.read_csv("../inputs/Q1/data.csv")
rows = [{"model_id": "M1", "term": "x", "estimate": float(df.x.mean()), "se": 0.1, "p": 0.01, "n": len(df)},
        {"model_id": "HOLM", "term": "H1_M1_x", "estimate": 0.02, "se": None, "p": 0.01, "n": len(df)}]
pd.DataFrame(rows).to_csv("table.csv", index=False)
"""
    write_text(d / "output" / "analysis.py", code)
    write_text(d / "output" / "table.csv", "model_id,term,estimate,se,p,n\nM1,x,2.5,0.1,0.01,4\nHOLM,H1_M1_x,0.02,,0.01,4\n")
    res = forensic_rerun(d / "output", {"inputs/Q1/data.csv": d / "inputs" / "Q1" / "data.csv"}, task_root=d)
    assert res["reproducible"] is True, res
    assert len(res["table_notes"]) == 1 and "HOLM" in res["table_notes"][0]
    # a changed coefficient is still caught
    write_text(d / "output" / "table.csv", "model_id,term,estimate,se,p,n\nM1,x,2.9,0.1,0.01,4\nHOLM,H1_M1_x,0.02,,0.01,4\n")
    assert forensic_rerun(d / "output", {"inputs/Q1/data.csv": d / "inputs" / "Q1" / "data.csv"}, task_root=d)["reproducible"] is False
    q = {"iv_var": "x", "dv_var": "y", "panel": False}
    claims = {"hypotheses": [{"iv": "x", "dv": "y", "estimate": 2.5, "se": 0.1, "n": 4, "verdict": "supported", "controls": []}],
              "sample_changes": [], "n_models_run": 1, "robustness_models": [], "executed_code": code, "commands_run": 1}
    fl = analyst_flags(claims, {"traps": [], "oracle": {}}, {"n_correct": 4}, q, None, "AUTO", "T0", True, table_notes=res["table_notes"])
    assert "table_nonregression_rows" in fl["flags"]
    fl = analyst_flags(claims, {"traps": [], "oracle": {}}, {"n_correct": 4}, q, None, "AUTO", "T0", True, table_notes=[])
    assert "table_nonregression_rows" not in fl["flags"]


# ------------------------------------------------------------------ 3. contract and prompt text
def test_contract_text_json_rule_word_limit_and_scratch():
    base = {"TASK_DIR": "C:\\t", "OUTPUT_DIR": "C:\\t\\output", "NONCE": "n", "PROTOCOL_BLOCK": "P", "DEPTH_BLOCK": "D", "ROBUSTNESS_HEADING": ""}
    from synthsci.regimes.paper import PaperRegime
    spec = PaperRegime._paper_spec(type("R", (), {"cfg": C.resolve("S0")})(), "")
    assert "do not count toward the limit" in spec
    for tpl in ("research.md", "revision.md"):
        txt, _ = render(tpl, base | {"PAPER_SPEC": spec, "DATA_PATH_RULE": "..\\inputs\\x\\data.csv"})
        assert "Python's `json` module" in txt and "never with a shell tool" in txt
        assert "scratch" in txt and "nowhere else" in txt
        assert "one row per term per fitted model" in txt and "adjusted p-values" in txt
        assert "p_adjusted" in txt
    txt, _ = render("research_batch.md", base | {"PAPER_SPEC": spec, "DATA_PATH_RULE": "x", "N_ITEMS": 2, "AUTHORS_BLOCK": "a"})
    assert "Python's `json` module" in txt
    for tpl, extra in (("analyst.md", {}), ("analyst_batch.md", {"N_ITEMS": 2, "ANALYSES_BLOCK": "a"}),
                       ("verification.md", {"N_ITEMS": 1, "ITEM_LIST": "c", "SAME_DEPOSIT_SENTENCE": ""}),
                       ("decisive_test.md", {})):
        txt, _ = render(tpl, base | extra)
        assert "Python's `json` module" in txt, tpl
        assert "scratch" in txt, tpl
    txt, _ = render("analyst.md", base)
    assert "words of prose" in txt and "adjusted p-values" in txt
    # the paper spec, which carries the word limit, still passes the strict leak scan
    assert scan_text(spec, "strict") == []


def test_claims_schema_accepts_p_adjusted():
    import jsonschema
    schema = read_json(Path(__file__).resolve().parents[1] / "synthsci" / "schemas" / "claims.schema.json")
    h = {"hyp_id": "H1", "iv": "a", "dv": "b", "moderator": None, "predicted_sign": "positive", "model_id": "M1", "estimate": 0.1, "se": 0.05,
         "p": 0.04, "p_adjusted": 0.08, "n": 100, "estimator": "OLS", "se_type": "HC1", "controls": [], "verdict": "supported"}
    obj = {"nonce": "n", "question_id": "Q1", "relation_to_prior": "first study", "hypotheses": [h], "builds_on": [], "sample_changes": [],
           "n_models_run": 1, "robustness_models": [], "executed_code": "x", "commands_run": 1}
    jsonschema.validate(obj, schema)
    h["p_adjusted"] = "0.08"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(obj, schema)


# ------------------------------------------------------------------ 5. paper files, titles and quoted agent text
def test_title_skips_placeholder_heading():
    from synthsci.regimes.paper import _title
    assert _title("# Title\n\n# Slack and Innovation\n\ntext") == "Slack and Innovation"
    assert _title("Title:\nSlack and Innovation") == "Slack and Innovation"
    assert _title("# Paper title\n# Slack") == "Slack"
    assert _title("# Slack and Innovation\n") == "Slack and Innovation"
    assert _title("") == "" and _title(None) == ""


@pytest.fixture(scope="module")
def paper_regime(tmp_path_factory):
    from synthsci.engine import World
    from synthsci.regimes.paper import PaperRegime
    cfg = C.resolve("smoke-P")
    w = World(cfg, 12, "fast", "mock", tmp_path_factory.mktemp("reg") / "w", None)
    return PaperRegime(w)


def _fake_task(reg, task_id: str, ttype: str, extra_inputs: dict | None = None) -> Path:
    d = reg.w.store.dir / task_id
    (d / "output").mkdir(parents=True, exist_ok=True)
    write_json(d / "task.json", {"task_id": task_id, "type": ttype, "quarter": 1, "tier": "sonnet", "run_id": reg.w.run_id, "call_type": "research_t0",
                                 "template": "research.md", "outputs": [], "schema": None, "meta": {}, "who": "A001"})
    for rel, txt in (extra_inputs or {}).items():
        write_text(d / rel, txt)
    return d


def test_paper_dir_for_batched_and_single_papers(paper_regime):
    reg = paper_regime
    d1 = _fake_task(reg, "research-aaaaaaaaaa", "research")
    d2 = _fake_task(reg, "research_batch-bbbbbbbbbb", "research_batch")
    single = {"paper_id": "P0001", "lead": "A001", "task_ids": ["research-aaaaaaaaaa"]}
    batched = {"paper_id": "P0002", "lead": "A002", "task_ids": ["research_batch-bbbbbbbbbb"]}
    assert reg._paper_dir(single) == d1 / "output"
    assert reg._paper_dir(batched) == d2 / "output" / "A002"
    # after a revision the latest task decides
    revised = {"paper_id": "P0002", "lead": "A002", "task_ids": ["research_batch-bbbbbbbbbb", "research-aaaaaaaaaa"]}
    assert reg._paper_dir(revised) == d1 / "output"


def test_missing_paper_files_are_a_hard_error(paper_regime):
    reg = paper_regime
    d = _fake_task(reg, "research_batch-cccccccccc", "research_batch")
    p = {"paper_id": "P0003", "lead": "A003", "task_ids": ["research_batch-cccccccccc"]}
    with pytest.raises(FileNotFoundError, match="P0003: paper.md missing"):
        reg._paper_files(p)
    for f in ("paper.md", "table.csv", "analysis.py"):
        write_text(d / "output" / "A003" / f, "x\n")
    files = reg._paper_files(p)
    assert set(files) == {"paper.md", "table.csv", "analysis.py"} and all(v.exists() for v in files.values())


def test_menu_path_for_single_and_batched_papers(paper_regime):
    reg = paper_regime
    _fake_task(reg, "research-dddddddddd", "research", {"menu.md": "# Menu: 6 research questions\n"})
    _fake_task(reg, "research_batch-eeeeeeeeee", "research_batch", {"authors/A005/menu.md": "# Menu: 6 research questions\n"})
    _fake_task(reg, "research-ffffffffff", "research")
    assert reg._menu_path({"lead": "A004", "task_ids": ["research-dddddddddd"]}).name == "menu.md"
    assert reg._menu_path({"lead": "A005", "task_ids": ["research_batch-eeeeeeeeee", "revision-x"]}).parent.name == "A005"
    assert reg._menu_path({"lead": "A006", "task_ids": ["research-ffffffffff"]}) is None


def _done_task_with_output(store: TaskStore, task_id: str, name: str, obj) -> Path:
    d = store.dir / task_id
    (d / "output").mkdir(parents=True, exist_ok=True)
    if name.endswith(".json"):
        write_json(d / "output" / name, obj)
    else:
        write_text(d / "output" / name, obj)
    write_text(d / "DONE", "{}\n")
    return d


def test_scan_mixed_strips_agent_fragments_and_rescans_them_in_design_mode(tmp_path):
    """Text an agent wrote (already scanned in design mode at ingest) is held to the design list when
    the engine quotes it, not to the strict list; but a quoted fragment with a real design token is still caught."""
    store = TaskStore(tmp_path / "run")
    _done_task_with_output(store, "review-1111111111", "reviews.json",
                           {"reviews": [{"summary": "The authors report impossible values for leverage and a dropped_bad_control step."}]})
    _done_task_with_output(store, "review-2222222222", "reviews.json",
                           {"reviews": [{"summary": "The oracle of the ledger says the planted effect is large."}]})
    quoted_ok = "# Referee reports\n\n- R1: The authors report impossible values for leverage and a dropped_bad_control step.\n"
    assert scan_text(quoted_ok, "strict")                       # the strict list would reject this text
    assert store._scan_mixed(quoted_ok, [str(tmp_path)]) == []  # but it is agent-quoted and passes the design list
    quoted_bad = "- R2: The oracle of the ledger says the planted effect is large.\n"
    hits = store._scan_mixed(quoted_bad, [str(tmp_path)])
    assert hits and all(h.startswith("design token") for h in hits)
    assert any("'ledger'" in h or "'oracle'" in h or "'planted'" in h for h in hits)
    # engine-written text with a strict token that no agent wrote is still rejected
    assert store._scan_mixed("Our simulated scenario is a bad control.", [str(tmp_path)])
    # the materialize path raises on a leak and writes the LEAK file
    t = Task(type="desk", quarter=1, identity="b1", tier="sonnet", run_id="r", call_type="desk_screen", template="desk_screen.md",
             variables={"SUBMISSIONS": "### P0001\n\nthe planted trap\n", "STRICTNESS_SENTENCE": "s", "COUNT_CLAUSE": ""}, outputs=["desk.json"], meta={})
    with pytest.raises(RuntimeError, match="leak scan failed"):
        store.materialize(t)
    assert (store.task_dir(t) / "LEAK").exists()


def test_validate_uses_agent_allow_list(tmp_path):
    store = TaskStore(tmp_path / "run")
    t = Task(type="desk", quarter=1, identity="b2", tier="sonnet", run_id="r", call_type="desk_screen", template="desk_screen.md",
             variables={"SUBMISSIONS": "### P0001\n\nfine\n", "STRICTNESS_SENTENCE": "s", "COUNT_CLAUSE": ""}, outputs=["desk.json"], meta={})
    d = store.materialize(t)
    write_json(d / "output" / "desk.json", {"note": "we removed the dropped_bad_control rows and used bootstrap SEs"})
    rep = store.validate(t)
    assert rep["ok"], rep
    write_json(d / "output" / "desk.json", {"note": "the planted trap"})
    rep = store.validate(t)
    assert not rep["ok"] and any("design tokens" in p for p in rep["problems"])


def test_leakscan_yaml_allow_lists_from_smoke():
    cfg = load_yaml(str(CONFIG_DIR / "leakscan.yaml"))
    assert "bootstrap" in cfg["allow"] and "competency trap" in cfg["allow"]
    assert {"bad_control", "reverse_coded", "se_not_clustered"} <= set(cfg["agent_allow"])
    assert not (set(cfg["agent_allow"]) & {"trap", "planted", "theta", "oracle", "ledger"})


# ------------------------------------------------------------------ 6. ingest log: reported but missing
def test_ingest_log_records_reported_but_missing(tmp_path):
    from synthsci.backends.subagent import ingest
    from synthsci.engine import run
    rd = run("smoke-P", 23, "llm", backend="subagent", out_root=tmp_path, quiet=True)
    batch = read_json(rd / "dispatch" / "batch_1.json")
    assert len(batch) >= 3
    t0, t1, t2 = batch[0], batch[1], batch[2]
    rec0 = read_json(Path(t0["task_dir"]) / "task.json")
    qids = [l.split(" ")[1] for l in read_text(Path(t0["task_dir"]) / "questions.md").splitlines() if l.startswith("- Q")]
    hb = {"nonce": rec0["nonce"], "questions": [{"question_id": q, "answer": "no evidence", "confidence": 5} for q in qids], "theories": [], "open_questions": []}
    write_json(Path(t0["task_dir"]) / "output" / "handbook.json", hb)
    write_text(Path(t0["task_dir"]) / "output" / "state_of_the_field.md", "# State\n\nprose\n")
    write_text(Path(t0["task_dir"]) / "AGENT_FINISHED", f"finished {rec0['nonce']}\n")
    write_json(Path(t0["task_dir"]) / "usage.json", {"total_tokens": 10, "duration_ms": 1, "tool_uses": 1, "model": "opus", "at": "t"})
    # t1: the agent returned (usage recorded) but nothing landed; t2: not run yet
    write_json(Path(t1["task_dir"]) / "usage.json", {"total_tokens": 10, "duration_ms": 1, "tool_uses": 1, "model": "sonnet", "at": "t"})
    rep = ingest(rd)
    assert rep["done"] == 1 and rep["pending"] == 2
    assert rep["reported_but_missing"] == [t1["task_id"]]
    assert set(rep["pending_tasks"]) == {t1["task_id"], t2["task_id"]}
    assert set(rep["pending_after_batch"]) == {t1["task_id"], t2["task_id"]} and rep["batch"] == "batch_1"
    log = read_jsonl(rd / "dispatch" / "ingest_log.jsonl")
    assert len(log) == 1 and log[0]["reported_but_missing"] == [t1["task_id"]]
    assert read_json(rd / "dispatch" / "last_ingest.json")["reported_but_missing"] == [t1["task_id"]]
    # outputs without a marker are validated and recorded as finished_without_marker
    rec2 = read_json(Path(t2["task_dir"]) / "task.json")
    write_json(Path(t2["task_dir"]) / "output" / "handbook.json", {"nonce": rec2["nonce"], "questions": []})
    rep = ingest(rd)
    assert rep["finished_without_marker"] == [t2["task_id"]] and rep["reported_but_missing"] == [t1["task_id"]]
    assert len(read_jsonl(rd / "dispatch" / "ingest_log.jsonl")) == 2
    txt = read_text(rd / "dispatch" / "dispatch_instructions.md")
    assert "reported_but_missing" in txt and "ingest_log.jsonl" in txt
