"""Agent task packets, the dispatch boundary, ingest, and the API backend stubs."""
import json
from pathlib import Path

import pytest

from synthsci import config as C
from synthsci.backends.base import PendingDispatch
from synthsci.engine import run
from synthsci.orchestration.tasks import PROMPTS, Task, TaskStore, render
from synthsci.util import read_json, read_text, write_json, write_text


def test_render_substitutes_and_includes():
    txt, sha = render("research.md", {"TASK_DIR": "C:\\x", "OUTPUT_DIR": "C:\\x\\output", "NONCE": "n1", "PROTOCOL_BLOCK": "P", "DEPTH_BLOCK": "D",
                                      "ROBUSTNESS_HEADING": "", "PAPER_SPEC": "1,200-1,800 words.", "DATA_PATH_RULE": "..\\inputs\\<question id>\\data.csv"})
    assert "{{" not in txt and "C:\\x\\output" in txt and len(sha) == 64
    with pytest.raises(KeyError):
        render("research.md", {"TASK_DIR": "x"})


def test_all_templates_exist_for_every_call_type():
    names = {p.name for p in PROMPTS.glob("*.md")}
    for t in ("research.md", "revision.md", "desk_screen.md", "review.md", "review_ai_written.md", "prescreen.md", "decision.md", "lab_decision.md",
              "analyst.md", "verification.md", "decisive_test.md", "handbook_writer.md", "prior_probe.md", "legacy_abstracts.md", "dispatch_stub.md"):
        assert t in names


def test_subagent_backend_writes_packets_and_stops(tmp_path):
    rd = run("smoke-P", 21, "llm", backend="subagent", out_root=tmp_path, quiet=True)
    d = rd / "dispatch"
    assert (d / "dispatch_instructions.md").exists() and (d / "batch_1.md").exists()
    txt = read_text(d / "dispatch_instructions.md")
    assert "verbatim" in txt and "ingest" in txt and "resume" in txt
    batch = read_json(d / "batch_1.json")
    assert batch and all("stub" in b and str(rd) in b["stub"] for b in batch)
    # the first boundary is the quarter-0 handbook (three writers); packets carry no hidden truth
    for b in batch:
        td = Path(b["task_dir"])
        assert (td / "prompt.md").exists() and not (td / "DONE").exists()
        for p in td.rglob("*"):
            if p.is_file():
                assert "theta" not in read_text(p).lower() or p.name == "task.json"
    state = read_json(rd / "state" / "state.json")
    assert state["quarter"] == 0   # paused inside quarter 0's handbook phase (the warm-start pre-history ran with the rule agents)
    # the warm-start pre-history never reaches the dispatch: every pre-history task is DONE by the rule agents, none is in the batch
    batch_ids = {b["task_id"] for b in batch}
    pre = [d for d in (rd / "tasks").iterdir() if d.is_dir() and (d / "task.json").exists()
           and read_json(d / "task.json")["quarter"] <= 0 and read_json(d / "task.json")["type"] != "handbook"]
    assert pre and all((d / "DONE").exists() and d.name not in batch_ids for d in pre)
    assert all(b["type"] == "handbook" for b in batch)


def test_ingest_validates_outputs(tmp_path):
    from synthsci.backends.subagent import ingest
    rd = run("smoke-P", 22, "llm", backend="subagent", out_root=tmp_path, quiet=True)
    batch = read_json(rd / "dispatch" / "batch_1.json")
    # fake a valid handbook output for the first writer and an invalid one for the second
    t0, t1 = batch[0], batch[1]
    rec0 = read_json(Path(t0["task_dir"]) / "task.json")
    qids = [l.split(" ")[1] for l in read_text(Path(t0["task_dir"]) / "questions.md").splitlines() if l.startswith("- Q")]
    hb = {"nonce": rec0["nonce"], "questions": [{"question_id": q, "answer": "no evidence", "confidence": 5} for q in qids], "theories": [], "open_questions": []}
    write_json(Path(t0["task_dir"]) / "output" / "handbook.json", hb)
    write_text(Path(t0["task_dir"]) / "output" / "state_of_the_field.md", "# State\n\nprose\n")
    write_text(Path(t0["task_dir"]) / "AGENT_FINISHED", f"finished {rec0['nonce']}\n")
    write_json(Path(t1["task_dir"]) / "output" / "handbook.json", {"nonce": "wrong", "questions": []})
    write_text(Path(t1["task_dir"]) / "AGENT_FINISHED", "finished wrong\n")
    rep = ingest(rd)
    assert rep["done"] == 1 and rep["failed"] == 1 and rep["pending"] == 1
    assert (Path(t0["task_dir"]) / "DONE").exists() and (Path(t1["task_dir"]) / "FAILED").exists()
    v = read_json(Path(t1["task_dir"]) / "validation.json")
    assert any("nonce" in p or "schema" in p or "missing" in p for p in v["problems"])


def test_api_stub_refuses_unpinned(tmp_path):
    from synthsci.backends.api_anthropic import AnthropicBackend
    cfg = C.resolve("S0")
    store = TaskStore(tmp_path)
    be = AnthropicBackend(cfg, tmp_path, store)
    with pytest.raises(RuntimeError):
        be.model_id("sonnet")


def test_freeze_roundtrip(monkeypatch, tmp_path):
    from synthsci import freeze as fz
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "FREEZE.sha256")
    assert fz.freeze() == 0
    assert fz.verify() == 0
