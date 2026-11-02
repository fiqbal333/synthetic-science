"""Scaling to 150 authors at a fixed cost: realistic productivity, batched research calls, lean paper contract."""
import json
import os
from pathlib import Path

import pytest

from synthsci import config as C
from synthsci.engine import run
from synthsci.population import effective_availability
from synthsci.util import hash_tree, read_json, read_jsonl, read_text


def test_default_availability_unchanged():
    cfg = C.resolve("S0")
    ea = effective_availability(cfg)
    assert ea["availability_p"] == 0.33 and ea["derived"] is False


def test_derived_availability_matches_target():
    cfg = C.resolve("S0-150")
    ea = effective_availability(cfg)
    assert ea["derived"] and ea["papers_per_author_year"] == 0.25
    # 0.25 papers/author-year, team 2.4, acceptance .6 -> .0434 starts per author-quarter; p = that / (1 slot x e^(.18))
    assert abs(ea["starts_per_author_quarter"] - 0.25 / (2.4 * 0.6 * 4)) < 1e-9
    assert 0.03 < ea["availability_p_unadjusted"] < 0.04
    # with calendar-time review a slot is held through review, so the drawn p is raised by the expected busy share
    assert ea["availability_p_unadjusted"] < ea["availability_p"] < 0.07
    # cost scales with papers: 150 authors at the grounded rate start about as many studies as the knob implies
    from synthsci.cost import estimate
    est = estimate(cfg)
    assert 5 < est["expected_starts_per_quarter"] < 9
    assert est["research_batch_size"] == 3 and est["paper_contract"] == "lean"
    est60 = estimate(C.resolve("S0"))
    assert est["paper_regime"]["tokens_per_quarter"] < est60["paper_regime"]["tokens_per_quarter"]


def test_paper_spec_variants():
    from synthsci.regimes.paper import PaperRegime
    full = PaperRegime._paper_spec(type("R", (), {"cfg": C.resolve("S0")})(), "")
    assert full.startswith("1,200-1,800 words of prose (the rendered table and the reference list do not count toward the limit) with these headings: "
                           "Title; Abstract (at most 150 words); Theory and Hypotheses")
    lean = PaperRegime._paper_spec(type("R", (), {"cfg": C.resolve("S0-150")})(), "")
    assert "500-800 words" in lean and "Discussion and Limitations" in lean


def test_grouping_rule(tmp_path):
    """Groups never share a menu question or a coauthor pair, and never exceed the batch size."""
    from synthsci.engine import World
    from synthsci.regimes.paper import PaperRegime
    cfg = C.resolve("smoke-P", {"journal.research_batch_size": 3, "population": 12})
    w = World(cfg, 41, "fast", "mock", tmp_path / "g", None)
    reg = PaperRegime(w)
    reg.state["coauthor_history"] = {"A001": ["A002"], "A002": ["A001"]}
    jobs = [{"kind": "research", "author": s, "slot": 0} for s in w.scientists]
    groups = reg._group_jobs(jobs, 1)
    assert sum(len(g) for g in groups) == len(jobs)
    assert all(len(g) <= 3 for g in groups)
    for g in groups:
        ids = [j["author"]["id"] for j in g]
        assert not ("A001" in ids and "A002" in ids)
        menus = [j["_menu_ids"] for j in g]
        for i in range(len(menus)):
            for j in range(i + 1, len(menus)):
                assert not (menus[i] & menus[j])
    assert any(len(g) > 1 for g in groups)


@pytest.fixture(scope="module")
def batched_mock(tmp_path_factory):
    root = tmp_path_factory.mktemp("batched")
    # no warm start: the test is about batched live calls, and with a warm pipeline the twelve authors' slots are held in
    # review at quarter 1, so the live quarters may never batch (seed 7 batched only in the pre-history, which the audit excludes)
    return run("smoke-P", 7, "mock", overrides={"journal.research_batch_size": 3, "journal.paper_contract": "lean",
                                               "journal.timing.warm_start.enabled": False}, out_root=root, quiet=True)


def test_batched_packets_and_ledger(batched_mock):
    rd = batched_mock
    man = read_jsonl(rd / "tasks" / "manifest.jsonl")
    batch_ids = [m["task_id"] for m in man if m["event"] == "created" and m["type"] == "research_batch"]
    assert batch_ids, "no batched research call was created"
    done = {m["task_id"] for m in man if m["event"] == "done"}
    assert all(b in done for b in batch_ids)
    d = rd / "tasks" / batch_ids[0]
    prompt = read_text(d / "prompt.md")
    assert "Never read, copy or mention another study's folder" in prompt and "500-800 words" in prompt
    authors = [p.name for p in (d / "authors").iterdir() if p.is_dir()]
    assert len(authors) >= 2
    for a in authors:
        assert (d / "authors" / a / "menu.md").exists() and (d / "authors" / a / "profile.md").exists()
        assert (d / "output" / a / "claims.json").exists() and (d / "output" / a / "paper.md").exists()
        assert read_text(d / "output" / a / "analysis.py").count(f"authors/{a}/inputs") >= 1   # each script reads only its own data
    papers = [p for p in read_json(rd / "ledger" / "papers_snapshot.json") if p.get("call_group")]
    assert papers and all(p["call_group_size"] >= 2 for p in papers)
    assert all(p["reproducible"] is True for p in papers if p["reproducible"] is not None)
    hyps = [h for h in read_jsonl(rd / "ledger" / "hypotheses.jsonl") if h.get("call_group")]
    assert hyps
    a = read_json(rd / "audit" / "audit.json")
    cg = a["call_groups"]
    assert cg["n_batched_units"] >= 2 and "false_discovery_agreement" in cg


def test_unbatched_runs_carry_no_call_group(tmp_path):
    rd = run("smoke-P", 8, "fast", out_root=tmp_path, quiet=True)
    assert not any(h.get("call_group") for h in read_jsonl(rd / "ledger" / "hypotheses.jsonl"))
    assert read_json(rd / "audit" / "audit.json")["call_groups"]["n_batched_units"] == 0


def test_150_pair_byte_identity_and_budget(tmp_path):
    ov = {"horizon_quarters": 8, "landscape.legacy.size": 40, "landscape.fringe_pool": 40}
    a = run("S0-150", 3, "fast", overrides=ov, out_root=tmp_path / "a", quiet=True)
    b = run("S0-150", 3, "fast", overrides=ov, out_root=tmp_path / "b", quiet=True)
    inc = ("ledger/", "audit/", "landscape/", "population/", "budget/")
    assert hash_tree(a, include=inc) == hash_tree(b, include=inc)
    r = run("R-b-150", 3, "fast", overrides=ov, out_root=tmp_path / "a", quiet=True)
    sp = read_json(a / "budget" / "spend_by_quarter.json")
    sr = read_json(r / "budget" / "spend_by_quarter.json")
    allot = read_json(r / "budget" / "summary.json")["allotments"]
    # pairing: R's quarterly compute allotment equals P's realized compute (plus carry-over); its data allotment is the
    # ceiling multiple x P's realized data; R's data never exceeds the ceiling (0.3% rounding) and R actually operates
    dcm = float(C.resolve("R-b-150")["budget"]["data_ceiling_multiple"])
    assert dcm == 4.0   # R-b-150 sets 4x (a 2x ceiling bound data at 150 scientists and left compute unspent)
    # (a quarter's allotment is the ceiling plus the carry-over, which is negative after an overspent quarter, so only the
    # first quarter is checked exactly; the cumulative cap is checked on the totals)
    assert abs(allot["1"]["data"] - dcm * sp["1"]["data"]) < 1e-6 and abs(allot["1"]["compute"] - sp["1"]["compute"]) < 1e-6
    tot_p = sum(v["data"] for v in sp.values())
    tot_r = sum(v["data"] for v in sr.values())
    assert tot_p > 0 and tot_r <= dcm * tot_p * 1.003, ("data", tot_p, tot_r)
    assert tot_r >= 0.6 * tot_p, (tot_p, tot_r)   # R operates (its data use is bounded by the compute the lab calls leave)
    # the audit reports data as an outcome beside the paired world and names the binding resource of every deferral
    ra = read_json(r / "audit" / "audit.json")["budget"]
    assert ra["per_unit"]["paired_world"]["data_ratio"] <= dcm * 1.003 and ra["per_unit"]["data_used"] == tot_r
    assert "data_per_correct_handbook_answer" in ra["per_unit"] and ra["per_unit"]["data_ceiling_multiple"] == dcm
    dbb = ra["deferrals_by_binding"]
    assert dbb["unrecorded"] == 0
    # compute: admission caps everything except the per-lab decision calls, which by design always run and are debited first;
    # at realistic productivity they alone can exceed the paper field's compute, so the invariant is on the admitted calls
    lab_calls = sum(b["amount"] for b in read_jsonl(r / "ledger" / "budget.jsonl") if b.get("resource") == "compute" and b.get("class") == "lab_decision")
    comp_p = sum(v["compute"] for v in sp.values())
    comp_r = sum(v["compute"] for v in sr.values())
    assert comp_r - lab_calls <= comp_p * 1.003, (comp_p, comp_r, lab_calls)
    assert read_json(a / "run.json")["population"] == 150 and read_json(a / "run.json")["availability"]["derived"] is True
    papers = [h for h in read_jsonl(a / "ledger" / "hypotheses.jsonl") if not h["legacy"]]
    assert papers


def test_relative_out_and_run_paths_are_resolved(tmp_path, monkeypatch):
    """A relative --out or --run must not put a relative path (which contains the mode name 'llm') into prompts."""
    from synthsci.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--scenario", "smoke-P", "--seed", "33", "--mode", "llm", "--backend", "subagent", "--out", "rel_runs", "--quiet"]) == 0
    rd = tmp_path / "rel_runs" / "smoke-P" / "33" / "llm"
    assert (rd / "dispatch" / "batch_1.md").exists()
    for p in (rd / "tasks").glob("*/prompt.md"):
        txt = read_text(p)
        assert "`rel_runs" not in txt and str(rd) in txt   # every path in the prompt is absolute
    # ingest / audit accept a relative --run path
    assert main(["ingest", "--run", "rel_runs/smoke-P/33/llm"]) == 0
    assert (rd / "dispatch" / "last_ingest.json").exists()


def test_agent_quoted_fragments_versus_engine_text(tmp_path):
    """(1) an engine-written line saying 'impossible values' fails the strict scan; (2) the same words inside a reviewer
    comment quoted from a DONE reviews.json pass (agent fragments get the design scan); (3) 'trap' fails anywhere."""
    from synthsci.orchestration.tasks import Task, TaskStore
    from synthsci.util import write_json, write_text
    store = TaskStore(tmp_path / "run")
    phrase = "The authors dropped observations with impossible values before estimation."
    # a DONE review task whose JSON output carries the phrase as a string field
    t0 = Task(type="review", quarter=1, identity="A009:0", tier="sonnet", run_id="r", call_type="review", template="review_ai_written.md",
              variables={"N_ITEMS": 1, "ITEM_LIST": "P0001"}, inputs={"inputs/P0001/paper.md": "ok"}, outputs=["reviews.json"], meta={})
    d0 = store.materialize(t0)
    write_json(d0 / "output" / "reviews.json", {"nonce": t0.nonce(), "reviews": [{"paper_id": "P0001", "recommendation": "accept", "flaw_found": False,
                                                 "issues": [{"location": "Method", "description": phrase, "severity": "minor", "kind": "data"}],
                                                 "validity_probability": 70, "relation_to_prior": "first study", "confidence": 0.7, "summary": "fine"}]})
    write_text(d0 / "DONE", "{}\n")

    def desk(identity, submissions):
        return Task(type="desk", quarter=1, identity=identity, tier="sonnet", run_id="r", call_type="desk_screen", template="desk_screen.md",
                    variables={"SUBMISSIONS": submissions, "STRICTNESS_SENTENCE": "s", "COUNT_CLAUSE": ""}, outputs=["desk.json"], meta={})
    # (2) the reviewer's words, quoted by the engine the way reviews.md renders them, pass
    store.materialize(desk("ok", f"### P0001\n\nIssues:\n- [minor, data] Method: {phrase}\n"))
    # (1) the engine's own line with the same words fails
    with pytest.raises(RuntimeError):
        store.materialize(desk("bad1", "### P0002\n\nThe outcome variable contains impossible values by construction.\n"))
    # (3) a design token fails even inside a quoted agent fragment
    t3 = Task(type="review", quarter=1, identity="A010:0", tier="sonnet", run_id="r", call_type="review", template="review_ai_written.md",
              variables={"N_ITEMS": 1, "ITEM_LIST": "P0003"}, inputs={"inputs/P0003/paper.md": "ok"}, outputs=["reviews.json"], meta={})
    d3 = store.materialize(t3)
    write_json(d3 / "output" / "reviews.json", {"nonce": t3.nonce(), "reviews": [{"paper_id": "P0003", "recommendation": "reject", "flaw_found": True,
                                                 "issues": [{"location": "Data", "description": "This looks like a planted trap in the dataset.", "severity": "major", "kind": "data"}],
                                                 "validity_probability": 20, "relation_to_prior": "unclear", "confidence": 0.6, "summary": "no"}]})
    write_text(d3 / "DONE", "{}\n")
    with pytest.raises(RuntimeError):
        store.materialize(desk("bad2", "### P0003\n\nIssues:\n- [major, data] Data: This looks like a planted trap in the dataset.\n"))
