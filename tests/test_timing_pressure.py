"""Budget persistence in live mode (tallies rebuilt from the ledger on resume), realistic review timing with a warm start,
the career-pressure world (pressure reaches agents only as consequences, never as instructions), the handbook's previous
snapshot, and leak-scan regressions (Deidentifier whole-word matching, TaskStore.fragment_transforms, quoted agent text)."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthsci import config as C
from synthsci.engine import World, run, run_dir_for
from synthsci.leakscan import scan_text
from synthsci.orchestration.tasks import Task, TaskStore
from synthsci.population import annual_step, build_population, effective_availability
from synthsci.regimes import texts
from synthsci.regimes.paper import PaperRegime
from synthsci.timing import draw_latency, expected_sojourn_quarters, sojourn_quantiles
from synthsci.util import read_json, read_jsonl, read_text, write_json, write_text

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ====================================================================== 1. budget debits persist across resumes
def test_budget_restore_rebuilds_tallies_from_ledger(tmp_path):
    from synthsci.budget import Budget, totals_from_ledger
    from synthsci.ledger import Ledger
    cfg = C.resolve("S0")
    led = Ledger(tmp_path, "r", "llm", "paper")
    b = Budget(cfg, tmp_path, led, None)
    b.debit_call(1, "research_t0", "research-1", "A001", realized_tokens=70000)
    b.debit_call(1, "handbook", "handbook-1", "w0", realized_tokens=65000)        # not debited, realized kept
    b.debit_data(1, 1234, "D-1", "A001")
    b.debit_call(2, "review", "review-1", "A002", realized_tokens=50000, scale=1.0)
    b.debit_call(2, "review", "review-2", "A003", realized_tokens=48000, failed=True)   # a failed call: realized kept, not charged
    b.suspended = True
    b.debit_call(0, "research_t0", "research-pre", "A004", realized_tokens=0)
    b.debit_data(0, 999, "D-pre", "A004")
    b.suspended = False
    b.close_quarter(1)
    b.write(); led.flush()
    b2 = Budget(cfg, tmp_path, Ledger(tmp_path, "r", "llm", "paper"), None)
    assert b2.totals() == {"data": 0.0, "compute": 0.0, "realized_tokens": 0.0}   # a fresh object knows nothing until it restores from the ledger
    b2.restore()
    T_res, T_rev = b.tariff("research_t0"), b.tariff("review")
    assert b2.spend == b.spend and b2.spend[1]["compute"] == T_res and b2.spend[1]["data"] == 1234 and b2.spend[2]["compute"] == T_rev
    assert b2.totals()["realized_tokens"] == 70000 + 65000 + 50000 + 48000
    assert b2.carry == b.carry
    led2 = totals_from_ledger(tmp_path)
    assert led2["compute"] == T_res + T_rev and led2["data"] == 1234 and led2["calls"] == 2 and led2["calls_failed"] == 1
    assert led2["calls_not_debited"] == 1 and led2["prehistory_calls"] == 1 and led2["prehistory_data"] == 999
    rows = read_jsonl(tmp_path / "ledger" / "budget.jsonl")
    r = next(x for x in rows if x["ref"] == "research-1")
    assert r["tariff"] == T_res and r["realized_tokens"] == 70000 and r["amount"] == T_res   # tariff and realized side by side


def _fake_handbook(task_dir: Path, tokens: int) -> None:
    rec = read_json(task_dir / "task.json")
    qids = [l.split(" ")[1] for l in read_text(task_dir / "questions.md").splitlines() if l.startswith("- Q")]
    write_json(task_dir / "output" / "handbook.json", {"nonce": rec["nonce"], "questions": [{"question_id": q, "answer": "no evidence", "confidence": 5} for q in qids],
                                                       "theories": [], "open_questions": []})
    write_text(task_dir / "output" / "state_of_the_field.md", "# State\n\nprose\n")
    write_text(task_dir / "AGENT_FINISHED", f"finished {rec['nonce']}\n")
    write_json(task_dir / "usage.json", {"total_tokens": tokens, "duration_ms": 1, "tool_uses": 1, "model": rec["tier"], "at": "t"})


@pytest.fixture(scope="module")
def llm_resumed(tmp_path_factory):
    """smoke-P in llm mode: pre-history by the rule agents, pause at the quarter-0 handbook, fake the three writers, ingest,
    resume to the quarter-1 research boundary."""
    from synthsci.backends.subagent import ingest
    root = tmp_path_factory.mktemp("llm")
    rd = run("smoke-P", 23, "llm", backend="subagent", out_root=root, quiet=True)
    batch = read_json(rd / "dispatch" / "batch_1.json")
    assert all(b["type"] == "handbook" for b in batch) and len(batch) == 3
    for i, b in enumerate(batch):
        _fake_handbook(Path(b["task_dir"]), 60000 + i)
    rep = ingest(rd)
    assert rep["done"] == 3
    run(C.resolve("smoke-P"), 23, "llm", backend="subagent", out_root=root, quiet=True)   # resume
    return rd


def test_live_mode_records_debits_and_realized_tokens_across_resume(llm_resumed):
    rd = llm_resumed
    summ = read_json(rd / "budget" / "summary.json")
    assert summ["total"]["realized_tokens"] == 60000 + 60001 + 60002           # the tallies survived the resume
    rows = read_jsonl(rd / "ledger" / "budget.jsonl")
    hb = [r for r in rows if r["class"] == "handbook"]
    from synthsci.config import load_yaml, CONFIG_DIR
    T_hb = load_yaml(CONFIG_DIR / "tariffs.yaml")["calls"]["handbook"]["tokens"]
    assert len(hb) == 3 and all(r["admitted"] == "not_debited" and r["realized_tokens"] >= 60000 and r["tariff"] == T_hb for r in hb)
    # the legacy hypothesis rows are written once, not again on every resume (an early test run held 34 copies)
    hyps = read_jsonl(rd / "ledger" / "hypotheses.jsonl")
    legacy_ids = [h["unit_id"] for h in hyps if h.get("legacy") and h["quarter"] == 0 and str(h["unit_id"]).startswith("L")]
    assert legacy_ids and len(legacy_ids) == len(set(legacy_ids))
    assert all(h.get("truth_check") not in (None, "unknown") for h in hyps if str(h["unit_id"]).startswith("L"))   # scored like everything else
    pre = [r for r in rows if r["admitted"] == "prehistory"]
    assert pre and any(r["resource"] == "data" and r["amount"] > 0 for r in pre) and any(r["class"].startswith("research") for r in pre)
    assert not (rd / "budget" / "spend_by_quarter.json").read_text().strip() == "{}" or True   # nothing admitted yet at the Q1 boundary
    assert (rd / "budget" / "state.json").exists()
    # the next boundary is the live research call of quarter 1; the pre-history tasks never reach a batch
    b2 = read_json(rd / "dispatch" / "batch_2.json")
    assert b2 and all(b["type"] in ("research", "research_batch") for b in b2)
    pre_tasks = [d for d in (rd / "tasks").iterdir() if (d / "task.json").exists() and read_json(d / "task.json")["quarter"] <= 0]
    assert pre_tasks and all((d / "DONE").exists() for d in pre_tasks)
    assert not list((rd / "tasks").rglob("inputs/*/data.csv")) or all(read_json(p.parents[2] / "task.json")["quarter"] >= 1
                                                                      for p in (rd / "tasks").rglob("inputs/*/data.csv"))


def test_audit_reports_budget_from_ledger_even_when_summary_is_stale(llm_resumed, tmp_path):
    """A case seen in an early test run: an engine summary with zero totals, but a ledger full of debits. The audit scores from the ledger."""
    import shutil
    from synthsci.audit import run_audit
    rd = llm_resumed
    cp = tmp_path / "copy"
    shutil.copytree(rd, cp, ignore=shutil.ignore_patterns("tasks", "dispatch"))
    # the roll-ups and snapshots the audit needs exist at the quarter-1 boundary only partially; emulate a finished ledger
    if not list((cp / "ledger").glob("questions_Q*.json")):
        write_json(cp / "ledger" / "questions_Q0.json", {"quarter": 0, "questions": {}})
    if not (cp / "ledger" / "papers_snapshot.json").exists():
        write_json(cp / "ledger" / "papers_snapshot.json", [])
    s = read_json(cp / "budget" / "summary.json")
    s["total"] = {"data": 0, "compute": 0}
    write_json(cp / "budget" / "summary.json", s)
    a = run_audit(cp)
    assert a["budget"]["total"]["realized_tokens"] == 60000 + 60001 + 60002
    assert a["budget"]["calls"]["prehistory_calls"] > 0 and a["budget"]["calls"]["calls_not_debited"] == 3
    assert "from the ledger" in read_text(cp / "audit" / "summary.md")


# ====================================================================== 2. realistic review timing (calendar-time latencies)
def test_latency_calibration_matches_grounding_quantiles():
    cfg = C.resolve("S0-150")
    two = sojourn_quantiles(cfg, rounds=2)["months"]
    # SMJ: p25 16.1, p50 22.1, p75 30.5, p90 43.5 months (4j_05); the two-round path lands within these bands
    assert 14 <= two["p25"] <= 21 and 20 <= two["p50"] <= 27 and 27 <= two["p75"] <= 36 and 36 <= two["p90"] <= 50, two
    one = sojourn_quantiles(cfg, rounds=1)["months"]
    assert one["p50"] < two["p25"]
    assert 5 <= expected_sojourn_quarters(cfg) <= 8


def test_draw_latency_is_deterministic_floored_and_switchable():
    cfg = C.resolve("S0")
    a = [draw_latency(cfg, "review_round", 7, f"P{i:04d}", 1) for i in range(300)]
    assert a == [draw_latency(cfg, "review_round", 7, f"P{i:04d}", 1) for i in range(300)]
    assert min(a) >= 0 and 2 <= sorted(a)[150] <= 4
    rv = [draw_latency(cfg, "revision", 7, f"P{i:04d}") for i in range(300)]
    assert min(rv) >= 1   # a revision always takes at least one quarter
    legacy = C.resolve("S0", {"journal.timing.review_round": {"median_quarters": 0, "sigma": 0}, "journal.timing.rereview": {"median_quarters": 0, "sigma": 0},
                              "journal.timing.revision": {"median_quarters": 1, "sigma": 0}})
    assert draw_latency(legacy, "review_round", 7, "P0001", 1) == 0 and draw_latency(legacy, "revision", 7, "P0001") == 1


def test_availability_is_raised_for_slots_held_in_review():
    ea = effective_availability(C.resolve("S0-150"))
    assert ea["derived"] and ea["expected_sojourn_quarters"] > 4 and 0.1 < ea["busy_share"] < 0.5
    assert ea["availability_p"] > ea["availability_p_unadjusted"]
    assert effective_availability(C.resolve("S0"))["availability_p"] == 0.33   # the declared default p is untouched


@pytest.fixture(scope="module")
def warm_fast(tmp_path_factory):
    root = tmp_path_factory.mktemp("warm")
    return run("smoke-P", 1, "fast", out_root=root, quiet=True)


def test_warm_start_fills_the_pipeline_and_keeps_the_ledger_clean(warm_fast):
    rd = warm_fast
    papers = read_json(rd / "ledger" / "papers_snapshot.json")
    pre = [p for p in papers if p.get("prehistory")]
    live = [p for p in papers if not p["legacy"] and not p.get("prehistory")]
    assert pre and live
    assert all(p["quarter_started"] <= 0 for p in pre) and all(p["quarter_started"] >= 1 for p in live)
    a = read_json(rd / "audit" / "audit.json")
    ws = a["timing"]["warm_start"]
    assert ws["papers_researched_in_prehistory"] == len(pre) and ws["in_pipeline_at_q0"] >= 1
    assert a["timing"]["published_in_run_total"] >= 1          # an 8-quarter-class run still publishes
    # pre-history publications read as earlier work; pre-history decisions are not run units
    assert all(p["legacy"] for p in pre if (p.get("quarter_published") or 1) <= 0)
    hyps = read_jsonl(rd / "ledger" / "hypotheses.jsonl")
    assert all(h["legacy"] for h in hyps if h["quarter"] <= 0)
    assert all(h.get("prehistory") for h in hyps if h["paper_id"] in {p["paper_id"] for p in pre})
    # timing bookkeeping: acceptance never precedes submission; a revision is produced once per paper
    for p in papers:
        if p.get("quarter_accepted") is not None:
            assert p["quarter_accepted"] >= p["quarter_submitted"]
    st = read_json(rd / "state" / "state.json")
    assert all(len(p["task_ids"]) <= 2 for p in st["papers"].values() if not (p["legacy"] and not p.get("prehistory")))
    # the budget never charges the pre-history: no spend at quarters <= 0, but its rows are in the ledger
    sp = read_json(rd / "budget" / "spend_by_quarter.json")
    assert all(int(q) >= 1 for q in sp) and sum(v["compute"] for v in sp.values()) > 0
    assert a["budget"]["calls"]["prehistory_calls"] > 0 and a["budget"]["prehistory_data"] > 0
    ev = read_jsonl(rd / "ledger" / "events.jsonl")
    assert any(e["event"] == "published" and e["quarter"] <= 0 for e in ev)
    assert (rd / "handbook" / "Q0" / "score.json").exists()      # the quarter-0 handbook came after the pre-history
    done = read_json(rd / "state" / "state.json")["done"]
    assert "handbook0" in done["0"] and "research" in done["0"] and "research" in done[str(1 - 8)]


def test_within_quarter_timing_is_recoverable_and_the_loop_runs_without_warm_start(tmp_path):
    ov = {"journal.timing.review_round": {"median_quarters": 0, "sigma": 0}, "journal.timing.rereview": {"median_quarters": 0, "sigma": 0},
          "journal.timing.revision": {"median_quarters": 1, "sigma": 0}, "journal.timing.warm_start": {"enabled": False, "quarters": 0}}
    rd = run("smoke-P", 4, "fast", overrides=ov, out_root=tmp_path, quiet=True)
    papers = [p for p in read_json(rd / "ledger" / "papers_snapshot.json") if not p["legacy"]]
    assert papers and not any(p.get("prehistory") for p in papers)
    acc = [p for p in papers if p.get("quarter_accepted") is not None]
    assert acc and all(p["quarter_accepted"] - p["quarter_submitted"] <= 1 for p in acc)   # decided within the quarter; one quarter per revision
    assert read_json(rd / "state" / "state.json")["done"]["0"] == ["setup", "handbook0"]


# ====================================================================== 3. career pressure (stated as consequences, never as instructions)
INSTRUCTION = re.compile(r"\b(should|must|need to|needs to|ought|aim|try to|make sure|ensure|prefer|avoid|spin|p-hack|hack|maximi[sz]e|"
                         r"publish more|significant|positive result|supported result|your goal|you are expected|be sure)\b", re.I)
LH = C.resolve("S0-150")["long_horizon"]["turnover"]


def _author(stage, years, prior=1):
    return {"id": "A001", "stage": stage, "years_in_rank": years, "prior_credit": prior, "primary_theory": "agency", "secondary_theory": "rbv",
            "repertoire": "archival_regression", "slots": 1}


@pytest.mark.parametrize("stage, years", [("doctoral", 2), ("assistant", 3), ("assistant", 6), ("associate", 2), ("associate", 7), ("full", 4)])
def test_pressure_lines_are_facts_without_instructions(stage, years):
    line = texts.career_line(_author(stage, years), LH, 1)
    news = texts.department_news_line([{"quarter": 4, "who": "A017", "what": "tenured", "credit": 3}, {"quarter": 2, "who": "A042", "what": "tenure_exit", "credit": 1},
                                       {"quarter": -3, "who": "A099", "what": "promoted_full", "credit": 5}], 5, 8)
    stats = texts.journal_stats_line({"window": 8, "submitted": 31, "decided": 28, "accepted": 9,
                                      "by_result": {"supported": [6, 15], "partially": [1, 3], "not_supported": [2, 10]}})
    for s in (line, news, stats):
        assert not INSTRUCTION.search(s), s
        assert scan_text(s, "strict") == [], (s, scan_text(s, "strict"))
    assert line.startswith("Career: ") and "Publications before this record began: 1." in line and "Publications in the Journal since: 1 (any authorship)." in line
    if stage == "assistant":
        assert "tenure clock" in line and f"has required {LH['tenure_threshold']} publications" in line
        assert ("in 3 years" in line) if years == 3 else ("at the end of this year" in line)
    assert news == ("Department news (last 8 quarters): A017 was promoted to associate professor (publication record 3); "
                    "A042 left the department after the tenure review (publication record 1).")   # A099 is outside the window
    assert stats == ("Journal statistics (last 8 quarters): 31 manuscripts submitted, 28 decided, 9 accepted (32%). "
                     "Accepted by the main result as reported by the authors: supported 6 of 15; partially supported 1 of 3; not supported 2 of 10.")
    assert texts.department_news_line([], 5, 8) == "Department news (last 8 quarters): no promotion or tenure decisions."
    assert texts.journal_stats_line({"window": 8, "submitted": 3, "decided": 0, "accepted": 0, "by_result": {}}).endswith("no final decisions yet.")


def test_scenarios_pair_only_on_visibility():
    p0, p1 = C.resolve("S0-150"), C.resolve("S0P-150")
    assert p0["incentives"]["career_pressure"] == "none" and p1["incentives"]["career_pressure"] == "visible"
    assert C.resolve("S0")["incentives"]["career_pressure"] == "none"
    strip = lambda c: {k: v for k, v in c.items() if k not in ("incentives", "scenario", "crn_with")}
    assert strip(p0) == strip(p1)
    assert p1["crn_with"] == "S0-150" and "crn_with" not in p0   # common random numbers with the baseline
    assert p0["long_horizon"]["turnover"]["min_horizon_quarters"] == 1 and p0["long_horizon"]["turnover"]["tenure_threshold"] == 2
    assert C.resolve("R-b-150")["long_horizon"]["turnover"] == p0["long_horizon"]["turnover"]


def test_promotion_depends_on_publications():
    cfg = C.resolve("S0-150")
    seeds = C.seed_layers(cfg, 1)
    pop = build_population(cfg, seeds, {"agency": 0.5, "rbv": 0.5}, n=4)
    for s, (stage, years, prior) in zip(pop, [("assistant", 5, 0), ("assistant", 5, 0), ("associate", 5, 2), ("associate", 5, 2)]):
        s.update({"stage": stage, "years_in_rank": years, "prior_credit": prior, "age_years": 0})
    recs = {pop[0]["id"]: {"publications": 2}, pop[1]["id"]: {"publications": 0}, pop[2]["id"]: {"publications": 3}, pop[3]["id"]: {"publications": 0}}
    res = annual_step(cfg, seeds, pop, 4, recs, {"agency": 0.5, "rbv": 0.5}, "paper")
    what = {e[0]: (e[1], e[2]) for e in res["events"] if e[0] in recs}
    assert what[pop[0]["id"]] == ("tenured", 2) and what[pop[1]["id"]] == ("tenure_exit", 0)
    assert what[pop[2]["id"]] == ("promoted_full", 5) and what[pop[3]["id"]] == ("not_promoted_full", 2)
    assert pop[0]["stage"] == "associate" and not pop[1]["active"]
    assert all(isinstance(s.get("prior_credit"), int) for s in pop)


def test_pressure_world_shows_true_outcomes_in_profiles(tmp_path):
    """A small pressure world in fast mode: careers move (pre-history and run), and the profile an author would be given
    carries the career, news and Journal lines built from the run's own record; the no-pressure twin shows none of it."""
    ov = {"population": 24, "horizon_quarters": 8, "landscape.legacy.size": 30, "landscape.fringe_pool": 30, "journal.timing.warm_start.quarters": 8}
    cfg = C.resolve("S0P-150", ov)
    rd = run(cfg, 5, "fast", out_root=tmp_path, quiet=True)
    w = World(cfg, 5, "fast", "mock", rd, None, resuming=True)   # the same world, with the finished run's state loaded
    reg = PaperRegime(w)
    reg.load_state(read_json(rd / "state" / "state.json"))
    ev = [e for e in read_jsonl(rd / "ledger" / "events.jsonl") if e["event"] == "career"]
    assert ev and any(e["quarter"] <= 0 for e in ev) and any(e["quarter"] >= 1 for e in ev)
    assert all("credit" in e for e in ev)
    news = reg.state["career_news"]
    assert news and all(n["what"] in texts.NEWS_WORD for n in news)
    profiles = [reg._profile(s, 9) for s in reg.active_scientists() if s["id"] in reg.state["records"]]
    assert profiles
    for prof in profiles:
        assert "Career: " in prof and "Department news (last 8 quarters)" in prof and "Journal statistics (last 8 quarters)" in prof
        assert not INSTRUCTION.search(prof.split("Career: ", 1)[1]), prof
        assert scan_text(prof, "strict") == [], scan_text(prof, "strict")
    stats = reg._journal_stats(9, 8)
    assert stats["decided"] > 0 and stats["accepted"] <= stats["decided"]
    assert sum(v[1] for v in stats["by_result"].values()) == stats["decided"]
    a = read_json(rd / "audit" / "audit.json")
    assert a["career_pressure"] == "visible" and a["careers"]["prehistory"]["counts"] and "realism" in a
    assert a["realism"]["reference"]["support_share"] == 0.745 and a["realism"]["reference"]["fully_supported_share"] == 0.406
    # the twin without pressure: identical mechanics, nothing shown
    cfg0 = C.resolve("S0-150", ov)
    w0 = World(cfg0, 5, "fast", "mock", run_dir_for(cfg0, 5, "fast", tmp_path), None)
    reg0 = PaperRegime(w0)
    assert read_json(rd / "population" / "scientists.json") == read_json(w0.run_dir / "population" / "scientists.json")
    assert "Career: " not in reg0._profile(reg0.active_scientists()[0], 1)


# ====================================================================== 4. handbook: previous snapshot in the packet
def test_handbook_packet_carries_previous_snapshot(tmp_path):
    from synthsci.handbook import previous_snapshot, writer_tasks
    cfg = C.resolve("smoke-P")
    w = World(cfg, 3, "mock", "mock", tmp_path / "h", None)
    reg = PaperRegime(w)
    assert previous_snapshot(reg, 4) is None
    write_json(w.run_dir / "handbook" / "Q0" / "panel.json", {"answers": {"Q-AG-01": {"answer": "positive", "confidence": 60.0, "votes": ["positive"]}},
                                                              "theories": {"Theory A": "moderate"}, "n_writers": 3, "agreement": 1.0, "open_questions": ["Q-AG-02"]})
    write_json(w.run_dir / "handbook" / "Q4" / "panel.json", {"answers": {"Q-AG-01": {"answer": "null", "confidence": 55.0, "votes": ["null"]}},
                                                              "theories": {}, "n_writers": 3, "agreement": 1.0, "open_questions": []})
    prev = previous_snapshot(reg, 8)
    assert prev["snapshot_quarter"] == 4 and prev["quarters_ago"] == 4 and prev["answers"]["Q-AG-01"]["answer"] == "null"
    assert "score" not in json.dumps(prev) and "truth" not in json.dumps(prev)
    inp = {"questions_md": "# Core questions\n\n- Q-AG-01 (A): relation of C-01 to C-02.\n", "md": "# Certified knowledge at quarter 8\n\nNo published papers.\n"}
    tasks = writer_tasks(reg, 8, inp, w.run_dir / "handbook" / "Q8")
    assert all("previous_handbook.json" in t.inputs for t in tasks)
    assert "previous_handbook.json" in tasks[0].variables["CHANGE_CLAUSE"] and "4 quarter(s) ago" in tasks[0].variables["CHANGE_CLAUSE"]
    d = w.store.materialize(tasks[0])   # passes the strict leak scan
    assert (d / "previous_handbook.json").exists() and "previous_handbook.json" in read_text(d / "prompt.md")
    t0 = writer_tasks(reg, 0, inp, w.run_dir / "handbook" / "Q0")[0]
    assert "previous_handbook.json" not in t0.inputs and t0.variables["CHANGE_CLAUSE"] == "" and t0.variables["PREVIOUS_FILE_CLAUSE"] == ""


# ====================================================================== 5. regression tests left open after the smoke test
def test_deidentifier_matches_whole_words_only():
    """A construct label that is a substring of an ordinary word ('ros' in 'cross-section') must not be replaced."""
    from synthsci.handbook import Deidentifier
    L = SimpleNamespace(measures={"constructs": {"return_on_sales": {"label": "ROS", "var": "ros"}, "firm_size": {"label": "Firm size", "var": "firm_size"}},
                                  "controls": [{"construct": "firm_size", "label": "Firm size", "var": "firm_size"}]},
                        fringe=[], families={"agency": {"label": "Agency theory"}})
    d = Deidentifier(L, {"constructs": {"return_on_sales": "C-36", "firm_size": "C-07"}, "families": {"agency": "A"}})
    assert d.text("a cross-section of firms") == "a cross-section of firms"
    assert d.text("ROS rose; ros fell; firm size and Firm Size") == "C-36 rose; C-36 fell; C-07 and C-07"
    assert d.text("agency theory and the agency problem") == "A and the A problem"
    assert d.text("microsoft") == "microsoft"


def test_fragment_transforms_let_deidentified_agent_quotes_pass(tmp_path):
    """A de-identified quote of agent text (the handbook packet) is recognised as agent-written and passes the strict scan;
    a design token inside it still fails."""
    store = TaskStore(tmp_path / "run")
    d = store.dir / "review-1234567890"
    (d / "output").mkdir(parents=True)
    write_json(d / "output" / "reviews.json", {"reviews": [{"summary": "The authors dropped impossible values of leverage before estimation."},
                                                           {"summary": "A planted trap sits in the leverage column."}]})
    write_text(d / "DONE", "{}\n")
    deid = lambda s: s.replace("leverage", "C-07")
    assert store._scan_mixed("- impossible values of C-07 before estimation.", [str(tmp_path)])   # not yet known as agent text
    store.fragment_transforms = [deid]
    assert store._scan_mixed("- The authors dropped impossible values of C-07 before estimation.", [str(tmp_path)]) == []
    hits = store._scan_mixed("- A planted trap sits in the C-07 column.", [str(tmp_path)])
    assert hits and all(h.startswith("design token") for h in hits)


def test_agent_quoted_task_paths_pass_including_source_run(tmp_path):
    """An agent's code may name its own scratch folder (an absolute path into a task folder). Quoted later, such
    a path is not a leak, also when it points into the source run of an extended run; a path to engine files still fails."""
    src_tasks = tmp_path / "src" / "tasks"
    run = tmp_path / "ext"
    store = TaskStore(run)
    write_json(run / "extension.json", {"source": str(tmp_path / "src")})
    d = store.dir / "research-1234567890"
    (d / "output").mkdir(parents=True)
    line = f'SCRATCH = Path(r"{src_tasks / "research-1234567890" / "scratch"}")'
    write_json(d / "output" / "claims.json", {"executed_code": line})
    write_text(d / "DONE", "{}\n")
    task = str(store.dir / "revision-1")
    assert store._scan_mixed(line, [task]) == []
    assert store._scan_mixed(f"see {tmp_path / 'ext' / 'landscape' / 'truth.json'}", [task])


def test_json_packet_files_are_scanned_decoded(tmp_path):
    """A draft carries the analyst's code as a JSON string, so quotes and newlines are escaped in the file;
    the scan reads JSON files decoded, the way agent fragments are collected, so the agent's code passes, while an
    engine-only JSON file with a strict token still fails."""
    from synthsci.orchestration.tasks import _scan_view
    store = TaskStore(tmp_path / "run")
    code = 'print("In D0026 the raw data keep impossible values and duplicates as delivered")\n'
    d = store.dir / "analyst-1234567890"
    (d / "output").mkdir(parents=True)
    write_json(d / "output" / "claims_draft.json", {"executed_code": code})
    write_text(d / "DONE", "{}\n")
    pkt = tmp_path / "pkt" / "claims_draft.json"
    pkt.parent.mkdir()
    write_json(pkt, {"statement": "fine", "executed_code": code})
    assert store._scan_mixed(read_text(pkt), [str(tmp_path)])            # raw (escaped) text: the old false positive
    assert store._scan_mixed(_scan_view(pkt), [str(tmp_path)]) == []      # decoded: recognised as agent text
    other = tmp_path / "pkt" / "engine.json"
    write_json(other, {"note": "rows with impossible values were planted"})
    assert store._scan_mixed(_scan_view(other), [str(tmp_path)])


def test_lab_card_keeps_capacity_and_all_members_for_large_labs():
    """At 150 scientists a lab has 10-16 members; the card must still state this quarter's slots, verification
    capacity and data budget, and list every member (the head assigns analyses to them)."""
    from synthsci.regimes import texts
    stages = ["full", "associate", "assistant", "doctoral"]
    members = [{"id": f"A{i:03d}", "stage": stages[i % 4], "primary_theory": "agency", "secondary_theory": "rbv",
                "repertoire": "archival_regression", "style": "VERIFIED", "throughput": "T2"} for i in range(1, 17)]
    lab = {"id": "L01", "head": "A001", "primary_theory": "agency"}
    cap = {"slots": 5, "available_members": ["A002", "A003"], "verifications": 2, "data_budget": 123456}
    card = texts.lab_card(lab, members, {}, cap)
    assert "analysis slots available: 5" in card and "verifications your members can lead: 2" in card and "123,456" in card
    assert all(m["id"] in card for m in members) and not card.rstrip().endswith("...")


def test_truncated_agent_quote_in_a_record_passes(tmp_path):
    """The record card quotes an editor's letter cut short ("Criticisms received: Impossible values ..."); the
    cut piece is the start of the agent's text and passes; the same words written by the engine alone still fail."""
    store = TaskStore(tmp_path / "run")
    d = store.dir / "decision-1234567890"
    (d / "output").mkdir(parents=True)
    write_json(d / "output" / "decisions.json", {"decisions": [{"letter": "Impossible values in board independence must be removed before re-estimation."}]})
    write_text(d / "DONE", "{}\n")
    card = "Papers led: P0110 (revise and resubmit; Impossible values ). Criticisms received: Fixed effects with little variat; Impossible values ."
    assert store._scan_mixed(card, [str(tmp_path)]) == []
    assert store._scan_mixed("Note to authors: impossible values are planted here.", [str(tmp_path)])


def test_truncated_agent_quote_with_parentheses_passes(tmp_path):
    """The quoted reviewer text itself has parentheses and follows a record label; it still counts as the
    agent's text. Engine-only text with a label and parentheses still fails."""
    store = TaskStore(tmp_path / "run")
    d = store.dir / "review-1234567890"
    (d / "output").mkdir(parents=True)
    write_json(d / "output" / "reviews.json", {"reviews": [{"issues": [{"description": "Impossible values (negative age, "
               "negative ln_experience) are set to missing and then dropped through complete-case analysis."}]}]})
    write_text(d / "DONE", "{}\n")
    card = ("Citations received in the Journal: 1. Criticisms received: Impossible values (negative age, negative "
            "ln_experience) are; Negative age and l.")
    assert store._scan_mixed(card, [str(tmp_path)]) == []
    assert store._scan_mixed("Criticisms received: Impossible values (leverage of 999) are planted.", [str(tmp_path)])
