"""Extend a finished run to a longer horizon under one rulebook (how the 8-quarter runs were continued to 20 quarters).

(a) a run made to the short horizon, closed, then extended equals the run made in one go with the extended config
    (paper world, and repository world paired with the extended paper world), quarter by quarter;
(b) the source run is byte-identical after `extend`;
(c) reopened papers keep their prior status, due quarters, queue place and slot;
(d) with the pin off the long-horizon behaviour is the old one, and existing configs keep their hashes;
(e) the pressure world's extension shares the baseline's random numbers (common random numbers, CRN).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from synthsci import config as C
from synthsci.cli import main as cli_main
from synthsci.engine import crn_namespace, run
from synthsci.extend import PIPELINE_STATUSES, extend, extended_config, manifest_lines, task_namespaces
from synthsci.util import read_json, read_jsonl, read_text

SHORT, LONG, SEED = 4, 8, 3
# careers move at every annual step (as in the Stage 1 scenarios), so the annual step and fringe renewal are exercised
OV = {"datasets": {"keep_csv": False}, "horizon_quarters": SHORT, "long_horizon": {"turnover": {"min_horizon_quarters": 1}}}
TABLES = ("hypotheses", "papers", "claims", "votes", "verifications", "reviews", "decisions", "datasets", "budget", "events")
WS = Path(__file__).resolve().parents[1]


def _cfg(name: str) -> dict:
    return C.deep_merge(C.resolve(name), OV)


def _stored(rd: Path) -> dict:
    return yaml.safe_load(read_text(rd / "config.resolved.yaml"))


def _is_extension_row(r: dict) -> bool:
    return r.get("event") == "extension_reopened" or (r.get("event") == "unfinished_at_horizon" and int(r.get("quarter", -1)) == SHORT)


def _ledger(rd: Path, table: str, drop_extension_rows: bool) -> list[dict]:
    rows = read_jsonl(rd / "ledger" / f"{table}.jsonl")
    if drop_extension_rows:
        rows = [r for r in rows if not _is_extension_row(r)]
    return [{k: v for k, v in r.items() if k != "run_id"} for r in rows]


def _assert_same_run(ext: Path, straight: Path, regime: str) -> None:
    import json
    for t in TABLES:
        a, b = _ledger(ext, t, True), _ledger(straight, t, False)
        assert len(a) == len(b), (t, len(a), len(b))
        if regime == "repository" and t == "hypotheses":
            # finish() writes this table by iterating state['claims']; a state reloaded from state.json (written with sorted
            # keys, as at every resume of a live run) lists C... before LC..., so only the row order can differ
            canon = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
            assert canon(a) == canon(b), t
        else:
            assert a == b, t
    rollups = lambda d: sorted(p.name for p in (d / "ledger").glob("questions_Q*.json"))
    assert rollups(ext) == rollups(straight)
    for name in rollups(ext):
        assert read_json(ext / "ledger" / name) == read_json(straight / "ledger" / name), name
    assert read_json(ext / "state" / "state.json") == read_json(straight / "state" / "state.json")
    assert read_json(ext / "state" / "world.json") == read_json(straight / "state" / "world.json")
    for f in ("spend_by_quarter.json", "state.json"):
        assert read_json(ext / "budget" / f) == read_json(straight / "budget" / f), f
    snap = "papers_snapshot.json" if regime == "paper" else "registry_snapshot.json"
    key = "paper_id" if regime == "paper" else "claim_id"   # rows follow the state dict's order (see the hypotheses note)
    srt = lambda rows: sorted(rows, key=lambda r: r[key])
    assert srt(read_json(ext / "ledger" / snap)) == srt(read_json(straight / "ledger" / snap))
    hb = lambda d: sorted(p.parent.name for p in (d / "handbook").glob("Q*/score.json"))
    assert hb(ext) == hb(straight)
    for q in hb(ext):
        assert read_json(ext / "handbook" / q / "score.json") == read_json(straight / "handbook" / q / "score.json"), q
    ae, as_ = read_json(ext / "audit" / "audit.json"), read_json(straight / "audit" / "audit.json")
    keys = ["field", "theory", "author", "handbook", "realism", "timing", "careers", "call_groups"]
    keys += ["failure_indices", "verification_power"] if regime == "repository" else []
    for k in keys:
        assert ae[k] == as_[k], k
    for k in ("total", "calls", "allotments", "deferrals_by_binding"):
        assert ae["budget"].get(k) == as_["budget"].get(k), k


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """Short runs (paper, repository paired with it) closed at quarter 4; the extended copies resumed to 8; the same
    configs made in one go."""
    root = tmp_path_factory.mktemp("extend")
    short = root / "short"
    p_short = run(_cfg("smoke-P"), SEED, "fast", out_root=short, quiet=True)
    r_short = run(_cfg("smoke-R"), SEED, "fast", out_root=short, quiet=True)
    before = {d: manifest_lines(d) for d in (p_short, r_short)}
    src_state = read_json(p_short / "state" / "state.json")

    ext_root = root / "ext"
    # budget pairing: the repository run's allotment comes from its paper run, so it cannot be extended before that run is
    with pytest.raises(RuntimeError, match="extend the paired paper run first"):
        extend(r_short, LONG, ext_root)
    assert cli_main(["extend", "--run", str(p_short), "--to", str(LONG), "--out-root", str(ext_root)]) == 0
    p_ext = ext_root / f"smoke-P-x{LONG}" / str(SEED) / "fast"
    reopened_state = read_json(p_ext / "state" / "state.json")
    with pytest.raises(FileExistsError):
        extend(p_short, LONG, ext_root)      # nothing is written over
    r_info = extend(r_short, LONG, ext_root)
    r_ext = Path(r_info["run_dir"])
    # the repository run waits at its own boundary while the extended paper run has not closed quarter 5
    assert cli_main(["resume", "--run", str(r_ext)]) == 0
    st = read_json(r_ext / "state" / "state.json")
    assert st["quarter"] == SHORT + 1 and not st["done"].get(str(SHORT + 1)), "no phase of quarter 5 may run before the paper run closes it"
    assert not (r_ext / "ledger" / "hypotheses.jsonl").exists()
    assert cli_main(["resume", "--run", str(p_ext)]) == 0
    assert cli_main(["resume", "--run", str(r_ext)]) == 0

    # the same config made in one go. It is built from the scenario file (not from the YAML-reloaded copy) because the first
    # process of a run draws the population in the config's key order (population_profile.stages), and the short run's
    # first process, like every live run's, started from the scenario file; config.resolved.yaml stores sorted keys
    straight = root / "straight"
    pin = {"horizon_quarters": LONG, "long_horizon": {"pin_short_rules": True, "extended_from": [SHORT]}}
    p_cfg, r_cfg = C.deep_merge(_cfg("smoke-P"), pin), C.deep_merge(_cfg("smoke-R"), pin)
    assert C.config_hash(p_cfg) == C.config_hash(extended_config(_stored(p_short), LONG))
    assert C.config_hash(r_cfg) == C.config_hash(extended_config(_stored(r_short), LONG))
    p_str = run(p_cfg, SEED, "fast", out_root=straight, quiet=True)
    r_str = run(r_cfg, SEED, "fast", out_root=straight, quiet=True)
    return SimpleNamespace(p_short=p_short, r_short=r_short, p_ext=p_ext, r_ext=r_ext, p_str=p_str, r_str=r_str, before=before,
                           src_state=src_state, reopened_state=reopened_state, r_info=r_info)


# ---------------------------------------------------------------------- (a) equivalence
def test_extended_paper_run_equals_straight_run(worlds):
    ext = read_json(worlds.p_ext / "extension.json")
    assert ext["reopen"]["papers_reopened"] > 0, "the test needs work in progress at the short horizon"
    _assert_same_run(worlds.p_ext, worlds.p_str, "paper")
    # the extension rows are there, one per reopened paper, and nothing else differs
    ev = read_jsonl(worlds.p_ext / "ledger" / "events.jsonl")
    assert sum(1 for e in ev if e["event"] == "extension_reopened") == ext["reopen"]["papers_reopened"]
    assert read_json(worlds.p_ext / "run.json")["run_id"] == read_json(worlds.p_str / "run.json")["run_id"]


def test_extended_repository_run_equals_straight_run_with_pairing(worlds):
    _assert_same_run(worlds.r_ext, worlds.r_str, "repository")
    meta = read_json(worlds.r_ext / "run.json")
    assert Path(meta["paired_run_dir"]).resolve() == worlds.p_ext.resolve()
    # quarters 1-4 keep the short paper run's allotments; quarters 5-8 are the extended paper run's spend, quarter by quarter
    allot = read_json(worlds.r_ext / "budget" / "summary.json")["allotments"]
    src_allot = read_json(worlds.r_short / "budget" / "summary.json")["allotments"]
    for q in range(1, SHORT + 1):
        assert allot[str(q)] == src_allot[str(q)]
    sp = read_json(worlds.p_ext / "budget" / "spend_by_quarter.json")
    st = read_json(worlds.r_ext / "budget" / "state.json")
    assert set(str(q) for q in range(1, LONG + 1)) <= set(allot)
    q5 = allot[str(SHORT + 1)]
    carry4 = read_json(worlds.r_short / "budget" / "state.json")["carry"]
    assert abs(q5["compute"] - max(0.0, sp[str(SHORT + 1)]["compute"] + carry4["compute"])) < 1e-6
    assert st["allotments"] == allot


# ---------------------------------------------------------------------- (b) source untouched
def test_source_runs_byte_identical(worlds):
    for d, lines in worlds.before.items():
        assert manifest_lines(d) == lines, d
    for d in (worlds.p_ext, worlds.r_ext):
        ext = read_json(d / "extension.json")
        assert ext["source_unchanged"] is True
        assert (d / "extension" / "source_manifest.sha256").exists()
        assert (d / "extension" / f"audit_at_Q{SHORT}" / "audit.json").exists()
        assert (d / "extension" / "removed_at_reopen" / "NOTE.md").exists()


# ---------------------------------------------------------------------- (c) reopened papers
def test_reopened_papers_keep_status_and_timing(worlds):
    src, re_ = worlds.src_state, worlds.reopened_state
    ev = [e for e in read_jsonl(worlds.p_short / "ledger" / "events.jsonl") if e["event"] == "unfinished_at_horizon"]
    assert ev and len({e["status"] for e in ev}) >= 2
    for e in ev:
        pid = e["paper_id"]
        assert src["papers"][pid]["status"] == "unfinished"
        assert re_["papers"][pid]["status"] == e["status"] and e["status"] in PIPELINE_STATUSES
        for k in ("review_due", "revision_due", "publish_due", "review_round", "reviewers", "reviews", "decisions", "revise_slot_set"):
            assert re_["papers"][pid].get(k) == src["papers"][pid].get(k), (pid, k)
    assert re_["review_queue"] == src["review_queue"]
    assert re_["authors"] == src["authors"]
    unchanged = {pid for pid in src["papers"] if pid not in {e["paper_id"] for e in ev}}
    assert all(re_["papers"][pid] == src["papers"][pid] for pid in unchanged)
    removed = read_jsonl(worlds.p_ext / "extension" / "removed_at_reopen" / "hypotheses.jsonl")
    assert removed and all(h["status"] == "unfinished" for h in removed)


# ---------------------------------------------------------------------- (d) pin off = old behaviour; hashes unchanged
def _regime(cfg: dict):
    from synthsci.agents.rules import load_params
    from synthsci.regimes.base import RegimeBase
    w = SimpleNamespace(cfg=cfg, scientists=[], landscape=SimpleNamespace(questions=[], fringe=[]), params=load_params(cfg))
    return RegimeBase(w)


def test_pin_off_keeps_long_horizon_rules_and_hashes():
    base = C.resolve("S0")                                     # min_horizon_quarters 20, handbook every 8, snapshots [4, 8]
    long_ = _regime(C.deep_merge(base, {"horizon_quarters": 20}))
    assert long_.long and long_.turnover_on and long_.handbook_quarters() == {4, 8, 16}
    short = _regime(base)
    assert not short.long and not short.turnover_on and short.handbook_quarters() == {4, 8}
    pinned = _regime(extended_config(base, 20))
    assert not pinned.long and not pinned.turnover_on, "pinned: the 8-quarter rules"
    assert pinned.handbook_quarters() == {4, 8, 16, 20}
    assert [q for q in range(1, 21) if not pinned.starts_allowed(q)] == [7, 8, 19, 20]
    assert [q for q in range(1, 21) if not long_.starts_allowed(q)] == [19, 20]
    s150 = _regime(extended_config(C.resolve("S0-150"), 20))   # min_horizon_quarters 1: careers move, as in the 8-quarter run
    assert s150.turnover_on and not s150.long
    with pytest.raises(ValueError):
        _regime(C.deep_merge(base, {"horizon_quarters": 20, "long_horizon": {"pin_short_rules": True}}))
    # the keys are absent from every config, so every existing hash (and the Stage 1 live runs' recorded hashes) stands
    for p in sorted((C.CONFIG_DIR / "scenarios").glob("*.yaml")):
        lh = C.resolve(p.stem).get("long_horizon", {})
        assert "pin_short_rules" not in lh and "extended_from" not in lh, p.stem
    live = {"S0-150": "44fa1851", "S0P-150": "6c1af2ba", "R-b-150": "affd5e6c"}
    for s, h in live.items():
        assert C.config_hash(C.deep_merge(C.resolve(s), {"datasets": {"keep_csv": True}}))[:8] == h, s


# ---------------------------------------------------------------------- (e) CRN
def test_pressure_world_extension_keeps_crn():
    for seed in (1, 2):
        s0 = C.deep_merge(C.resolve("S0-150"), {"datasets": {"keep_csv": True}})
        s0p = C.deep_merge(C.resolve("S0P-150"), {"datasets": {"keep_csv": True}})
        ns = C.run_id(s0, seed, "llm")
        assert crn_namespace(extended_config(s0, 20), seed, "llm") == ns
        assert crn_namespace(extended_config(s0p, 20), seed, "llm") == ns
        assert C.run_id(extended_config(s0p, 20), seed, "llm") != C.run_id(s0p, seed, "llm")
    # the live Stage 1 runs: the tasks of both worlds were named in S0-150's namespace, which the extensions keep
    for seed in (1, 2):
        a, b = WS / "runs" / "S0-150" / str(seed) / "llm", WS / "runs" / "S0P-150" / str(seed) / "llm"
        if not (a / "run.json").exists() or not (b / "run.json").exists():
            continue
        na, nb = task_namespaces(a), task_namespaces(b)
        assert na == nb == {crn_namespace(extended_config(_stored(b), 20), seed, "llm")}


def test_extension_waits_for_paired_quarter(worlds):
    from synthsci.budget import paired_closed_through
    assert paired_closed_through(worlds.p_short) == SHORT
    assert paired_closed_through(worlds.p_ext) == LONG
