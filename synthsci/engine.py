"""The run driver: run = config + seed + mode; phases are barriers; state is persisted after every phase so a run
resumes after an interruption or a dispatch boundary."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import config as C
from .backends import make_backend
from .backends.base import PendingDispatch
from .budget import Budget, PairingNotReady
from .datasets import DatasetGenerator
from .landscape import build_landscape
from .ledger import Ledger
from .legacy import build_legacy
from .orchestration.tasks import TaskStore
from .population import build_population, form_labs
from .seeding import SeedingTables
from .util import read_json, write_json, write_text


class World:
    """Everything both regimes share at a seed: landscape, truth, population, legacy, ledger, budget, tasks, backend."""

    def __init__(self, cfg: dict, master_seed: int, mode: str, backend_name: str, run_dir: Path, paired_run_dir: Path | None,
                 allow_unpaired: bool = False, resuming: bool = False):
        self.cfg, self.seed, self.mode = cfg, int(master_seed), mode
        self.resuming = bool(resuming)    # a resumed run must not re-append the setup's ledger rows
        self.seeds = C.seed_layers(cfg, master_seed)
        self.run_id = C.run_id(cfg, master_seed, mode)
        # task, group and dataset ids (and so the rule agents' random streams, which are keyed on task ids) are named in this
        # namespace; a treatment world that only changes what agents see (`crn_with`, e.g. S0P-150 -> S0-150) borrows its
        # baseline's, so both share the same warm start and random numbers (common random numbers)
        self.task_ns = crn_namespace(cfg, master_seed, mode)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tables = SeedingTables(cfg["seeding"]["table_set"])
        self.landscape = build_landscape(cfg, self.seeds, self.tables)
        self.deid = self.landscape.write(self.run_dir / "landscape", self.seeds["run"])
        self.gen = DatasetGenerator(cfg, self.tables, self.landscape, self.run_dir)
        self.theory_shares = self.tables.theory_shares()
        self.scientists = build_population(cfg, self.seeds, self.theory_shares)
        self.labs = form_labs(cfg, self.seeds, self.scientists) if cfg["regime"] == "repository" else []
        write_json(self.run_dir / "population" / "scientists.json", self.scientists)
        if self.labs:
            write_json(self.run_dir / "population" / "labs.json", self.labs)
        self.legacy = build_legacy(cfg, self.seeds, self.landscape, self.gen)
        write_json(self.run_dir / "legacy" / "papers.json", self.legacy)
        self.ledger = Ledger(self.run_dir, self.run_id, mode, cfg["regime"])
        self.budget = Budget(cfg, self.run_dir, self.ledger, paired_run_dir, allow_unpaired=allow_unpaired)
        if resuming:   # the tallies live in the ledger; a resumed run must not restart them from zero
            self.budget.restore()
        self.store = TaskStore(self.run_dir)
        self.backend_name = backend_name
        self.backend = make_backend(backend_name, cfg, self.run_dir, self.store) if mode != "fast" else None
        from .agents.rules import load_params
        self.params = load_params(cfg)
        self.log_dir = self.run_dir / "log"
        self.log_dir.mkdir(exist_ok=True)

    def log(self, msg: str) -> None:
        with open(self.log_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def crn_namespace(cfg: dict, master_seed: int, mode: str) -> str:
    """The run id of the baseline this world shares random numbers with: this run's own config (overrides included) with
    the scenario name and the treatment keys (`incentives`) put back to the baseline's. Without `crn_with`, the run's own id."""
    import copy
    ext = (cfg.get("long_horizon") or {}).get("extended_from")
    if ext:
        # an extended run keeps the namespace of the run it extends (its first segment), so the ids and random
        # streams of quarters after the old horizon continue the original run's, and S0P's extension still shares S0's
        cfg = copy.deepcopy(cfg)
        cfg["horizon_quarters"] = int(ext[0])
        for k in ("pin_short_rules", "extended_from"):
            cfg["long_horizon"].pop(k, None)
    pre = (cfg.get("fastmode") or {}).get("prehistory_params")
    if pre:
        # a run whose warm start uses its own rule file is named as if that file were used throughout, so the
        # warm start (and every later random draw) matches the run that used the pre-history file for everything
        cfg = copy.deepcopy(cfg)
        cfg["fastmode"] = {k: v for k, v in cfg["fastmode"].items() if k not in ("params", "prehistory_params")} | {"params": pre}
    crn = cfg.get("crn_with")
    if not crn:
        return C.run_id(cfg, master_seed, mode)
    base = C.resolve(crn)
    t = copy.deepcopy(cfg)
    t.pop("crn_with", None)
    t["scenario"] = base["scenario"]
    if "incentives" in base:
        t["incentives"] = copy.deepcopy(base["incentives"])
    else:
        t.pop("incentives", None)
    return C.run_id(t, master_seed, mode)


def run_dir_for(cfg: dict, seed: int, mode: str, out_root: Path | None = None) -> Path:
    # always absolute: a relative root would put a relative path (containing e.g. the mode name "llm") into prompts
    root = Path(out_root).resolve() if out_root else C.WS_ROOT / "runs"
    return root / cfg["scenario"] / str(seed) / mode


def _availability_note(cfg: dict) -> dict:
    from .population import effective_availability
    return effective_availability(cfg)


def run(scenario: str | dict, seed: int, mode: str = "mock", backend: str | None = None, overrides: dict | None = None,
        out_root: Path | None = None, paired_run_dir: Path | None = None, resume: bool = True, quiet: bool = False,
        allow_unpaired: bool = False, run_dir: Path | None = None) -> Path:
    """`run_dir` (used by resume) names the run folder explicitly; otherwise it is <out_root>/<scenario>/<seed>/<mode>. An
    extended run lives in <scenario>-x<H>/..., which the scenario name alone would not find."""
    cfg = C.resolve(scenario, overrides)
    if C.get(cfg, "datasets.keep_csv") is None:
        cfg = C.deep_merge(cfg, {"datasets": {"keep_csv": mode == "llm"}})
    backend = backend or {"mock": "mock", "fast": "mock", "llm": "subagent"}[mode]
    rd = Path(run_dir).resolve() if run_dir else run_dir_for(cfg, seed, mode, out_root)
    if paired_run_dir is None and cfg["regime"] == "repository" and cfg["budget"]["rule"] == "pairing":
        paired_run_dir = run_dir_for({"scenario": cfg["budget"]["paired_scenario"]}, seed, mode, out_root)
    state_path = rd / "state" / "state.json"
    resuming = bool(resume and state_path.exists())
    if resuming and (rd / "run.json").exists():
        prior = read_json(rd / "run.json")
        if prior["config_hash"] != C.config_hash(cfg):
            raise RuntimeError(f"run dir {rd} holds a different config (hash {prior['config_hash'][:8]}); use a new out_root")
    world = World(cfg, seed, mode, backend, rd, paired_run_dir, allow_unpaired=allow_unpaired, resuming=resuming)
    write_text(rd / "config.resolved.yaml", C.dump_yaml(cfg))
    write_json(rd / "run.json", {"run_id": world.run_id, "scenario": cfg["scenario"], "seed": seed, "mode": mode,
                                  "backend": backend, "config_hash": C.config_hash(cfg), "seeds": world.seeds,
                                  "seeding_manifest": world.tables.manifest(), "regime": cfg["regime"],
                                  "paired_run_dir": str(paired_run_dir) if paired_run_dir else None,
                                  "pairing_fallback": world.budget.pairing_fallback, "horizon_quarters": cfg["horizon_quarters"],
                                  "population": cfg["population"], "availability": _availability_note(cfg),
                                  "timing": C.get(cfg, "journal.timing"), "career_pressure": C.get(cfg, "incentives.career_pressure", "none")})
    if cfg["regime"] == "paper":
        from .regimes.paper import PaperRegime as Regime
    else:
        from .regimes.repository import RepositoryRegime as Regime
    regime = Regime(world)
    if resuming:
        regime.load_state(read_json(state_path))
        world.log(f"resumed at quarter {regime.state['quarter']} phase {regime.state['phase']}")
    try:
        regime.run_all()
    except PendingDispatch as pd_:
        regime.save_state()
        world.ledger.flush()
        world.budget.write()
        msg = f"PENDING: {len(pd_.tasks)} task(s) await dispatch. Instructions: {pd_.instructions_path}"
        world.log(msg)
        if not quiet:
            print(msg)
        return rd
    except PairingNotReady as pn:
        # raised before the first phase of the quarter, so nothing of that quarter has run; resume once the paper run is ahead
        regime.save_state()
        world.ledger.flush()
        world.budget.write()
        world.log(str(pn))
        if not quiet:
            print(str(pn))
        return rd
    regime.save_state()
    world.ledger.flush()
    world.budget.write()
    from .audit import run_audit
    run_audit(rd, quiet=quiet)
    world.log("run complete")
    if not quiet:
        print(f"DONE {world.run_id} -> {rd}")
    return rd
