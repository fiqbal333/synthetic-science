"""Shared regime machinery: phase barriers and resumable state, research-question menus, bounded views, the observatory
roll-up, fringe promotion, the annual career step and handbook snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..belief import all_rules, score_belief
from ..config import rng
from ..population import annual_step, availability
from ..util import read_text, seed_from, short_id, write_json
from . import texts


class RegimeBase:
    PHASES: list[str] = []
    PREHISTORY_PHASES: list[str] = []     # the warm-start loop (quarters 1-K..0), run by the rule agents; empty = no warm start
    REGIME = "base"

    def __init__(self, world):
        self.w = world
        self.cfg = world.cfg
        self.H = int(self.cfg["horizon_quarters"])
        # a run extended past its original horizon keeps one rulebook. `long_horizon.extended_from` lists
        # the horizons at which the run was closed and reopened (history); `long_horizon.pin_short_rules: true` makes every
        # horizon-gated rule follow the FIRST segment's horizon (H_rules) instead of the new one. Both keys are absent from every
        # existing config, so nothing changes (and no config hash changes) unless `synthsci extend` sets them.
        lh = self.cfg["long_horizon"]
        self.extended_from = [int(h) for h in (lh.get("extended_from") or [])]
        self.pinned = bool(lh.get("pin_short_rules", False))
        if self.pinned and not self.extended_from:
            raise ValueError("long_horizon.pin_short_rules needs long_horizon.extended_from (the horizon whose rules are pinned)")
        self.H_rules = self.extended_from[0] if self.pinned else self.H
        self.long = self.H_rules >= 20
        # careers move at every annual step in long horizons or when the scenario lowers min_horizon_quarters (the 150-scientist worlds)
        self.turnover_on = bool(lh["turnover"].get("enabled", True)) and \
            self.H_rules >= int(lh["turnover"].get("min_horizon_quarters", 20))
        self.K = self.warm_start_quarters()
        self.prehistory = False
        self._mock_backend = None
        # the clock opens at 1-K (warm start), else at quarter 0, whose only phase is the quarter-0 handbook
        self.state: dict = {"quarter": 1 - self.K if (self.PREHISTORY_PHASES and self.K > 0) else 0, "phase": None, "done": {}, "authors": {}, "counts": {},
                            "promoted": [], "handbooks": {}, "fringe_size": len(world.landscape.fringe), "records": {}, "career_news": []}
        self._task_ctx: dict = {}
        from ..agents.rules import RuleAuthor, RuleEditor, RuleHandbook, RuleLabHead, RuleReviewer, load_params

        def _rules(p):
            return {"author": RuleAuthor(p), "reviewer": RuleReviewer(p), "editor": RuleEditor(p), "lab": RuleLabHead(p), "handbook": RuleHandbook(p)}
        # the warm-start loop may run on its own rule file (`fastmode.prehistory_params`, default = fastmode.params)
        self._params_live = world.params
        self._rules_live = _rules(self._params_live)
        if (self.cfg.get("fastmode") or {}).get("prehistory_params"):
            self._params_pre = load_params(self.cfg, prehistory=True)
            self._rules_pre = _rules(self._params_pre)
        else:
            self._params_pre, self._rules_pre = self._params_live, self._rules_live
        self.params, self.rules = self._params_live, self._rules_live
        self.init_state()

    def _select_rules(self) -> None:
        """The rule agents (and their parameters) of the current phase: pre-history or live period."""
        self.params, self.rules = (self._params_pre, self._rules_pre) if self.prehistory else (self._params_live, self._rules_live)

    def warm_start_quarters(self) -> int:
        ws = (self.cfg["journal"].get("timing") or {}).get("warm_start") or {}
        return int(ws.get("quarters", 0)) if ws.get("enabled", False) else 0

    # ------------------------------------------------------------ state
    def init_state(self) -> None:
        for s in self.w.scientists:
            self.state["authors"][s["id"]] = {"slots": [{"state": "free", "paper_id": None, "reserved": None} for _ in range(s["slots"])]}
            self.state["records"][s["id"]] = {"papers": [], "papers_published": [], "reviews_written": 0, "reviews_matched": 0,
                                               "citations": 0, "coauthored": 0, "criticisms": []}
        for q in self.w.landscape.questions + self.w.landscape.fringe:
            self.state["counts"][q["id"]] = 0

    def save_state(self) -> None:
        write_json(self.w.run_dir / "state" / "state.json", self.state, indent=None)
        # the world's mutable parts (careers, labs, fringe, truth of new fringe cards) change in the annual step; World()
        # rebuilds them from the seeds, so a resumed run must restore them from here
        write_json(self.w.run_dir / "state" / "world.json", {"scientists": self.w.scientists, "labs": self.w.labs,
                                                              "fringe": self.w.landscape.fringe, "truth": self.w.landscape.truth},
                   indent=None)
        self.w.ledger.flush()

    def load_state(self, st: dict) -> None:
        self.state = st
        self.state.setdefault("career_news", [])
        self._restore_world()

    def _restore_world(self) -> None:
        wp =self.w.run_dir / "state" / "world.json"
        if wp.exists():
            ws = json.loads(wp.read_text(encoding="utf-8"))
            self.w.scientists[:] = ws["scientists"]
            self.w.labs[:] = ws.get("labs") or []
            self.w.landscape.fringe[:] = ws["fringe"]
            self.w.landscape.truth.clear(); self.w.landscape.truth.update(ws["truth"])
            # World() rewrote these two files from the rebuilt landscape; the audit reads them
            write_json(self.w.run_dir / "landscape" / "fringe.json", self.w.landscape.fringe)
            write_json(self.w.run_dir / "landscape" / "truth.json", self.w.landscape.truth)
            src = "state/world.json"
        else:
            # runs saved before world state was written to state/world.json: the latest annual snapshot at or before the saved quarter is the population
            # (careers only move in the annual step); fringe and truth only change from quarter 4, so the rebuilt ones hold
            q = int(self.state.get("quarter", 0))
            snaps = []
            for f in (self.w.run_dir / "population").glob("scientists_Q*.json"):
                try:
                    sq = int(f.stem.split("_Q")[1])
                except ValueError:
                    continue
                if sq <= q:
                    snaps.append((sq, f))
            if not snaps:
                return
            sq, f = max(snaps)
            if sq >= 4:
                raise RuntimeError(f"cannot restore fringe/truth after quarter {sq} for a run saved without state/world.json: {self.w.run_dir}")
            self.w.scientists[:] = json.loads(f.read_text(encoding="utf-8"))
            src = f.name
        for s in self.w.scientists:   # entrants that the saved state does not know yet (cannot happen; guards old runs)
            self.state["authors"].setdefault(s["id"], {"slots": [{"state": "free", "paper_id": None, "reserved": None}
                                                                   for _ in range(s["slots"])]})
        self.w.log(f"restored world from {src}: {len(self.w.scientists)} scientists, "
                   f"{sum(1 for s in self.w.scientists if s.get('active', True))} active")

    def phase_done(self, q: int, phase: str) -> bool:
        return phase in self.state["done"].get(str(q), [])

    def phases_for(self, q: int) -> list[str]:
        """Quarters >= 1 run the regime's loop; the pre-history quarters (<= 0) run the warm-start loop with the rule agents,
        and quarter 0 ends with the quarter-0 handbook (the state of knowledge before any live agent acts)."""
        if q >= 1:
            return self.PHASES
        pre = list(self.PREHISTORY_PHASES) if (self.K and q > -self.K) else []
        return pre + (["handbook0"] if q == 0 else [])

    def mark_phase(self, q: int, phase: str) -> None:
        self.state["done"].setdefault(str(q), []).append(phase)
        self.state["phase"] = phase
        # fast mode has no dispatch boundaries: persist once a year and at the end (serialising the state after every
        # phase was 90% of a 30-year run's time); mock and llm modes persist after every phase so a run resumes anywhere
        if self.w.mode != "fast" or (phase == self.phases_for(q)[-1] and q % 4 == 0):
            self.save_state()

    def run_all(self) -> None:
        if not self.phase_done(0, "setup"):
            self.setup()
            self.mark_phase(0, "setup")
        for q in range(int(self.state["quarter"]), self.H + 1):
            self.state["quarter"] = q
            if q >= 1 and not all(self.phase_done(q, ph) for ph in self.phases_for(q)):
                self.before_quarter(q)   # e.g. the paired paper run must have closed this quarter (its realized spend is this run's allotment)
            for phase in self.phases_for(q):
                if self.phase_done(q, phase):
                    continue
                # the warm-start loop is run by the rule agents and nothing in it is charged to the field's budget;
                # the quarter-0 handbook is a live call like every later one
                self.prehistory = q <= 0 and phase != "handbook0"
                self.w.budget.suspended = self.prehistory
                self._select_rules()
                self.w.log(f"Q{q} {phase}")
                getattr(self, phase)(q)
                self.mark_phase(q, phase)
            if q >= 1:
                self.w.budget.close_quarter(q)
        self.prehistory = False
        self.w.budget.suspended = False
        self._select_rules()
        self.finish()

    def setup(self) -> None:
        pass

    def before_quarter(self, quarter: int) -> None:
        """Hook run once before the first phase of a live quarter that still has phases to run."""
        pass

    def handbook0(self, quarter: int) -> None:
        if self.cfg["handbook"].get("quarter0", True):
            self.handbook_snapshot(0)

    def finish(self) -> None:
        pass

    def backend(self):
        """The live backend, except in the pre-history, where the rule agents (mock backend) stand in for every call in
        every mode: the warm start never dispatches a task and never calls a model."""
        if self.prehistory and self.w.backend_name != "mock":
            if self._mock_backend is None:
                from ..backends.mock import MockBackend
                self._mock_backend = MockBackend(self.cfg, self.w.run_dir, self.w.store, params=self._params_pre)
            return self._mock_backend
        return self.w.backend

    # ------------------------------------------------------------ helpers
    def scientist(self, aid: str) -> dict:
        return next(s for s in self.w.scientists if s["id"] == aid)

    def active_scientists(self) -> list[dict]:
        return [s for s in self.w.scientists if s.get("active", True)]

    def rng(self, *tags) -> np.random.Generator:
        return rng(self.w.seeds["run"], self.REGIME, *tags)

    def available(self, aid: str, quarter: int, slot: int) -> bool:
        s = self.scientist(aid)
        return availability(self.w.seeds, aid, quarter, slot, self.availability_p, s["productivity"])

    @property
    def availability_p(self) -> float:
        from ..population import effective_availability
        return effective_availability(self.cfg)["availability_p"]

    def starts_allowed(self, quarter: int) -> bool:
        stop = int(self.cfg["population_profile"]["starts_stop_before_end"])
        if quarter > self.H - stop:
            return False
        # an extended run already lived through the no-start window before each earlier horizon (history); a run
        # with the same config made in one go replays that window, so the two are the same run
        return not any(h - stop < quarter <= h for h in self.extended_from)

    # ------------------------------------------------------------ menus
    def menu_entries(self, home_theory: str, quarter: int, who: str) -> list[dict]:
        L = self.w.landscape
        mc = self.cfg["landscape"]["menu"]
        r = self.rng("menu", quarter, who)
        counts = self.state["counts"]
        core = [q for q in L.questions] + [f for f in L.fringe if f["id"] in self.state["promoted"]]
        home = [q for q in core if q["family"] == home_theory] or core
        w = np.array([(1.0 / (1.0 + counts.get(q["id"], 0))) ** mc["weight_power"] for q in home])
        k = min(mc["affinity"], len(home))
        aff = [home[int(i)] for i in r.choice(len(home), size=k, replace=False, p=w / w.sum())]
        entries = [{"question": q, "slot": "affinity"} for q in aff]
        crowded = max(home, key=lambda q: (counts.get(q["id"], 0), q["id"]))
        if crowded["id"] not in {e["question"]["id"] for e in entries}:
            entries.append({"question": crowded, "slot": "crowded"})
        rest = [q for q in core if q["id"] not in {e["question"]["id"] for e in entries}]
        for _ in range(mc["random"]):
            if rest:
                pick = rest[int(r.integers(len(rest)))]
                entries.append({"question": pick, "slot": "random"})
                rest.remove(pick)
        fringe = [f for f in L.fringe if f["id"] not in self.state["promoted"]]
        fam = [f for f in fringe if f["family"] == home_theory] or fringe
        for _ in range(mc["fringe"]):
            if fam:
                entries.append({"question": fam[int(r.integers(len(fam)))], "slot": "fringe"})
        # the menu always has the configured number of entries: when the crowded question coincides with an affinity pick
        # (or a pool is exhausted) the shortfall is filled with further random core questions
        size = int(mc["affinity"]) + int(mc["crowded"]) + int(mc["random"]) + int(mc["fringe"])
        rest = [q for q in core if q["id"] not in {e["question"]["id"] for e in entries}]
        while len(entries) < size and rest:
            pick = rest[int(r.integers(len(rest)))]
            entries.append({"question": pick, "slot": "random"})
            rest.remove(pick)
        for e in entries:
            qid = e["question"]["id"]
            e["question_id"] = qid
            e["family"] = e["question"]["family"]
            e["count"] = counts.get(qid, 0)
            e["panel"] = e["question"]["panel"]
            e["handbook_entry"] = self.latest_handbook_entry(qid)
        return entries

    def latest_handbook_entry(self, qid: str) -> str | None:
        hb = self.state.get("handbooks", {})
        if not hb or not self.long or not self.cfg["long_horizon"].get("textbook", True):
            return None   # the handbook enters the menu only in long horizons
        last = hb.get(str(max(int(k) for k in hb)))
        if not last:
            return None
        ans = (last.get("answers") or {}).get(qid)
        return f"{ans['answer']} (confidence {ans['confidence']:.0f})" if ans else None

    # ------------------------------------------------------------ observatory
    def certified_units(self, qid: str) -> list[dict]:
        """Override: list of certified units on the focal relation of qid with estimate (partial r), se, n, verdict, attention."""
        raise NotImplementedError

    def rollup(self, quarter: int) -> dict:
        L = self.w.landscape
        out = {"quarter": quarter, "questions": {}}
        one_per_deposit = self.REGIME == "repository"
        for q in L.questions + [f for f in L.fringe if self.state["counts"].get(f["id"], 0) > 0]:
            units = self.certified_units(q["id"])
            theta = L.truth_of(q["id"])["theta"]
            rules = all_rules(units, share=float(self.cfg["repository"]["endorsement_share"]) if one_per_deposit else 2 / 3,
                              one_per_deposit=one_per_deposit)
            scored = {k: score_belief(v, theta) for k, v in rules.items()}
            out["questions"][q["id"]] = {"family": q["family"], "core": q["core"], "prediction_sign": q["prediction_sign"],
                                         "theta": theta, "n_units": len(units), "n_run_units": sum(1 for u in units if not u.get("legacy")),
                                         "rules": rules, "scored": scored,
                                         "units": [{k: u.get(k) for k in ("unit_id", "estimate", "se", "n", "verdict", "attention", "quarter", "legacy", "truth_check", "deposit_id")} for u in units]}
        # fast mode keeps annual and final roll-ups on disk (the quarterly field metrics live in events.jsonl)
        if self.w.mode != "fast" or quarter % 4 == 0 or quarter == self.H or quarter <= 1 or quarter in self.extended_from:
            self.w.ledger.write_rollup(quarter, out)
        return out

    def check_promotion(self, quarter: int) -> None:
        thr = int(self.cfg["landscape"]["fringe_promotion_units"])
        for f in self.w.landscape.fringe:
            if f["id"] not in self.state["promoted"] and self.state["counts"].get(f["id"], 0) >= thr:
                self.state["promoted"].append(f["id"])
                f["promoted_quarter"] = quarter
                self.w.ledger.add("events", {"quarter": quarter, "event": "fringe_promoted", "question_id": f["id"]})

    # ------------------------------------------------------------ long horizons
    def annual(self, quarter: int) -> None:
        if not self.turnover_on or quarter % 4 != 0:
            return
        recs = {aid: {"publications": len(r.get("papers_published", [])), "endorsed_claims": r.get("endorsed", 0),
                      "verifications": r.get("verifications", 0)} for aid, r in self.state["records"].items()}
        before = {s["id"] for s in self.w.scientists}
        stage_before = {s["id"]: s["stage"] for s in self.w.scientists}
        res = annual_step(self.cfg, self.w.seeds, self.w.scientists, quarter, recs, self.w.theory_shares, self.REGIME,
                          self.w.labs if self.REGIME == "repository" else None)
        for s in self.w.scientists:
            if s["id"] not in before:
                self.state["authors"][s["id"]] = {"slots": [{"state": "free", "paper_id": None, "reserved": None} for _ in range(s["slots"])]}
                self.state["records"][s["id"]] = {"papers": [], "papers_published": [], "reviews_written": 0, "reviews_matched": 0,
                                                   "citations": 0, "coauthored": 0, "criticisms": []}
        for ev in res.get("events", []):
            credit = ev[2] if len(ev) > 2 else None
            self.w.ledger.add("events", {"quarter": quarter, "event": "career", "who": ev[0], "what": ev[1], "credit": credit,
                                         "stage_before": stage_before.get(ev[0])})
            if ev[1] in ("tenured", "tenure_exit", "promoted_full", "not_promoted_full", "retired", "exit"):
                # public department news (true outcomes of the simulated population; shown only when career pressure is visible)
                self.state["career_news"].append({"quarter": quarter, "who": ev[0], "what": ev[1], "credit": credit})
        if quarter >= 1:   # the pre-history's renewal would only churn the fringe before anyone has seen it
            target = int(self.cfg["long_horizon"]["renewal"]["fringe_refill"])
            self.w.landscape.replenish_fringe(target + len(self.state["promoted"]))
            for f in self.w.landscape.fringe:
                self.state["counts"].setdefault(f["id"], 0)
            self.state["fringe_size"] = len(self.w.landscape.fringe)
        write_json(self.w.run_dir / "population" / f"scientists_Q{quarter}.json", self.w.scientists)
        write_json(self.w.run_dir / "landscape" / "fringe.json", self.w.landscape.fringe)
        write_json(self.w.run_dir / "landscape" / "truth.json", self.w.landscape.truth)

    # ------------------------------------------------------------ handbook
    def handbook_quarters(self) -> set[int]:
        """The live quarters with a handbook snapshot (quarter 0 is separate: `handbook.quarter0`)."""
        snaps = {int(q) for q in self.cfg["handbook"]["snapshots"]}
        every = self.cfg["long_horizon"].get("handbook_every_quarters")
        if self.pinned:
            # up to the first horizon the schedule is exactly the original run's; after it, the configured
            # snapshots, every `handbook_every_quarters`, and the end of the run (S0-150 extended to 20: Q4, Q8, Q16, Q20)
            if every:
                snaps |= {q for q in range(int(every), self.H + 1, int(every)) if q > self.H_rules}
            snaps.add(self.H)
        elif self.long and every:
            snaps |= {q for q in range(int(every), self.H + 1, int(every))}
        return {q for q in snaps if q <= self.H}

    def handbook_snapshot(self, quarter: int) -> None:
        from ..handbook import build_handbook_input, panel_aggregate, score_handbook, writer_tasks
        if quarter != 0 and quarter not in self.handbook_quarters():
            return
        inp = build_handbook_input(self, quarter)
        hdir = self.w.run_dir / "handbook" / f"Q{quarter}"
        write_json(hdir / "input.json", inp["json"])
        from ..util import write_text as _wt
        _wt(hdir / "input.md", inp["md"])
        _wt(hdir / "questions.md", inp["questions_md"])
        results = []
        if self.w.mode == "fast":
            for i, tier in enumerate(self.cfg["handbook"]["panel"]["writers"]):
                r = self.rng("handbook", quarter, i)
                hb = self.rules["handbook"].write(inp["rules"], inp["theory_questions"], r, nonce="fast", predictions=inp["predictions"])
                hb["tier"] = tier
                results.append(hb)
        else:
            tasks = writer_tasks(self, quarter, inp, hdir)
            for t in tasks:
                self._task_ctx[t.id] = {"type": "handbook", "rules": inp["rules"], "theory_questions": inp["theory_questions"], "quarter": quarter,
                                        "predictions": inp["predictions"]}
            self.run_tasks(tasks, quarter)
            for t in tasks:
                if self.w.store.status(t) == "done":
                    hb = self.w.store.read_output(t, "handbook.json")
                    hb["tier"] = t.tier
                    results.append(hb)
        if results:
            panel = panel_aggregate(results)
            score = score_handbook(panel, results, self, quarter)
            write_json(hdir / "panel.json", panel)
            write_json(hdir / "score.json", score)
            self.state["handbooks"][str(quarter)] = {"answers": panel["answers"], "score": {k: v for k, v in score.items() if k != "per_question"}}
            self.w.ledger.add("events", {"quarter": quarter, "event": "handbook", "accuracy": score.get("accuracy"), "brier": score.get("brier")})

    # ------------------------------------------------------------ tasks
    def run_tasks(self, tasks: list, quarter: int) -> None:
        if not tasks:
            return
        for t in tasks:
            self.w.store.materialize(t)
        ctx = {"world": self.w, "regime": self, "task_ctx": self._task_ctx}
        backend = self.backend()
        backend.run(tasks, ctx)
        lean = self.cfg["journal"].get("paper_contract", "full") == "lean"
        for t in tasks:
            status = self.w.store.status(t)
            if status not in ("done", "failed"):
                continue
            usage = backend.usage(t)
            # a batched research/analyst call carries several studies: debit once at the batch multiplier (same rule as fast mode);
            # the realized tokens the backend reports (usage.json) travel in the same ledger row as the tariff
            scale = self.w.budget.call_scale(int(t.meta.get("batch", 1)), lean and t.type in ("research", "research_batch", "revision"))
            self.w.budget.debit_call(quarter, t.call_type, t.id, t.who, realized_tokens=usage.get("total_tokens"), scale=scale,
                                     failed=(status == "failed"))
