"""The paper regime: authors, the Journal, blind review, decisions, the literature index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..belief import truth_class
from ..orchestration.tasks import PROMPTS, Task
from ..population import draw_team
from ..scorer import analyst_flags, claim_sign, forensic_rerun, relation_theta, score_hypothesis, truth_check
from ..stats import partial_r
from ..util import read_text, seed_from, short_id, write_json, write_text
from . import texts
from .base import RegimeBase

PROTOCOL = {"VERIFIED": "_protocol_verified.md", "AUTO": "_protocol_auto.md"}


class PaperRegime(RegimeBase):
    PHASES = ["research", "desk", "assign", "review", "decide", "publish", "score", "handbook", "annual"]
    # warm start: the same loop without the observatory's roll-ups and handbooks, run by the rule agents
    PREHISTORY_PHASES = ["research", "desk", "assign", "review", "decide", "publish", "annual"]
    REGIME = "paper"

    def init_state(self) -> None:
        super().init_state()
        st = self.state
        st.update({"papers": {}, "paper_counter": 0, "literature": {}, "review_queue": [], "coauthor_history": {},
                   "reviewer_last": {}, "legacy_ids": []})
        for lp in self.w.legacy:
            pid = lp["paper_id"]
            # the starting literature is scored like everything else: the legacy hypotheses carry their truth
            # check (focal: the question's theta; moderations: 0; a secondary outcome: its own theta, else 0), as the
            # repository regime's legacy claims already did, so 'inherited' certification is comparable across regimes
            q = self.w.landscape.question(lp["question_id"])
            for h in lp["hypotheses"]:
                if h.get("truth_check") is None:
                    theta = legacy_theta(lp, h, q)
                    h["theta"], h["truth_class"], h["truth_check"] = theta, truth_class(theta), truth_check(h["verdict"], h.get("estimate"), theta)
            st["papers"][pid] = {"paper_id": pid, "question_id": lp["question_id"], "legacy": True, "status": "published",
                                 "quarter_published": 0, "title": lp["title"], "authors": lp["authors"], "lead": lp["authors"][0],
                                 "hypotheses": lp["hypotheses"], "abstract": lp["abstract"], "abstract_template": lp["abstract_template"],
                                 "citations": 0, "cited_by": [], "builds_on": [], "relation_to_prior": "first study",
                                 "design": lp["design"], "estimator": lp["estimator"], "n": lp["n"], "theory": lp["theory"],
                                 "hidden": lp["hidden"], "style": "legacy", "throughput": "legacy", "tier": "rule", "core": lp["core"]}
            st["literature"].setdefault(lp["question_id"], []).append(pid)
            st["counts"][lp["question_id"]] = st["counts"].get(lp["question_id"], 0) + 1
            st["legacy_ids"].append(pid)
        # legacy hypotheses into the ledger (certified at quarter 0); once per run, not again on every resume (otherwise
        # each dispatch pause would add another copy of every legacy hypothesis)
        if not getattr(self.w, "resuming", False):
            for lp in self.w.legacy:
                self._ledger_hypotheses(st["papers"][lp["paper_id"]], certified=True, quarter=0)

    # ======================================================================== views
    def view(self, qid: str) -> list[dict]:
        cap = int(self.cfg["long_horizon"]["memory"]["abstracts"])
        pids = self.state["literature"].get(qid, [])
        papers = [self.state["papers"][p] for p in pids]
        if len(papers) <= cap:
            chosen = papers
        else:
            by_cite = sorted(papers, key=lambda p: (-p["citations"], -p["quarter_published"], p["paper_id"]))[:cap // 2]
            rest = [p for p in papers if p not in by_cite]
            by_recent = sorted(rest, key=lambda p: (-p["quarter_published"], p["paper_id"]))[:cap - len(by_cite)]
            chosen = by_cite + by_recent
        return [self._abstract_view(p) for p in sorted(chosen, key=lambda p: (-p["quarter_published"], p["paper_id"]))]

    def _abstract_view(self, p: dict) -> dict:
        h1 = p["hypotheses"][0]
        return {"paper_id": p["paper_id"], "title": p["title"], "abstract": p.get("abstract"), "abstract_template": p.get("abstract_template", ""),
                "hypotheses": p["hypotheses"], "citations": p["citations"], "quarter_published": p["quarter_published"],
                "legacy": bool(p.get("legacy", False) or (p.get("quarter_published") or 1) <= 0),   # pre-history publications read as earlier work
                "focal_estimate": h1.get("estimate"), "focal_verdict": h1.get("verdict")}

    # ======================================================================== timing and profiles
    def latency(self, kind: str, *tags) -> int:
        from ..timing import draw_latency
        return draw_latency(self.cfg, kind, self.w.seeds["run"], *tags)

    def _keep_csv(self) -> bool:
        """llm mode keeps data.csv for the forensic re-run at ingest; the pre-history's forensics run in-process, so its
        menu datasets are deleted after use in every mode (regenerable from seed + id)."""
        return bool(self.cfg["datasets"].get("keep_csv", True)) and not self.prehistory

    def _profile(self, a: dict, quarter: int) -> str:
        """The record card; with `incentives.career_pressure: visible` it also carries the career, department-news and
        Journal-statistics lines (facts from the run's own record, never instructions)."""
        rec = self.state["records"][a["id"]]
        pressure = None
        if (self.cfg.get("incentives") or {}).get("career_pressure", "none") == "visible":
            inc = self.cfg["incentives"]
            lh = self.cfg["long_horizon"]["turnover"]
            pressure = {"career": texts.career_line(a, lh, len(rec.get("papers_published", []))),
                        "news": texts.department_news_line(self.state.get("career_news", []), quarter, int(inc.get("news_window_quarters", 8))),
                        "journal": texts.journal_stats_line(self._journal_stats(quarter, int(inc.get("journal_stats_window_quarters", 8))))}
        return texts.record_card(a, rec, "paper", quarter, pressure=pressure)

    def _journal_stats(self, quarter: int, window: int) -> dict:
        """The Journal's public record over the last `window` quarters: submissions, final decisions, acceptances, and
        acceptance by the main result as the authors reported it (their own focal verdict; nothing hidden enters here)."""
        lo = quarter - window
        out = {"window": window, "submitted": 0, "decided": 0, "accepted": 0,
               "by_result": {"supported": [0, 0], "partially": [0, 0], "not_supported": [0, 0]}}
        for p in self.state["papers"].values():
            if p.get("legacy") and not p.get("prehistory"):
                continue
            qs = p.get("quarter_submitted")
            if qs is not None and lo <= qs < quarter and p.get("desk") is not None:
                out["submitted"] += 1
            final = None
            if (p.get("desk") or {}).get("decision") == "desk_reject":
                final = ("reject", p["desk"]["quarter"])
            for d in (p.get("decisions") or {}).values():
                if d["decision"] in ("accept", "reject"):
                    final = (d["decision"], d["quarter"])
            if final and lo <= final[1] < quarter:
                out["decided"] += 1
                v = p["hypotheses"][0].get("verdict", "not_supported") if p.get("hypotheses") else "not_supported"
                v = v if v in out["by_result"] else "not_supported"
                out["by_result"][v][1] += 1
                if final[0] == "accept":
                    out["accepted"] += 1
                    out["by_result"][v][0] += 1
        return out

    def certified_units(self, qid: str) -> list[dict]:
        q = self.w.landscape.question(qid)
        units = []
        for pid in self.state["literature"].get(qid, []):
            p = self.state["papers"][pid]
            for h in p["hypotheses"]:
                if h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator"):
                    pr = h.get("partial_r")
                    if pr is None and h.get("se"):
                        pr = partial_r(h["estimate"] / h["se"], max((h.get("n") or 10) - 10, 1))
                    # what the paper states for the observatory is the direction of a significant estimate (the same rule as the
                    # repository regime's claim sign); the author's verdict (support of the prediction) is kept beside it
                    units.append({"unit_id": f"{pid}:{h['hyp_id']}", "estimate": pr, "se": h.get("se"), "n": h.get("n"),
                                  "verdict": "supported" if claim_sign(h) != "none" else "not_supported", "author_verdict": h["verdict"],
                                  "attention": p["citations"], "quarter": p["quarter_published"],
                                  "legacy": p.get("legacy", False), "truth_check": h.get("truth_check"), "paper_id": pid})
                    break
        return units

    # ======================================================================== phase 1: research and revision
    def research(self, quarter: int) -> None:
        st = self.state
        jobs = []
        for s in self.active_scientists():
            a = st["authors"][s["id"]]
            for si, slot in enumerate(a["slots"]):
                if slot["state"] == "revising" and slot["paper_id"]:
                    if st["papers"][slot["paper_id"]].get("revision_due", quarter) > quarter:
                        continue   # the revision is still being worked on (journal.timing.revision)
                    jobs.append({"kind": "revision", "author": s, "slot": si, "paper_id": slot["paper_id"]})
                elif slot["state"] == "waiting":
                    continue  # the paper was already produced; it enters the submission list this quarter (see desk)
                elif slot["state"] == "free" and self.starts_allowed(quarter) and self.available(s["id"], quarter, si):
                    jobs.append({"kind": "research", "author": s, "slot": si})
        research_jobs = [j for j in jobs if j["kind"] == "research"]
        revision_jobs = [j for j in jobs if j["kind"] == "revision"]
        groups = self._group_jobs(research_jobs, quarter)   # one call per group (journal.research_batch_size; size 1 = current behaviour)
        lean = self.cfg["journal"].get("paper_contract", "full") == "lean"
        if self.w.mode == "fast":
            for g in groups:
                gid = f"G-{short_id(self.w.task_ns, quarter, *[f'{j['author']['id']}:{j['slot']}' for j in g], n=8)}" if len(g) > 1 else None
                for job in g:
                    self._fast_job(job, quarter, call_group=gid, call_group_size=len(g))
                # the call the rule agents stand in for is debited at tariff (once per group), so the paired budget is the same in every mode
                tp = "t1" if any(j["author"]["throughput"] == "T1" for j in g) else "t0"
                self.w.budget.debit_call(quarter, f"research_{tp}", f"fast-research-{g[0]['author']['id']}-{g[0]['slot']}-Q{quarter}",
                                         g[0]["author"]["id"], scale=self.w.budget.call_scale(len(g), lean))
            for job in revision_jobs:
                self._fast_job(job, quarter)
                self.w.budget.debit_call(quarter, f"revision_{job['author']['throughput'].lower()}",
                                         f"fast-revision-{job['author']['id']}-{job['slot']}-Q{quarter}", job["author"]["id"],
                                         scale=self.w.budget.call_scale(1, lean))
        else:
            tasks = [self._make_task(g[0], quarter) if len(g) == 1 else self._make_batch_task(g, quarter) for g in groups]
            tasks += [self._make_task(j, quarter) for j in revision_jobs]
            self.run_tasks(tasks, quarter)
            for t in tasks:
                if t.type == "research_batch":
                    self._ingest_research_batch(t, quarter)
                else:
                    self._ingest_research(t, quarter)
        # slots in 'waiting' become their paper's submission this quarter (handled in desk); mark them busy
        for s in self.active_scientists():
            for slot in st["authors"][s["id"]]["slots"]:
                if slot["state"] == "waiting":
                    slot["state"] = "busy"

    # ------------------------------------------------------------------ batching (journal.research_batch_size)
    def _group_jobs(self, jobs: list[dict], quarter: int) -> list[list[dict]]:
        """Greedy grouping of this quarter's research jobs into calls of at most `research_batch_size` studies.
        A job joins a group only if its six menu questions are disjoint from every member's menu, it shares no coauthor
        history with any member, and (repository regime) it comes from a different lab. Deterministic (jobs sorted by id)."""
        B = int(self.cfg["journal"].get("research_batch_size", 1))
        jobs = sorted(jobs, key=lambda j: (j["author"]["id"], j["slot"]))
        if B <= 1:
            return [[j] for j in jobs]
        hist = {k: set(v) for k, v in self.state["coauthor_history"].items()}
        for j in jobs:
            entries, _ = self._menu_for(j["author"], quarter, j["slot"], with_data=False)
            j["_menu_ids"] = {e["question_id"] for e in entries}
        groups: list[list[dict]] = []
        for j in jobs:
            placed = False
            for g in groups:
                if len(g) >= B:
                    continue
                ok = all(not (j["_menu_ids"] & m["_menu_ids"]) and m["author"]["id"] not in hist.get(j["author"]["id"], set())
                         and j["author"]["id"] not in hist.get(m["author"]["id"], set()) and m["author"]["id"] != j["author"]["id"]
                         and self._lab_of(m["author"]["id"]) != self._lab_of(j["author"]["id"]) for m in g)
                if ok:
                    g.append(j)
                    placed = True
                    break
            if not placed:
                groups.append([j])
        return groups

    def _lab_of(self, aid: str):
        return self.state.get("member_lab", {}).get(aid, aid)   # paper regime: no labs, every author is its own "lab"

    def _paper_spec(self, robustness_heading: str) -> str:
        jc = self.cfg["journal"]
        contract = jc.get("paper_contract", "full")
        lo, hi = jc.get("paper_words", {}).get(contract, [1200, 1800] if contract == "full" else [500, 800])
        words = f"{lo:,}-{hi:,} words of prose (the rendered table and the reference list do not count toward the limit)"
        if contract == "lean":
            return (f"{words} with these headings: Title; Abstract (at most 100 words); Hypotheses (one paragraph of motivation, "
                    f"then the list); Data and Method; Results (the table, rendered); {robustness_heading}Discussion and Limitations; References (by paper id).")
        return (f"{words} with these headings: Title; Abstract (at most 150 words); Theory and Hypotheses; Data and Sample; Method; "
                f"Results (the table, rendered); {robustness_heading}Discussion; Limitations; References (by paper id).")

    def _make_batch_task(self, jobs: list[dict], quarter: int) -> Task:
        st = self.state
        ids = [f"{j['author']['id']}:{j['slot']}" for j in jobs]
        tp = "t1" if any(j["author"]["throughput"] == "T1" for j in jobs) else "t0"
        tier = jobs[0]["author"].get("tier", "sonnet")
        outputs = [f"{j['author']['id']}/{f}" for j in jobs for f in ("analysis.py", "table.csv", "claims.json", "paper.md")]
        t = Task(type="research_batch", quarter=quarter, identity="+".join(ids), tier=tier, run_id=self.w.task_ns, call_type=f"research_{tp}",
                 template="research_batch.md", outputs=outputs, schema="claims.schema.json",
                 meta={"authors": [j["author"]["id"] for j in jobs], "slots": [j["slot"] for j in jobs], "batch": len(jobs)},
                 who="+".join(j["author"]["id"] for j in jobs))
        d = self.w.store.task_dir(t)
        block, inputs, ctx_jobs = [], {}, []
        for j in jobs:
            a = j["author"]
            protocol = read_text(PROMPTS / PROTOCOL[a["style"]]).strip()
            depth = read_text(PROMPTS / "_depth_t1.md").strip() if a["throughput"] == "T1" else "No additional specifications are required."
            block.append(f"- Researcher `{a['id']}`: folder `{d}\\authors\\{a['id']}\\`; write its outputs into `{d}\\output\\{a['id']}\\`. "
                         f"Working protocol: {protocol} {depth}")
            if self.w.store.status(t) == "new":
                entries, hidden = self._menu_for(a, quarter, j["slot"], with_data=True, task_dir_rel=str(d / "authors" / a["id"]))
                inputs[f"authors/{a['id']}/profile.md"] = self._profile(a, quarter)
                inputs[f"authors/{a['id']}/menu.md"] = texts.menu_md(entries)
            else:
                entries, hidden = self._menu_for(a, quarter, j["slot"], with_data=False)
                hidden = self._regen_hidden(entries)
            ctx_jobs.append({"author": a, "slot": j["slot"], "entries": entries, "hidden": hidden})
        t.inputs = inputs
        t.variables = {"N_ITEMS": len(jobs), "AUTHORS_BLOCK": "\n".join(block), "OUTPUT_DIR": str(d / "output" / "<researcher id>"),
                       "PAPER_SPEC": self._paper_spec("Robustness (where the researcher's protocol requires it); "),
                       "ROBUSTNESS_HEADING": "Robustness (where required); ",
                       "DATA_PATH_RULE": "..\\..\\authors\\<researcher id>\\inputs\\<question id>\\data.csv"}
        self._task_ctx[t.id] = {"type": "research_batch", "jobs": ctx_jobs, "quarter": quarter}
        return t

    def _ingest_research_batch(self, t: Task, quarter: int) -> None:
        ctx = self._task_ctx.get(t.id, {})
        d = self.w.store.task_dir(t)
        if self.w.store.status(t) != "done":
            self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": t.who, "batch": t.meta.get("batch")})
            return
        n = len(ctx["jobs"])
        for j in ctx["jobs"]:
            a = j["author"]
            claims = self.w.store.read_output(t, f"{a['id']}/claims.json")
            paper_md = self.w.store.read_output(t, f"{a['id']}/paper.md")
            entry = next((e for e in j["entries"] if e["question_id"] == claims["question_id"]), None)
            if entry is None:
                self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": a["id"], "reason": "question not on menu"})
                continue
            qid = entry["question_id"]
            rel = f"authors/{a['id']}/inputs/{qid}/data.csv"
            forensic = forensic_rerun(d / "output" / a["id"], {rel: d / rel}, self.cfg["datasets"]["forensics"]["mode"],
                                      self.cfg["datasets"]["forensics"]["tolerance"], work_rel=f"output/{a['id']}", task_root=d) \
                if (d / rel).exists() else {"reproducible": None, "reason": "data not kept"}
            self._register_paper(a, j["slot"], qid, claims, paper_md, j["hidden"][qid]["hidden"], j["hidden"][qid]["meta"], entry, t.id, quarter, forensic,
                                 call_group=t.id, call_group_size=n)
            if not self._keep_csv():
                for e in j["entries"]:
                    f = d / "authors" / a["id"] / "inputs" / e["question_id"] / "data.csv"
                    if f.exists():
                        f.unlink()

    def _menu_for(self, author: dict, quarter: int, slot: int, with_data: bool, task_dir_rel: str | None = None) -> tuple[list[dict], dict]:
        entries = self.menu_entries(author["primary_theory"], quarter, f"{author['id']}:{slot}")
        hidden = {}
        for e in entries:
            qid = e["question_id"]
            ds_seed = seed_from(self.w.seeds["run"], "dataset", quarter, author["id"], slot, qid)
            ds_id = f"D-{short_id(self.w.task_ns, quarter, author['id'], slot, qid, n=8)}"
            e["dataset_seed"], e["dataset_id"] = ds_seed, ds_id
            e["view"] = self.view(qid)
            e["upper_half"] = author["throughput"] == "T1"
            if with_data:
                res = self.w.gen.generate(qid, ds_seed, ds_id, upper_half=e["upper_half"], write_dir=Path(task_dir_rel) / "inputs" / qid)
                e["n_preview"] = res["meta"]["n_rows"]
                hidden[qid] = {"hidden": res["hidden"], "meta": res["meta"], "structure": None}
            else:
                from ..datasets import draw_frame
                from ..config import rng as _rng
                fr = draw_frame(e["question"], _rng(ds_seed, "frame"), self.cfg["datasets"]["n_cap"], upper_half=e["upper_half"])
                e["n_preview"] = fr["n_firms"] * fr["n_years"] if fr["panel"] else fr["n_target"]
        ids = [e["question_id"] for e in entries]
        for e in entries:
            e["_siblings_ids"] = ids
        return entries, hidden

    def _make_task(self, job: dict, quarter: int) -> Task:
        a = job["author"]
        st = self.state
        profile = self._profile(a, quarter)
        protocol = read_text(PROMPTS / PROTOCOL[a["style"]])
        depth = read_text(PROMPTS / "_depth_t1.md") if a["throughput"] == "T1" else "(No additional specifications are required.)"
        rob = "Robustness; " if a["throughput"] == "T1" else ""
        tier = a.get("tier", "sonnet")
        spec = self._paper_spec(rob)
        if job["kind"] == "research":
            t = Task(type="research", quarter=quarter, identity=f"{a['id']}:{job['slot']}", tier=tier, run_id=self.w.task_ns,
                     call_type=f"research_{a['throughput'].lower()}", template="research.md",
                     variables={"PROTOCOL_BLOCK": protocol, "DEPTH_BLOCK": depth, "ROBUSTNESS_HEADING": rob, "PAPER_SPEC": spec,
                                "DATA_PATH_RULE": "..\\inputs\\<question id>\\data.csv"},
                     outputs=["analysis.py", "table.csv", "claims.json", "paper.md"], schema="claims.schema.json",
                     meta={"author_id": a["id"], "slot": job["slot"]}, who=a["id"])
            d = self.w.store.task_dir(t)
            if self.w.store.status(t) == "new":
                entries, hidden = self._menu_for(a, quarter, job["slot"], with_data=True, task_dir_rel=str(d))
                t.inputs = {"profile.md": profile, "menu.md": texts.menu_md(entries)}
            else:  # resumed: rebuild the menu context (hidden truth) without rewriting files
                entries, hidden = self._menu_for(a, quarter, job["slot"], with_data=False)
                hidden = self._regen_hidden(entries)
            self._task_ctx[t.id] = {"type": "research", "author": a, "entries": entries, "hidden": hidden, "quarter": quarter}
            return t
        # revision
        p = st["papers"][job["paper_id"]]
        t = Task(type="revision", quarter=quarter, identity=p["paper_id"], tier=tier, run_id=self.w.task_ns,
                 call_type=f"revision_{a['throughput'].lower()}", template="revision.md",
                 variables={"PROTOCOL_BLOCK": protocol, "DEPTH_BLOCK": depth, "ROBUSTNESS_HEADING": rob, "PAPER_SPEC": spec,
                            "DATA_PATH_RULE": "..\\inputs\\data\\data.csv"},
                 outputs=["analysis.py", "table.csv", "claims.json", "paper.md", "response.md"], schema="claims.schema.json",
                 meta={"author_id": a["id"], "paper_id": p["paper_id"]}, who=a["id"])
        if self.w.store.status(t) == "new":
            d = self.w.store.task_dir(t)
            self._ensure_dataset_files(p, d / "inputs" / "data")
            orig = self._paper_dir(p)
            inputs = {"profile.md": profile, "reviews.md": texts.reviews_md(p["reviews"]["1"]),
                      "prior.md": texts.prior_md([v for v in self.view(p["question_id"]) if v["paper_id"] != p["paper_id"]], p["question_id"]),
                      "editor_letter.md": f"# Editor's letter\n\n{p['decisions']['1']['letter']}\n\nRequired changes:\n" +
                                          "\n".join(f"- {c}" for c in p["decisions"]["1"].get("required_changes", [])) + "\n"}
            for f in ("analysis.py", "table.csv", "claims.json", "paper.md"):
                if (orig / f).exists():
                    inputs[f"inputs/original/{f}"] = orig / f
            menu = self._menu_path(p)   # the question's menu card; prior.md carries what is published now
            if menu is not None:
                inputs["menu.md"] = menu
            else:
                q = self.w.landscape.question(p["question_id"])
                entry = {"question": q, "n_preview": p["n"], "count": st["counts"].get(p["question_id"], 0), "view": [], "slot": p.get("menu_slot", 0)}
                inputs["menu.md"] = texts.menu_md([entry])
            t.inputs = inputs
        self._task_ctx[t.id] = {"type": "revision", "author": a, "paper": p, "quarter": quarter}
        return t

    def _regen_hidden(self, entries: list[dict]) -> dict:
        hidden = {}
        for e in entries:
            res = self.w.gen.generate(e["question_id"], e["dataset_seed"], e["dataset_id"], upper_half=e["upper_half"], write_dir=None)
            e["n_preview"] = res["meta"]["n_rows"]
            hidden[e["question_id"]] = {"hidden": res["hidden"], "meta": res["meta"], "structure": None}
        return hidden

    def _paper_dir(self, p: dict) -> Path:
        """Folder holding a paper's current files. A batched research call writes each study to output/<lead id>/,
        so review packets for batched papers must be built from there."""
        tid = p["task_ids"][-1]
        d = self.w.store.task_dir(self._task_from_record(tid)) / "output"
        return d / p["lead"] if tid.startswith("research_batch-") else d

    def _paper_files(self, p: dict, names: tuple[str, ...] = ("paper.md", "table.csv", "analysis.py")) -> dict[str, Path]:
        """The paper's files that go into a review or pre-screen packet; a missing file is a hard error (never send an
        empty packet)."""
        pdir = self._paper_dir(p)
        out = {}
        for f in names:
            if not (pdir / f).exists():
                raise FileNotFoundError(f"{p['paper_id']}: {f} missing in {pdir}")
            out[f] = pdir / f
        return out

    def _menu_path(self, p: dict) -> Path | None:
        """The menu card the lead author chose the question from (its original research packet): `menu.md` of a single
        research task, `authors/<lead>/menu.md` of a batched one."""
        t0 = self.w.store.task_dir(self._task_from_record(p["task_ids"][0]))
        for cand in (t0 / "menu.md", t0 / "authors" / p["lead"] / "menu.md"):
            if cand.exists():
                return cand
        return None

    def _task_from_record(self, task_id: str) -> Task:
        rec = json.loads(read_text(self.w.store.dir / task_id / "task.json"))
        return Task(type=rec["type"], quarter=rec["quarter"], identity="", tier=rec["tier"], run_id=rec["run_id"],
                    call_type=rec["call_type"], template=rec["template"], outputs=rec["outputs"], schema=rec["schema"],
                    meta=rec["meta"], fixed_id=task_id)

    def _ensure_dataset_files(self, p: dict, dest: Path) -> None:
        """Regenerate data.csv/codebook.md/sample_frame.md for a paper's dataset (deterministic in seed + id)."""
        if (dest / "data.csv").exists():
            return
        self.w.gen.generate(p["question_id"], p["dataset_seed"], p["dataset_id"], upper_half=p["throughput"] == "T1", write_dir=dest)

    # ------------------------------------------------------------------ ingest
    def _ingest_research(self, t: Task, quarter: int) -> None:
        ctx = self._task_ctx.get(t.id, {})
        a = ctx["author"]
        slot = t.meta.get("slot")
        d = self.w.store.task_dir(t)
        if self.w.store.status(t) != "done":
            self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": a["id"]})
            if t.type == "revision":
                self._abandon(ctx["paper"], quarter, "revision_failed")
            return
        claims = self.w.store.read_output(t, "claims.json")
        paper_md = self.w.store.read_output(t, "paper.md")
        if t.type == "research":
            entry = next((e for e in ctx["entries"] if e["question_id"] == claims["question_id"]), None)
            if entry is None:
                self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "reason": "question not on menu"})
                return
            qid = entry["question_id"]
            hid = ctx["hidden"][qid]
            rel = f"inputs/{qid}/data.csv"
            forensic = forensic_rerun(d / "output", {rel: d / rel}, self.cfg["datasets"]["forensics"]["mode"],
                                      self.cfg["datasets"]["forensics"]["tolerance"], task_root=d) if (d / rel).exists() else {"reproducible": None, "reason": "data not kept"}
            self._register_paper(a, slot, qid, claims, paper_md, hid["hidden"], hid["meta"], entry, t.id, quarter, forensic)
            if not self._keep_csv():
                for e in ctx["entries"]:
                    f = d / "inputs" / e["question_id"] / "data.csv"
                    if f.exists():
                        f.unlink()
        else:
            p = ctx["paper"]
            rel = "inputs/data/data.csv"
            forensic = forensic_rerun(d / "output", {rel: d / rel}, self.cfg["datasets"]["forensics"]["mode"],
                                      self.cfg["datasets"]["forensics"]["tolerance"], task_root=d) if (d / rel).exists() else {"reproducible": None, "reason": "data not kept"}
            self._apply_revision(p, claims, paper_md, t.id, quarter, forensic)
            if not self._keep_csv() and (d / rel).exists():
                (d / rel).unlink()

    def _score_claims(self, claims: dict, hidden: dict, meta: dict, q: dict, paper_md: str | None, style: str, throughput: str,
                      reproducible: bool | None, table_notes: list[str] | None = None) -> tuple[list[dict], dict]:
        hyps = []
        controls = meta["variables"]["controls"]
        for h in claims["hypotheses"]:
            theta = relation_theta(hidden, h["iv"], h["dv"], h.get("moderator"), h.get("kind", ""), q, controls=controls)
            theta_fe = hidden["theta_fe"] if (h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator")) else theta
            sc = score_hypothesis(h, theta, theta_fe)
            hh = dict(h)
            hh.update(sc)
            if hh.get("estimate") is not None and hh.get("se"):
                k = len(h.get("controls", [])) + 3
                hh["partial_r"] = partial_r(hh["estimate"] / hh["se"], max((hh.get("n") or 10) - k, 1))
            hyps.append(hh)
        flags = analyst_flags(claims, hidden, meta, q, paper_md, style, throughput, reproducible, table_notes=table_notes)
        return hyps, flags

    def _register_paper(self, a: dict, slot: int, qid: str, claims: dict, paper_md: str, hidden: dict, meta: dict, entry: dict,
                        task_id: str, quarter: int, forensic: dict, fast_study: dict | None = None,
                        call_group: str | None = None, call_group_size: int = 1) -> dict:
        st = self.state
        q = self.w.landscape.question(qid)
        hyps, flags = self._score_claims(claims, hidden, meta, q, paper_md, a["style"], a["throughput"], forensic.get("reproducible"),
                                         table_notes=forensic.get("table_notes"))
        st["paper_counter"] += 1
        pid = f"P{st['paper_counter']:04d}"
        team = draw_team(self.w.seeds, a, self.active_scientists(), quarter, st["paper_counter"],
                         self.cfg["population_profile"]["teams"], {k: set(v) for k, v in st["coauthor_history"].items()})
        for m in team:
            for n in team:
                if m != n:
                    st["coauthor_history"].setdefault(m, [])
                    if n not in st["coauthor_history"][m]:
                        st["coauthor_history"][m].append(n)
        title = _title(paper_md) or f"{q['iv_label']} and {q['dv_label']}"
        abstract = _section(paper_md, "Abstract") if paper_md else ""
        costly_delay = 1 if (q["access_cost"] == "costly" and a["throughput"] == "T0") else 0
        # a study takes `research_quarters` quarters from start to submission (the call happens at the start; the slot is held)
        research_q = max(1, int((self.cfg["journal"].get("timing") or {}).get("research_quarters", 1)))
        submit_q = quarter + costly_delay + research_q - 1
        p = {"paper_id": pid, "question_id": qid, "core": q["core"], "legacy": False, "prehistory": bool(self.prehistory),
             "lead": a["id"], "authors": team,
             "style": a["style"], "throughput": a["throughput"], "tier": a.get("tier", "sonnet"), "quarter_started": quarter,
             "quarter_submitted": submit_q, "status": "submitted" if submit_q == quarter else "waiting",
             "dataset_id": meta["dataset_id"], "dataset_seed": meta["dataset_seed"], "n": meta["n_rows"], "n_correct": meta["n_correct"],
             "design": q["design"], "estimator": hyps[0]["estimator"] if hyps else "", "theory": a["primary_theory"],
             "title": title, "abstract": abstract, "hypotheses": hyps, "relation_to_prior": claims["relation_to_prior"],
             "builds_on": claims.get("builds_on", []), "sample_changes": claims.get("sample_changes", []),
             "n_models_run": claims.get("n_models_run", 0), "robustness_models": claims.get("robustness_models", []),
             "flags": flags, "forensic": forensic, "hidden": {k: hidden[k] for k in ("traps", "theta", "theta_fe", "theta_naive", "target", "moderators",
                                                                                  "subsample_offset", "theta_secondary", "mediator_var", "cov", "cov_names", "oracle")},
             "task_ids": [task_id], "reviews": {}, "decisions": {}, "desk": None, "reviewers": [], "merit": None,
             "citations": 0, "cited_by": [], "quarter_published": None, "menu_slot": entry["slot"], "menu_ids": [e["question_id"] for e in self._entries_of(entry)],
             "self_cites": sum(1 for b in claims.get("builds_on", []) if b.get("paper_id") in st["papers"] and set(st["papers"][b["paper_id"]]["authors"]) & set(team))}
        if call_group:  # recorded only when research calls are batched, so unbatched runs keep their byte-identical ledgers
            p["call_group"], p["call_group_size"] = call_group, int(call_group_size)
        st["papers"][pid] = p
        aslot = st["authors"][a["id"]]["slots"][slot]
        aslot.update({"state": "waiting" if submit_q > quarter else "busy", "paper_id": pid})
        st["records"][a["id"]]["papers"].append({"paper_id": pid, "title": title, "question_id": qid, "status": "under review", "gist": ""})
        for m in team[1:]:
            st["records"][m]["coauthored"] = st["records"][m].get("coauthored", 0) + 1
        self.w.budget.debit_data(quarter, meta["n_rows"], meta["dataset_id"], a["id"])
        self.w.ledger.add("datasets", {"quarter": quarter, "dataset_id": meta["dataset_id"], "question_id": qid, "who": a["id"],
                                       "meta": {k: v for k, v in meta.items() if k != "columns"}, "hidden": {k: v for k, v in hidden.items() if k not in ("cov",)}})
        self.w.ledger.add("events", {"quarter": quarter, "event": "paper_started", "paper_id": pid, "question_id": qid, "who": a["id"],
                                     "slot": entry["slot"], "traps": hidden["traps"], "flags": flags["flags"]})
        return p

    def _entries_of(self, entry: dict) -> list[dict]:
        return [{"question_id": x} for x in entry.get("_siblings_ids", [entry["question_id"]])]

    def _apply_revision(self, p: dict, claims: dict, paper_md: str, task_id: str, quarter: int, forensic: dict) -> None:
        q = self.w.landscape.question(p["question_id"])
        a = self.scientist(p["lead"])
        meta = {"variables": {"controls": [c for c in q["controls_pool"] if c not in ("industry", "year")]}, "n_correct": p["n_correct"]}
        hyps, flags = self._score_claims(claims, p["hidden"], meta, q, paper_md, a["style"], a["throughput"], forensic.get("reproducible"),
                                         table_notes=forensic.get("table_notes"))
        p.update({"hypotheses": hyps, "flags": flags, "forensic": forensic, "relation_to_prior": claims["relation_to_prior"],
                  "builds_on": claims.get("builds_on", p["builds_on"]), "sample_changes": claims.get("sample_changes", []),
                  "status": "resubmitted", "quarter_resubmitted": quarter, "abstract": _section(paper_md, "Abstract") or p["abstract"],
                  "n_models_run": claims.get("n_models_run", 0), "robustness_models": claims.get("robustness_models", [])})
        p["task_ids"].append(task_id)
        slot = self._slot_of(p)   # the slot is held through the re-review (it may take several quarters); never a second revision call
        if slot:
            slot["state"] = "busy"

    def _abandon(self, p: dict, quarter: int, reason: str) -> None:
        p["status"] = "abandoned"
        p["quarter_final_decision"] = quarter
        self._free_slot(p)
        self._ledger_hypotheses(p, certified=False, quarter=quarter)
        self.w.ledger.add("events", {"quarter": quarter, "event": "abandoned", "paper_id": p["paper_id"], "reason": reason})

    # ------------------------------------------------------------------ fast path
    def _fast_job(self, job: dict, quarter: int, call_group: str | None = None, call_group_size: int = 1) -> None:
        a = job["author"]
        ra = self.rules["author"]
        if job["kind"] == "research":
            entries, _ = self._menu_for(a, quarter, job["slot"], with_data=False)
            r = self.rng("fast_author", quarter, a["id"], job["slot"])
            entry = ra.choose(entries, a, r)
            q = entry["question"]
            spec = self.w.gen.spec_for(q)
            from ..traps import draw_traps
            from ..config import rng as _rng
            traps = draw_traps(spec, _rng(entry["dataset_seed"], "traps"), self.cfg["datasets"]["traps"]["density"], self.cfg["datasets"]["traps"]["registry"])
            handled = ra.decide_handling(traps, a["style"], r)
            use_fe = ra.use_fe(q, a, r)
            variables = {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")}
            res = ra.analyse_analytic(self.w.gen, q, entry["dataset_seed"], a, handled, use_fe, ra.n_hypotheses(r), variables, entry["upper_half"])
            rel = ra.relation_to_prior(res["hypotheses"][0]["estimate"], entry["view"], r)
            builds = [{"paper_id": v["paper_id"], "informed": ["H1"]} for v in entry["view"][:3]]
            claims = ra.claims_json(q, res, rel, builds, nonce="fast")
            s = res["study"]
            hidden = dict(s["hidden"])
            hidden.update({"oracle": {"correct": {"estimate": s["hidden"]["target"], "se": s["focal"]["se"], "n": s["n"]}, "naive": {}},
                           "cov": None, "cov_names": None})
            meta = {"dataset_id": f"D-{short_id(self.w.task_ns, quarter, a['id'], job['slot'], q['id'], n=8)}", "dataset_seed": entry["dataset_seed"],
                    "n_rows": s["n"], "n_correct": s["n"], "frame": s["frame"], "variables": variables, "admissible": True}
            paper_md = None
            forensic = {"reproducible": None, "reason": "fast mode"}
            p = self._register_paper(a, job["slot"], q["id"], claims, paper_md, hidden, meta, entry, "fast", quarter, forensic,
                                     call_group=call_group, call_group_size=call_group_size)
            p["title"] = f"{q['iv_label']} and {q['dv_label']}"
            p["abstract"] = ra.paper_md(q, res, rel, a, entry["view"], handled).split("## Abstract")[1].split("##")[0].strip()
            # analytic flags: unhandled traps, duplicates -> n inflated, no clustering when no FE on a panel
            fl = [{"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed", "reverse_coded": "reverse_coding_missed",
                   "selection_on_outcome": "selection_unacknowledged", "bad_control": "bad_control_included", "miscoded_values": "miscoded_values_kept",
                   "unit_mismatch": "unit_mismatch_unfixed", "omitted_standard_control": "omitted_standard_control"}[t] for t in traps if not handled.get(t)]
            if "duplicates" in traps and not handled.get("duplicates"):
                fl.append("n_inflated")
            if q["panel"] and not use_fe:
                fl.append("se_not_clustered")
            p["flags"] = {"flags": fl, "trap_handled": handled, "n_models_run": res["n_models_run"],
                          "robustness_block_present": a["throughput"] == "T1", "specification_search": res["n_models_run"] > 3}
        else:
            p = self.state["papers"][job["paper_id"]]
            r = self.rng("fast_revision", quarter, p["paper_id"])
            # the author fixes problems the referees named with a style-dependent probability (a design default, not grounded in the corpus)
            named = {i["kind"] for rv in p["reviews"]["1"] for i in rv["issues"] if i["severity"] == "major"}
            fix_p = 0.8 if p["style"] == "VERIFIED" else 0.3
            handled = dict(p["flags"]["trap_handled"])
            for t, h in handled.items():
                if not h and named and r.uniform() < fix_p:
                    handled[t] = True
            q = self.w.landscape.question(p["question_id"])
            a = self.scientist(p["lead"])
            spec = self.w.gen.spec_for(q)
            variables = {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")}
            res = ra.analyse_analytic(self.w.gen, q, p["dataset_seed"], a, handled, p["hypotheses"][0]["estimator"] == "fixed effects",
                                      len(p["hypotheses"]), variables, p["throughput"] == "T1")
            claims = ra.claims_json(q, res, p["relation_to_prior"], p["builds_on"], nonce="fast")
            self._apply_revision(p, claims, None, "fast", quarter, {"reproducible": None, "reason": "fast mode"})
            fl = [f for f in p["flags"]["flags"] if _flag_trap(f) not in {t for t, h in handled.items() if h}]
            p["flags"] = {"flags": fl, "trap_handled": handled, "n_models_run": res["n_models_run"],
                          "robustness_block_present": p["throughput"] == "T1", "specification_search": res["n_models_run"] > 3}

    # ======================================================================== phase 2-3: desk screen
    def _submissions(self, quarter: int) -> list[dict]:
        st = self.state
        subs = []
        for p in st["papers"].values():
            if p.get("legacy"):
                continue
            if p["status"] == "waiting" and p["quarter_submitted"] <= quarter:
                p["status"] = "submitted"
            if p["status"] == "submitted" and p["quarter_submitted"] <= quarter and p["desk"] is None:
                subs.append(p)
        return sorted(subs, key=lambda p: p["paper_id"])

    def desk(self, quarter: int) -> None:
        subs = self._submissions(quarter)
        if not subs:
            return
        jc = self.cfg["journal"]
        strict = jc["desk_strictness"]
        if self.w.mode == "fast":
            for p in subs:
                r = self.rng("desk", quarter, p["paper_id"])
                dec = self.rules["editor"].desk({"paper_id": p["paper_id"], "hypotheses": p["hypotheses"]}, strict, r)
                self._apply_desk(p, dec, quarter)
            for i in range(0, len(subs), 10):
                self.w.budget.debit_call(quarter, "desk_screen", f"fast-desk-Q{quarter}-{i // 10}", "editor")
            return
        tasks = []
        for i in range(0, len(subs), 10):
            batch = subs[i:i + 10]
            # the editor sees the author status line at every blinding level except triple-blind
            text = "\n".join(texts.submission_summary(p, self.state["counts"].get(p["question_id"], 0) if jc["show_question_count_to_editor"] else None,
                                                       None if jc["blinding"] == "triple" else self._status_line(p))
                             for p in batch)
            strict_sentence = {"scope_only": "Desk-reject only submissions that are out of scope or unintelligible.",
                               "contribution": "Desk-reject submissions that are out of scope, unintelligible, or whose contribution is insufficient for the Journal.",
                               "design": "Desk-reject submissions that are out of scope, unintelligible, insufficient in contribution, or whose design cannot support their claims."}[strict]
            t = Task(type="desk", quarter=quarter, identity=f"batch{i // 10}", tier=jc["editor_tier"], run_id=self.w.task_ns,
                     call_type="desk_screen", template="desk_screen.md",
                     variables={"SUBMISSIONS": text, "STRICTNESS_SENTENCE": strict_sentence,
                                "COUNT_CLAUSE": ", and the number of papers already published on the manuscript's question" if jc["show_question_count_to_editor"] else ""},
                     outputs=["desk.json"], schema="desk.schema.json", meta={"paper_ids": [p["paper_id"] for p in batch]}, who="editor")
            self._task_ctx[t.id] = {"type": "desk", "items": [{"paper_id": p["paper_id"], "hypotheses": p["hypotheses"]} for p in batch], "strictness": strict}
            tasks.append(t)
        self.run_tasks(tasks, quarter)
        for t in tasks:
            if self.w.store.status(t) != "done":
                for pid in t.meta["paper_ids"]:  # a failed editor call: send out by rule (logged)
                    self._apply_desk(self.state["papers"][pid], {"decision": "send_out", "reason": "editor call failed; sent by rule"}, quarter)
                continue
            out = self.w.store.read_output(t, "desk.json")
            for dec in out["decisions"]:
                if dec["paper_id"] in self.state["papers"]:
                    self._apply_desk(self.state["papers"][dec["paper_id"]], dec, quarter)

    def _status_line(self, p: dict) -> str:
        return texts.author_status_line([self.scientist(a) for a in p["authors"]], self.state["records"])

    def _apply_desk(self, p: dict, dec: dict, quarter: int) -> None:
        p["desk"] = {"decision": dec["decision"], "reason": dec.get("reason", ""), "quarter": quarter}
        self.w.ledger.add("decisions", {"quarter": quarter, "paper_id": p["paper_id"], "stage": "desk", "decision": dec["decision"],
                                        "question_id": p["question_id"], "truth_check_focal": p["hypotheses"][0].get("truth_check")})
        if dec["decision"] == "desk_reject":
            p["status"] = "desk_rejected"
            p["quarter_final_decision"] = quarter
            self._free_slot(p)
            self._ledger_hypotheses(p, certified=False, quarter=quarter)
            self._record_status(p, "desk rejected", dec.get("reason", ""))
        else:
            p["status"] = "sent_out"
            self.state["review_queue"].append(p["paper_id"])

    # ======================================================================== phase 4: reviewer assignment (rule)
    def assign(self, quarter: int) -> None:
        st = self.state
        jc = self.cfg["journal"]
        cap = int(jc["reviewer_cap"])
        k = int(jc["reviewers_per_paper"])
        eligible = [s for s in self.active_scientists() if s["stage"] != "doctoral"]
        load = {s["id"]: 0 for s in eligible}
        # re-reviews first (same reviewers)
        resub = [p for p in st["papers"].values() if p["status"] == "resubmitted"]
        for p in resub:
            for rid in p["reviewers"]:
                load[rid] = load.get(rid, 0) + 1
            p["status"] = "under_review"
            p["review_round"] = 2
            p["review_due"] = quarter + self.latency("rereview", p["paper_id"], 2)   # the re-review takes its time
        queue = [pid for pid in st["review_queue"]]
        assigned, waiting = [], []
        r = self.rng("assign", quarter)
        for pid in queue:
            p = st["papers"][pid]
            q = self.w.landscape.question(p["question_id"])
            same_q_authors = {a for pp in st["papers"].values() if pp["question_id"] == p["question_id"] and not pp.get("legacy") and pp["paper_id"] != pid for a in pp["authors"]}
            coauth = set()
            for a in p["authors"]:
                coauth |= set(st["coauthor_history"].get(a, []))
            # reviewer exclusion ladder: authors always excluded; same-question authors excluded when >= k remain;
            # the coauthor-history and cap exclusions are relaxed, in that order, only when fewer than k candidates remain
            base = [s for s in eligible if s["id"] not in p["authors"]]
            cands = [s for s in base if s["id"] not in coauth and load.get(s["id"], 0) < cap]
            relaxed = None
            if jc["exclude_same_question_reviewers"]:
                c2 = [s for s in cands if s["id"] not in same_q_authors]
                if len(c2) >= k:
                    cands = c2
            if len(cands) < k:
                cands = [s for s in base if load.get(s["id"], 0) < cap]
                relaxed = "coauthor_history"
            if len(cands) < k:
                cands = base
                relaxed = "cap"
            if not cands:
                waiting.append(pid)
                continue
            def score(s):
                sc = 0.0
                if s["primary_theory"] == q["family"]:
                    sc += 2
                elif s["secondary_theory"] == q["family"]:
                    sc += 1
                if s["repertoire"] == "survey" and "survey" in q["design"]:
                    sc += 1
                if s["repertoire"] == "causal_design" and q["identification"]["instrument_variable"]:
                    sc += 1
                debt = quarter - st["reviewer_last"].get(s["id"], 0)
                return (sc, debt, r.uniform())
            ranked = sorted(cands, key=score, reverse=True)
            chosen = [s["id"] for s in ranked[:k]]
            for rid in chosen:
                load[rid] = load.get(rid, 0) + 1
                st["reviewer_last"][rid] = quarter
            p["reviewers"] = chosen
            p["status"] = "under_review"
            p["review_round"] = 1
            p["review_due"] = quarter + self.latency("review_round", pid, 1)   # reviews come in `review_due - quarter` quarters later
            assigned.append(pid)
            self.w.ledger.add("events", {"quarter": quarter, "event": "assigned", "paper_id": pid, "reviewers": chosen,
                                         "scores": [score(s)[0] for s in ranked[:k]], "relaxed": relaxed})
        st["review_queue"] = waiting
        demand = k * len(queue) + k * len(resub)
        capacity = cap * len(eligible)
        self.w.ledger.add("events", {"quarter": quarter, "event": "review_capacity", "demand": demand, "capacity": capacity,
                                     "queued": len(waiting), "assigned": len(assigned), "re_reviews": len(resub)})

    # ======================================================================== phase 5: review
    def review(self, quarter: int) -> None:
        st = self.state
        jc = self.cfg["journal"]
        # only papers whose reviews are due this quarter are reviewed (the call is made when the reviews come in, so the
        # referees see the literature of that quarter and the calls are spread over the latency)
        under = [p for p in st["papers"].values() if p["status"] == "under_review" and p.get("review_due", quarter) <= quarter]
        if not under:
            return
        assistance = jc["reviewer_assistance"]
        if self.w.mode == "fast":
            load: dict[str, int] = {}
            for p in under:
                rnd = str(p.get("review_round", 1))
                revs = []
                for rid in p["reviewers"]:
                    load[rid] = load.get(rid, 0) + 1
                    r = self.rng("review", quarter, p["paper_id"], rid)
                    rev = self.rules["reviewer"].review(self._review_item(p), r, assistance, batch_size=3)
                    rev["reviewer_id"] = rid
                    revs.append(rev)
                p["reviews"][rnd] = revs
                self._ledger_reviews(p, revs, quarter)
            batch_cap = 6 if assistance == "ai_written" else int(jc["reviewer_cap"])
            call = "review_ai_written" if assistance == "ai_written" else "review"
            for rid, n in sorted(load.items()):
                for b in range(0, n, batch_cap):
                    self.w.budget.debit_call(quarter, call, f"fast-review-{rid}-Q{quarter}-{b}", rid)
            if assistance == "prescreen":
                for i in range(0, len(under), 4):
                    self.w.budget.debit_call(quarter, "prescreen", f"fast-prescreen-Q{quarter}-{i // 4}", "prescreen")
            return
        # batches per reviewer (reviewer load)
        by_rev: dict[str, list[dict]] = {}
        for p in under:
            for rid in p["reviewers"]:
                by_rev.setdefault(rid, []).append(p)
        tasks = []
        prescreens = {}
        if assistance == "prescreen":
            prescreens = self._prescreen(under, quarter)
        for rid, papers in sorted(by_rev.items()):
            s = self.scientist(rid)
            batch_cap = 6 if assistance == "ai_written" else int(jc["reviewer_cap"])
            for bi in range(0, len(papers), batch_cap):
                batch = papers[bi:bi + batch_cap]
                inputs = {"profile.md": self._profile(s, quarter)}
                for p in batch:
                    pdir = self._paper_dir(p)
                    for f, src in self._paper_files(p).items():
                        inputs[f"inputs/{p['paper_id']}/{f}"] = src
                    cb = self._codebook_path(p)
                    if cb and cb.exists():
                        inputs[f"inputs/{p['paper_id']}/codebook.md"] = cb
                    inputs[f"inputs/{p['paper_id']}/prior.md"] = texts.prior_md([v for v in self.view(p["question_id"]) if v["paper_id"] != p["paper_id"]], p["question_id"])
                    if p.get("review_round", 1) == 2:
                        prev = next((rv for rv in p["reviews"]["1"] if rv["reviewer_id"] == rid), None)
                        inputs[f"inputs/{p['paper_id']}/your_previous_review.md"] = texts.reviews_md([prev]) if prev else ""
                        inputs[f"inputs/{p['paper_id']}/response.md"] = pdir / "response.md" if (pdir / "response.md").exists() else "(no response file)"
                    if p["paper_id"] in prescreens:
                        inputs[f"inputs/{p['paper_id']}/prescreen.md"] = f"# Pre-screen flags\n\n{prescreens[p['paper_id']]}\n"
                status = ""
                if jc["blinding"] in ("open", "signed", "single"):
                    status = "Author status for each submission is given in its folder's status.md."
                    for p in batch:
                        inputs[f"inputs/{p['paper_id']}/status.md"] = f"Authors: {self._status_line(p)}\n"
                extra = ""
                if any(pp.get("review_round", 1) == 2 for pp in batch):
                    extra = ", your previous review (`your_previous_review.md`) and the authors' response (`response.md`) for resubmissions"
                if prescreens:
                    extra += ", and a pre-screen flag list (`prescreen.md`)"
                t = Task(type="review", quarter=quarter, identity=f"{rid}:{bi}", tier=jc["reviewer_tier"], run_id=self.w.task_ns,
                         call_type="review_ai_written" if assistance == "ai_written" else "review",
                         template="review_ai_written.md" if assistance == "ai_written" else "review.md",
                         variables={"N_ITEMS": len(batch), "ITEM_LIST": ", ".join(p["paper_id"] for p in batch),
                                    "EXTRA_MATERIALS": extra, "AUTHOR_STATUS_SENTENCE": status},
                         inputs=inputs, outputs=["reviews.json"], schema="reviews.schema.json",
                         meta={"reviewer_id": rid, "paper_ids": [p["paper_id"] for p in batch]}, who=rid)
                self._task_ctx[t.id] = {"type": "review", "items": [self._review_item(p) for p in batch], "assistance": assistance,
                                        "batch_size": len(batch), "reviewer_id": rid}
                tasks.append(t)
        self.run_tasks(tasks, quarter)
        collected: dict[str, list[dict]] = {}
        for t in tasks:
            rid = t.meta["reviewer_id"]
            if self.w.store.status(t) != "done":
                for pid in t.meta["paper_ids"]:  # a failed review call: the paper waits for a fresh attempt next quarter
                    self.w.ledger.add("events", {"quarter": quarter, "event": "review_failed", "paper_id": pid, "reviewer": rid})
                continue
            out = self.w.store.read_output(t, "reviews.json")
            for rev in out["reviews"]:
                rev["reviewer_id"] = rid
                collected.setdefault(rev["paper_id"], []).append(rev)
        for p in under:
            revs = collected.get(p["paper_id"], [])
            rnd = str(p.get("review_round", 1))
            if len(revs) >= max(1, len(p["reviewers"]) - 1):
                p["reviews"][rnd] = revs
                self._ledger_reviews(p, revs, quarter)
            else:
                p["status"] = "review_pending"  # retried next quarter by assign (stays with its reviewers)

    def _prescreen(self, papers: list[dict], quarter: int) -> dict[str, str]:
        out = {}
        tasks = []
        for i in range(0, len(papers), 4):
            batch = papers[i:i + 4]
            inputs = {}
            for p in batch:
                for f, src in self._paper_files(p).items():
                    inputs[f"inputs/{p['paper_id']}/{f}"] = src
                cb = self._codebook_path(p)
                if cb and cb.exists():
                    inputs[f"inputs/{p['paper_id']}/codebook.md"] = cb
            t = Task(type="prescreen", quarter=quarter, identity=f"batch{i // 4}", tier="haiku", run_id=self.w.task_ns, call_type="prescreen",
                     template="prescreen.md", variables={"ITEM_LIST": ", ".join(p["paper_id"] for p in batch)}, inputs=inputs,
                     outputs=["prescreen.json"], schema="prescreen.schema.json", meta={"paper_ids": [p["paper_id"] for p in batch]}, who="prescreen")
            self._task_ctx[t.id] = {"type": "prescreen", "items": [self._review_item(p) for p in batch]}
            tasks.append(t)
        self.run_tasks(tasks, quarter)
        for t in tasks:
            if self.w.store.status(t) == "done":
                for f in self.w.store.read_output(t, "prescreen.json")["flags"]:
                    out[f["paper_id"]] = f["flags"]
        return out

    def _codebook_path(self, p: dict) -> Path | None:
        t0 = self.w.store.task_dir(self._task_from_record(p["task_ids"][0]))
        cb = t0 / "inputs" / p["question_id"] / "codebook.md"
        if cb.exists():
            return cb
        self._ensure_dataset_files(p, t0 / "inputs" / p["question_id"])
        return cb

    def _review_item(self, p: dict) -> dict:
        return {"paper_id": p["paper_id"], "hypotheses": [{k: h.get(k) for k in ("hyp_id", "iv", "dv", "moderator", "estimate", "se", "p", "n", "verdict")} for h in p["hypotheses"]],
                "visible_problems": list(p["flags"]["flags"]), "relation_to_prior": p["relation_to_prior"],
                "n_prior": self.state["counts"].get(p["question_id"], 0)}

    def _ledger_reviews(self, p: dict, revs: list[dict], quarter: int) -> None:
        traps = set(p["hidden"]["traps"])
        for rv in revs:
            kinds = [i["kind"] for i in rv["issues"]]
            self.w.ledger.add("reviews", {"quarter": quarter, "paper_id": p["paper_id"], "round": p.get("review_round", 1), "reviewer_id": rv["reviewer_id"],
                                          "recommendation": rv["recommendation"], "flaw_found": rv["flaw_found"], "validity_probability": rv["validity_probability"],
                                          "relation_to_prior": rv["relation_to_prior"], "confidence": rv["confidence"], "issue_kinds": kinds,
                                          "n_issues_major": sum(1 for i in rv["issues"] if i["severity"] == "major"),
                                          "truth_check_focal": p["hypotheses"][0].get("truth_check"), "analyst_flags": p["flags"]["flags"], "traps": sorted(traps),
                                          "noticed_trap_kind": any(i["kind"] in ("data", "specification", "inference") for i in rv["issues"]) and bool(p["flags"]["flags"])})
            rec = self.state["records"][rv["reviewer_id"]]
            rec["reviews_written"] = rec.get("reviews_written", 0) + 1

    # ======================================================================== phase 6: decisions
    def decide(self, quarter: int) -> None:
        st = self.state
        jc = self.cfg["journal"]
        ready = [p for p in st["papers"].values() if p["status"] == "under_review" and str(p.get("review_round", 1)) in p["reviews"]]
        if not ready:
            return
        if self.w.mode == "fast" or jc["decision_process"] == "reviewer_vote":
            for p in ready:
                rnd = p.get("review_round", 1)
                revs = p["reviews"][str(rnd)]
                if jc["decision_process"] == "reviewer_vote":
                    dec = self._aggregate_votes(p, revs, rnd)
                else:
                    r = self.rng("decide", quarter, p["paper_id"])
                    dec = self.rules["editor"].decide({"paper_id": p["paper_id"], "relation_to_prior": p["relation_to_prior"]}, revs, rnd,
                                                      st["counts"].get(p["question_id"], 0), r)
                self._apply_decision(p, dec, quarter)
            if self.w.mode == "fast" and jc["decision_process"] != "reviewer_vote":
                for i in range(0, len(ready), 8):
                    self.w.budget.debit_call(quarter, "decision", f"fast-decision-Q{quarter}-{i // 8}", "editor")
            return
        tasks = []
        for i in range(0, len(ready), 8):
            batch = ready[i:i + 8]
            inputs = {}
            for p in batch:
                rnd = p.get("review_round", 1)
                inputs[f"inputs/{p['paper_id']}/summary.md"] = texts.submission_summary(
                    p, st["counts"].get(p["question_id"], 0) if jc["show_question_count_to_editor"] else None,
                    None if jc["blinding"] == "triple" else self._status_line(p))
                inputs[f"inputs/{p['paper_id']}/reviews.md"] = texts.reviews_md(p["reviews"][str(rnd)], names=jc["blinding"] in ("open", "signed"))
                pdir = self._paper_dir(p)
                if (pdir / "table.csv").exists():
                    inputs[f"inputs/{p['paper_id']}/table.csv"] = pdir / "table.csv"
                if rnd == 2 and (pdir / "response.md").exists():
                    inputs[f"inputs/{p['paper_id']}/response.md"] = pdir / "response.md"
            round_sentence = " Manuscripts marked as resubmissions have already had one revision; a second revision is not possible, so decide accept or reject." if any(p.get("review_round", 1) == 2 for p in batch) else ""
            t = Task(type="decision", quarter=quarter, identity=f"batch{i // 8}", tier=jc["editor_tier"], run_id=self.w.task_ns, call_type="decision",
                     template="decision.md", variables={"N_ITEMS": len(batch), "ITEM_LIST": ", ".join(f"{p['paper_id']}{' (resubmission)' if p.get('review_round', 1) == 2 else ''}" for p in batch),
                                                        "COUNT_CLAUSE": ", the number of papers already published on the manuscript's question" if jc["show_question_count_to_editor"] else "",
                                                        "AUTHOR_STATUS_SENTENCE": "" if jc["blinding"] == "triple" else " and the authors' status line",
                                                        "ROUND_SENTENCE": round_sentence},
                     inputs=inputs, outputs=["decisions.json"], schema="decisions.schema.json", meta={"paper_ids": [p["paper_id"] for p in batch]}, who="editor")
            self._task_ctx[t.id] = {"type": "decision", "items": [{"paper_id": p["paper_id"], "relation_to_prior": p["relation_to_prior"], "round": p.get("review_round", 1),
                                                                   "reviews": p["reviews"][str(p.get("review_round", 1))], "count": st["counts"].get(p["question_id"], 0)} for p in batch]}
            tasks.append(t)
        self.run_tasks(tasks, quarter)
        for t in tasks:
            if self.w.store.status(t) != "done":
                for pid in t.meta["paper_ids"]:
                    p = st["papers"][pid]
                    self._apply_decision(p, self._aggregate_votes(p, p["reviews"][str(p.get("review_round", 1))], p.get("review_round", 1), fallback=True), quarter)
                continue
            out = self.w.store.read_output(t, "decisions.json")
            for dec in out["decisions"]:
                if dec["paper_id"] in st["papers"]:
                    self._apply_decision(st["papers"][dec["paper_id"]], dec, quarter)

    def _aggregate_votes(self, p: dict, revs: list[dict], rnd: int, fallback: bool = False) -> dict:
        """Rule-based ablation / reviewer vote: any reject -> reject; >=2 major -> revise; otherwise accept."""
        recs = [r["recommendation"] for r in revs]
        if "reject" in recs:
            d = "reject"
        elif recs.count("major_revision") >= 2:
            d = "revise" if rnd == 1 else "reject"
        else:
            d = "accept"
        merit = float(np.mean([r["validity_probability"] for r in revs])) if revs else 50.0
        return {"paper_id": p["paper_id"], "decision": d, "merit": merit, "required_changes": [], "letter": "Decision by aggregation rule." + (" (editor call failed)" if fallback else "")}

    def _apply_decision(self, p: dict, dec: dict, quarter: int) -> None:
        rnd = p.get("review_round", 1)
        d = dec["decision"]
        if rnd == 2 and d == "revise":
            d = "reject"
        p["decisions"][str(rnd)] = {"decision": d, "merit": dec.get("merit"), "letter": dec.get("letter", ""), "required_changes": dec.get("required_changes", []), "quarter": quarter}
        p["merit"] = dec.get("merit")
        revs = p["reviews"][str(rnd)]
        for rv in revs:
            rec = self.state["records"][rv["reviewer_id"]]
            match = (rv["recommendation"] in ("accept", "minor_revision") and d == "accept") or (rv["recommendation"] == "reject" and d == "reject") or \
                    (rv["recommendation"] == "major_revision" and d == "revise")
            if match:
                rec["reviews_matched"] = rec.get("reviews_matched", 0) + 1
        self.w.ledger.add("decisions", {"quarter": quarter, "paper_id": p["paper_id"], "stage": f"round{rnd}", "decision": d, "merit": dec.get("merit"),
                                        "question_id": p["question_id"], "truth_check_focal": p["hypotheses"][0].get("truth_check"),
                                        "recommendations": [r["recommendation"] for r in revs], "mean_validity": float(np.mean([r["validity_probability"] for r in revs])) if revs else None,
                                        "relation_to_prior": p["relation_to_prior"], "count_on_question": self.state["counts"].get(p["question_id"], 0)})
        gist = "; ".join(i["description"][:60] for rv in revs for i in rv["issues"] if i["severity"] == "major")[:160]
        p.setdefault("quarter_first_decision", quarter)
        if d in ("accept", "reject"):
            p["quarter_final_decision"] = quarter
        if d == "accept":
            p["status"] = "accepted"
            p["quarter_accepted"] = quarter
            p["publish_due"] = quarter + int((self.cfg["journal"].get("timing") or {}).get("accept_to_publish_quarters", 0))
            self._record_status(p, "accepted", gist)
        elif d == "revise":
            p["status"] = "revise"
            p["revision_due"] = quarter + self.latency("revision", p["paper_id"])   # the author's turnaround
            self._record_status(p, "revise and resubmit", gist)
            self.state["records"][p["lead"]]["criticisms"].append(gist[:80] or "revise requested")
        else:
            p["status"] = "rejected"
            self._free_slot(p)
            self._ledger_hypotheses(p, certified=False, quarter=quarter)
            self._record_status(p, "rejected", gist)

    def _record_status(self, p: dict, status: str, gist: str) -> None:
        for e in self.state["records"][p["lead"]]["papers"]:
            if e["paper_id"] == p["paper_id"]:
                e["status"], e["gist"] = status, gist[:80]

    # ======================================================================== phase 7: publish and update
    def publish(self, quarter: int) -> None:
        st = self.state
        for p in list(st["papers"].values()):
            if p.get("legacy"):
                continue
            if p["status"] == "accepted" and p.get("publish_due", quarter) <= quarter:
                p["status"], p["quarter_published"] = "published", quarter
                if quarter <= 0:   # published in the pre-history: part of the starting literature, read as earlier work (like the legacy library)
                    p["legacy"] = True
                st["literature"].setdefault(p["question_id"], []).append(p["paper_id"])
                st["counts"][p["question_id"]] = st["counts"].get(p["question_id"], 0) + 1
                for b in p.get("builds_on", []):
                    cited = st["papers"].get(b.get("paper_id"))
                    if cited and cited["status"] == "published":
                        cited["citations"] += 1
                        cited["cited_by"].append(p["paper_id"])
                        for a in cited["authors"]:
                            if a in st["records"]:
                                st["records"][a]["citations"] = st["records"][a].get("citations", 0) + 1
                        self.w.ledger.add("events", {"quarter": quarter, "event": "citation", "from": p["paper_id"], "to": cited["paper_id"],
                                                     "to_truth_focal": cited["hypotheses"][0].get("truth_check"), "informed": b.get("informed", [])})
                for a in p["authors"]:
                    st["records"][a]["papers_published"].append(p["paper_id"])
                self._record_status(p, f"published Q{quarter}", "")
                self._free_slot(p)
                self._ledger_hypotheses(p, certified=True, quarter=quarter)
                self.w.ledger.add("events", {"quarter": quarter, "event": "published", "paper_id": p["paper_id"], "question_id": p["question_id"],
                                             "truth_check_focal": p["hypotheses"][0].get("truth_check"), "relation_to_prior": p["relation_to_prior"]})
            elif p["status"] == "revise" and not p.get("revise_slot_set"):
                # once per revise decision: the slot turns to revising and the author decides whether to revise at all
                # (a paper now waits `revision_due` quarters in this status, so the lottery must not be redrawn each quarter)
                p["revise_slot_set"] = True
                slot = self._slot_of(p)
                if slot:
                    slot["state"] = "revising"
                r = self.rng("revise_choice", quarter, p["paper_id"])
                if r.uniform() > self.params["author"]["p_revise_when_asked"]:
                    self._abandon(p, quarter, "author_did_not_revise")
            elif p["status"] in ("review_pending",):
                p["status"] = "under_review"
        self.check_promotion(quarter)
        if self.w.mode != "fast" or quarter % 4 == 0:
            self._write_papers_snapshot()

    def _slot_of(self, p: dict) -> dict | None:
        for slot in self.state["authors"].get(p["lead"], {}).get("slots", []):
            if slot.get("paper_id") == p["paper_id"]:
                return slot
        return None

    def _free_slot(self, p: dict) -> None:
        slot = self._slot_of(p)
        if slot:
            slot.update({"state": "free", "paper_id": None})

    def _ledger_hypotheses(self, p: dict, certified: bool, quarter: int) -> None:
        for h in p["hypotheses"]:
            self.w.ledger.add("hypotheses", {"unit_id": f"{p['paper_id']}:{h['hyp_id']}", "paper_id": p["paper_id"], "hyp_id": h["hyp_id"],
                                             "question_id": p["question_id"], "core": p.get("core", True), "quarter": quarter,
                                             "author_id": p["lead"], "authors": p["authors"], "style": p["style"], "throughput": p["throughput"], "tier": p["tier"],
                                             "iv": h["iv"], "dv": h["dv"], "moderator": h.get("moderator"), "kind": h.get("kind"),
                                             "predicted_sign": h["predicted_sign"], "estimate": h.get("estimate"), "se": h.get("se"), "p": h.get("p"),
                                             "n": h.get("n"), "n_correct": p.get("n_correct"), "estimator": h.get("estimator"), "se_type": h.get("se_type"),
                                             "controls": h.get("controls"), "verdict": h["verdict"], "partial_r": h.get("partial_r"),
                                             "relation_to_prior": p["relation_to_prior"], "theta": h.get("theta"), "theta_fe": p["hidden"].get("theta_fe"),
                                             "theta_naive": p["hidden"].get("theta_naive"), "truth_class": h.get("truth_class"), "truth_check": h.get("truth_check"),
                                             "truth_check_lenient": h.get("truth_check_lenient"), "magnitude_ratio": h.get("magnitude_ratio"), "ci_covers": h.get("ci_covers"),
                                             "traps": p["hidden"].get("traps"), "flags": (p.get("flags") or {}).get("flags", []), "certified": certified,
                                             "certified_quarter": quarter if certified else None, "status": p["status"],
                                             # rows settled in the pre-history (quarter <= 0) are earlier work for the audit; a paper researched
                                             # in the pre-history but decided by the live journal keeps prehistory=True for the realism checks
                                             "legacy": bool(p.get("legacy", False) or quarter <= 0), "prehistory": bool(p.get("prehistory", False)),
                                             "dataset_id": p.get("dataset_id"), "menu_slot": p.get("menu_slot")}
                               | ({"call_group": p["call_group"], "call_group_size": p["call_group_size"]} if p.get("call_group") else {}))

    def _write_papers_snapshot(self) -> None:
        rows = []
        for p in self.state["papers"].values():
            rows.append({k: p.get(k) for k in ("paper_id", "question_id", "core", "legacy", "prehistory", "lead", "authors", "style", "throughput", "tier", "status",
                                                "quarter_started", "quarter_submitted", "quarter_first_decision", "quarter_final_decision", "quarter_accepted", "quarter_published",
                                                "review_due", "revision_due", "citations", "cited_by", "relation_to_prior",
                                                "merit", "n", "n_correct", "menu_slot", "self_cites", "n_models_run", "robustness_models", "dataset_id")}
                        | {"truth_check_focal": p["hypotheses"][0].get("truth_check"), "flags": (p.get("flags") or {}).get("flags", []),
                           "traps": p["hidden"].get("traps"), "reproducible": (p.get("forensic") or {}).get("reproducible"),
                           "n_hypotheses": len(p["hypotheses"]), "n_supported": sum(1 for h in p["hypotheses"] if h["verdict"] == "supported"),
                           "desk": (p.get("desk") or {}).get("decision"), "decisions": {k: v["decision"] for k, v in p.get("decisions", {}).items()}}
                        | ({"call_group": p["call_group"], "call_group_size": p["call_group_size"]} if p.get("call_group") else {}))
        write_json(self.w.run_dir / "ledger" / "papers_snapshot.json", rows)

    # ======================================================================== phases 8-9
    def score(self, quarter: int) -> None:
        roll = self.rollup(quarter)
        st = self.state
        pubs = [p for p in st["papers"].values() if not p.get("legacy") and p.get("quarter_published") == quarter]
        decided = [p for p in st["papers"].values() if not p.get("legacy") and any(d["quarter"] == quarter for d in p["decisions"].values())]
        subs = [p for p in st["papers"].values() if not p.get("legacy") and p.get("quarter_submitted") == quarter]
        desk_rej = sum(1 for p in subs if (p.get("desk") or {}).get("decision") == "desk_reject")
        true_pub = sum(1 for p in pubs if p["hypotheses"][0].get("truth_check") in ("true_discovery", "correct_null", "true_discovery_negligible", "correct_null_negligible"))
        self.w.ledger.add("events", {"quarter": quarter, "event": "field_metrics", "submissions": len(subs), "desk_rejected": desk_rej,
                                     "decided": len(decided), "published": len(pubs), "published_true_focal": true_pub,
                                     "accept_rate": (len(pubs) / len(decided)) if decided else None,
                                     "core_share": (sum(1 for p in pubs if p["core"]) / len(pubs)) if pubs else None,
                                     "n_covered_core": sum(1 for qid, v in roll["questions"].items() if v["core"] and v["n_run_units"] > 0),
                                     "net_knowledge_B2": sum(1 if v["scored"]["B2"]["correct"] else (-1 if v["scored"]["B2"]["correct"] is False else 0)
                                                             for v in roll["questions"].values() if v["core"] and v["scored"]["B2"]["confidence"] >= 0.7)})

    def handbook(self, quarter: int) -> None:
        self.handbook_snapshot(quarter)

    def finish(self) -> None:
        # anything still in the pipeline at the horizon is recorded as uncertified (file drawer by horizon)
        for p in list(self.state["papers"].values()):
            if not p.get("legacy") and p["status"] in ("submitted", "waiting", "sent_out", "under_review", "revise", "resubmitted", "review_pending", "accepted"):
                self.w.ledger.add("events", {"quarter": self.H, "event": "unfinished_at_horizon", "paper_id": p["paper_id"], "status": p["status"]})
                p["status"] = "unfinished"
                self._ledger_hypotheses(p, certified=False, quarter=self.H)
        self._write_papers_snapshot()
        write_json(self.w.run_dir / "ledger" / "citations.json", {pid: p["citations"] for pid, p in self.state["papers"].items()})


def legacy_theta(lp: dict, h: dict, q: dict) -> float:
    """Truth for a legacy hypothesis (the same rule as the repository regime's legacy claims): the question's theta for the
    focal relation, 0 for a moderation, the secondary outcome's theta when the library recorded one, else 0."""
    if h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator"):
        return float(lp["hidden"]["theta"])
    if h.get("moderator"):
        return 0.0
    return float(lp["hidden"].get("theta_secondary", 0.0)) if h["dv"] == q.get("secondary_dv_var") else 0.0


def _title(text: str | None) -> str:
    """First non-empty line, minus heading marks; a bare placeholder heading ('# Title') is skipped for the next line
    (smoke test: an author wrote '# Title' above the real title)."""
    lines = [ln.lstrip("# ").strip() for ln in (text or "").splitlines() if ln.strip()]
    while lines and lines[0].lower().rstrip(":") in ("title", "paper title"):
        lines.pop(0)
    return lines[0] if lines else ""


def _section(text: str | None, name: str) -> str:
    if not text:
        return ""
    import re
    m = re.search(rf"^#+\s*{name}\s*$(.*?)(?=^#+\s|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def _flag_trap(flag: str) -> str | None:
    from ..agents.rules import _flag_to_trap
    return _flag_to_trap(flag)
