"""The repository regime: labs, shared data repositories, claims, four certification variants, reputation,
under the data and compute budget paired with the journal world."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..orchestration.tasks import PROMPTS, Task
from ..scorer import analyst_flags, claim_sign as _claim_sign, forensic_rerun, relation_theta, score_hypothesis, verification_power
from ..stats import partial_r
from ..traps import draw_traps
from ..util import read_text, seed_from, short_id, write_json, write_text
from . import texts
from .base import RegimeBase

PROTOCOL = {"VERIFIED": "_protocol_verified.md", "AUTO": "_protocol_auto.md"}
RANK = {"doctoral": 0, "assistant": 1, "associate": 2, "full": 3}


class RepositoryRegime(RegimeBase):
    PHASES = ["repo_update", "labdecide", "budget_admit", "draws", "analyse", "verify", "tests", "certify", "registry", "score", "handbook", "annual"]
    REGIME = "repository"

    # ======================================================================== state
    def init_state(self) -> None:
        super().init_state()
        st = self.state
        rc = self.cfg["repository"]
        st.update({"labs": {}, "repos": {}, "claims": {}, "drafts": {}, "analyses": {}, "claim_counter": 0, "deposit_counter": 0,
                   "analysis_counter": 0, "draft_counter": 0, "votes_this_quarter": [], "offers": [], "tests": {}, "agenda_history": {},
                   "admitted": {"analyses": [], "verifications": [], "tests": []}, "member_lab": {}})
        for lab in self.w.labs:
            st["labs"][lab["id"]] = {"id": lab["id"], "head": lab["head"], "members": list(lab["members"]), "primary_theory": lab["primary_theory"],
                                     "secondary_theory": lab["secondary_theory"], "points": float(rc["stakes"]["endowment"]), "reputation": 0.0,
                                     "record": {"claims": 0, "endorsed": 0, "contested": 0, "retracted": 0, "lapsed": 0, "verifications": 0, "deposits": 0,
                                                "challenges_won": 0, "challenges_lost": 0, "endorsed_by": [], "deferred": 0, "analyses": 0},
                                     "size": len(lab["members"])}
            for m in lab["members"]:
                st["member_lab"][m] = lab["id"]
        for q in self.w.landscape.questions:
            st["repos"][q["id"]] = {"deposits": {}}
        # legacy claims: endorsed (earlier work), one per legacy hypothesis, no deposit
        for lp in self.w.legacy:
            q = self.w.landscape.question(lp["question_id"])
            for h in lp["hypotheses"]:
                cid = f"LC{st['claim_counter'] + 1:04d}"
                st["claim_counter"] += 1
                sign = _claim_sign(h)   # direction from the estimate and p, not from support of the prediction (a significant wrong-sign result is not "no relation")
                st["claims"][cid] = {"claim_id": cid, "question_id": q["id"], "core": q["core"], "legacy": True, "iv": h["iv"], "dv": h["dv"],
                                     "moderator": h.get("moderator"), "kind": h.get("kind", "focal"), "sign": sign,
                                     "statement": f"{h['iv']} has {'no reliable' if sign == 'none' else 'a ' + sign} relation with {h['dv']}" + (f" conditional on {h['moderator']}" if h.get("moderator") else ""),
                                     "status": "endorsed_legacy", "quarter_filed": 0, "quarter_endorsed": 0, "proposer_lab": "earlier", "leads": lp["authors"],
                                     "evidence": [{"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"], "estimator": lp["estimator"], "partial_r": h.get("partial_r")}],
                                     "deposit_ids": [], "endorsements": [], "challenges": [], "verifications": [], "tests": [], "stakes": [], "links": [],
                                     "status_history": [{"quarter": 0, "status": "endorsed_legacy", "event": "legacy"}], "confidence_stated": 60.0,
                                     "confidence_certified": 0.6, "note": "", "relation_to_prior": "first claim", "route": "legacy",
                                     "hidden": {"theta": lp["hidden"]["theta"], "theta_fe": lp["hidden"]["theta_fe"], "theta_naive": lp["hidden"]["theta_naive"],
                                                "traps": lp["hidden"]["traps"], "truth_check": _truth_check_claim(sign, h["estimate"], _theta_for_legacy(lp, h, q))}}
                if h.get("kind", "focal") == "focal" or (h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator")):
                    st["counts"][q["id"]] = st["counts"].get(q["id"], 0) + 1

    def before_quarter(self, quarter: int) -> None:
        # budget pairing: this quarter's compute allotment is the paired paper run's realized spend in the same quarter, so the paper run
        # must have closed it (an extended paper run that is still mid-quarter would hand over a partial spend)
        self.w.budget.require_paired(quarter)

    def lab_of(self, member: str) -> str | None:
        return self.state["member_lab"].get(member)

    def lab_members(self, lab_id: str) -> list[dict]:
        return [self.scientist(m) for m in self.state["labs"][lab_id]["members"] if self.scientist(m).get("active", True)]

    # ======================================================================== views
    def catalogue(self, qid: str, quarter: int, lab_id: str | None = None) -> list[dict]:
        rc = self.cfg["repository"]
        out = []
        for d in self.state["repos"].get(qid, {"deposits": {}})["deposits"].values():
            avail = quarter >= d["available_from"]
            if rc["access_rule"] == "embargo" and d["lab"] != lab_id:
                avail = quarter >= d["quarter"] + int(rc["embargo_quarters"]) + 1
            if rc["access_rule"] == "lab_restricted" and d["lab"] != lab_id and lab_id not in d.get("shared_with", []):
                continue
            out.append({"deposit_id": d["deposit_id"], "version": d["version"], "n": d["n"], "unit": d["unit"], "lab": d["lab"], "quarter": d["quarter"],
                        "n_analyses": d["n_analyses"], "n_claims": len(d["claims"]), "available": avail})
        return sorted(out, key=lambda d: (-d["quarter"], d["deposit_id"]))

    def claims_on(self, qid: str) -> list[dict]:
        return [c for c in self.state["claims"].values() if c["question_id"] == qid]

    def registry_view(self, qid: str, quarter: int, lab_id: str | None) -> str:
        rc = self.cfg["repository"]
        q = self.w.landscape.question(qid)
        labs = {lid: {"size": L["size"], "reputation": L["points"] if rc["variant"] == "d" else L["reputation"], "endorsed_claims": L["record"]["endorsed"]}
                for lid, L in self.state["labs"].items()}
        claims = [self._public_claim(c, quarter) for c in self.claims_on(qid)]
        return texts.registry_view_md(q, claims, self.catalogue(qid, quarter, lab_id), int(self.cfg["long_horizon"]["memory"]["registry_claims"]),
                                      hidden_tally=rc["hidden_tally"], blinded=rc["blinded_registry"], labs=labs, handbook_entry=self.latest_handbook_entry(qid))

    def _public_claim(self, c: dict, quarter: int) -> dict:
        """What agents may see: tallies from previous quarters only (votes are simultaneous within a quarter)."""
        pc = {k: v for k, v in c.items() if k != "hidden"}
        pc["endorsements"] = [e for e in c["endorsements"] if e["quarter"] < quarter]
        pc["challenges"] = [e for e in c["challenges"] if e["quarter"] < quarter]
        return pc

    def certified_units(self, qid: str) -> list[dict]:
        q = self.w.landscape.question(qid)
        units = []
        for c in self.claims_on(qid):
            if c["status"] not in ("endorsed", "endorsed_legacy"):
                continue
            if not (c["iv"] == q["iv_var"] and c["dv"] == q["dv_var"] and not c.get("moderator")):
                continue
            ev = c["evidence"][0] if c["evidence"] else {}
            pr = ev.get("partial_r")
            if pr is None and ev.get("se"):
                pr = partial_r(ev["estimate"] / ev["se"], max((ev.get("n") or 10) - 10, 1))
            attention = len(c["endorsements"]) + (sum(s["amount"] for s in c.get("stakes", [])) if self.cfg["repository"]["variant"] == "d" else 0)
            units.append({"unit_id": c["claim_id"], "estimate": pr, "se": ev.get("se"), "n": ev.get("n"), "verdict": "supported" if c["sign"] != "none" else "not_supported",
                          "attention": attention, "quarter": c.get("quarter_endorsed") or c["quarter_filed"], "legacy": c.get("legacy", False),
                          "truth_check": c["hidden"].get("truth_check"), "deposit_id": (c["deposit_ids"] or [c["claim_id"]])[0], "route": c.get("route")})
        return units

    # ======================================================================== phase 1: repository update
    def repo_update(self, quarter: int) -> None:
        for repo in self.state["repos"].values():
            for d in repo["deposits"].values():
                d.setdefault("available_from", d["quarter"] + 1)
        self.state["votes_this_quarter"] = []
        self.state["offers"] = []
        self.state["admitted"] = {"analyses": [], "verifications": [], "tests": []}

    # ======================================================================== phase 2: lab decisions
    def _capacity(self, lab: dict, quarter: int) -> dict:
        avail = []
        for m in self.lab_members(lab["id"]):
            for si, slot in enumerate(self.state["authors"][m["id"]]["slots"]):
                if slot["state"] == "free" and self.starts_allowed(quarter) and self.available(m["id"], quarter, si):
                    avail.append((m["id"], si, m["style"], m["throughput"]))
        ver = sum(int(self.cfg["repository"]["verifications_per_member"]) for m in self.lab_members(lab["id"]) if m["stage"] != "doctoral")
        return {"available": avail, "verifications": ver}

    def _in_view(self, c: dict, lab: dict, menu_ids: set[str], quarter: int) -> bool:
        if self.cfg["repository"]["view_rule"] == "all":
            return True
        hist = set(self.state["agenda_history"].get(lab["id"], {}).get(str(quarter - 1), [])) | set(self.state["agenda_history"].get(lab["id"], {}).get(str(quarter - 2), []))
        fam = self.w.landscape.question(c["question_id"])["family"]
        return c["question_id"] in menu_ids or c["question_id"] in hist or fam == lab["primary_theory"]

    def _open_claims_for(self, lab: dict, menu_ids: set[str], quarter: int) -> list[dict]:
        cap = int(self.cfg["repository"]["view_caps"]["open_claims"])
        cands = [c for c in self.state["claims"].values() if c["status"] in ("proposed", "contested") and c["proposer_lab"] != lab["id"] and not c.get("legacy")
                 and c["quarter_filed"] <= quarter and self._in_view(c, lab, menu_ids, quarter)]
        own_q = set(self.state["agenda_history"].get(lab["id"], {}).get(str(quarter - 1), []))
        cands.sort(key=lambda c: (0 if c["question_id"] in own_q else (1 if self.w.landscape.question(c["question_id"])["family"] == lab["primary_theory"] else 2),
                                  -c["quarter_filed"], c["claim_id"]))
        return cands[:cap]

    def _lab_ctx(self, lab: dict, quarter: int) -> dict:
        rc = self.cfg["repository"]
        cap = self._capacity(lab, quarter)
        entries = self.menu_entries(lab["primary_theory"], quarter, lab["id"])
        for e in entries:
            e["catalogue"] = self.catalogue(e["question_id"], quarter, lab["id"])
        menu_ids = {e["question_id"] for e in entries}
        drafts = [d for d in self.state["drafts"].values() if d["lab"] == lab["id"] and d["status"] == "ready"][:int(rc["view_caps"]["drafts"])]
        open_claims = self._open_claims_for(lab, menu_ids, quarter)
        challenges = [c for c in self.state["claims"].values() if c["proposer_lab"] == lab["id"] and c["status"] == "contested"
                      and any(ch["quarter"] == quarter - 1 and not ch.get("responded") for ch in c["challenges"])][:int(rc["view_caps"]["challenges"])]
        endorsed_us = set(lab["record"].get("endorsed_by", [])[-10:])
        oc = []
        for c in open_claims:
            ev = c["evidence"][0] if c["evidence"] else {}
            pc = self._public_claim(c, quarter)
            oc.append({"claim_id": c["claim_id"], "question_id": c["question_id"], "p": ev.get("p"), "n": ev.get("n"), "significant": (ev.get("p") or 1) < 0.05,
                       "endorsements": len(pc["endorsements"]), "challenges": len(pc["challenges"]), "proposer_lab": c["proposer_lab"],
                       "proposer_size": self.state["labs"].get(c["proposer_lab"], {}).get("size", 0),
                       "same_theory": self.w.landscape.question(c["question_id"])["family"] == lab["primary_theory"],
                       "reciprocity": c["proposer_lab"] in endorsed_us,
                       "verifiable": len([v for v in c["verifications"]]) < int(rc["reproductions_k"]) and rc["variant"] in ("b", "d"),
                       "test": next((t for t in c["tests"] if t.get("status") == "proposed"), None)})
        return {"lab": lab, "members_available": [(m, s, st, tp) for m, s, st, tp in cap["available"]], "verification_capacity": cap["verifications"],
                "menu": entries, "drafts": [{"draft_id": d["draft_id"], "entries": [{"entry_id": e["entry_id"], "p": e.get("p"), "estimate": e.get("estimate"), "kind": e.get("kind")} for e in d["entries"]]} for d in drafts],
                "open_claims": oc, "challenges": [{"claim_id": c["claim_id"], "kind": c["challenges"][-1]["kind"], "test": next((t for t in c["tests"] if t.get("status") == "proposed"), None)} for c in challenges],
                "variant": rc["variant"], "pooling": rc["pooling"]["enabled"], "autonomous_filing": rc["autonomous_filing"],
                "remaining_budget": self.w.budget.remaining(quarter), "author_params": self.params["author"], "menu_ids": sorted(menu_ids)}

    def labdecide(self, quarter: int) -> None:
        tasks, ctxs = [], {}
        for lid, lab in sorted(self.state["labs"].items()):
            if not self.lab_members(lid):
                continue
            ctx = self._lab_ctx(lab, quarter)
            ctxs[lid] = ctx
            if self.cfg["repository"].get("skip_idle_lab_calls", False) and not ctx["members_available"] and not ctx["drafts"] \
                    and not ctx["open_claims"] and not ctx["challenges"]:
                # nothing to decide this quarter: no call, no debit (knob; default off keeps one call per lab per quarter)
                self.w.ledger.add("events", {"quarter": quarter, "event": "lab_idle", "lab": lid})
                self.state["agenda_history"].setdefault(lid, {})[str(quarter)] = []
                continue
            if self.w.mode == "fast":
                r = self.rng("lab", quarter, lid)
                dec = self.rules["lab"].decide(ctx | {"nonce": "fast"}, r)
                self.w.budget.debit_call(quarter, "lab_decision", f"lab-{lid}-Q{quarter}", lid)
                self._ingest_lab_decision(lab, ctx, dec, quarter)
                continue
            tasks.append(self._lab_task(lab, ctx, quarter))
        if self.w.mode != "fast":
            self.run_tasks(tasks, quarter)
            for t in tasks:
                lab = self.state["labs"][t.meta["lab_id"]]
                if self.w.store.status(t) != "done":
                    self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": lab["id"]})
                    continue
                self._ingest_lab_decision(lab, ctxs[lab["id"]], self.w.store.read_output(t, "decisions.json"), quarter)

    def _lab_task(self, lab: dict, ctx: dict, quarter: int) -> Task:
        rc = self.cfg["repository"]
        members = self.lab_members(lab["id"])
        cap = {"slots": len(ctx["members_available"]), "available_members": [m for m, *_ in ctx["members_available"]],
               "verifications": ctx["verification_capacity"], "data_budget": int(max(0, ctx["remaining_budget"]["data"]))}
        inputs = {"lab.md": texts.lab_card(lab, members, lab["record"] | {"points": lab["points"]}, cap)}
        for d in ctx["drafts"]:
            dd = self.state["drafts"][d["draft_id"]]
            inputs[f"drafts/{d['draft_id']}/claims_draft.json"] = json.dumps(dd["public_draft"], indent=2)
            inputs[f"drafts/{d['draft_id']}/note.md"] = dd["note"]
        for e in ctx["menu"]:
            inputs[f"registry/{e['question_id']}.md"] = self.registry_view(e["question_id"], quarter, lab["id"])
        oc_lines = ["# Open claims you may endorse, challenge or offer to verify", ""]
        labs_pub = {lid: {"size": L["size"], "reputation": L["points"] if rc["variant"] == "d" else L["reputation"], "endorsed_claims": L["record"]["endorsed"]} for lid, L in self.state["labs"].items()}
        for c in ctx["open_claims"]:
            cl = self.state["claims"][c["claim_id"]]
            oc_lines.append(f"- {texts.claim_entry(self._public_claim(cl, quarter), rc['hidden_tally'], rc['blinded_registry'], labs_pub)}")
            an = self.state["analyses"].get(cl.get("analysis_id"))
            if an and an.get("code"):
                inputs[f"open_claims/{c['claim_id']}/analysis.py"] = an["code"]
            oc_lines.append(f"  Code: `open_claims\\{c['claim_id']}\\analysis.py`." if an and an.get("code") else "  Code: not available.")
        inputs["open_claims.md"] = "\n".join(oc_lines) + "\n"
        ch_lines = ["# Challenges to your claims", ""]
        for c in ctx["challenges"]:
            cl = self.state["claims"][c["claim_id"]]
            last = cl["challenges"][-1]
            ch_lines.append(f"- {c['claim_id']} ({cl['statement']}): objection of kind {last['kind']} by group {last['lab']}: {last.get('text', '')}"
                            + (f" Proposed decisive test: {json.dumps(c['test'])}" if c.get("test") else ""))
        if not ctx["challenges"]:
            ch_lines.append("None this quarter.")
        inputs["challenges.md"] = "\n".join(ch_lines) + "\n"
        variant_block = ""
        if rc["variant"] == "c":
            variant_block = read_text(PROMPTS / "_variant_c.md")
        elif rc["variant"] == "d":
            variant_block = read_text(PROMPTS / "_variant_d.md")
        t = Task(type="lab_decision", quarter=quarter, identity=lab["id"], tier=self.scientist(lab["head"]).get("tier", "sonnet"), run_id=self.w.task_ns,
                 call_type="lab_decision", template="lab_decision.md", variables={"VARIANT_BLOCK": variant_block}, inputs=inputs,
                 outputs=["decisions.json"], schema="lab_decision.schema.json", meta={"lab_id": lab["id"]}, who=lab["id"])
        self._task_ctx[t.id] = {"type": "lab_decision", "ctx": {k: v for k, v in ctx.items() if k != "lab"} | {"lab": {"id": lab["id"], "primary_theory": lab["primary_theory"]}}}
        return t

    def _ingest_lab_decision(self, lab: dict, ctx: dict, dec: dict, quarter: int) -> None:
        st = self.state
        rc = self.cfg["repository"]
        lid = lab["id"]
        avail = {m: (s, sty, tp) for m, s, sty, tp in ctx["members_available"]}
        agenda_q = []
        for a in dec.get("agenda", [])[:len(avail)]:
            lead = a.get("lead") if a.get("lead") in avail else (next(iter(avail)) if avail else None)
            if lead is None or a["question_id"] not in {e["question_id"] for e in ctx["menu"]}:
                continue
            slot, style, tp = avail.pop(lead)
            data = a["data"]
            deposits = [] if data == "new" else ([data] if isinstance(data, str) else list(data))
            # the registry lists deposits as "D0001 v1"; heads sometimes write "D0001:v1" or "D0001 v1":
            # match on the deposit id, or a reuse request silently became a fresh (charged) draw
            deposits = [re.split(r"[:\s]", str(d).strip())[0] for d in deposits]
            cat = {d["deposit_id"]: d for e in ctx["menu"] if e["question_id"] == a["question_id"] for d in e.get("catalogue", [])}
            deposits = [d for d in deposits if d in cat and cat[d]["available"]]
            if not rc["pooling"]["enabled"]:
                deposits = deposits[:1]
            deposits = deposits[:int(rc["pooling"]["max_deposits"])]
            fresh = len(deposits) == 0
            st["analysis_counter"] += 1
            aid = f"AN{st['analysis_counter']:04d}"
            q = self.w.landscape.question(a["question_id"])
            n_est, n_requested, n_override = None, None, None
            ds_seed = seed_from(self.w.seeds["run"], "deposit", quarter, lid, aid)
            if fresh:
                from ..datasets import draw_frame
                from ..config import rng as _rng
                fr = draw_frame(q, _rng(ds_seed, "frame"), self.cfg["datasets"]["n_cap"], upper_half=tp == "T1")
                n_est = fr["n_firms"] * fr["n_years"] if fr["panel"] else fr["n_target"]
                # the head may name the dataset size it wants: honoured within the data budget of the quarter
                # (the remaining allotment at request time; the quarter's requests are admitted later in priority order)
                # and never above the largest sample the field has seen (datasets.n_cap)
                req_n = a.get("n")
                if isinstance(req_n, (int, float)) and not isinstance(req_n, bool) and int(req_n) >= 60:
                    n_requested = int(req_n)
                    remaining = float(ctx["remaining_budget"]["data"])
                    granted = min(n_requested, int(self.cfg["datasets"]["n_cap"]))
                    if granted > remaining >= 60:
                        granted = int(remaining)
                    n_override = granted
                    n_est = granted
                    self.w.ledger.add("events", {"quarter": quarter, "event": "dataset_size_request", "lab": lid, "analysis_id": aid,
                                                 "question_id": q["id"], "n_requested": n_requested, "n_granted": granted,
                                                 "frame_draw": fr["n_firms"] * fr["n_years"] if fr["panel"] else fr["n_target"],
                                                 "reason": "as requested" if granted == n_requested else ("capped at n_cap" if granted == int(self.cfg["datasets"]["n_cap"]) else "clamped to the remaining data allotment")})
            st["analyses"][aid] = {"analysis_id": aid, "lab": lid, "lead": lead, "slot": slot, "question_id": a["question_id"], "quarter": quarter,
                                   "fresh": fresh, "deposits": deposits, "brief": a.get("brief", ""), "status": "requested", "style": style, "throughput": tp,
                                   "n_est": n_est, "n_requested": n_requested, "n_override": n_override, "lab_size": lab["size"], "ds_seed": ds_seed}
            agenda_q.append(a["question_id"])
            st["authors"][lead]["slots"][slot].update({"state": "busy", "paper_id": aid})
        st["agenda_history"].setdefault(lid, {})[str(quarter)] = agenda_q
        # filings
        for f in dec.get("filings", []):
            d = st["drafts"].get(f["draft_id"])
            if not d or d["lab"] != lid or d["status"] != "ready":
                continue
            e = next((x for x in d["entries"] if x["entry_id"] == f["entry"]), None)
            if e is None or e.get("filed"):
                continue
            if f["action"] == "file" or rc["autonomous_filing"]:
                self._file_claim(lab, d, e, float(f.get("probability", 50)), quarter)
            else:
                e["filed"] = "dropped"
                self.w.ledger.add("events", {"quarter": quarter, "event": "draft_dropped", "draft_id": d["draft_id"], "entry": e["entry_id"], "lab": lid,
                                             "truth_check": e.get("truth_check")})
        for d in st["drafts"].values():
            if d["lab"] == lid and d["status"] == "ready":
                d["status"] = "decided"
        # votes (recorded now, applied at certify; tallies visible next quarter)
        in_view = {c["claim_id"] for c in ctx["open_claims"]}
        analysed_qs = {an["question_id"] for an in st["analyses"].values() if an["lab"] == lid}
        n_stakes = 0
        for v in dec.get("votes", []):
            if v["claim_id"] not in in_view:
                continue
            c = st["claims"][v["claim_id"]]
            prior_e = len([e for e in c["endorsements"] if e["quarter"] < quarter])
            rec = {"quarter": quarter, "claim_id": c["claim_id"], "voter_lab": lid, "proposer_lab": c["proposer_lab"], "vote": v["vote"],
                   "probability": float(v.get("probability", 50)), "reason": v.get("reason", ""), "objection_kind": v.get("objection_kind"),
                   "prior_endorsements": prior_e, "prior_challenges": len([e for e in c["challenges"] if e["quarter"] < quarter]),
                   "voter_analysed_question": c["question_id"] in analysed_qs, "voter_theory": lab["primary_theory"],
                   "same_theory": self.w.landscape.question(c["question_id"])["family"] == lab["primary_theory"],
                   "reciprocity": c["proposer_lab"] in set(lab["record"].get("endorsed_by", [])[-10:]),
                   "proposer_size": st["labs"].get(c["proposer_lab"], {}).get("size", 0), "significant": (c["evidence"][0].get("p") or 1) < 0.05 if c["evidence"] else False,
                   "question_id": c["question_id"], "claim_true": c["hidden"].get("truth_check") in ("true_discovery", "correct_null", "true_discovery_negligible", "correct_null_negligible"),
                   "test": v.get("test") if rc["variant"] == "c" else None}
            if rc["variant"] == "d" and v.get("stake") and n_stakes < int(rc["stakes"]["max_per_quarter"]):
                amt = int(v["stake"]["amount"])
                if amt in rc["stakes"]["menu"] and lab["points"] >= amt:
                    rec["stake"] = {"side": v["stake"]["side"], "amount": amt}
                    lab["points"] -= amt
                    n_stakes += 1
            st["votes_this_quarter"].append(rec)
            self.w.ledger.add("votes", rec)
        # verification offers
        cap = int(ctx["verification_capacity"])
        for cid in dec.get("verifications", [])[:cap]:
            if cid in in_view:
                st["offers"].append({"quarter": quarter, "claim_id": cid, "lab": lid})
                self.w.ledger.add("votes", {"quarter": quarter, "claim_id": cid, "voter_lab": lid, "vote": "offer", "proposer_lab": st["claims"][cid]["proposer_lab"],
                                            "question_id": st["claims"][cid]["question_id"]})
        # responses to challenges
        for rsp in dec.get("responses", []):
            c = st["claims"].get(rsp["claim_id"])
            if not c or c["proposer_lab"] != lid:
                continue
            for ch in c["challenges"]:
                ch["responded"] = True
            act = rsp["action"]
            c["responses"] = c.get("responses", []) + [{"quarter": quarter, "action": act, "text": rsp.get("text", "")}]
            if act == "withdraw":
                self._set_status(c, "withdrawn", quarter, "proposer withdrew")
            elif act == "revise":
                c["version"] = c.get("version", 1) + 1
                self._set_status(c, "proposed", quarter, "revised after challenge")
            elif act in ("accept_test", "amend_test") and rc["variant"] == "c":
                t = next((t for t in c["tests"] if t.get("status") == "proposed"), None)
                if t:
                    if act == "amend_test" and rsp.get("test") and not t.get("amended"):
                        t["specification"] = rsp["test"].get("specification", t["specification"])
                        t["n_option"] = rsp["test"].get("n_option", t["n_option"])
                        t["amended"] = True
                    t["status"] = "agreed"
                    t["agreed_quarter"] = quarter

    def _file_claim(self, lab: dict, draft: dict, e: dict, prob: float, quarter: int) -> dict:
        st = self.state
        rc = self.cfg["repository"]
        st["claim_counter"] += 1
        cid = f"C{st['claim_counter']:04d}"
        q = self.w.landscape.question(draft["question_id"])
        est = e.get("estimate") or 0.0
        sign = e.get("sign") or ("none" if e["verdict"] == "not_supported" else ("positive" if est > 0 else "negative"))
        pr = e.get("partial_r") if e.get("partial_r") is not None else (partial_r(est / e["se"], max((e.get("n") or 10) - 10, 1)) if e.get("se") else None)
        c = {"claim_id": cid, "question_id": q["id"], "core": q["core"], "legacy": False, "iv": e["iv"], "dv": e["dv"], "moderator": e.get("moderator"),
             "kind": e.get("kind", "focal"), "sign": sign, "statement": e.get("statement") or f"{e['iv']} -> {e['dv']}: {sign}", "magnitude_band": e.get("magnitude_band"),
             "status": "proposed", "quarter_filed": quarter, "quarter_endorsed": None, "proposer_lab": lab["id"], "leads": [draft["lead"]],
             "analysis_id": draft["analysis_id"], "draft_id": draft["draft_id"],
             "evidence": [{"analysis_id": draft["analysis_id"], "estimate": e.get("estimate"), "se": e.get("se"), "p": e.get("p"), "n": e.get("n"),
                           "estimator": e.get("estimator"), "controls": e.get("controls"), "se_type": e.get("se_type"), "partial_r": pr}],
             "deposit_ids": list(draft["deposits"]), "endorsements": [], "challenges": [], "verifications": [], "tests": [], "stakes": [], "links": [],
             "status_history": [{"quarter": quarter, "status": "proposed", "event": "filed"}], "confidence_stated": prob, "confidence_certified": None,
             "note": draft.get("note", "")[:1600], "relation_to_prior": draft.get("relation_to_prior", "first claim"), "route": None, "version": 1,
             "hidden": {"theta": e.get("theta"), "theta_fe": draft["hidden"].get("theta_fe"), "theta_naive": draft["hidden"].get("theta_naive"),
                        "traps": draft["hidden"].get("traps"), "truth_check": e.get("truth_check"), "truth_class": e.get("truth_class"), "flags": draft.get("flags", {}).get("flags", [])}}
        if rc["variant"] == "d":
            stake = float(rc["stakes"]["proposal_stake"])
            lab["points"] -= stake
            c["stakes"].append({"lab": lab["id"], "side": "for", "amount": stake, "quarter": quarter, "probability": prob, "proposal": True})
        st["claims"][cid] = c
        e["filed"] = cid
        lab["record"]["claims"] += 1
        for dep in draft["deposits"]:
            for repo in st["repos"].values():
                if dep in repo["deposits"]:
                    repo["deposits"][dep]["claims"].append(cid)
        # links to earlier claims on the same relation
        for other in self.claims_on(q["id"]):
            if other["claim_id"] != cid and other["iv"] == c["iv"] and other["dv"] == c["dv"] and other.get("moderator") == c.get("moderator") and other["status"] in ("endorsed", "endorsed_legacy", "proposed", "contested"):
                c["links"].append({"type": "contradicts" if other["sign"] != c["sign"] else "replicates", "claim_id": other["claim_id"]})
        self._ledger_claim(c, quarter, "filed")
        return c

    # ======================================================================== phase 3-4: budget admission and draws
    def budget_admit(self, quarter: int) -> None:
        st = self.state
        rc = self.cfg["repository"]
        # prune stale deferred requests: verifications of claims no longer open, duplicates of the same (claim, lab)
        keep, seen = [], set()
        for r in self.w.budget.deferred:
            if r["class"] == "verifications":
                c = st["claims"].get(r.get("claim_id"))
                key = (r.get("claim_id"), r.get("lab"))
                if not c or c["status"] not in ("proposed", "contested") or key in seen or any(v["lab"] == r["lab"] for v in c["verifications"]):
                    self.w.ledger.add("events", {"quarter": quarter, "event": "deferred_request_dropped", "ref": r["id"], "class": r["class"], "lab": r.get("lab")})
                    continue
                seen.add(key)
            keep.append(r)
        self.w.budget.deferred = keep
        reqs = []
        for an in st["analyses"].values():
            if an["status"] == "requested":
                tp = an["throughput"].lower()
                reqs.append({"id": an["analysis_id"], "class": "analyses", "lab": an["lab"], "lab_size": an["lab_size"],
                             "compute": self.w.budget.tariff(f"analyst_{tp}"), "data": an["n_est"] or 0})
        # verification offers -> verifications (variant b and d), first come first served, k per claim, independent labs
        if rc["variant"] in ("b", "d"):
            k = int(rc["reproductions_k"])
            per_claim: dict[str, list] = {}
            for off in st["offers"]:
                c = st["claims"][off["claim_id"]]
                if c["status"] not in ("proposed", "contested"):
                    continue
                done = {v["lab"] for v in c["verifications"]} | {x["lab"] for x in per_claim.get(c["claim_id"], [])}
                if off["lab"] in done or len(done) >= k:
                    continue
                if rc["reciprocity_rule"] == "forbid_verification" and any(v["lab"] == c["proposer_lab"] and v["quarter"] >= quarter - 2
                                                                           for cc in st["claims"].values() if cc["proposer_lab"] == off["lab"] for v in cc["verifications"]):
                    continue
                per_claim.setdefault(c["claim_id"], []).append(off)
            if rc["variant"] == "d":  # the verification queue follows total stake; offers are preferred, else rotation
                ranked = sorted([c for c in st["claims"].values() if c["status"] in ("proposed", "contested") and not c.get("legacy")],
                                key=lambda c: (-sum(s["amount"] for s in c["stakes"]), c["claim_id"]))
                labs_cycle = sorted(st["labs"])
                for c in ranked[: max(1, len(labs_cycle))]:
                    if c["claim_id"] not in per_claim and len(c["verifications"]) < k:
                        others = [l for l in labs_cycle if l != c["proposer_lab"] and not any(s["lab"] == l for s in c["stakes"])]
                        if others:
                            per_claim[c["claim_id"]] = [{"quarter": quarter, "claim_id": c["claim_id"], "lab": others[(quarter + len(per_claim)) % len(others)], "assigned": True}]
            if rc["verification_assignment"] == "rotation":
                labs_cycle = sorted(st["labs"])
                i = 0
                for c in st["claims"].values():
                    if c["status"] == "proposed" and not c.get("legacy") and c["claim_id"] not in per_claim and len(c["verifications"]) < k:
                        others = [l for l in labs_cycle if l != c["proposer_lab"]]
                        per_claim[c["claim_id"]] = [{"quarter": quarter, "claim_id": c["claim_id"], "lab": others[(quarter + i) % len(others)], "assigned": True}]
                        i += 1
            cap_used: dict[str, int] = {}
            for cid, offs in per_claim.items():
                c = st["claims"][cid]
                q = self.w.landscape.question(c["question_id"])
                for off in offs:
                    cap = self._capacity(st["labs"][off["lab"]], quarter)["verifications"]
                    if cap_used.get(off["lab"], 0) >= cap:
                        continue
                    cap_used[off["lab"]] = cap_used.get(off["lab"], 0) + 1
                    if any(r.get("claim_id") == cid and r.get("lab") == off["lab"] for r in self.w.budget.deferred):
                        continue  # already queued from an earlier quarter
                    vid = f"V{short_id(self.w.task_ns, quarter, cid, off['lab'], n=6)}"
                    fresh = not rc["reanalysis_instead_of_fresh"]
                    n_est, n_override, n_basis = 0, None, None
                    ds_seed = seed_from(self.w.seeds["run"], "verification", quarter, cid, off["lab"])
                    if fresh:
                        from ..datasets import draw_frame
                        from ..config import rng as _rng
                        fr = draw_frame(q, _rng(ds_seed, "frame"), self.cfg["datasets"]["n_cap"])
                        n_frame = fr["n_firms"] * fr["n_years"] if fr["panel"] else fr["n_target"]
                        # a fresh verification draw is at least as large as the claim's own sample where the data
                        # allotment allows (a smaller draw makes 'reproduced' uninformative for null claims); else the frame draw
                        claim_n = int((c["evidence"][0].get("n") if c["evidence"] else 0) or 0)
                        n_target = min(max(n_frame, claim_n), int(self.cfg["datasets"]["n_cap"]))
                        remaining = float(self.w.budget.remaining(quarter)["data"])
                        if n_target > n_frame and n_target <= remaining:
                            n_est, n_override, n_basis = n_target, n_target, "claim_n"
                        else:
                            n_est, n_basis = n_frame, ("frame" if n_target <= n_frame else "frame (claim N exceeds the remaining data allotment)")
                    reqs.append({"id": vid, "class": "verifications", "lab": off["lab"], "lab_size": st["labs"][off["lab"]]["size"],
                                 "compute": self.w.budget.tariff("verification") / 2, "data": n_est, "claim_id": cid, "fresh": fresh,
                                 "assigned": off.get("assigned", False), "ds_seed": ds_seed, "n_override": n_override, "n_basis": n_basis})
        # decisive tests agreed last quarter (variant c)
        for c in st["claims"].values():
            for t in c["tests"]:
                if t.get("status") == "agreed" and t.get("agreed_quarter", quarter) < quarter and not t.get("test_id"):
                    q = self.w.landscape.question(c["question_id"])
                    med = q["frame_quantiles"]["p50"]
                    n_req = int({"median": med, "2x_median": 2 * med, "p75": q["frame_quantiles"]["p75"]}.get(t.get("n_option", "median"), med))
                    tid = f"T{short_id(self.w.task_ns, quarter, c['claim_id'], n=6)}"
                    t["test_id"] = tid
                    reqs.append({"id": tid, "class": "responses_tests", "lab": t["challenger"], "lab_size": st["labs"][t["challenger"]]["size"],
                                 "compute": self.w.budget.tariff("decisive_test"), "data": n_req, "claim_id": c["claim_id"], "n": n_req})
        admitted, deferred = self.w.budget.admit(quarter, reqs)
        for r in admitted:
            if r["class"] == "analyses":
                st["analyses"][r["id"]]["status"] = "admitted"
                st["admitted"]["analyses"].append(r["id"])
            elif r["class"] == "verifications":
                st["admitted"]["verifications"].append(r)
            else:
                st["admitted"]["tests"].append(r)
        for r in deferred:
            st["labs"][r["lab"]]["record"]["deferred"] += 1
            if r["class"] == "analyses" and r["id"] in st["analyses"]:
                st["analyses"][r["id"]]["status"] = "deferred"
        la = self.w.budget.last_admission
        # the event names the binding resource of every deferred request (compute or data); `remaining_before` is the allotment net of
        # the quarter's earlier debits (the lab calls), `remaining_after_admission` what the admitted requests leave
        self.w.ledger.add("events", {"quarter": quarter, "event": "budget_admission", "requested": len(reqs), "admitted": len(admitted), "deferred": len(deferred),
                                     "allotment": self.w.budget.allotment(quarter), "remaining_before": la.get("remaining_before"),
                                     "remaining_after_admission": la.get("remaining_after_admission"),
                                     "deferred_by_binding": la.get("deferred_by_binding"), "deferred_requests": la.get("deferred", [])})

    def _new_deposit(self, qid: str, lab_id: str, quarter: int, ds_seed: int, upper_half: bool, purpose: str, write: bool,
                     n_override: int | None = None) -> dict:
        st = self.state
        st["deposit_counter"] += 1
        dep_id = f"D{st['deposit_counter']:04d}"
        dest = self.w.run_dir / "datasets" / dep_id if write else None
        res = self.w.gen.generate(qid, ds_seed, dep_id, upper_half=upper_half, write_dir=dest, n_override=n_override)
        q = self.w.landscape.question(qid)
        d = {"deposit_id": dep_id, "version": 1, "question_id": qid, "lab": lab_id, "quarter": quarter, "available_from": quarter + 1,
             "n": res["meta"]["n_rows"], "unit": q["unit"], "dataset_seed": ds_seed, "upper_half": upper_half, "n_override": n_override, "n_analyses": 0, "claims": [],
             "versions": [{"version": 1, "quarter": quarter, "by": lab_id}], "purpose": purpose, "hidden": {k: v for k, v in res["hidden"].items() if k != "cov"},
             "meta": {k: v for k, v in res["meta"].items() if k != "columns"}, "data_sha256": res["meta"]["data_sha256"], "corrections": []}
        st["repos"].setdefault(qid, {"deposits": {}})["deposits"][dep_id] = d
        st["labs"][lab_id]["record"]["deposits"] += 1
        self.w.budget.debit_data(quarter, res["meta"]["n_rows"], dep_id, lab_id)
        self.w.ledger.add("datasets", {"quarter": quarter, "dataset_id": dep_id, "question_id": qid, "who": lab_id, "purpose": purpose,
                                       "meta": d["meta"], "hidden": {k: v for k, v in res["hidden"].items() if k not in ("cov",)}})
        return d

    def _ensure_deposit_files(self, d: dict) -> Path:
        dest = self.w.run_dir / "datasets" / d["deposit_id"]
        if not (dest / "data.csv").exists():
            self.w.gen.generate(d["question_id"], d["dataset_seed"], d["deposit_id"], upper_half=d["upper_half"], write_dir=dest, n_override=d.get("n_override"))
            for corr in d.get("corrections", []):
                self._apply_correction_script(dest, corr["script"])
        return dest

    def draws(self, quarter: int) -> None:
        st = self.state
        for aid in st["admitted"]["analyses"]:
            an = st["analyses"][aid]
            if an["fresh"]:
                ds_seed = an.get("ds_seed") or seed_from(self.w.seeds["run"], "deposit", quarter, an["lab"], aid)   # the seed the admission costed
                d = self._new_deposit(an["question_id"], an["lab"], quarter, ds_seed, an["throughput"] == "T1", "analysis", write=self.w.mode != "fast",
                                      n_override=an.get("n_override"))
                an["deposits"] = [d["deposit_id"]]
                d["available_from"] = quarter  # the commissioning lab analyses it this quarter; others from the next
                d["owner_only_until"] = quarter + 1
        for r in st["admitted"]["verifications"]:
            if r.get("fresh"):
                ds_seed = r.get("ds_seed") or seed_from(self.w.seeds["run"], "verification", quarter, r["claim_id"], r["lab"])
                c = st["claims"][r["claim_id"]]
                d = self._new_deposit(c["question_id"], r["lab"], quarter, ds_seed, False, "verification", write=self.w.mode != "fast",
                                      n_override=r.get("n_override"))
                r["deposit_id"] = d["deposit_id"]
            else:
                r["deposit_id"] = (st["claims"][r["claim_id"]]["deposit_ids"] or [None])[0]
        for r in st["admitted"]["tests"]:
            c = st["claims"][r["claim_id"]]
            ds_seed = seed_from(self.w.seeds["run"], "test", quarter, r["claim_id"])
            d = self._new_deposit(c["question_id"], r["lab"], quarter, ds_seed, False, "decisive_test", write=self.w.mode != "fast")
            r["deposit_id"] = d["deposit_id"]

    # ======================================================================== phase 5: analyses
    def _deposit(self, dep_id: str) -> dict:
        for repo in self.state["repos"].values():
            if dep_id in repo["deposits"]:
                return repo["deposits"][dep_id]
        raise KeyError(dep_id)

    def _group_analyses(self, prepared: list[dict]) -> list[list[dict]]:
        """Group admitted analyses into calls of at most `analyst_batch_size`: different labs, different questions."""
        B = int(self.cfg["repository"].get("analyst_batch_size", 1))
        prepared = sorted(prepared, key=lambda x: x["aid"])
        if B <= 1:
            return [[x] for x in prepared]
        groups: list[list[dict]] = []
        for x in prepared:
            for g in groups:
                if len(g) < B and all(m["an"]["lab"] != x["an"]["lab"] and m["q"]["id"] != x["q"]["id"] for m in g):
                    g.append(x)
                    break
            else:
                groups.append([x])
        return groups

    def analyse(self, quarter: int) -> None:
        st = self.state
        tasks = []
        prepared = []
        for aid in st["admitted"]["analyses"]:
            an = st["analyses"][aid]
            lead = self.scientist(an["lead"])
            q = self.w.landscape.question(an["question_id"])
            deps = [self._deposit(d) for d in an["deposits"]]
            for d in deps:
                d["n_analyses"] += 1
            spec = self.w.gen.spec_for(q)
            variables = {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")}
            traps = sorted({t for d in deps for t in d["hidden"]["traps"]})
            view = [{"paper_id": c["claim_id"], "focal_estimate": (c["evidence"][0].get("estimate") if c["evidence"] else 0.0),
                     "focal_verdict": "supported" if c["sign"] != "none" else "not_supported"} for c in self.claims_on(q["id"]) if c["status"] in ("endorsed", "endorsed_legacy")]
            if self.w.mode == "fast":
                self._fast_analysis(an, lead, q, deps, variables, traps, view, quarter)
                continue
            protocol = read_text(PROMPTS / PROTOCOL[lead["style"]])
            depth = read_text(PROMPTS / "_depth_t1.md") if lead["throughput"] == "T1" else "(No additional specifications are required.)"
            prepared.append({"aid": aid, "an": an, "lead": lead, "q": q, "deps": deps, "variables": variables, "traps": traps, "view": view,
                             "protocol": protocol, "depth": depth})
        if self.w.mode == "fast":
            return
        for g in self._group_analyses(prepared):
            if len(g) == 1:
                x = g[0]
                an, lead, q, deps = x["an"], x["lead"], x["q"], x["deps"]
                inputs = {"brief.md": f"# Brief from the group head\n\n{an['brief']}\n", "registry.md": self.registry_view(q["id"], quarter, an["lab"])}
                for d in deps:
                    dest = self._ensure_deposit_files(d)
                    for f in ("data.csv", "codebook.md", "sample_frame.md"):
                        inputs[f"inputs/{d['deposit_id']}/{f}"] = dest / f
                t = Task(type="analyst", quarter=quarter, identity=x["aid"], tier=lead.get("tier", "sonnet"), run_id=self.w.task_ns, call_type=f"analyst_{lead['throughput'].lower()}",
                         template="analyst.md", variables={"PROTOCOL_BLOCK": x["protocol"], "DEPTH_BLOCK": x["depth"]}, inputs=inputs,
                         outputs=["analysis.py", "table.csv", "claims_draft.json", "note.md"], schema="claims_draft.schema.json",
                         meta={"analysis_id": x["aid"], "lab_id": an["lab"]}, who=an["lab"])
                self._task_ctx[t.id] = {"type": "analyst", "lead": lead, "question": q, "traps": x["traps"], "variables": x["variables"],
                                        "deposit_dirs": [d["deposit_id"] for d in deps], "deposit_versions": [f"{d['deposit_id']}:{d['version']}" for d in deps], "view": x["view"]}
                tasks.append(t)
            else:
                tasks.append(self._make_analyst_batch_task(g, quarter))
        self.run_tasks(tasks, quarter)
        for t in tasks:
            items = self._task_ctx[t.id]["items"] if t.type == "analyst_batch" else [self._task_ctx[t.id] | {"analysis_id": t.meta["analysis_id"]}]
            for item in items:
                aid = item["analysis_id"]
                an = st["analyses"][aid]
                sub = f"output/{aid}" if t.type == "analyst_batch" else "output"
                prefix = f"analyses/{aid}/inputs" if t.type == "analyst_batch" else "inputs"
                d = self.w.store.task_dir(t)
                if self.w.store.status(t) != "done":
                    an["status"] = "failed"
                    self._free_member(an)
                    self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": an["lab"]})
                    continue
                draft = self.w.store.read_output(t, f"{aid}/claims_draft.json" if t.type == "analyst_batch" else "claims_draft.json")
                note = self.w.store.read_output(t, f"{aid}/note.md" if t.type == "analyst_batch" else "note.md")
                data_files = {f"{prefix}/{dep}/data.csv": d / prefix / dep / "data.csv" for dep in item["deposit_dirs"]}
                forensic = forensic_rerun(d / sub, data_files, self.cfg["datasets"]["forensics"]["mode"], self.cfg["datasets"]["forensics"]["tolerance"], work_rel=sub, task_root=d) \
                    if all(p.exists() for p in data_files.values()) else {"reproducible": None, "reason": "data not kept"}
                if t.type == "analyst_batch":
                    an["call_group"], an["call_group_size"] = t.id, len(items)
                self._finish_analysis(an, t, d, sub, draft, note, forensic, quarter, data_files)

    def _make_analyst_batch_task(self, g: list[dict], quarter: int) -> Task:
        tp = "t1" if any(x["lead"]["throughput"] == "T1" for x in g) else "t0"
        outputs = [f"{x['aid']}/{f}" for x in g for f in ("analysis.py", "table.csv", "claims_draft.json", "note.md")]
        t = Task(type="analyst_batch", quarter=quarter, identity="+".join(x["aid"] for x in g), tier=g[0]["lead"].get("tier", "sonnet"), run_id=self.w.task_ns,
                 call_type=f"analyst_{tp}", template="analyst_batch.md", outputs=outputs, schema="claims_draft.schema.json",
                 meta={"analysis_ids": [x["aid"] for x in g], "lab_ids": [x["an"]["lab"] for x in g], "batch": len(g)}, who="+".join(x["an"]["lab"] for x in g))
        d = self.w.store.task_dir(t)
        inputs, block, items = {}, [], []
        for x in g:
            an, q, deps = x["an"], x["q"], x["deps"]
            inputs[f"analyses/{x['aid']}/brief.md"] = f"# Brief from the group head\n\n{an['brief']}\n"
            inputs[f"analyses/{x['aid']}/registry.md"] = self.registry_view(q["id"], quarter, an["lab"])
            for dep in deps:
                dest = self._ensure_deposit_files(dep)
                for f in ("data.csv", "codebook.md", "sample_frame.md"):
                    inputs[f"analyses/{x['aid']}/inputs/{dep['deposit_id']}/{f}"] = dest / f
            block.append(f"- Analysis `{x['aid']}`: folder `{d}\\analyses\\{x['aid']}\\`; outputs into `{d}\\output\\{x['aid']}\\`. "
                         f"Working protocol: {x['protocol'].strip()} {x['depth'].strip()}")
            items.append({"analysis_id": x["aid"], "lead": x["lead"], "question": q, "traps": x["traps"], "variables": x["variables"],
                          "deposit_dirs": [dep["deposit_id"] for dep in deps], "deposit_versions": [f"{dep['deposit_id']}:{dep['version']}" for dep in deps], "view": x["view"]})
        t.inputs = inputs
        t.variables = {"N_ITEMS": len(g), "ANALYSES_BLOCK": "\n".join(block)}
        self._task_ctx[t.id] = {"type": "analyst_batch", "items": items}
        return t

    def _finish_analysis(self, an, t, d, sub, draft, note, forensic, quarter, data_files) -> None:
        if True:
            if True:
                code = read_text(d / sub / "analysis.py")
                self._register_draft(an, draft, note, code, forensic, quarter, t.id)
                if (d / sub / "correction.py").exists() and draft.get("correction"):
                    self._file_correction(an, d / sub / "correction.py", draft["correction"].get("changelog", ""), quarter)
                if not self.cfg["datasets"].get("keep_csv", True):
                    for p in data_files.values():
                        if p.exists():
                            p.unlink()

    def _free_member(self, an: dict) -> None:
        slot = self.state["authors"][an["lead"]]["slots"][an["slot"]]
        slot.update({"state": "free", "paper_id": None})

    def _register_draft(self, an: dict, draft: dict, note: str, code: str, forensic: dict, quarter: int, task_id: str, fast_hidden: dict | None = None) -> dict:
        st = self.state
        q = self.w.landscape.question(an["question_id"])
        lead = self.scientist(an["lead"])
        deps = [self._deposit(d) for d in an["deposits"]]
        hidden = fast_hidden or dict(deps[0]["hidden"])
        hidden["traps"] = sorted({t for d in deps for t in d["hidden"]["traps"]}) if not fast_hidden else hidden["traps"]
        meta = {"variables": {"controls": [c for c in q["controls_pool"] if c not in ("industry", "year")]}, "n_correct": sum(d["meta"]["n_correct"] for d in deps) if deps else an.get("n_est")}
        claims_like = {"hypotheses": [{k: e.get(k) for k in ("hyp_id", "iv", "dv", "moderator", "kind", "predicted_sign", "model_id", "estimate", "se", "p", "n", "estimator", "se_type", "controls", "verdict")} | {"hyp_id": e["entry_id"]} for e in draft["entries"]],
                       "sample_changes": draft.get("sample_changes", []), "n_models_run": draft.get("n_models_run", 0), "robustness_models": draft.get("robustness_models", []),
                       "executed_code": draft.get("executed_code", ""), "commands_run": draft.get("commands_run", 1)}
        for e in draft["entries"]:
            theta = relation_theta(hidden, e["iv"], e["dv"], e.get("moderator"), e.get("kind", ""), q, controls=meta["variables"]["controls"])
            theta_fe = hidden.get("theta_fe") if (e["iv"] == q["iv_var"] and e["dv"] == q["dv_var"] and not e.get("moderator")) else theta
            e.update(score_hypothesis(e, theta, theta_fe))
            if e.get("estimate") is not None and e.get("se"):
                e["partial_r"] = partial_r(e["estimate"] / e["se"], max((e.get("n") or 10) - len(e.get("controls", [])) - 3, 1))
            # the stated sign is what the claim will say
            e["truth_check"] = _truth_check_claim(e.get("sign") or ("none" if e["verdict"] == "not_supported" else ("positive" if (e.get("estimate") or 0) > 0 else "negative")), e.get("estimate") or 0.0, theta) if theta is not None else "unknown"
        flags = analyst_flags(claims_like, hidden, meta, q, note, lead["style"], lead["throughput"], forensic.get("reproducible"))
        st["draft_counter"] += 1
        did = f"DR{st['draft_counter']:04d}"
        public = {k: v for k, v in draft.items() if k not in ("nonce",)}
        public["entries"] = [{k: v for k, v in e.items() if k not in ("theta", "truth_class", "truth_check", "truth_check_lenient", "magnitude_ratio", "ci_covers", "partial_r")} for e in draft["entries"]]
        st["drafts"][did] = {"draft_id": did, "analysis_id": an["analysis_id"], "lab": an["lab"], "lead": an["lead"], "question_id": q["id"], "deposits": an["deposits"],
                             "entries": draft["entries"], "note": note, "public_draft": public, "status": "ready", "quarter": quarter, "relation_to_prior": draft.get("relation_to_prior", "first claim"),
                             "hidden": {k: hidden.get(k) for k in ("theta", "theta_fe", "theta_naive", "traps", "target")}, "flags": flags, "forensic": forensic}
        an.update({"status": "done", "draft_id": did, "code": code, "flags": flags, "forensic": forensic, "n_entries": len(draft["entries"]), "task_id": task_id})
        self._free_member(an)
        st["labs"][an["lab"]]["record"]["analyses"] += 1
        for d in deps:
            d["n_analyses"] = d.get("n_analyses", 0)
        self.w.ledger.add("papers", ({"call_group": an["call_group"], "call_group_size": an["call_group_size"]} if an.get("call_group") else {})
                          | {"analysis_id": an["analysis_id"], "lab": an["lab"], "lab_size": an["lab_size"], "lead": an["lead"], "style": lead["style"],
                                     "throughput": lead["throughput"], "question_id": q["id"], "core": q["core"], "quarter": quarter, "fresh": an["fresh"],
                                     "deposits": an["deposits"], "pooled": len(an["deposits"]) > 1, "n_entries": len(draft["entries"]), "flags": flags["flags"],
                                     "traps": hidden["traps"], "reproducible": forensic.get("reproducible"), "truth_check_focal": draft["entries"][0].get("truth_check"),
                                     "relation_to_prior": draft.get("relation_to_prior"), "n_models_run": draft.get("n_models_run")})
        return st["drafts"][did]

    def _fast_analysis(self, an, lead, q, deps, variables, traps, view, quarter) -> None:
        ra = self.rules["author"]
        r = self.rng("fast_analysis", quarter, an["analysis_id"])
        handled = ra.decide_handling(traps, lead["style"], r)
        use_fe = ra.use_fe(q, lead, r)
        n_total = sum(d["n"] for d in deps)
        # rule parameter `author.analyst_reports_found_sign` (default false = the old predicted-sign verdict)
        res = ra.analyse_analytic(self.w.gen, q, deps[0]["dataset_seed"], lead, handled, use_fe, ra.n_hypotheses(r), variables, lead["throughput"] == "T1",
                                  # with `fast_honours_deposit_n` the analytic study has the deposit's size when the deposit
                                  # was drawn at a requested size (default false: the frame is redrawn from the dataset seed, as before)
                                  n_override=n_total if (len(deps) > 1 or (self.params.get("fast_honours_deposit_n") and any(d.get("n_override") for d in deps))) else None,
                                  predict_sign=not bool(ra.p.get("analyst_reports_found_sign", False)))
        self.w.budget.debit_call(quarter, f"analyst_{lead['throughput'].lower()}", an["analysis_id"], an["lab"])
        rel = ra.relation_to_prior(res["hypotheses"][0]["estimate"], view, r, first_word="first claim")
        entries = []
        for h in res["hypotheses"]:
            est = h["estimate"]
            pr = abs(h.get("partial_r") or 0.0)
            band = "negligible" if pr < 0.02 else ("small" if pr < 0.05 else ("moderate" if pr < 0.10 else "large"))
            sign = _claim_sign(h)   # direction from the estimate and p, not from support of the prediction (a significant wrong-sign result is not "no relation")
            entries.append({"entry_id": f"E{h['hyp_id'][1:]}", "iv": h["iv"], "dv": h["dv"], "moderator": h.get("moderator"), "kind": h["kind"],
                            "statement": f"{h['iv']} has {'no reliable' if sign == 'none' else 'a ' + band + ' ' + sign} relation with {h['dv']}" + (f" conditional on {h['moderator']}" if h.get("moderator") else ""),
                            "sign": sign, "magnitude_band": band, "confidence": float(np.clip(r.normal(70 if h["verdict"] == "supported" else 45, 10), 5, 95)),
                            "model_id": h["model_id"], "estimate": est, "se": h["se"], "p": h["p"], "n": h["n"], "estimator": h["estimator"], "se_type": h["se_type"],
                            "controls": h["controls"], "verdict": h["verdict"], "predicted_sign": h["predicted_sign"], "partial_r": h.get("partial_r")})
        draft = {"question_id": q["id"], "deposits": [f"{d['deposit_id']}:{d['version']}" for d in deps], "relation_to_prior": rel, "entries": entries,
                 "sample_changes": res["sample_changes"], "n_models_run": res["n_models_run"], "robustness_models": res["robustness_models"],
                 "executed_code": "", "commands_run": 1, "correction": None}
        s = res["study"]
        hidden = dict(s["hidden"])
        hidden.update({"oracle": {"correct": {"estimate": s["hidden"]["target"], "se": s["focal"]["se"], "n": s["n"]}, "naive": {}}, "cov": None, "cov_names": None})
        d0 = self._register_draft(an, draft, "", "", {"reproducible": None, "reason": "fast mode"}, quarter, "fast", fast_hidden=hidden)
        fl = [{"duplicates": "duplicates_not_removed", "scale_mixture": "scale_mixture_unfixed", "reverse_coded": "reverse_coding_missed", "selection_on_outcome": "selection_unacknowledged",
               "bad_control": "bad_control_included", "miscoded_values": "miscoded_values_kept", "unit_mismatch": "unit_mismatch_unfixed", "omitted_standard_control": "omitted_standard_control"}[t]
              for t in traps if not handled.get(t)]
        if q["panel"] and not use_fe:
            fl.append("se_not_clustered")
        d0["flags"] = {"flags": fl, "trap_handled": handled, "n_models_run": res["n_models_run"], "robustness_block_present": lead["throughput"] == "T1", "specification_search": False}
        an["flags"] = d0["flags"]

    def _file_correction(self, an: dict, script: Path, changelog: str, quarter: int) -> None:
        """Apply correction.py in a clean folder, check determinism, file version v+1."""
        st = self.state
        dep = self._deposit(an["deposits"][0])
        dest = self._ensure_deposit_files(dep)
        code = read_text(script)
        import shutil, tempfile
        tmp = Path(tempfile.mkdtemp(prefix="corr_"))
        try:
            shutil.copy(dest / "data.csv", tmp / "data.csv")
            write_text(tmp / "correction.py", code)
            h1 = self._apply_correction_script(tmp, code)
            shutil.copy(dest / "data.csv", tmp / "data.csv")
            h2 = self._apply_correction_script(tmp, code)
            deterministic = h1 == h2 and h1 is not None
            verdict = "unknown"
            if deterministic:
                import pandas as pd
                from ..datasets import fit_focal
                q = self.w.landscape.question(dep["question_id"])
                df = pd.read_csv(tmp / "data.csv")
                controls = [c for c in q["controls_pool"] if c not in ("industry", "year")]
                try:
                    f = fit_focal(df, q, controls, q["panel"])
                    est = f["coef"][q["iv_var"]]["partial_r"]
                    correct = dep["hidden"]["oracle"]["correct"]["partial_r"]
                    se = 1.0 / np.sqrt(max(f.get("df") or 10, 10))
                    verdict = "fixed" if abs(est - correct) <= se else "no_change_or_error"
                except Exception:  # noqa: BLE001
                    verdict = "fit_failed"
                shutil.copy(tmp / "data.csv", dest / "data.csv")
                dep["version"] += 1
                dep["versions"].append({"version": dep["version"], "quarter": quarter, "by": an["lab"], "changelog": changelog})
                dep["corrections"].append({"script": code, "quarter": quarter, "lab": an["lab"], "verdict": verdict})
                st["labs"][an["lab"]]["record"]["corrections"] = st["labs"][an["lab"]]["record"].get("corrections", 0) + 1
            self.w.ledger.add("events", {"quarter": quarter, "event": "correction", "deposit_id": dep["deposit_id"], "lab": an["lab"], "deterministic": deterministic, "verdict": verdict})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _apply_correction_script(self, folder: Path, code: str) -> str | None:
        import os, runpy, sys
        from ..util import sha256_file
        write_text(folder / "correction.py", code)
        cwd = os.getcwd()
        try:
            os.chdir(folder)
            runpy.run_path(str(folder / "correction.py"), run_name="__main__")
        except Exception:  # noqa: BLE001
            return None
        finally:
            os.chdir(cwd)
        return sha256_file(folder / "data.csv") if (folder / "data.csv").exists() else None

    # ======================================================================== phase 6: verification
    def verify(self, quarter: int) -> None:
        st = self.state
        reqs = st["admitted"]["verifications"]
        if not reqs:
            return
        if self.w.mode == "fast":
            for r in reqs:
                self._fast_verification(r, quarter)
            return
        tasks = []
        for i, batch in enumerate(verification_batches(reqs)):
            lab = st["labs"][batch[0]["lab"]]
            members = [m for m in self.lab_members(lab["id"]) if m["stage"] != "doctoral"] or self.lab_members(lab["id"])
            if not members:
                for r in batch:
                    self.w.ledger.add("events", {"quarter": quarter, "event": "verification_dropped", "claim_id": r["claim_id"], "lab": lab["id"], "reason": "no members"})
                continue
            verifier = members[(quarter + i) % len(members)]
            inputs, items = {}, []
            for r in batch:
                c = st["claims"][r["claim_id"]]
                q = self.w.landscape.question(c["question_id"])
                dep = self._deposit(r["deposit_id"])
                dest = self._ensure_deposit_files(dep)
                for f in ("data.csv", "codebook.md", "sample_frame.md"):
                    inputs[f"inputs/{c['claim_id']}/{f}"] = dest / f
                an = st["analyses"].get(c.get("analysis_id"), {})
                inputs[f"inputs/{c['claim_id']}/analysis.py"] = an.get("code") or "# original code not available\n"
                orig = self._deposit(c["deposit_ids"][0]) if c["deposit_ids"] else None
                if orig:
                    inputs[f"inputs/{c['claim_id']}/original_codebook.md"] = self._ensure_deposit_files(orig) / "codebook.md"
                ev = c["evidence"][0]
                inputs[f"inputs/{c['claim_id']}/claim.md"] = (f"# Claim {c['claim_id']}\n\n{c['statement']}.\n\nEvidence: estimate {ev['estimate']:.4f}, SE {ev['se']:.4f}, p {ev['p']:.4f}, "
                                                              f"N {ev['n']}, {ev.get('estimator', '')}; controls {', '.join(ev.get('controls') or [])}. Data: {', '.join(c['deposit_ids'])}.\n")
                spec = self.w.gen.spec_for(q)
                items.append({"claim_id": c["claim_id"], "verifier": verifier, "question": q, "traps": dep["hidden"]["traps"],
                              "variables": {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")},
                              "claim_estimate": ev["estimate"], "claim_p": ev.get("p"), "claim_sign": c["sign"]})
            protocol = read_text(PROMPTS / PROTOCOL[verifier["style"]])
            t = Task(type="verification", quarter=quarter, identity=f"{lab['id']}:{i}", tier=verifier.get("tier", "sonnet"), run_id=self.w.task_ns, call_type="verification",
                     template="verification.md", variables={"N_ITEMS": len(batch), "ITEM_LIST": ", ".join(r["claim_id"] for r in batch), "PROTOCOL_BLOCK": protocol,
                                                            "SAME_DEPOSIT_SENTENCE": "" if batch[0].get("fresh") else " (here the same dataset the claim used)"},
                     inputs=inputs, outputs=["verifications.json"], schema="verifications.schema.json",
                     meta={"lab_id": lab["id"], "verifier": verifier["id"], "requests": [{"claim_id": r["claim_id"], "deposit_id": r["deposit_id"], "fresh": r.get("fresh", True)} for r in batch]}, who=lab["id"])
            self._task_ctx[t.id] = {"type": "verification", "items": items}
            tasks.append(t)
        self.run_tasks(tasks, quarter)
        for t in tasks:
            if self.w.store.status(t) != "done":
                self.w.ledger.add("events", {"quarter": quarter, "event": "task_failed", "task_id": t.id, "who": t.meta["lab_id"]})
                continue
            out = self.w.store.read_output(t, "verifications.json")
            for v, r in zip(out["verifications"], t.meta["requests"]):
                self._record_verification(r["claim_id"], t.meta["lab_id"], t.meta["verifier"], r["deposit_id"], r["fresh"], v, quarter)
            if not self.cfg["datasets"].get("keep_csv", True):
                for p in (self.w.store.task_dir(t) / "inputs").rglob("data.csv"):
                    p.unlink()

    def _fast_verification(self, r: dict, quarter: int) -> None:
        st = self.state
        c = st["claims"][r["claim_id"]]
        q = self.w.landscape.question(c["question_id"])
        lab = st["labs"][r["lab"]]
        members = [m for m in self.lab_members(lab["id"]) if m["stage"] != "doctoral"] or self.lab_members(lab["id"])
        if not members:  # the lab lost its members to turnover between the offer and the draw
            self.w.ledger.add("events", {"quarter": quarter, "event": "verification_dropped", "claim_id": c["claim_id"], "lab": lab["id"], "reason": "no members"})
            return
        verifier = members[quarter % len(members)]
        dep = self._deposit(r["deposit_id"])
        rr = self.rng("fast_verify", quarter, c["claim_id"], lab["id"])
        handled = self.rules["author"].decide_handling(dep["hidden"]["traps"], verifier["style"], rr)
        spec = self.w.gen.spec_for(q)
        res = self.rules["author"].analyse_analytic(self.w.gen, q, dep["dataset_seed"], verifier, handled, q["panel"], 1,
                                                    {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")}, False,
                                                    n_override=dep.get("n_override") if self.params.get("fast_honours_deposit_n") else None)   # see fast_honours_deposit_n above
        self.w.budget.debit_call(quarter, "verification", f"{c['claim_id']}-{lab['id']}", lab["id"])
        h = res["hypotheses"][0]
        from ..agents.rules import reproduction_result
        ev = c["evidence"][0]
        result = reproduction_result(ev["estimate"], ev.get("p"), h["estimate"], h["se"], self.params["verifier"]["factor_tolerance"], claim_sign=c["sign"])
        self._record_verification(c["claim_id"], lab["id"], verifier["id"], r["deposit_id"], r.get("fresh", True),
                                  {"own": {"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"]}, "result": result, "confidence": 0.7, "discrepancies": ""}, quarter)

    def _record_verification(self, cid: str, lab_id: str, verifier: str, deposit_id: str | None, fresh: bool, v: dict, quarter: int) -> None:
        st = self.state
        c = st["claims"][cid]
        from ..agents.rules import reproduction_result
        own = v.get("own") or {}
        rule_result = reproduction_result(c["evidence"][0]["estimate"], c["evidence"][0].get("p"), own.get("estimate") or 0.0, own.get("se") or 0.0,
                                          self.params["verifier"]["factor_tolerance"], claim_sign=c["sign"]) if own.get("se") else "inconclusive"
        ev = c["evidence"][0]
        # verification power: the minimum detectable partial r at the verifier's N beside the verdict
        dep = next((d for repo in st["repos"].values() for d in repo["deposits"].values() if d["deposit_id"] == deposit_id), None)
        n_ver = own.get("n") or (dep["n"] if dep else None)
        power = verification_power(n_ver, len(own.get("controls") or ev.get("controls") or []), ev.get("partial_r"), c["sign"], ev.get("n"))
        rec = {"lab": lab_id, "verifier": verifier, "quarter": quarter, "deposit_id": deposit_id, "fresh": fresh, "result": rule_result, "reported_result": v.get("result"),
               "estimate": own.get("estimate"), "se": own.get("se"), "n": own.get("n"), "confidence": v.get("confidence"), "discrepancies": v.get("discrepancies", ""),
               "mde_partial_r": power["mde_partial_r"], "powered": power["powered"]}
        c["verifications"].append(rec)
        st["labs"][lab_id]["record"]["verifications"] += 1
        truth_agrees = (rule_result == "reproduced") == (c["hidden"].get("truth_check") in ("true_discovery", "true_discovery_negligible"))
        self.w.ledger.add("verifications", {"quarter": quarter, "claim_id": cid, "lab": lab_id, "verifier": verifier, "deposit_id": deposit_id, "fresh": fresh,
                                            "result": rule_result, "reported_result": v.get("result"), "estimate": own.get("estimate"), "se": own.get("se"), "n": own.get("n"),
                                            "claim_estimate": ev["estimate"], "claim_n": ev.get("n"), "claim_partial_r": ev.get("partial_r"), "claim_sign": c["sign"],
                                            "claim_truth_check": c["hidden"].get("truth_check"), "agrees_with_truth": truth_agrees,
                                            "question_id": c["question_id"], "proposer_lab": c["proposer_lab"], **power})
        rec_s = self.state["records"].setdefault(verifier, {})
        rec_s["verifications"] = rec_s.get("verifications", 0) + 1

    # ======================================================================== phase 7: decisive tests (variant c)
    def tests(self, quarter: int) -> None:
        st = self.state
        for r in st["admitted"]["tests"]:
            c = st["claims"][r["claim_id"]]
            t = next((t for t in c["tests"] if t.get("test_id") == r["id"]), None)
            if not t:
                continue
            q = self.w.landscape.question(c["question_id"])
            dep = self._deposit(r["deposit_id"])
            executor_lab = sorted(l for l in st["labs"] if l not in (c["proposer_lab"], t["challenger"]) and self.lab_members(l))
            ex_lab = executor_lab[quarter % len(executor_lab)] if executor_lab else t["challenger"]
            members = self.lab_members(ex_lab)
            if not members:
                self.w.ledger.add("events", {"quarter": quarter, "event": "test_dropped", "claim_id": c["claim_id"], "reason": "no executing members"})
                continue
            executor = members[quarter % len(members)]
            rr = self.rng("fast_test", quarter, c["claim_id"])
            handled = self.rules["author"].decide_handling(dep["hidden"]["traps"], executor["style"], rr)
            spec = self.w.gen.spec_for(q)
            variables = {"controls": spec["controls"], "moderators": q["moderator_vars"], "secondary_dv": q.get("secondary_dv_var")}
            if self.w.mode == "fast":
                res = self.rules["author"].analyse_analytic(self.w.gen, q, dep["dataset_seed"], executor, handled, q["panel"], 1, variables, False)
                h = res["hypotheses"][0]
                focal = {"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"]}
                self.w.budget.debit_call(quarter, "decisive_test", r["id"], ex_lab)
            else:
                dest = self._ensure_deposit_files(dep)
                tk = Task(type="decisive_test", quarter=quarter, identity=r["id"], tier=executor.get("tier", "sonnet"), run_id=self.w.task_ns, call_type="decisive_test",
                          template="decisive_test.md", variables={"PROTOCOL_BLOCK": read_text(PROMPTS / PROTOCOL[executor["style"]])},
                          inputs={"test.md": f"# Pre-registered test for {c['claim_id']}\n\nSpecification: {t['specification']}\n\nDecision rule: {t['decision_rule']}\n",
                                  **{f"inputs/data/{f}": dest / f for f in ("data.csv", "codebook.md", "sample_frame.md")}},
                          outputs=["analysis.py", "table.csv", "test_result.json"], schema="test_result.schema.json", meta={"claim_id": c["claim_id"]}, who=ex_lab)
                self._task_ctx[tk.id] = {"type": "decisive_test", "executor": executor, "question": q, "traps": dep["hidden"]["traps"], "variables": variables, "claim_id": c["claim_id"]}
                self.run_tasks([tk], quarter)
                if self.w.store.status(tk) != "done":
                    continue
                focal = self.w.store.read_output(tk, "test_result.json")["focal"]
            # settlement by the pre-registered rule: the claim holds if the focal estimate has the claimed sign with p < .05
            holds = focal.get("p") is not None and focal["p"] < 0.05 and (c["sign"] == "none" or np.sign(focal["estimate"]) == (1 if c["sign"] == "positive" else -1))
            if c["sign"] == "none":
                holds = focal.get("p") is not None and focal["p"] >= 0.05
            t.update({"status": "settled", "result": "holds" if holds else "fails", "focal": focal, "executor": executor["id"], "executor_lab": ex_lab, "quarter_settled": quarter})
            if holds:
                c["route"] = "test"
                c["confidence_certified"] = 0.9
                self._set_status(c, "endorsed", quarter, "decisive test")
                st["labs"][t["challenger"]]["record"]["challenges_lost"] += 1
            else:
                self._set_status(c, "retracted", quarter, "decisive test failed")
                st["labs"][t["challenger"]]["record"]["challenges_won"] += 1
            self.w.ledger.add("events", {"quarter": quarter, "event": "decisive_test", "claim_id": c["claim_id"], "result": t["result"], "claim_truth_check": c["hidden"].get("truth_check")})

    # ======================================================================== phase 8: certification rules
    def certify(self, quarter: int) -> None:
        st = self.state
        rc = self.cfg["repository"]
        variant = rc["variant"]
        # apply this quarter's votes
        for v in st["votes_this_quarter"]:
            c = st["claims"][v["claim_id"]]
            if v["vote"] == "endorse":
                c["endorsements"].append({"lab": v["voter_lab"], "quarter": quarter, "probability": v["probability"], "reason": v["reason"]})
                prop = st["labs"].get(c["proposer_lab"])
                if prop:
                    prop["record"].setdefault("endorsed_by", []).append(v["voter_lab"])
            elif v["vote"] == "challenge":
                c["challenges"].append({"lab": v["voter_lab"], "quarter": quarter, "kind": v.get("objection_kind") or "specification", "probability": v["probability"], "text": v["reason"]})
                if variant == "c" and v.get("test"):
                    c["tests"].append({"challenger": v["voter_lab"], "quarter": quarter, "status": "proposed", "specification": v["test"].get("specification", ""),
                                       "n_option": v["test"].get("n_option", "median"), "decision_rule": v["test"].get("decision_rule", "")})
                if c["status"] == "proposed" or (c["status"] == "endorsed" and variant in ("c",)):
                    self._set_status(c, "contested", quarter, "challenge")
            if v.get("stake"):
                c["stakes"].append({"lab": v["voter_lab"], "side": v["stake"]["side"], "amount": float(v["stake"]["amount"]), "quarter": quarter, "probability": v["probability"]})
        n_labs = len(st["labs"])
        for c in list(st["claims"].values()):
            if c.get("legacy") or c["status"] in ("endorsed", "superseded", "retracted", "lapsed", "withdrawn"):
                if c["status"] == "endorsed" and variant in ("b", "d"):
                    fails = [v for v in c["verifications"] if v["result"] == "not_reproduced"]
                    if fails and fails[-1]["quarter"] == quarter:
                        self._set_status(c, "contested", quarter, "later verification failed")
                continue
            labs_endorsing = {e["lab"] for e in c["endorsements"]}
            n_e, n_c = len(labs_endorsing), len({ch["lab"] for ch in c["challenges"]})
            if variant == "a":
                if n_e >= int(rc["endorsement_k"]) and n_e / max(1, n_e + n_c) >= float(rc["endorsement_share"]):
                    c["route"] = "vote"
                    c["confidence_certified"] = float(np.mean([e["probability"] for e in c["endorsements"]])) / 100.0
                    self._set_status(c, "endorsed", quarter, "endorsement threshold")
                elif n_c > 0 and c["status"] == "proposed":
                    self._set_status(c, "contested", quarter, "challenge")
            elif variant == "b":
                reps = [v for v in c["verifications"] if v["result"] == "reproduced"]
                fails = [v for v in c["verifications"] if v["result"] == "not_reproduced"]
                indep = {v["lab"] for v in reps}
                if len(fails) >= 2 and quarter - fails[0]["quarter"] >= 2 and not any(r_["quarter"] > fails[-1]["quarter"] for r_ in reps):
                    self._set_status(c, "retracted", quarter, "two failed reproductions")
                elif fails and c["status"] == "proposed":
                    self._set_status(c, "contested", quarter, "failed reproduction")
                elif len(indep) >= int(rc["reproductions_k"]) and (not fails or fails[-1]["quarter"] < reps[-1]["quarter"]):
                    c["route"] = "reproduction"
                    c["confidence_certified"] = max(0.5, 1 - 0.5 ** len(reps))
                    self._set_status(c, "endorsed", quarter, f"{len(indep)} independent reproductions")
            elif variant == "c":
                sil = rc["silence_endorsement"]
                in_view_n = sum(1 for lab in st["labs"].values() if self._in_view(c, lab, set(), quarter))
                age = quarter - c["quarter_filed"]
                if sil["enabled"] and c["status"] == "proposed" and n_c == 0 and in_view_n >= int(sil["labs_in_view"]) and age >= int(sil["quarters"]):
                    c["route"] = "silence"
                    c["confidence_certified"] = 0.6
                    self._set_status(c, "endorsed", quarter, "no challenge (silence)")
            elif variant == "d":
                price = self._price(c)
                c["price"] = price
                reps = [v for v in c["verifications"] if v["result"] == "reproduced"]
                fails = [v for v in c["verifications"] if v["result"] == "not_reproduced"]
                if c["verifications"] and c["verifications"][-1]["quarter"] == quarter:
                    self._settle_stakes(c, c["verifications"][-1]["result"] == "reproduced", quarter)
                if reps and price >= 0.6:
                    c["route"] = "market"
                    c["confidence_certified"] = price
                    self._set_status(c, "endorsed", quarter, "reproduction and price >= .6")
                elif fails and price < 0.4:
                    self._set_status(c, "retracted", quarter, "failed reproduction and price < .4")
                elif (reps or fails) and c["status"] == "proposed":
                    self._set_status(c, "contested", quarter, "mixed market evidence")
                elif quarter - c["quarter_filed"] >= 4 and c["status"] in ("proposed", "contested") and not c["verifications"]:
                    for s in c["stakes"]:
                        st["labs"][s["lab"]]["points"] += s["amount"]
                    self._set_status(c, "lapsed", quarter, "unresolved after four quarters; stakes returned")
            # lapse: no vote, verification or stake for `lapse_quarters`
            if c["status"] == "proposed":
                last_act = max([c["quarter_filed"]] + [e["quarter"] for e in c["endorsements"]] + [ch["quarter"] for ch in c["challenges"]]
                               + [v["quarter"] for v in c["verifications"]] + [s["quarter"] for s in c["stakes"]])
                if quarter - last_act >= int(rc["lapse_quarters"]):
                    self._set_status(c, "lapsed", quarter, "no action for three quarters")
            # contested claims under a: re-vote window of two quarters then stays contested (unsettled)
        # supersession: a newly endorsed claim that contradicts an earlier endorsed claim on the same relation
        for c in st["claims"].values():
            if c["status"] == "endorsed" and c.get("quarter_endorsed") == quarter:
                for link in c["links"]:
                    if link["type"] == "contradicts":
                        other = st["claims"].get(link["claim_id"])
                        if other and other["status"] in ("endorsed", "endorsed_legacy") and other.get("quarter_endorsed", 0) < quarter:
                            other["links"].append({"type": "superseded_by", "claim_id": c["claim_id"]})
                            self._set_status(other, "superseded", quarter, f"superseded by {c['claim_id']}")
        self._reputation(quarter)
        # verification debt and consensus bookkeeping
        open_claims = [c for c in st["claims"].values() if c["status"] in ("proposed", "contested") and not c.get("legacy")]
        offered = {o["claim_id"] for o in st["offers"]} | {v["lab"] for c in open_claims for v in c["verifications"]}
        self.w.ledger.add("events", {"quarter": quarter, "event": "verification_debt", "open_claims": len(open_claims),
                                     "open_without_offers": sum(1 for c in open_claims if c["claim_id"] not in offered and not c["verifications"])})

    def _price(self, c: dict) -> float:
        if not c["stakes"]:
            return 0.5
        num = sum(s["amount"] * (s.get("probability", 50) / 100.0 if s["side"] == "for" else 1 - s.get("probability", 50) / 100.0) for s in c["stakes"])
        return float(num / sum(s["amount"] for s in c["stakes"]))

    def _settle_stakes(self, c: dict, reproduced: bool, quarter: int) -> None:
        st = self.state
        winners = [s for s in c["stakes"] if (s["side"] == "for") == reproduced and not s.get("settled")]
        losers = [s for s in c["stakes"] if (s["side"] == "for") != reproduced and not s.get("settled")]
        pot = sum(s["amount"] for s in losers)
        wsum = sum(s["amount"] for s in winners)
        for s in winners:
            st["labs"][s["lab"]]["points"] += s["amount"] + (pot * s["amount"] / wsum if wsum else 0.0)
            s["settled"] = quarter
        for s in losers:
            s["settled"] = quarter
        self.w.ledger.add("events", {"quarter": quarter, "event": "stake_settlement", "claim_id": c["claim_id"], "reproduced": reproduced, "pot": pot,
                                     "claim_truth_check": c["hidden"].get("truth_check")})

    def _set_status(self, c: dict, status: str, quarter: int, event: str) -> None:
        if c["status"] == status:
            return
        prev = c["status"]
        c["status"] = status
        c["status_history"].append({"quarter": quarter, "status": status, "event": event})
        lab = self.state["labs"].get(c["proposer_lab"])
        if status == "endorsed":
            c["quarter_endorsed"] = quarter
            if lab:
                lab["record"]["endorsed"] += 1
            q = self.w.landscape.question(c["question_id"])
            if c["iv"] == q["iv_var"] and c["dv"] == q["dv_var"] and not c.get("moderator"):
                self.state["counts"][c["question_id"]] = self.state["counts"].get(c["question_id"], 0) + 1
        elif prev == "endorsed" and lab:
            lab["record"]["endorsed"] = max(0, lab["record"]["endorsed"] - 1)
        if status == "contested" and lab:
            lab["record"]["contested"] += 1
        if status == "retracted" and lab:
            lab["record"]["retracted"] += 1
        if status == "lapsed" and lab:
            lab["record"]["lapsed"] += 1
        self._ledger_claim(c, quarter, event)

    def _ledger_claim(self, c: dict, quarter: int, event: str) -> None:
        ev = c["evidence"][0] if c["evidence"] else {}
        self.w.ledger.add("claims", {"quarter": quarter, "claim_id": c["claim_id"], "question_id": c["question_id"], "core": c["core"], "status": c["status"], "event": event,
                                     "proposer_lab": c["proposer_lab"], "proposer_size": self.state["labs"].get(c["proposer_lab"], {}).get("size", 0), "leads": c["leads"],
                                     "sign": c["sign"], "kind": c["kind"], "iv": c["iv"], "dv": c["dv"], "moderator": c.get("moderator"), "estimate": ev.get("estimate"), "se": ev.get("se"),
                                     "p": ev.get("p"), "n": ev.get("n"), "deposit_ids": c["deposit_ids"], "n_endorsements": len(c["endorsements"]), "n_challenges": len(c["challenges"]),
                                     "n_verifications": len(c["verifications"]), "route": c.get("route"), "confidence_stated": c.get("confidence_stated"),
                                     "confidence_certified": c.get("confidence_certified"), "relation_to_prior": c.get("relation_to_prior"), "quarter_filed": c["quarter_filed"],
                                     "truth_check": c["hidden"].get("truth_check"), "theta": c["hidden"].get("theta"),
                                     "claim_true": c["hidden"].get("truth_check") in ("true_discovery", "correct_null", "true_discovery_negligible", "correct_null_negligible"),
                                     "legacy": c.get("legacy", False), "price": c.get("price")})

    def _reputation(self, quarter: int) -> None:
        """Baseline reputation formula, recomputed from scratch each quarter; two alternative formulas are offline sweeps."""
        st = self.state
        f = self.cfg["repository"]["reputation_formula"]
        w = {"baseline": (3, 1, 2, 1, 1, -2, -1), "proposal_heavy": (5, 0.5, 2, 1, 1, -2, -1), "verification_heavy": (2, 2, 3, 1, 1, -2, -1)}[f]
        pts = {lid: 0.0 for lid in st["labs"]}
        sci = {}
        for c in st["claims"].values():
            if c.get("legacy"):
                continue
            lab = c["proposer_lab"]
            if c["status"] == "endorsed":
                pts[lab] += w[0]
                for m in c["leads"]:
                    sci[m] = sci.get(m, 0.0) + w[0]
            if c["status"] == "retracted":
                pts[lab] += w[5]
            for v in c["verifications"]:
                if v["lab"] in pts:
                    pts[v["lab"]] += w[1]
                    sci[v["verifier"]] = sci.get(v["verifier"], 0.0) + w[1]
            if c["status"] in ("retracted", "superseded"):
                for e in c["endorsements"]:
                    if e["lab"] in pts:
                        pts[e["lab"]] += w[6]
                for ch in c["challenges"]:
                    if ch["lab"] in pts:
                        pts[ch["lab"]] += w[2]
        for repo in st["repos"].values():
            for d in repo["deposits"].values():
                reusers = {a["lab"] for a in st["analyses"].values() if d["deposit_id"] in a["deposits"] and a["lab"] != d["lab"]}
                pts[d["lab"]] = pts.get(d["lab"], 0.0) + w[3] * len(reusers)
                for corr in d.get("corrections", []):
                    pts[corr["lab"]] = pts.get(corr["lab"], 0.0) + w[4]
        for lid, L in st["labs"].items():
            L["reputation"] = pts.get(lid, 0.0)
        write_json(self.w.run_dir / "ledger" / "reputation.json", {"quarter": quarter, "labs": pts, "scientists": sci, "formula": f})

    # ======================================================================== phases 9-11
    def registry(self, quarter: int) -> None:
        st = self.state
        for lid, L in st["labs"].items():
            L["size"] = len(self.lab_members(lid))
            for m in self.lab_members(lid):
                rec = self.state["records"].setdefault(m["id"], {})
                rec["lab_id"] = lid
                rec["points"] = L["points"] if self.cfg["repository"]["variant"] == "d" else L["reputation"]
        self.check_promotion(quarter)
        write_json(self.w.run_dir / "ledger" / "registry_snapshot.json",
                   [{k: v for k, v in c.items() if k != "hidden"} | {"truth_check": c["hidden"].get("truth_check")} for c in st["claims"].values()])

    def score(self, quarter: int) -> None:
        roll = self.rollup(quarter)
        st = self.state
        cl = [c for c in st["claims"].values() if not c.get("legacy")]
        by_status = {}
        for c in cl:
            by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        endorsed_q = [c for c in cl if c.get("quarter_endorsed") == quarter]
        self.w.ledger.add("events", {"quarter": quarter, "event": "field_metrics", "analyses_started": sum(1 for a in st["analyses"].values() if a["quarter"] == quarter),
                                     "fresh_draws": sum(1 for a in st["analyses"].values() if a["quarter"] == quarter and a["fresh"]),
                                     "claims_filed": sum(1 for c in cl if c["quarter_filed"] == quarter), "claims_by_status": by_status,
                                     "endorsed_this_quarter": len(endorsed_q),
                                     "endorsed_true_focal": sum(1 for c in endorsed_q if c["hidden"].get("truth_check") in ("true_discovery", "correct_null", "true_discovery_negligible", "correct_null_negligible")),
                                     "verification_offers": len(st["offers"]), "votes": len(st["votes_this_quarter"]),
                                     "unverified_endorsements_share": (sum(1 for v in st["votes_this_quarter"] if v["vote"] == "endorse" and not v["voter_analysed_question"]) /
                                                                       max(1, sum(1 for v in st["votes_this_quarter"] if v["vote"] == "endorse"))),
                                     "n_covered_core": sum(1 for qid, v in roll["questions"].items() if v["core"] and v["n_run_units"] > 0),
                                     "budget_remaining": self.w.budget.remaining(quarter)})

    def handbook(self, quarter: int) -> None:
        self.handbook_snapshot(quarter)

    def finish(self) -> None:
        st = self.state
        # every claim entry (filed or dropped) as a hypothesis row with its final status (one row per unit)
        for c in st["claims"].values():
            ev = c["evidence"][0] if c["evidence"] else {}
            an = st["analyses"].get(c.get("analysis_id"), {})
            self.w.ledger.add("hypotheses", {"unit_id": c["claim_id"], "claim_id": c["claim_id"], "analysis_id": c.get("analysis_id"), "question_id": c["question_id"], "core": c["core"],
                                             "quarter": c["quarter_filed"], "author_id": (c["leads"] or [None])[0], "authors": c["leads"], "lab_id": c["proposer_lab"],
                                             "style": an.get("style", "legacy"), "throughput": an.get("throughput", "legacy"), "tier": "rule" if c.get("legacy") else self.scientist(c["leads"][0]).get("tier", "sonnet") if c["leads"] and c["leads"][0] in {s["id"] for s in self.w.scientists} else "rule",
                                             "iv": c["iv"], "dv": c["dv"], "moderator": c.get("moderator"), "kind": c["kind"], "predicted_sign": None, "estimate": ev.get("estimate"), "se": ev.get("se"),
                                             "p": ev.get("p"), "n": ev.get("n"), "estimator": ev.get("estimator"), "se_type": ev.get("se_type"), "controls": ev.get("controls"),
                                             "verdict": "supported" if c["sign"] != "none" else "not_supported", "partial_r": ev.get("partial_r"), "relation_to_prior": c.get("relation_to_prior"),
                                             "theta": c["hidden"].get("theta"), "theta_fe": c["hidden"].get("theta_fe"), "theta_naive": c["hidden"].get("theta_naive"),
                                             "truth_class": c["hidden"].get("truth_class"), "truth_check": c["hidden"].get("truth_check"), "traps": c["hidden"].get("traps"),
                                             "flags": c["hidden"].get("flags", []), "certified": c["status"] in ("endorsed", "endorsed_legacy"), "certified_quarter": c.get("quarter_endorsed"),
                                             "status": c["status"], "legacy": c.get("legacy", False), "deposit_ids": c["deposit_ids"], "route": c.get("route"),
                                             "attention": len(c["endorsements"])})
        for d in st["drafts"].values():
            for e in d["entries"]:
                if e.get("filed") in (None, "dropped"):
                    self.w.ledger.add("hypotheses", {"unit_id": f"{d['draft_id']}:{e['entry_id']}", "analysis_id": d["analysis_id"], "question_id": d["question_id"], "core": self.w.landscape.question(d["question_id"])["core"],
                                                     "quarter": d["quarter"], "author_id": d["lead"], "authors": [d["lead"]], "lab_id": d["lab"], "iv": e["iv"], "dv": e["dv"], "moderator": e.get("moderator"),
                                                     "kind": e.get("kind"), "estimate": e.get("estimate"), "se": e.get("se"), "p": e.get("p"), "n": e.get("n"), "verdict": e["verdict"],
                                                     "partial_r": e.get("partial_r"), "theta": e.get("theta"), "truth_class": e.get("truth_class"), "truth_check": e.get("truth_check"),
                                                     "traps": d["hidden"].get("traps"), "flags": d.get("flags", {}).get("flags", []), "certified": False, "certified_quarter": None,
                                                     "status": "never_filed" if e.get("filed") is None else "dropped", "legacy": False, "deposit_ids": d["deposits"], "attention": 0,
                                                     "style": self.scientist(d["lead"])["style"], "throughput": self.scientist(d["lead"])["throughput"], "tier": self.scientist(d["lead"]).get("tier", "sonnet")})
        self.registry(self.H)


def verification_batches(reqs: list[dict], size: int = 2) -> list[list[dict]]:
    """Verification calls carry up to `size` claims for ONE lab (its member is the verifier) and never the same claim twice
    (a 150-author mock run batched two labs' requests on one claim into one packet, so the second overwrote the first)."""
    by_lab: dict[str, list[dict]] = {}
    for r in reqs:
        by_lab.setdefault(r["lab"], []).append(r)
    out: list[list[dict]] = []
    for lab in sorted(by_lab):
        pending = list(by_lab[lab])
        while pending:
            batch, rest = [], []
            for r in pending:
                if len(batch) < size and all(x["claim_id"] != r["claim_id"] for x in batch):
                    batch.append(r)
                else:
                    rest.append(r)
            out.append(batch)
            pending = rest
    return out


def _truth_check_claim(sign: str, estimate: float, theta: float | None) -> str:
    if theta is None:
        return "unknown"
    if sign == "none":
        if theta == 0.0:
            return "correct_null"
        return "missed_effect" if abs(theta) >= 0.02 else "correct_null_negligible"
    s = 1 if sign == "positive" else -1
    if theta == 0.0 or np.sign(theta) != s:
        return "false_discovery"
    return "true_discovery" if abs(theta) >= 0.02 else "true_discovery_negligible"


def _theta_for_legacy(lp: dict, h: dict, q: dict) -> float:
    if h["iv"] == q["iv_var"] and h["dv"] == q["dv_var"] and not h.get("moderator"):
        return lp["hidden"]["theta"]
    if h.get("moderator"):
        return 0.0
    return lp["hidden"].get("theta_secondary", 0.0) if h["dv"] == q.get("secondary_dv_var") else 0.0
