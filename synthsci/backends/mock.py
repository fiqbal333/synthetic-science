"""Mock backend: runs the rule agents on real task packets, writes schema-valid outputs, validates and marks DONE.

It receives engine context (`ctx['task_ctx'][task.id]`) in memory; the planted-trap list reaches only the rule author,
which uses it to decide, with the parameter file's probability, whether its code handles each trap. Nothing hidden is written
into the task folder. The rule author's analysis.py is really executed, so the forensic re-run is exercised end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..agents.rules import RuleAuthor, RuleEditor, RuleHandbook, RuleLabHead, RuleReviewer, load_params, reproduction_result
from ..config import rng
from ..scorer import claim_sign as _claim_sign
from ..util import write_json, write_text
from .base import Backend


class MockBackend(Backend):
    name = "mock"
    executes_code = True

    def __init__(self, cfg, run_dir, store, params: dict | None = None):
        super().__init__(cfg, run_dir, store)
        self.params = params if params is not None else load_params(cfg)   # the warm start may pass its own rule-parameter file
        self.author = RuleAuthor(self.params)
        self.reviewer = RuleReviewer(self.params)
        self.editor = RuleEditor(self.params)
        self.lab = RuleLabHead(self.params)
        self.handbook = RuleHandbook(self.params)

    def run(self, tasks: list, ctx: dict) -> list:
        for t in tasks:
            if self.store.status(t) == "done":
                continue
            tctx = ctx["task_ctx"].get(t.id, {})
            r = rng(ctx["world"].seeds["run"], "mock", t.id)
            d = self.store.task_dir(t)
            out = d / "output"
            out.mkdir(exist_ok=True)
            handler = getattr(self, f"_do_{t.type}", None)
            if handler is None:
                raise NotImplementedError(f"mock backend: no handler for task type {t.type!r}")
            try:
                handler(t, tctx, ctx, d, out, r)
            except Exception as e:  # noqa: BLE001
                write_text(d / "mock_error.txt", f"{type(e).__name__}: {e}")
                self.store.mark(t, False, {"ok": False, "reason": f"mock error: {type(e).__name__}: {e}"[:300], "output_hashes": {}})
                continue
            rep = self.store.validate(t)
            self.store.mark(t, rep["ok"], rep)
        return tasks

    def usage(self, task) -> dict:
        return {"total_tokens": 0, "note": "mock"}

    # ------------------------------------------------------------------ paper regime
    def _do_research(self, t, tctx, ctx, d, out, r):
        a = tctx["author"]
        entries = tctx["entries"]
        entry = self.author.choose(entries, a, r)
        q = entry["question"]
        hid = tctx["hidden"][q["id"]]
        traps = hid["hidden"]["traps"]
        handled = self.author.decide_handling(traps, a["style"], r)
        use_fe = self.author.use_fe(q, a, r)
        variables = hid["meta"]["variables"]
        n_hyp = self.author.n_hypotheses(r)
        res = self.author.analyse_full(q, [f"../inputs/{q['id']}/data.csv"], d, a, handled, variables, use_fe, n_hyp, r,
                                       a["throughput"], traps)
        rel = self.author.relation_to_prior(res["hypotheses"][0]["estimate"] or 0.0, entry["view"], r)
        builds = [{"paper_id": v["paper_id"], "informed": ["H1"]} for v in entry["view"][:3]]
        claims = self.author.claims_json(q, res, rel, builds, nonce=t.nonce())
        write_json(out / "claims.json", claims)
        write_text(out / "paper.md", self.author.paper_md(q, res, rel, a, entry["view"], handled))

    def _do_research_batch(self, t, tctx, ctx, d, out, r):
        """Several authors' studies in one packet: each is done strictly inside its own folder (the rule author never
        touches another study's data), outputs go to output/<author id>/."""
        for j in tctx["jobs"]:
            a = j["author"]
            rr = rng(ctx["world"].seeds["run"], "mock", t.id, a["id"])
            entry = self.author.choose(j["entries"], a, rr)
            q = entry["question"]
            hid = j["hidden"][q["id"]]
            traps = hid["hidden"]["traps"]
            handled = self.author.decide_handling(traps, a["style"], rr)
            use_fe = self.author.use_fe(q, a, rr)
            n_hyp = self.author.n_hypotheses(rr)
            res = self.author.analyse_full(q, [f"../../authors/{a['id']}/inputs/{q['id']}/data.csv"], d, a, handled, hid["meta"]["variables"], use_fe,
                                           n_hyp, rr, a["throughput"], traps, out_subdir=f"output/{a['id']}")
            rel = self.author.relation_to_prior(res["hypotheses"][0]["estimate"] or 0.0, entry["view"], rr)
            builds = [{"paper_id": v["paper_id"], "informed": ["H1"]} for v in entry["view"][:3]]
            write_json(out / a["id"] / "claims.json", self.author.claims_json(q, res, rel, builds, nonce=t.nonce()))
            write_text(out / a["id"] / "paper.md", self.author.paper_md(q, res, rel, a, entry["view"], handled))

    def _do_revision(self, t, tctx, ctx, d, out, r):
        a, p = tctx["author"], tctx["paper"]
        regime = ctx["regime"]
        q = regime.w.landscape.question(p["question_id"])
        named = {i["kind"] for rv in p["reviews"]["1"] for i in rv["issues"] if i["severity"] == "major"}
        fix_p = 0.8 if a["style"] == "VERIFIED" else 0.3
        handled = dict(p["flags"]["trap_handled"])
        for trap, h in handled.items():
            if not h and named and r.uniform() < fix_p:
                handled[trap] = True
        traps = p["hidden"]["traps"]
        variables = {"controls": [c for c in q["controls_pool"] if c not in ("industry", "year")], "moderators": q["moderator_vars"],
                     "secondary_dv": q.get("secondary_dv_var")}
        use_fe = p["hypotheses"][0]["estimator"] == "fixed effects"
        res = self.author.analyse_full(q, ["../inputs/data/data.csv"], d, a, handled, variables, use_fe, len(p["hypotheses"]), r,
                                       a["throughput"], traps)
        view = [v for v in regime.view(q["id"]) if v["paper_id"] != p["paper_id"]]
        claims = self.author.claims_json(q, res, p["relation_to_prior"], p["builds_on"], nonce=t.nonce())
        write_json(out / "claims.json", claims)
        write_text(out / "paper.md", self.author.paper_md(q, res, p["relation_to_prior"], a, view, handled))
        write_text(out / "response.md", "# Response to the referees\n\n" + "\n".join(
            f"- Referee point ({k}): addressed in the revised analysis." for k in sorted(named)) + "\n- Editor: all required changes considered.\n")

    def _do_desk(self, t, tctx, ctx, d, out, r):
        decs = [self.editor.desk(item, tctx["strictness"], r) for item in tctx["items"]]
        write_json(out / "desk.json", {"nonce": t.nonce(), "decisions": decs})

    def _do_review(self, t, tctx, ctx, d, out, r):
        revs = [self.reviewer.review(item, r, tctx.get("assistance", "none"), tctx.get("batch_size", 1)) for item in tctx["items"]]
        write_json(out / "reviews.json", {"nonce": t.nonce(), "reviews": revs})

    def _do_prescreen(self, t, tctx, ctx, d, out, r):
        flags = []
        for item in tctx["items"]:
            probs = item.get("visible_problems", [])
            seen = [p for p in probs if r.uniform() < 0.6]
            flags.append({"paper_id": item["paper_id"], "flags": ("Possible issues: " + ", ".join(x.replace("_", " ") for x in seen)) if seen else "No specific flags."})
        write_json(out / "prescreen.json", {"nonce": t.nonce(), "flags": flags})

    def _do_decision(self, t, tctx, ctx, d, out, r):
        decs = [self.editor.decide({"paper_id": it["paper_id"], "relation_to_prior": it["relation_to_prior"]}, it["reviews"], it["round"], it["count"], r)
                for it in tctx["items"]]
        write_json(out / "decisions.json", {"nonce": t.nonce(), "decisions": decs})

    def _do_handbook(self, t, tctx, ctx, d, out, r):
        hb = self.handbook.write(tctx["rules"], tctx["theory_questions"], r, nonce=t.nonce(), predictions=tctx.get("predictions"))
        write_json(out / "handbook.json", hb)
        write_text(out / "state_of_the_field.md", "# State of the field\n\nA rule-generated placeholder: the panel's structured answers "
                   "are in handbook.json. In an LLM run this file carries the writer's 600-900 word prose.\n")

    def _do_prior_probe(self, t, tctx, ctx, d, out, r):
        qs = [{"question_id": qid, "answer": "no evidence", "confidence": 5.0, "n_read": 0, "key_units": [], "agree": True,
               "reasoning": "No literature was provided."} for qid in tctx["question_ids"]]
        write_json(out / "handbook.json", {"nonce": t.nonce(), "questions": qs, "theories": [], "open_questions": []})

    def _do_legacy_abstracts(self, t, tctx, ctx, d, out, r):
        write_json(out / "abstracts.json", {"nonce": t.nonce(), "abstracts": [{"paper_id": pid, "abstract": txt} for pid, txt in tctx["items"]]})

    # ------------------------------------------------------------------ repository regime
    def _do_lab_decision(self, t, tctx, ctx, d, out, r):
        dec = self.lab.decide(tctx["ctx"] | {"nonce": t.nonce()}, r)
        write_json(out / "decisions.json", dec)

    def _do_analyst_batch(self, t, tctx, ctx, d, out, r):
        for item in tctx["items"]:
            aid = item["analysis_id"]
            rr = rng(ctx["world"].seeds["run"], "mock", t.id, aid)
            self._do_analyst(t, item, ctx, d, out / aid, rr, data_prefix=f"../../analyses/{aid}/inputs", out_subdir=f"output/{aid}")

    def _do_analyst(self, t, tctx, ctx, d, out, r, data_prefix: str = "../inputs", out_subdir: str = "output"):
        lead = tctx["lead"]
        q = tctx["question"]
        traps = tctx["traps"]
        handled = self.author.decide_handling(traps, lead["style"], r)
        use_fe = self.author.use_fe(q, lead, r)
        variables = tctx["variables"]
        n_hyp = min(self.author.n_hypotheses(r), 8)
        res = self.author.analyse_full(q, [f"{data_prefix}/{dep}/data.csv" for dep in tctx["deposit_dirs"]], d, lead, handled, variables, use_fe,
                                       n_hyp, r, lead["throughput"], traps, out_subdir=out_subdir)
        rel = self.author.relation_to_prior(res["hypotheses"][0]["estimate"] or 0.0, tctx.get("view", []), r, first_word="first claim")
        entries = []
        for h in res["hypotheses"]:
            est = h["estimate"] or 0.0
            pr = abs(h.get("partial_r") or 0.0)
            band = "negligible" if pr < 0.02 else ("small" if pr < 0.05 else ("moderate" if pr < 0.10 else "large"))
            sign = _claim_sign(h)   # direction from the estimate and p, not from support of the prediction (a significant wrong-sign result is not "no relation")
            conf = float(np.clip(r.normal(70 if h["verdict"] == "supported" else 45, 10), 5, 95))
            stmt = (f"{h['iv']} has {'no reliable' if sign == 'none' else 'a ' + band + ' ' + sign} relation with {h['dv']}"
                    + (f" conditional on {h['moderator']}" if h.get("moderator") else ""))
            entries.append({"entry_id": f"E{h['hyp_id'][1:]}", "iv": h["iv"], "dv": h["dv"], "moderator": h.get("moderator"), "kind": h["kind"],
                            "statement": stmt, "sign": sign, "magnitude_band": band, "confidence": round(conf, 1), "model_id": h["model_id"],
                            "estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"], "estimator": h["estimator"], "se_type": h["se_type"],
                            "controls": h["controls"], "verdict": h["verdict"], "predicted_sign": h["predicted_sign"]})
        draft = {"nonce": t.nonce(), "question_id": q["id"], "deposits": tctx["deposit_versions"], "relation_to_prior": rel, "entries": entries,
                 "sample_changes": res["sample_changes"], "n_models_run": int(res["n_models_run"]), "robustness_models": res["robustness_models"],
                 "executed_code": res["executed_code"], "commands_run": int(res["commands_run"]), "correction": None}
        if any(handled.get(tp) for tp in ("duplicates", "scale_mixture", "miscoded_values", "unit_mismatch")) and r.uniform() < 0.5:
            draft["correction"] = {"script": "correction.py", "changelog": "Applied the sample corrections recorded in sample_changes to the deposit."}
            write_text(out / "correction.py", _correction_script(q, handled))
        write_json(out / "claims_draft.json", draft)
        write_text(out / "note.md", f"# Analysis note\n\nAnalysed {q['id']} on {', '.join(tctx['deposit_versions'])}. "
                   f"{len(entries)} relations examined; focal estimate {entries[0]['estimate']:.3f} (SE {entries[0]['se']:.3f}, p {entries[0]['p']:.3f}, N {entries[0]['n']}). "
                   f"Checks performed: {', '.join(res['sample_changes']) or 'none beyond listwise deletion'}. "
                   f"Limitations: proxies and unobserved heterogeneity.\n")

    def _do_verification(self, t, tctx, ctx, d, out, r):
        vers = []
        for item in tctx["items"]:
            lead = item["verifier"]
            q = item["question"]
            handled = self.author.decide_handling(item["traps"], lead["style"], r)
            use_fe = self.author.use_fe(q, lead, r)
            sub = d / "output" / item["claim_id"]
            sub.mkdir(parents=True, exist_ok=True)
            res = self.author.analyse_full(q, [f"../../inputs/{item['claim_id']}/data.csv"], d, lead, handled, item["variables"], use_fe, 1, r,
                                           lead["throughput"], item["traps"], out_subdir=f"output/{item['claim_id']}")
            (sub / "analysis.py").replace(sub / "own_analysis.py")   # replace, never rename: a second item on the same claim must not crash the call
            (sub / "table.csv").replace(sub / "own_table.csv")
            h = res["hypotheses"][0]
            result = reproduction_result(item["claim_estimate"], item.get("claim_p"), h["estimate"] or 0.0, h["se"] or 0.0,
                                         self.params["verifier"]["factor_tolerance"], claim_sign=item.get("claim_sign"))
            vers.append({"claim_id": item["claim_id"], "rerun": {"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"]},
                         "own": {"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"], "estimator": h["estimator"], "controls": h["controls"]},
                         "discrepancies": f"Claimed {item['claim_estimate']:.3f}; own estimate {(h['estimate'] or 0.0):.3f}.",
                         "result": result, "confidence": float(np.clip(r.normal(0.7, 0.1), 0.1, 0.95)), "sample_changes": res["sample_changes"],
                         "executed_code": res["executed_code"], "commands_run": 1})
        write_json(out / "verifications.json", {"nonce": t.nonce(), "verifications": vers})

    def _do_decisive_test(self, t, tctx, ctx, d, out, r):
        lead, q = tctx["executor"], tctx["question"]
        handled = self.author.decide_handling(tctx["traps"], lead["style"], r)
        res = self.author.analyse_full(q, ["../inputs/data/data.csv"], d, lead, handled, tctx["variables"], q["panel"], 1, r, "T0", tctx["traps"])
        h = res["hypotheses"][0]
        write_json(out / "test_result.json", {"nonce": t.nonce(), "claim_id": tctx["claim_id"],
                                              "focal": {"estimate": h["estimate"], "se": h["se"], "p": h["p"], "n": h["n"]},
                                              "notes": "Specification run as written.", "executed_code": res["executed_code"], "commands_run": 1})


def _correction_script(q: dict, handled: dict) -> str:
    L = ["import pandas as pd, numpy as np", 'df = pd.read_csv("data.csv")']
    if handled.get("duplicates"):
        L.append("df = df.drop_duplicates()")
    if handled.get("scale_mixture"):
        L.append(f'm = df["data_source"] == "B"; df.loc[m, "{q["dv_var"]}"] = df.loc[m, "{q["dv_var"]}"] / 100.0')
    if handled.get("miscoded_values"):
        L.append(f'mask = (df["{q["iv_var"]}"] >= 900)')
        L.append('if "leverage" in df.columns: mask |= (df["leverage"] >= 900)')
        L.append('if "firm_age" in df.columns: mask |= (df["firm_age"] < 0)')
        L.append("df = df[~mask]")
    if handled.get("unit_mismatch"):
        L.append(f'med = df.groupby("year")["{q["dv_var"]}"].median(); overall = med.median()')
        L.append(f'for yr, mv in med.items():\n    if abs(mv) > 50 * (abs(overall) + 0.05): df.loc[df["year"] == yr, "{q["dv_var"]}"] /= 1000.0')
    L.append('df.to_csv("data.csv", index=False)')
    return "\n".join(L) + "\n"
