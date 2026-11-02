"""The end-of-run handbook: de-identified input builder, writer tasks, panel aggregation, B4 scoring,
prior-probe packets."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .belief import is_correct, truth_class
from .orchestration.tasks import Task
from .regimes import texts
from .util import write_json, write_text


# ---------------------------------------------------------------------- de-identification
class Deidentifier:
    def __init__(self, landscape, deid: dict):
        self.L = landscape
        self.cmap, self.fmap = deid["constructs"], deid["families"]
        measures = landscape.measures["constructs"]
        pairs = []
        for construct, code in self.cmap.items():
            m = measures.get(construct)
            if m:
                pairs.append((m["label"], code))
                pairs.append((m["var"], code))
                if m.get("secondary"):
                    pairs.append((m["secondary"]["label"], code + "b"))
                    pairs.append((m["secondary"]["var"], code + "b"))
            pairs.append((construct.replace("_", " "), code))
            pairs.append((construct, code))
        for f in landscape.fringe:
            pairs.append((f["coined"], self.cmap.get(f["iv"], "C-F")))
            pairs.append((f["iv_var"], self.cmap.get(f["iv"], "C-F")))
        for fam, letter in self.fmap.items():
            pairs.append((landscape.families[fam]["label"], letter))
            pairs.append((texts.FAMILY_LABEL.get(fam, fam), letter))
            pairs.append((fam.replace("_", " "), letter))
            pairs.append((fam, letter))
        for c in landscape.measures["controls"]:
            pairs.append((c["label"], self.cmap.get(c["construct"], c["var"])))
            pairs.append((c["var"], self.cmap.get(c["construct"], c["var"])))
        pairs = [(a, b) for a, b in pairs if len(a) >= 3]
        pairs.sort(key=lambda x: -len(x[0]))
        self.lookup = {}
        for a, b in pairs:
            self.lookup.setdefault(a.lower(), b)
        # one alternation (longest first) and one pass: the per-pattern loop was the handbook builder's main cost
        # whole words only: a substring match turned "cross-section" into "cC-36bs-section" via the label "ros"
        self.rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a, _ in pairs) + r")(?![A-Za-z0-9])", re.I) if pairs else None

    def text(self, s: str) -> str:
        if self.rx is not None:
            s = self.rx.sub(lambda m: self.lookup.get(m.group(0).lower(), m.group(0)), s)
        return re.sub(r"\bthe view\b", "Theory", s)

    def prediction(self, q: dict) -> str:
        return f"{self.fmap[q['family']]} predicts that {self.cmap.get(q['iv'], 'C-F')} {'raises' if q['prediction_sign'] == 'positive' else 'lowers'} {self.cmap.get(q['dv'], '?')}."


# ---------------------------------------------------------------------- input builder
def build_handbook_input(regime, quarter: int) -> dict:
    L = regime.w.landscape
    deid = Deidentifier(L, regime.w.deid)
    # quoted abstracts reach the handbook packet de-identified; let the leak scan recognise them as agent-written text
    regime.w.store.fragment_transforms = [deid.text]
    core =L.questions + [f for f in L.fringe if f["id"] in regime.state["promoted"]]
    qlines = ["# Core questions", ""]
    theory_questions: dict[str, list[str]] = {}
    for q in core:
        qlines.append(f"- {q['id']} ({deid.fmap[q['family']]}): relation of {deid.cmap.get(q['iv'], 'C-F')} to {deid.cmap.get(q['dv'])}. {deid.prediction(q)}")
        theory_questions.setdefault(deid.fmap[q["family"]], []).append(q["id"])
    rules = {}
    roll_units = {}
    for q in core:
        units = regime.certified_units(q["id"])
        from .belief import all_rules
        rr = all_rules(units, one_per_deposit=regime.REGIME == "repository")
        rr["n_units"] = len(units)
        rules[q["id"]] = rr
        roll_units[q["id"]] = units
    md = [f"# Certified knowledge at quarter {quarter}", ""]
    js = {"quarter": quarter, "regime": regime.REGIME, "questions": []}
    if regime.REGIME == "paper":
        for q in core:
            pids = regime.state["literature"].get(q["id"], [])
            entries = []
            for pid in pids:
                p = regime.state["papers"][pid]
                if quarter < (p.get("quarter_published") or 0):
                    continue
                e = {"id": pid, "earlier_work": bool(p.get("legacy")), "quarter": p.get("quarter_published"), "citations": p["citations"],
                     "title": deid.text(p["title"]), "abstract": deid.text(p.get("abstract") or p.get("abstract_template") or ""),
                     "n": p.get("n"), "estimator": p.get("estimator"),
                     "hypotheses": [{"id": h["hyp_id"], "iv": deid.text(h["iv"]), "dv": deid.text(h["dv"]), "moderator": deid.text(h["moderator"]) if h.get("moderator") else None,
                                     "verdict": h["verdict"], "estimate": h.get("estimate"), "se": h.get("se"), "n": h.get("n")} for h in p["hypotheses"]]}
                entries.append(e)
            js["questions"].append({"question_id": q["id"], "theory": deid.fmap[q["family"]], "prediction": deid.prediction(q), "papers": entries})
            md += [f"## {q['id']} ({deid.fmap[q['family']]}): {deid.cmap.get(q['iv'], 'C-F')} and {deid.cmap.get(q['dv'])}", "", deid.prediction(q), ""]
            if not entries:
                md.append("No published papers on this question.")
            for e in entries:
                hy = "; ".join(f"{h['id']} {h['iv']}{' x ' + h['moderator'] if h['moderator'] else ''} -> {h['dv']}: {h['verdict'].replace('_', ' ')}"
                               + (f" (est {h['estimate']:.3f}, SE {h['se']:.3f}, N {h['n']})" if h.get("estimate") is not None else "") for h in e["hypotheses"])
                md.append(f"- **{e['id']}** ({'earlier work' if e['earlier_work'] else 'Q' + str(e['quarter'])}; cited {e['citations']}): *{e['title']}*. {e['abstract']} Hypotheses: {hy}.")
            md.append("")
    else:
        for q in core:
            cl = [c for c in regime.state["claims"].values() if c["question_id"] == q["id"] and c["status"] not in ("lapsed", "withdrawn")
                  and (c.get("quarter_filed") or 0) <= quarter]
            if regime.cfg["handbook"].get("endorsed_only"):
                cl = [c for c in cl if c["status"] in ("endorsed", "endorsed_legacy")]
            entries = []
            for c in cl:
                ev = c.get("evidence", [{}])[0] if c.get("evidence") else {}
                entries.append({"id": c["claim_id"], "earlier_work": c["status"] == "endorsed_legacy", "status": "endorsed" if c["status"] == "endorsed_legacy" else c["status"],
                                "statement": deid.text(c["statement"]), "quarter": c.get("quarter_filed"), "estimate": ev.get("estimate"), "se": ev.get("se"), "n": ev.get("n"),
                                "endorsements": len(c.get("endorsements", [])), "challenges": [ch["kind"] for ch in c.get("challenges", [])],
                                "verifications": [v["result"] for v in c.get("verifications", [])], "deposits": c.get("deposit_ids", []),
                                "note": deid.text(c.get("note", ""))[:600]})
            js["questions"].append({"question_id": q["id"], "theory": deid.fmap[q["family"]], "prediction": deid.prediction(q), "claims": entries})
            md += [f"## {q['id']} ({deid.fmap[q['family']]}): {deid.cmap.get(q['iv'], 'C-F')} and {deid.cmap.get(q['dv'])}", "", deid.prediction(q), ""]
            if not entries:
                md.append("No claims filed on this question.")
            for e in entries:
                md.append(f"- **{e['id']}** [{e['status']}{'; earlier work' if e['earlier_work'] else ''}]: {e['statement']}"
                          + (f" Evidence: est {e['estimate']:.3f}, SE {e['se']:.3f}, N {e['n']}." if e.get("estimate") is not None else "")
                          + f" Endorsements {e['endorsements']}; challenges {len(e['challenges'])}; verifications {', '.join(e['verifications']) or 'none'}; data {', '.join(e['deposits']) or 'earlier study'}.")
            md.append("")
    return {"json": js, "md": "\n".join(md), "questions_md": "\n".join(qlines), "rules": rules, "theory_questions": theory_questions, "deid": deid,
            "predictions": {q["id"]: q["prediction_sign"] for q in core}}   # the `agree` field's reference


def previous_snapshot(regime, quarter: int) -> dict | None:
    """The panel's answers at the latest earlier snapshot (handbook/Q<n>/panel.json: answers, theories, open questions; no
    truth, no scores), so that 'how knowledge has changed' is answerable from the packet (smoke-test finding 2026-10-14)."""
    root = regime.w.run_dir / "handbook"
    prev = sorted([int(d.name[1:]) for d in root.glob("Q*") if d.name[1:].lstrip("-").isdigit() and int(d.name[1:]) < quarter
                   and (d / "panel.json").exists()]) if root.exists() else []
    if not prev:
        return None
    q = prev[-1]
    from .util import read_json
    panel = read_json(root / f"Q{q}" / "panel.json")
    return {"snapshot_quarter": q, "quarters_ago": quarter - q,
            "answers": {k: {"answer": v["answer"], "confidence": v["confidence"]} for k, v in sorted(panel.get("answers", {}).items())},
            "theories": panel.get("theories", {}), "open_questions": sorted(set(panel.get("open_questions", [])))}


def writer_tasks(regime, quarter: int, inp: dict, hdir: Path) -> list[Task]:
    tasks = []
    P = regime.REGIME == "paper"
    prev = previous_snapshot(regime, quarter) if quarter > 0 else None
    change = (f", and how the knowledge has changed since the previous snapshot (`previous_handbook.json`: the panel's answers "
              f"{prev['quarters_ago']} quarter(s) ago, one entry per question with its confidence, the theories' standing and the open questions then)") if prev else ""
    for i, tier in enumerate(regime.cfg["handbook"]["panel"]["writers"]):
        inputs = {"questions.md": inp["questions_md"], ("literature.md" if P else "registry.md"): inp["md"]}
        if prev:
            inputs["previous_handbook.json"] = json.dumps(prev, indent=1)
        t = Task(type="handbook", quarter=quarter, identity=f"writer{i}", tier=tier, run_id=regime.w.task_ns, call_type="handbook",
                 template="handbook_writer.md",
                 variables={"KNOWLEDGE_WORD": "published literature" if P else "claim registry", "KNOWLEDGE_FILE": "literature.md" if P else "registry.md",
                            "UNIT_WORD": "paper" if P else "claim", "ATTENTION_CLAUSE": " and citation count" if P else ", status, endorsements, challenges and verifications",
                            "CHANGE_CLAUSE": change, "PREVIOUS_FILE_CLAUSE": " and `previous_handbook.json` (the previous snapshot's panel answers)" if prev else ""},
                 inputs=inputs,
                 outputs=["handbook.json", "state_of_the_field.md"], schema="handbook.schema.json", meta={"writer": i, "snapshot": quarter}, who=f"handbook_writer{i}")
        tasks.append(t)
    return tasks


def prior_probe_tasks(regime, inp: dict) -> list[Task]:
    tasks = []
    for i, tier in enumerate(regime.cfg["handbook"]["panel"]["writers"]):
        t = Task(type="prior_probe", quarter=0, identity=f"probe{i}", tier=tier, run_id=regime.w.task_ns, call_type="prior_probe",
                 template="prior_probe.md", inputs={"questions.md": inp["questions_md"]}, outputs=["handbook.json"], schema="handbook.schema.json",
                 meta={"writer": i}, who=f"prior_probe{i}")
        tasks.append(t)
    return tasks


# ---------------------------------------------------------------------- panel and scoring
def panel_aggregate(results: list[dict]) -> dict:
    answers: dict[str, dict] = {}
    qids = sorted({q["question_id"] for hb in results for q in hb["questions"]})
    for qid in qids:
        votes = [q for hb in results for q in hb["questions"] if q["question_id"] == qid]
        cnt = Counter(v["answer"] for v in votes)
        top, n = cnt.most_common(1)[0]
        if n == 1 and len(votes) > 1:
            top = "contested"
        answers[qid] = {"answer": top, "confidence": float(np.median([v["confidence"] for v in votes])), "votes": [v["answer"] for v in votes]}
    theories = {}
    for hb in results:
        for th in hb.get("theories", []):
            theories.setdefault(th["theory"], []).append(th["standing"])
    tpanel = {k: Counter(v).most_common(1)[0][0] for k, v in theories.items()}
    kappa = _fleiss_like_agreement([[v["answer"] for v in [q for q in hb["questions"]]] for hb in results]) if len(results) > 1 else None
    return {"answers": answers, "theories": tpanel, "n_writers": len(results), "agreement": kappa,
            "open_questions": [oq["question_id"] for hb in results for oq in hb.get("open_questions", [])]}


def _fleiss_like_agreement(cols: list[list[str]]) -> float | None:
    try:
        n = min(len(c) for c in cols)
        if n == 0:
            return None
        agree = sum(1 for i in range(n) if len({c[i] for c in cols}) == 1)
        return agree / n
    except Exception:  # noqa: BLE001
        return None


def score_handbook(panel: dict, results: list[dict], regime, quarter: int) -> dict:
    L = regime.w.landscape
    per_q = {}
    correct, brier, cw, n_cov = 0, [], 0, 0
    nulls = {"null": 0, "effect": 0, "contested": 0, "no evidence": 0, "n": 0}
    agree = {"B1": 0, "B2": 0, "B3": 0, "B5": 0, "n": 0}
    for qid, a in panel["answers"].items():
        try:
            tr = L.truth_of(qid)
        except KeyError:
            continue
        tc = truth_class(tr["theta"])
        ans = a["answer"]
        corr = None if ans in ("no evidence", "contested") else is_correct(ans, tc)
        conf = a["confidence"] / 100.0
        per_q[qid] = {"answer": ans, "confidence": a["confidence"], "truth_class": tc, "correct": corr}
        if ans != "no evidence":
            n_cov += 1
            if corr is not None:
                correct += int(corr)
                brier.append((1 - conf) ** 2 if corr else conf ** 2)
                if corr is False and conf >= 0.7:
                    cw += 1
        if tc == "null":
            nulls["n"] += 1
            nulls["null" if ans == "null" else ("effect" if ans in ("positive", "negative") else ans)] += 1
        units = regime.certified_units(qid)
        if units:
            from .belief import all_rules
            rr = all_rules(units, one_per_deposit=regime.REGIME == "repository")
            agree["n"] += 1
            for k in ("B1", "B2", "B3", "B5"):
                agree[k] += int(rr[k]["belief"] == ans)
    n_scored = sum(1 for v in per_q.values() if v["correct"] is not None)
    # the `agree` field: true when the writer's answer is the direction the theory predicts. Reported as the
    # share of answered questions agreeing with the prediction and the share of writers' agree fields consistent with it.
    pred = {q["id"]: q["prediction_sign"] for q in L.questions + L.fringe}
    agree_pred, agree_consistent, n_ag = 0, 0, 0
    for hb in results:
        for q in hb.get("questions", []):
            if q["question_id"] not in pred or q["answer"] in ("no evidence", "contested"):
                continue
            n_ag += 1
            is_pred = q["answer"] == pred[q["question_id"]]
            agree_pred += int(is_pred)
            if "agree" in q:
                agree_consistent += int(bool(q["agree"]) == is_pred)
    return {"quarter": quarter, "n_questions": len(per_q), "n_answered": n_cov, "n_scored": n_scored,
            "accuracy": (correct / n_scored) if n_scored else None, "correct_answers": correct,
            "brier": float(np.mean(brier)) if brier else None,
            "confidently_wrong": cw, "nulls": nulls, "agreement_with_rules": {k: (agree[k] / agree["n"]) if agree["n"] else None for k in ("B1", "B2", "B3", "B5")},
            "agree_with_prediction_share": (agree_pred / n_ag) if n_ag else None, "agree_field_consistent_share": (agree_consistent / n_ag) if n_ag else None,
            "inter_writer_agreement": panel.get("agreement"), "per_question": per_q,
            "per_writer_accuracy": [_writer_accuracy(hb, L) for hb in results]}


def _writer_accuracy(hb: dict, L) -> dict:
    c, n = 0, 0
    for q in hb["questions"]:
        try:
            tc = truth_class(L.truth_of(q["question_id"])["theta"])
        except KeyError:
            continue
        if q["answer"] in ("no evidence", "contested"):
            continue
        n += 1
        c += int(bool(is_correct(q["answer"], tc)))
    return {"tier": hb.get("tier"), "accuracy": (c / n) if n else None, "n": n}
