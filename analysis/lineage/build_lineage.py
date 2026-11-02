"""Knowledge-lineage datasets for the finished 20-quarter runs (descriptive only).

Paper world: citation network (legacy library, warm-start publications, run papers); edges from each study's
`builds_on` (state.json, the engine's paper record), realized citations from the `citation` events, reconciled with
papers_snapshot `cited_by`, ledger/citations.json and each task's output claims.json.
Repository world: evidence lineage of every claim (deposits, who drew them, analyses that reused them, verifications,
votes, links to earlier claims on the same relation).

Usage:
  set PYTHONDONTWRITEBYTECODE=1
  python analysis/lineage/build_lineage.py          (run folders: runs/, or set SYNTHSCI_RUNS)

READS ONLY runs/{S0-150,S0P-150,R-b-150}-x20/<seed>/llm/ (state/state.json, ledger/*, audit/audit.json, tasks/*/output/claims.json).
Imports synthsci.audit / synthsci.scorer for definitions only.
WRITES analysis/lineage/*.csv, analysis/lineage/headline_table.md, analysis/lineage/run_log.txt, analysis/lineage/figures/*.png
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
W = HERE.parents[1]
sys.path.insert(0, str(W))
# run folders: runs/ under the repository root, or the folder named by SYNTHSCI_RUNS (e.g. the unpacked run archive)
RUNS_ROOT = Path(os.environ.get("SYNTHSCI_RUNS") or (W / "runs"))
if not RUNS_ROOT.is_dir():
    sys.exit(f"STOP: run folder {RUNS_ROOT} not found; unpack the run archive there or set SYNTHSCI_RUNS to its location")

import numpy as np
import pandas as pd

from synthsci.audit import TRUE_CHECKS, _cert_q, _corrected_by, _dedupe, _is_focal, _stated_truth_check  # definitions only
from synthsci.scorer import claim_sign

OUT = HERE
FIG = HERE / "figures"
RUNS = [("S0-150", 1), ("S0-150", 2), ("S0-150", 3), ("S0P-150", 1), ("S0P-150", 2),
        ("R-b-150", 1), ("R-b-150", 2), ("R-b-150", 3)]
WORLDS = ["S0-150", "S0P-150", "R-b-150"]
WORLD_LABEL = {"S0-150": "Paper (S0)", "S0P-150": "Paper + career pressure (S0P)", "R-b-150": "Repository (R)"}
FALSE_CHECKS = {"false_discovery", "missed_effect"}
H = 20
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def rd(s, k):
    return RUNS_ROOT / f"{s}-x20" / str(k) / "llm"


def ld(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def div(a, b):
    return (a / b) if (a is not None and b) else None


def tclass(tc):
    if tc in TRUE_CHECKS:
        return "true"
    if tc in FALSE_CHECKS:
        return "false"
    return "unknown"


def stc(h):
    return h.get("truth_check_stated") or h.get("truth_check")


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return None
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top_share(x, frac=0.10):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    if len(x) == 0 or x.sum() == 0:
        return None
    k = max(1, int(math.ceil(frac * len(x))))
    return float(x[:k].sum() / x.sum())


def depth_map(nodes, out_edges):
    """Longest chain of references ending at each node: depth 0 = cites nothing in the graph; else 1 + max depth of what it cites."""
    memo = {}

    def d(n, stack=()):
        if n in memo:
            return memo[n]
        if n in stack:
            return 0
        targets = out_edges.get(n, [])
        v = 0 if not targets else 1 + max(d(t, stack + (n,)) for t in targets)
        memo[n] = v
        return v
    return {n: d(n) for n in nodes}


def load_hyps(d):
    hyps = _dedupe(jl(d / "ledger" / "hypotheses.jsonl"))
    for h in hyps:
        h["truth_check_stated"] = _stated_truth_check(h)
    return hyps


# ====================================================================================== paper world
def paper_run(key):
    s, k = key
    d = rd(s, k)
    S = ld(d / "state" / "state.json")
    log(f"{s} s{k}: state quarter {S.get('quarter')}, phase {S.get('phase')}")
    P = S["papers"]
    snap = {p["paper_id"]: p for p in ld(d / "ledger" / "papers_snapshot.json")}
    citj = ld(d / "ledger" / "citations.json")
    events = jl(d / "ledger" / "events.jsonl")
    hyps = load_hyps(d)
    audit = ld(d / "audit" / "audit.json")

    # focal row per paper (the certified row when there is one, else the last row written)
    focal = {}
    for h in hyps:
        if not _is_focal(h) or not h.get("paper_id"):
            continue
        pid = h["paper_id"]
        if pid not in focal or h.get("certified") or not focal[pid].get("certified"):
            if pid in focal and focal[pid].get("certified") and not h.get("certified"):
                continue
            focal[pid] = h
    units_all = [h for h in hyps if h.get("certified") and _is_focal(h)]
    run_units = [h for h in units_all if not h.get("legacy") and 1 <= _cert_q(h) <= H]
    if len(run_units) != g(audit, "field", "certification", "n_certified_units"):
        log(f"WARNING {key}: run units {len(run_units)} != audit {g(audit, 'field', 'certification', 'n_certified_units')}")
    live_an = [h for h in hyps if not h.get("legacy") and _is_focal(h) and not h.get("prehistory") and int(h.get("quarter") or 0) >= 1]

    def origin(p):
        if p.get("legacy") and str(p["paper_id"]).startswith("L") and not p.get("prehistory"):
            return "legacy_library"
        if p.get("legacy"):
            return "warmstart_published"
        if p.get("status") == "published" and (p.get("quarter_published") or 0) >= 1:
            return "run_published"
        return "not_published"

    nodes = {}
    for pid, p in P.items():
        f = focal.get(pid) or {}
        tc = stc(f) if f else None
        o = origin(p)
        n = {"world": s, "seed": k, "paper_id": pid, "origin": o, "inherited": o in ("legacy_library", "warmstart_published"),
             "question_id": p.get("question_id"), "core": p.get("core"), "status": p.get("status"),
             "prehistory_authored": bool(p.get("prehistory")), "quarter_started": p.get("quarter_started"),
             "quarter_published": p.get("quarter_published"), "lead": p.get("lead"), "authors": ";".join(p.get("authors") or []),
             "relation_to_prior": p.get("relation_to_prior"), "dataset_id": p.get("dataset_id"),
             "focal_truth_check_stated": tc, "truth": tclass(tc), "focal_sign": claim_sign(f) if f else None,
             "n_refs_made": len(p.get("builds_on") or []), "self_cites_engine": p.get("self_cites"),
             "citations_engine": p.get("citations", 0)}
        # overturn (a later certified unit on the question states another direction) and challenge (any live analysis does)
        if f and f.get("certified") and n["truth"] == "false":
            c = _corrected_by(f, units_all)
            n["overturned_quarter"] = _cert_q(c) if c else None
            fs = claim_sign(f)
            ch = [int(h["quarter"]) for h in live_an if h.get("question_id") == f.get("question_id") and claim_sign(h) != fs
                  and int(h["quarter"]) >= max(1, _cert_q(f))]
            n["challenged_quarter"] = min(ch) if ch else None
        nodes[pid] = n

    # ---------------- edges: builds_on of every study (intent); realized = the engine's citation event
    cit_events = [e for e in events if e.get("event") == "citation"]
    ev_set = collections.Counter((e["from"], e["to"]) for e in cit_events)
    ev_q = {(e["from"], e["to"]): e["quarter"] for e in cit_events}
    edges = []
    for pid, p in P.items():
        for b in p.get("builds_on") or []:
            t = b.get("paper_id")
            tp = P.get(t)
            src, tgt = nodes[pid], nodes.get(t, {})
            au_s, au_t = set(p.get("authors") or []), set((tp or {}).get("authors") or [])
            realized = (pid, t) in ev_set
            same_q = tp is not None and tp.get("question_id") == p.get("question_id")
            e = {"world": s, "seed": k, "from": pid, "to": t, "to_exists": tp is not None,
                 "from_origin": src["origin"], "from_status": p.get("status"), "from_live": not p.get("prehistory") and (p.get("quarter_started") or 0) >= 1,
                 "from_relation": p.get("relation_to_prior"), "from_quarter_started": p.get("quarter_started"),
                 "from_quarter_published": p.get("quarter_published"), "realized_citation": realized,
                 "citation_quarter": ev_q.get((pid, t)), "to_origin": tgt.get("origin"), "to_truth": tgt.get("truth"),
                 "to_truth_check": tgt.get("focal_truth_check_stated"), "to_relation": (tp or {}).get("relation_to_prior"),
                 "same_question": same_q, "self_citation": bool(au_s & au_t),
                 "agrees_with_cited": (same_q and src["focal_sign"] is not None and src["focal_sign"] == tgt.get("focal_sign")),
                 "informed": ";".join(b.get("informed") or [])}
            oq = tgt.get("overturned_quarter")
            for qname, qv in (("intent_quarter", p.get("quarter_started")), ("citation_quarter", ev_q.get((pid, t)))):
                if qv is None:
                    e[f"{qname}_vs_overturn"] = None
                elif tgt.get("truth") == "false" and tgt.get("inherited"):
                    e[f"{qname}_vs_overturn"] = "never_overturned" if oq is None else ("before" if qv < oq else "after")
                else:
                    e[f"{qname}_vs_overturn"] = None
            edges.append(e)
    E = pd.DataFrame(edges)

    # ---------------- reconciliation
    rec = {"world": s, "seed": k, "builds_on_refs": len(E), "builds_on_targets_missing": int((~E.to_exists).sum()) if len(E) else 0,
           "citation_events": len(cit_events), "citation_events_duplicated": sum(v - 1 for v in ev_set.values() if v > 1)}
    real_from_builds = set(zip(E[E.realized_citation]["from"], E[E.realized_citation]["to"])) if len(E) else set()
    exp_from_builds = set((r["from"], r["to"]) for _, r in E.iterrows() if r["from_status"] == "published"
                          and P.get(r["to"], {}).get("status") == "published") if len(E) else set()
    ev_pairs = set(ev_set)
    snap_pairs = set((c, pid) for pid, sp in snap.items() for c in (sp.get("cited_by") or []))
    state_pairs = set((c, pid) for pid, p in P.items() for c in (p.get("cited_by") or []))
    rec["realized_pairs_from_events"] = len(ev_pairs)
    rec["cited_by_pairs_snapshot"] = len(snap_pairs)
    rec["cited_by_pairs_state"] = len(state_pairs)
    rec["events_not_in_snapshot_cited_by"] = len(ev_pairs - snap_pairs)
    rec["snapshot_cited_by_not_in_events"] = len(snap_pairs - ev_pairs)
    rec["events_not_in_builds_on"] = len(ev_pairs - set(zip(E["from"], E["to"]))) if len(E) else len(ev_pairs)
    rec["published_builds_on_not_cited"] = len(exp_from_builds - ev_pairs)
    rec["citations_json_total"] = sum(citj.values())
    rec["snapshot_citations_total"] = sum((sp.get("citations") or 0) for sp in snap.values())
    rec["citations_json_vs_snapshot_mismatched_papers"] = sum(1 for pid, v in citj.items() if (snap.get(pid) or {}).get("citations") != v)
    rec["citations_count_vs_cited_by_len_mismatch"] = sum(1 for sp in snap.values() if (sp.get("citations") or 0) != len(sp.get("cited_by") or []))
    # engine self_cites (counted at registration on builds_on) vs author-overlap on builds_on edges
    sc_engine = sum((p.get("self_cites") or 0) for p in P.values() if not p.get("legacy") or p.get("prehistory"))
    rec["self_cites_engine_total"] = sc_engine
    rec["self_cites_from_edges_total"] = int(E.self_citation.sum()) if len(E) else 0
    # claims.json builds_on vs the engine record (last research/revision task of each study)
    n_ok = n_mis = n_nofile = 0
    for pid, p in P.items():
        tids = [t for t in (p.get("task_ids") or []) if t.startswith(("research", "revision"))]
        if not tids:
            continue
        t = tids[-1]
        cands = [d / "tasks" / t / "output" / "claims.json", d / "tasks" / t / "output" / str(p.get("lead")) / "claims.json"]
        f = next((c for c in cands if c.exists()), None)
        if f is None:
            n_nofile += 1
            continue
        cj = ld(f)
        a = sorted(b.get("paper_id") for b in (cj.get("builds_on") or []))
        bb = sorted(b.get("paper_id") for b in (p.get("builds_on") or []))
        if a == bb:
            n_ok += 1
        else:
            n_mis += 1
            if n_mis <= 5:
                log(f"  {key} builds_on mismatch {pid} task {t}: claims.json {a} vs engine {bb}")
    rec.update({"claims_json_builds_on_match": n_ok, "claims_json_builds_on_mismatch": n_mis, "claims_json_missing": n_nofile})
    rec["state_citations_vs_snapshot_mismatch"] = sum(1 for pid, p in P.items() if (snap.get(pid) or {}).get("citations") != p.get("citations"))

    # citations received per node
    N = pd.DataFrame(nodes.values())
    if len(E):
        rr = E[E.realized_citation]
        inrun = rr[rr.citation_quarter >= 1]
        live = E[E.from_live]
        N["citations_received_all"] = N.paper_id.map(rr.groupby("to").size()).fillna(0).astype(int)
        N["citations_received_in_run"] = N.paper_id.map(inrun.groupby("to").size()).fillna(0).astype(int)
        N["built_on_by_live_studies"] = N.paper_id.map(live.groupby("to").size()).fillna(0).astype(int)
        N["built_on_by_live_studies_agreeing"] = N.paper_id.map(live[live.agrees_with_cited].groupby("to").size()).fillna(0).astype(int)
    else:
        for c in ("citations_received_all", "citations_received_in_run", "built_on_by_live_studies", "built_on_by_live_studies_agreeing"):
            N[c] = 0
    if (N.citations_received_all != N.citations_engine).any():
        log(f"  {key}: {(N.citations_received_all != N.citations_engine).sum()} papers where event citations != engine count")
    # depth
    out_r = collections.defaultdict(list)
    out_i = collections.defaultdict(list)
    for _, e in E.iterrows():
        if e.realized_citation:
            out_r[e["from"]].append(e["to"])
        out_i[e["from"]].append(e["to"])
    dr = depth_map(list(nodes), out_r)
    di = depth_map(list(nodes), out_i)
    N["citation_depth"] = N.paper_id.map(dr)
    N["builds_on_depth"] = N.paper_id.map(di)

    # independence of evidence behind run-certified units (P: one dataset, one team per paper)
    ds_who = {x["dataset_id"]: x.get("who") for x in jl(d / "ledger" / "datasets.jsonl")}
    unit_rows = []
    for u in run_units:
        p = P.get(u["paper_id"], {})
        src = {p.get("dataset_id")} - {None}
        groups = {ds_who.get(x) for x in src} - {None}
        q = _cert_q(u)
        sign = claim_sign(u)
        au = set(p.get("authors") or [])
        corr = [v for v in units_all if v is not u and v.get("question_id") == u.get("question_id") and _cert_q(v) <= q
                and claim_sign(v) == sign and v.get("paper_id") != u.get("paper_id")
                and not (au & set(P.get(v.get("paper_id"), {}).get("authors") or []))]
        unit_rows.append({"world": s, "seed": k, "unit_id": u["unit_id"], "question_id": u.get("question_id"), "certified_quarter": q,
                          "truth": tclass(stc(u)), "n_data_sources": len(src), "n_data_groups": max(len(groups), len(src)),
                          "n_independent_corroborating_units_at_cert": len(corr)})
    return {"nodes": N, "edges": E, "rec": rec, "units": pd.DataFrame(unit_rows), "P": P, "run_units": run_units,
            "units_all": units_all}


def paper_summary(key, R):
    s, k = key
    N, E = R["nodes"], R["edges"]
    r = {"world": s, "seed": k, "regime": "paper"}
    pub = N[N.origin.isin(["legacy_library", "warmstart_published", "run_published"])]
    runp = N[N.origin == "run_published"]
    r["nodes_legacy_library"] = int((N.origin == "legacy_library").sum())
    r["nodes_warmstart_published"] = int((N.origin == "warmstart_published").sum())
    r["nodes_run_published"] = len(runp)
    r["nodes_run_published_live_authored"] = int((~runp.prehistory_authored).sum())
    r["studies_not_published"] = int((N.origin == "not_published").sum())
    rr = E[E.realized_citation] if len(E) else E
    inrun = rr[rr.citation_quarter >= 1] if len(rr) else rr
    r["citations_all"] = len(rr)
    r["citations_in_run"] = len(inrun)
    r["builds_on_refs_all_studies"] = len(E)
    live = E[E.from_live] if len(E) else E
    r["builds_on_refs_live_studies"] = len(live)
    r["refs_per_live_study"] = div(len(live), int(((~N.prehistory_authored) & (N.quarter_started.fillna(0) >= 1)).sum()))
    r["refs_made_per_run_published_paper"] = float(runp.n_refs_made.mean()) if len(runp) else None
    r["share_run_published_citing_anything"] = float((runp.n_refs_made > 0).mean()) if len(runp) else None
    r["citations_in_run_per_published_node"] = div(len(inrun), len(pub))
    for o in ("legacy_library", "warmstart_published", "run_published"):
        sub = pub[pub.origin == o]
        r[f"mean_citations_in_run_{o}"] = float(sub.citations_received_in_run.mean()) if len(sub) else None
    r["share_in_run_citations_to_inherited"] = div(int(inrun.to_origin.isin(["legacy_library", "warmstart_published"]).sum()), len(inrun))
    r["share_in_run_citations_to_legacy_library"] = div(int((inrun.to_origin == "legacy_library").sum()), len(inrun))
    r["share_in_run_citations_to_run_work"] = div(int((inrun.to_origin == "run_published").sum()), len(inrun))
    r["share_live_refs_to_inherited"] = div(int(live.to_origin.isin(["legacy_library", "warmstart_published"]).sum()), len(live))
    r["self_citation_share_in_run_citations"] = div(int(inrun.self_citation.sum()), len(inrun))
    r["self_citation_share_live_refs"] = div(int(live.self_citation.sum()), len(live))
    r["self_cites_engine_total"] = R["rec"]["self_cites_engine_total"]
    r["same_question_share_live_refs"] = div(int(live.same_question.sum()), len(live))
    # true vs false targets
    for tv in ("true", "false", "unknown"):
        sub = pub[pub.truth == tv]
        r[f"published_units_{tv}"] = len(sub)
        r[f"mean_citations_in_run_per_{tv}_unit"] = float(sub.citations_received_in_run.mean()) if len(sub) else None
        r[f"share_{tv}_units_cited_in_run"] = float((sub.citations_received_in_run > 0).mean()) if len(sub) else None
        r[f"mean_live_refs_per_{tv}_unit"] = float(sub.built_on_by_live_studies.mean()) if len(sub) else None
        r[f"share_in_run_citations_to_{tv}"] = div(int((inrun.to_truth == tv).sum()), len(inrun))
        r[f"share_live_refs_to_{tv}"] = div(int((live.to_truth == tv).sum()), len(live))
    # inherited false units: citations before vs after overturn (rates per unit-quarter of exposure)
    inh_f = pub[(pub.inherited) & (pub.truth == "false")]
    r["inherited_false_units"] = len(inh_f)
    r["inherited_false_overturned_by_Q20"] = int(inh_f.overturned_quarter.notna().sum()) if "overturned_quarter" in inh_f else 0
    fd = inh_f[inh_f.focal_truth_check_stated == "false_discovery"]
    r["inherited_false_discovery_units"] = len(fd)
    r["inherited_false_discovery_overturned_by_Q20"] = int(fd.overturned_quarter.notna().sum()) if "overturned_quarter" in fd else 0
    for kind, col, qcol in (("citations", "citation_quarter_vs_overturn", "citation_quarter"), ("live_refs", "intent_quarter_vs_overturn", "from_quarter_started")):
        src = inrun if kind == "citations" else live
        sub = src[src.to.isin(set(inh_f.paper_id))] if len(src) else src
        before = int((sub[col] == "before").sum()) if len(sub) else 0
        after = int((sub[col] == "after").sum()) if len(sub) else 0
        never = int((sub[col] == "never_overturned").sum()) if len(sub) else 0
        exp_b = exp_a = exp_n = 0
        for _, u in inh_f.iterrows():
            oq = u.get("overturned_quarter")
            if oq is None or (isinstance(oq, float) and math.isnan(oq)):
                exp_n += H
            else:
                exp_b += int(oq) - 1
                exp_a += H - int(oq) + 1
        r[f"inherited_false_{kind}_before_overturn"] = before
        r[f"inherited_false_{kind}_after_overturn"] = after
        r[f"inherited_false_{kind}_never_overturned"] = never
        r[f"inherited_false_{kind}_per_unit_quarter_before"] = div(before, exp_b)
        r[f"inherited_false_{kind}_per_unit_quarter_after"] = div(after, exp_a)
        r[f"inherited_false_{kind}_per_unit_quarter_never_overturned"] = div(never, exp_n)
        inh_t = pub[(pub.inherited) & (pub.truth == "true")]
        tsub = src[src.to.isin(set(inh_t.paper_id))] if len(src) else src
        r[f"inherited_true_{kind}_per_unit_quarter"] = div(len(tsub), len(inh_t) * H)
    # relation_to_prior
    for rel in ("first study", "replication", "extension", "contradiction"):
        r[f"in_run_citations_made_by_{rel.replace(' ', '_')}"] = int((inrun.from_relation == rel).sum()) if len(inrun) else 0
        sub = runp[runp.relation_to_prior == rel]
        r[f"run_papers_{rel.replace(' ', '_')}"] = len(sub)
        r[f"mean_citations_in_run_received_by_{rel.replace(' ', '_')}"] = float(sub.citations_received_in_run.mean()) if len(sub) else None
        r[f"mean_refs_made_by_{rel.replace(' ', '_')}_run_papers"] = float(sub.n_refs_made.mean()) if len(sub) else None
    # concentration
    r["gini_in_run_citations_published_nodes"] = gini(pub.citations_received_in_run)
    r["top10pct_share_in_run_citations"] = top_share(pub.citations_received_in_run)
    r["gini_live_refs_published_nodes"] = gini(pub.built_on_by_live_studies)
    r["top10pct_share_live_refs"] = top_share(pub.built_on_by_live_studies)
    r["share_published_nodes_never_cited_in_run"] = float((pub.citations_received_in_run == 0).mean()) if len(pub) else None
    # earliest finding on a question vs later ones (legacy library first, then by quarter published)
    pubo = pub.copy()
    pubo["order"] = np.where(pubo.origin == "legacy_library", -999, pubo.quarter_published.fillna(0))
    e_c, l_c, e_r, l_r = [], [], [], []
    for qid, sub in pubo.groupby("question_id"):
        if len(sub) < 2:
            continue
        m = sub.order.min()
        if (sub.order == m).all():
            continue
        e_c += list(sub[sub.order == m].citations_received_in_run)
        l_c += list(sub[sub.order > m].citations_received_in_run)
        e_r += list(sub[sub.order == m].built_on_by_live_studies)
        l_r += list(sub[sub.order > m].built_on_by_live_studies)
    r["questions_with_earliest_and_later"] = None if not e_c else len(set(pubo[pubo.question_id.isin(pubo.groupby("question_id").filter(lambda x: len(x) >= 2 and x.order.nunique() > 1).question_id)].question_id))
    r["mean_in_run_citations_earliest_cohort"] = float(np.mean(e_c)) if e_c else None
    r["mean_in_run_citations_later_papers"] = float(np.mean(l_c)) if l_c else None
    r["mean_live_refs_earliest_cohort"] = float(np.mean(e_r)) if e_r else None
    r["mean_live_refs_later_papers"] = float(np.mean(l_r)) if l_r else None
    # depth
    r["max_citation_depth"] = int(N.citation_depth.max())
    r["mean_citation_depth_run_published"] = float(runp.citation_depth.mean()) if len(runp) else None
    r["max_builds_on_depth"] = int(N.builds_on_depth.max())
    r["mean_builds_on_depth_live_studies"] = float(N[(~N.prehistory_authored) & (N.quarter_started.fillna(0) >= 1)].builds_on_depth.mean())
    return r


# ====================================================================================== repository world
def repo_run(key):
    s, k = key
    d = rd(s, k)
    REG = {c["claim_id"]: c for c in ld(d / "ledger" / "registry_snapshot.json")}
    hist = jl(d / "ledger" / "claims.jsonl")
    hyps = load_hyps(d)
    audit = ld(d / "audit" / "audit.json")
    DS = jl(d / "ledger" / "datasets.jsonl")
    AN = jl(d / "ledger" / "papers.jsonl")
    VE = jl(d / "ledger" / "verifications.jsonl")
    VO = jl(d / "ledger" / "votes.jsonl")
    who = {x["dataset_id"]: x.get("who") for x in DS}
    purpose = {x["dataset_id"]: x.get("purpose") for x in DS}
    dq = {x["dataset_id"]: x.get("quarter") for x in DS}
    hrow = {h["claim_id"]: h for h in hyps if h.get("claim_id")}
    units_all = [h for h in hyps if h.get("certified") and _is_focal(h)]
    run_units = [h for h in units_all if not h.get("legacy") and 1 <= _cert_q(h) <= H]
    if len(run_units) != g(audit, "field", "certification", "n_certified_units"):
        log(f"WARNING {key}: run units {len(run_units)} != audit {g(audit, 'field', 'certification', 'n_certified_units')}")
    live_an = [h for h in hyps if not h.get("legacy") and _is_focal(h) and int(h.get("quarter") or 0) >= 1]
    # status history
    ever = collections.defaultdict(set)
    first_q = {}
    for c in hist:
        ever[c["claim_id"]].add(c["status"])
        if c["status"] in ("superseded", "retracted"):
            first_q.setdefault((c["claim_id"], "removed"), c["quarter"])
    # analyses that used each deposit
    uses = collections.defaultdict(list)
    for a in AN:
        for dep in a.get("deposits") or []:
            uses[dep].append(a)
    # incoming references: later claims linking to an earlier claim on the same relation (links are written at filing)
    inc = collections.defaultdict(list)
    edges = []
    for cid, c in REG.items():
        for l in c.get("links") or []:
            if l["type"] not in ("replicates", "contradicts"):
                continue
            t = REG.get(l["claim_id"], {})
            e = {"world": s, "seed": k, "edge_type": "claim_references_claim", "from": cid, "to": l["claim_id"], "link": l["type"],
                 "quarter": c.get("quarter_filed"), "from_relation": c.get("relation_to_prior"), "from_lab": c.get("proposer_lab"),
                 "to_lab": t.get("proposer_lab"), "to_legacy": bool(t.get("legacy")),
                 "to_truth": tclass(stc(hrow.get(l["claim_id"], {})) or t.get("truth_check")),
                 "to_kind": t.get("kind"), "stated_relation_not_first": c.get("relation_to_prior") not in (None, "first claim"),
                 "to_certified_at_link": t.get("status") in ("endorsed_legacy", "superseded") and t.get("legacy")
                 or (t.get("quarter_endorsed") is not None and t["quarter_endorsed"] <= (c.get("quarter_filed") or 0))}
            edges.append(e)
            inc[l["claim_id"]].append(e)
        for dep in c.get("deposit_ids") or []:
            edges.append({"world": s, "seed": k, "edge_type": "claim_rests_on_deposit", "from": cid, "to": dep, "quarter": c.get("quarter_filed"),
                          "from_lab": c.get("proposer_lab"), "to_lab": who.get(dep), "deposit_purpose": purpose.get(dep)})
        for v in c.get("verifications") or []:
            edges.append({"world": s, "seed": k, "edge_type": "verification", "from": v.get("lab"), "to": cid, "quarter": v.get("quarter"),
                          "from_lab": v.get("lab"), "to_lab": c.get("proposer_lab"), "deposit": v.get("deposit_id"), "fresh": v.get("fresh"),
                          "result": v.get("result"), "powered": v.get("powered")})
    for v in VO:
        if v.get("vote") in ("endorse", "challenge"):
            edges.append({"world": s, "seed": k, "edge_type": f"vote_{v['vote']}", "from": v.get("voter_lab"), "to": v["claim_id"],
                          "quarter": v.get("quarter"), "from_lab": v.get("voter_lab"), "to_lab": v.get("proposer_lab")})
    for a in AN:
        for dep in a.get("deposits") or []:
            edges.append({"world": s, "seed": k, "edge_type": "analysis_uses_deposit", "from": a["analysis_id"], "to": dep, "quarter": a.get("quarter"),
                          "from_lab": a.get("lab"), "to_lab": who.get(dep), "fresh": a.get("fresh"), "pooled": a.get("pooled")})
    E = pd.DataFrame(edges)
    votes_by = collections.defaultdict(lambda: collections.Counter())
    vlabs = collections.defaultdict(lambda: collections.defaultdict(set))
    for v in VO:
        votes_by[v["claim_id"]][v.get("vote")] += 1
        vlabs[v["claim_id"]][v.get("vote")].add(v.get("voter_lab"))

    rows = []
    for cid, c in REG.items():
        h = hrow.get(cid, {})
        tc = stc(h) if h else c.get("truth_check")
        deps = list(c.get("deposit_ids") or [])
        vers = c.get("verifications") or []
        rep = [v for v in vers if v.get("result") == "reproduced"]
        supp_deps = set(deps) | {v.get("deposit_id") for v in rep if v.get("deposit_id")}
        data_labs = {who.get(x) for x in supp_deps} - {None}
        an = next((a for a in AN if a["analysis_id"] == c.get("analysis_id")), {})
        later_uses = [a for dep in deps for a in uses.get(dep, []) if a["analysis_id"] != c.get("analysis_id")
                      and a.get("quarter", 0) > (c.get("quarter_filed") or 0)]
        incl = inc.get(cid, [])
        sign = c.get("sign")
        focal = _is_focal(h) if h else (c.get("kind") == "focal")
        row = {"world": s, "seed": k, "claim_id": cid, "question_id": c.get("question_id"), "kind": c.get("kind"), "focal": focal,
               "legacy": bool(c.get("legacy")), "quarter_filed": c.get("quarter_filed"), "status": c.get("status"),
               "certified_now": c.get("status") in ("endorsed", "endorsed_legacy"), "ever_endorsed": "endorsed" in ever[cid] or bool(c.get("legacy")),
               "quarter_endorsed": c.get("quarter_endorsed"), "route": c.get("route"), "relation_to_prior": c.get("relation_to_prior"),
               "sign": sign, "truth_check_stated": tc, "truth": tclass(tc), "proposer_lab": c.get("proposer_lab"),
               "analysis_id": c.get("analysis_id"), "analysis_fresh": an.get("fresh"), "pooled": len(deps) > 1,
               "deposit_ids": ";".join(deps), "deposit_origin_labs": ";".join(sorted({who.get(x) or "?" for x in deps})),
               "rests_on_other_labs_data": any(who.get(x) not in (None, c.get("proposer_lab")) for x in deps),
               "n_verifications": len(vers), "n_verifications_fresh": sum(1 for v in vers if v.get("fresh")),
               "n_reproduced": len(rep), "n_not_reproduced": sum(1 for v in vers if v.get("result") == "not_reproduced"),
               "verifier_labs": ";".join(sorted({v.get("lab") for v in vers})),
               "reproducing_labs_fresh_data": ";".join(sorted({v.get("lab") for v in rep if v.get("fresh")})),
               "n_endorse_votes": votes_by[cid]["endorse"], "n_challenge_votes": votes_by[cid]["challenge"],
               "n_endorsing_labs": len(vlabs[cid]["endorse"]), "n_challenging_labs": len(vlabs[cid]["challenge"]),
               "n_verification_offers": votes_by[cid]["offer"],
               "n_data_labs_supporting": len(data_labs), "data_labs_supporting": ";".join(sorted(data_labs)),
               "n_data_labs_independent_of_proposer": len(data_labs - {c.get("proposer_lab")}),
               "n_deposits_supporting": len(supp_deps),
               "n_later_analyses_reusing_its_deposits": len(later_uses),
               "n_later_claims_referencing": len(incl), "n_later_claims_referencing_stated": sum(1 for e in incl if e["stated_relation_not_first"]),
               "n_later_claims_replicating": sum(1 for e in incl if e["link"] == "replicates"),
               "n_later_claims_contradicting": sum(1 for e in incl if e["link"] == "contradicts"),
               "n_links_out": sum(1 for l in c.get("links") or [] if l["type"] in ("replicates", "contradicts"))}
        # a superseded legacy claim lost its certified flag; keep it in the inherited set (as compute_results.py does)
        if h and (h.get("certified") or (c.get("legacy") and c.get("status") == "superseded")) and focal and row["truth"] == "false":
            cb = _corrected_by(h, units_all)
            oq = [x for x in (_cert_q(cb) if cb else None, first_q.get((cid, "removed"))) if x is not None]
            row["overturned_quarter"] = min(oq) if oq else None
            ch = [int(x["quarter"]) for x in live_an if x.get("question_id") == h.get("question_id") and claim_sign(x) != claim_sign(h)
                  and int(x["quarter"]) >= max(1, _cert_q(h))]
            row["challenged_quarter"] = min(ch) if ch else None
        # corroboration at certification (focal run units): another certified focal unit, same question & direction,
        # different proposer lab and no shared deposit (legacy claims count as independent earlier work)
        if h and h.get("certified") and focal and not h.get("legacy"):
            q = _cert_q(h)
            corr = []
            for v in units_all:
                if v is h or v.get("question_id") != h.get("question_id") or _cert_q(v) > q or claim_sign(v) != claim_sign(h):
                    continue
                vc = REG.get(v.get("claim_id"), {})
                if vc.get("proposer_lab") == c.get("proposer_lab") or (set(vc.get("deposit_ids") or []) & set(deps)):
                    continue
                corr.append(v)
            row["n_independent_corroborating_units_at_cert"] = len(corr)
        rows.append(row)
    C = pd.DataFrame(rows)
    # depth over claim references
    out = collections.defaultdict(list)
    for e in edges:
        if e["edge_type"] == "claim_references_claim":
            out[e["from"]].append(e["to"])
    dm = depth_map(list(REG), out)
    C["reference_depth"] = C.claim_id.map(dm)
    # deposits table
    drows = []
    for x in DS:
        dep = x["dataset_id"]
        us = uses.get(dep, [])
        drows.append({"world": s, "seed": k, "deposit_id": dep, "question_id": x.get("question_id"), "drawn_by_lab": x.get("who"),
                      "purpose": x.get("purpose"), "quarter": x.get("quarter"), "n_rows": g(x, "meta", "n_rows"),
                      "n_analyses_using": len(us), "n_analyses_other_labs": sum(1 for a in us if a.get("lab") != x.get("who")),
                      "n_labs_using": len({a.get("lab") for a in us}),
                      "n_claims_resting_on": sum(1 for c in REG.values() if dep in (c.get("deposit_ids") or []))})
    D = pd.DataFrame(drows)
    rec = {"world": s, "seed": k, "registry_claims": len(REG), "claims_ledger_ids": len({c["claim_id"] for c in hist}),
           "claims_ledger_ids_missing_from_registry": len({c["claim_id"] for c in hist} - set(REG)),
           "registry_deposits_missing_from_datasets_ledger": int(sum(1 for c in REG.values() for x in (c.get("deposit_ids") or []) if x not in who)),
           "verifications_ledger": len(VE), "verifications_in_registry": int(sum(len(c.get("verifications") or []) for c in REG.values())),
           "endorse_votes_ledger": sum(1 for v in VO if v.get("vote") == "endorse"),
           "endorsements_in_registry": int(sum(len(c.get("endorsements") or []) for c in REG.values())),
           "status_final_ledger_vs_registry_mismatch": 0}
    lastst = {}
    for c in hist:
        lastst[c["claim_id"]] = c["status"]
    rec["status_final_ledger_vs_registry_mismatch"] = sum(1 for cid, st_ in lastst.items() if REG.get(cid, {}).get("status") != st_)
    return {"claims": C, "edges": E, "deposits": D, "analyses": AN, "rec": rec, "run_units": run_units}


def repo_summary(key, R):
    s, k = key
    C, D, AN, E = R["claims"], R["deposits"], R["analyses"], R["edges"]
    r = {"world": s, "seed": k, "regime": "repository"}
    live = C[~C.legacy]
    lf = live[live.focal]
    cert = lf[lf.status == "endorsed"]
    cert_all = live[live.status == "endorsed"]
    r["claims_filed"] = len(live)
    r["claims_filed_focal"] = len(lf)
    r["certified_focal_run_units"] = len(cert)
    r["certified_all_kinds_run"] = len(cert_all)
    r["share_certified_focal_2plus_data_labs"] = float((cert.n_data_labs_supporting >= 2).mean()) if len(cert) else None
    r["share_certified_all_kinds_2plus_data_labs"] = float((cert_all.n_data_labs_supporting >= 2).mean()) if len(cert_all) else None
    r["mean_data_labs_per_certified_focal"] = float(cert.n_data_labs_supporting.mean()) if len(cert) else None
    r["share_certified_focal_2plus_labs_indep_of_proposer"] = float((cert.n_data_labs_independent_of_proposer >= 2).mean()) if len(cert) else None
    r["share_certified_focal_corroborated_at_cert"] = float((cert.n_independent_corroborating_units_at_cert >= 1).mean()) if len(cert) else None
    r["share_certified_focal_on_reused_data"] = float((cert.analysis_fresh == False).mean()) if len(cert) else None  # noqa: E712
    r["share_certified_focal_on_other_labs_data"] = float(cert.rests_on_other_labs_data.mean()) if len(cert) else None
    # data reuse
    an = [a for a in AN if int(a.get("quarter") or 0) >= 1]
    r["analyses"] = len(an)
    r["share_analyses_reusing_deposit"] = div(sum(1 for a in an if not a.get("fresh")), len(an))
    whom = dict(zip(D.deposit_id, D.drawn_by_lab))
    r["share_analyses_using_other_labs_deposit"] = div(sum(1 for a in an if any(whom.get(x) != a.get("lab") for x in a.get("deposits") or [])), len(an))
    r["share_analyses_pooled"] = div(sum(1 for a in an if a.get("pooled")), len(an))
    ad = D[D.purpose == "analysis"]
    r["deposits_analysis"] = len(ad)
    r["deposits_verification"] = int((D.purpose == "verification").sum())
    r["deposits_decisive_test"] = int((D.purpose == "decisive_test").sum())
    r["share_analysis_deposits_reused"] = float((ad.n_analyses_using >= 2).mean()) if len(ad) else None
    r["share_analysis_deposits_used_by_other_lab"] = float((ad.n_analyses_other_labs >= 1).mean()) if len(ad) else None
    r["mean_analyses_per_analysis_deposit"] = float(ad.n_analyses_using.mean()) if len(ad) else None
    r["share_verification_deposits_reused_in_analyses"] = float((D[D.purpose == "verification"].n_analyses_using >= 1).mean()) if (D.purpose == "verification").any() else None
    # building on prior claims
    for rel in ("first claim", "replication", "extension", "contradiction"):
        r[f"claims_{rel.replace(' ', '_')}"] = int((live.relation_to_prior == rel).sum())
    r["share_claims_linked_to_earlier_claim"] = float((live.n_links_out > 0).mean()) if len(live) else None
    r["share_focal_claims_linked_to_earlier_claim"] = float((lf.n_links_out > 0).mean()) if len(lf) else None
    cr = E[(E.edge_type == "claim_references_claim")] if len(E) else E
    crs = cr[cr.stated_relation_not_first] if len(cr) else cr
    r["claim_references_all"] = len(cr)
    r["claim_references_stated"] = len(crs)
    r["share_stated_refs_to_inherited"] = div(int(crs.to_legacy.sum()), len(crs))
    r["share_stated_refs_to_own_lab"] = div(int((crs.from_lab == crs.to_lab).sum()), len(crs))
    r["share_stated_refs_to_false"] = div(int((crs.to_truth == "false").sum()), len(crs))
    r["share_stated_refs_to_true"] = div(int((crs.to_truth == "true").sum()), len(crs))
    r["max_reference_depth"] = int(C.reference_depth.max())
    r["mean_reference_depth_live_claims"] = float(live.reference_depth.mean()) if len(live) else None
    r["mean_reference_depth_live_focal"] = float(lf.reference_depth.mean()) if len(lf) else None
    # true vs false claims filed (focal)
    for tv in ("true", "false", "unknown"):
        sub = lf[lf.truth == tv]
        r[f"filed_focal_{tv}"] = len(sub)
        r[f"share_{tv}_focal_verified"] = float((sub.n_verifications > 0).mean()) if len(sub) else None
        r[f"share_{tv}_focal_reproduced_at_least_once"] = float((sub.n_reproduced > 0).mean()) if len(sub) else None
        r[f"share_{tv}_focal_ever_endorsed"] = float(sub.ever_endorsed.mean()) if len(sub) else None
        r[f"share_{tv}_focal_certified_at_Q20"] = float((sub.status == "endorsed").mean()) if len(sub) else None
        r[f"share_{tv}_focal_with_endorse_vote"] = float((sub.n_endorse_votes > 0).mean()) if len(sub) else None
        r[f"share_{tv}_focal_with_challenge_vote"] = float((sub.n_challenge_votes > 0).mean()) if len(sub) else None
        r[f"share_{tv}_focal_referenced_by_later_claim"] = float((sub.n_later_claims_referencing > 0).mean()) if len(sub) else None
        r[f"share_{tv}_focal_deposits_reused_later"] = float((sub.n_later_analyses_reusing_its_deposits > 0).mean()) if len(sub) else None
        r[f"mean_endorse_votes_{tv}_focal"] = float(sub.n_endorse_votes.mean()) if len(sub) else None
    # inherited false legacy claims: references before / after overturn
    leg = C[C.legacy & C.focal]
    lf_ = leg[leg.truth == "false"]
    lt_ = leg[leg.truth == "true"]
    ce = E[E.edge_type == "claim_references_claim"] if len(E) else E
    r["inherited_false_units"] = len(lf_)
    r["inherited_false_overturned_by_Q20"] = int(lf_.overturned_quarter.notna().sum()) if "overturned_quarter" in lf_ else 0
    fd = lf_[lf_.truth_check_stated == "false_discovery"]
    r["inherited_false_discovery_units"] = len(fd)
    r["inherited_false_discovery_overturned_by_Q20"] = int(fd.overturned_quarter.notna().sum()) if "overturned_quarter" in fd else 0
    for kind, flt in (("refs", None), ("refs_stated", "stated_relation_not_first")):
        sub = ce[ce.to.isin(set(lf_.claim_id))]
        if flt:
            sub = sub[sub[flt]]
        b = a_ = nv = 0
        exp_b = exp_a = exp_n = 0
        oqm = dict(zip(lf_.claim_id, lf_.get("overturned_quarter", pd.Series([None] * len(lf_)))))
        for _, e in sub.iterrows():
            oq = oqm.get(e["to"])
            if oq is None or (isinstance(oq, float) and math.isnan(oq)):
                nv += 1
            elif e["quarter"] < oq:
                b += 1
            else:
                a_ += 1
        for cid, oq in oqm.items():
            if oq is None or (isinstance(oq, float) and math.isnan(oq)):
                exp_n += H
            else:
                exp_b += int(oq) - 1
                exp_a += H - int(oq) + 1
        r[f"inherited_false_{kind}_before_overturn"] = b
        r[f"inherited_false_{kind}_after_overturn"] = a_
        r[f"inherited_false_{kind}_never_overturned"] = nv
        r[f"inherited_false_{kind}_per_unit_quarter_before"] = div(b, exp_b)
        r[f"inherited_false_{kind}_per_unit_quarter_after"] = div(a_, exp_a)
        r[f"inherited_false_{kind}_per_unit_quarter_never_overturned"] = div(nv, exp_n)
        tsub = ce[ce.to.isin(set(lt_.claim_id))]
        if flt:
            tsub = tsub[tsub[flt]]
        r[f"inherited_true_{kind}_per_unit_quarter"] = div(len(tsub), len(lt_) * H)
    # concentration of attention (endorse votes; references received) across certified focal units (inherited + run)
    cf = C[C.focal & C.certified_now]
    r["gini_endorse_votes_certified_focal_run"] = gini(cert.n_endorse_votes)
    r["gini_refs_received_certified_focal_all"] = gini(cf.n_later_claims_referencing)
    r["top10pct_share_refs_received_certified_focal_all"] = top_share(cf.n_later_claims_referencing)
    return r


# ====================================================================================== cross-world measures
def crossworld(key, R, regime):
    """Comparable definitions (see the note): computed on the same unit definitions in both worlds."""
    s, k = key
    r = {"world": s, "seed": k}
    if regime == "paper":
        U, N = R["units"], R["nodes"]
        r["run_certified_units"] = len(U)
        r["I1_share_units_2plus_independent_data_groups"] = float((U.n_data_groups >= 2).mean()) if len(U) else None
        r["I1_mean_data_groups_per_unit"] = float(U.n_data_groups.mean()) if len(U) else None
        r["I2_share_units_corroborated_at_cert"] = float((U.n_independent_corroborating_units_at_cert >= 1).mean()) if len(U) else None
        lib = N[(N.origin == "legacy_library")]
        runp = N[N.origin == "run_published"]
        sets = {"legacy_library": lib, "run_certified": runp}
        bcol, acol = "built_on_by_live_studies", "built_on_by_live_studies_agreeing"
    else:
        C = R["claims"]
        cert = C[(~C.legacy) & C.focal & (C.status == "endorsed")]
        r["run_certified_units"] = len(cert)
        r["I1_share_units_2plus_independent_data_groups"] = float((cert.n_data_labs_supporting >= 2).mean()) if len(cert) else None
        r["I1_mean_data_groups_per_unit"] = float(cert.n_data_labs_supporting.mean()) if len(cert) else None
        r["I2_share_units_corroborated_at_cert"] = float((cert.n_independent_corroborating_units_at_cert >= 1).mean()) if len(cert) else None
        lib = C[C.legacy & C.focal]
        sets = {"legacy_library": lib, "run_certified": cert}
        bcol, acol = "n_later_claims_referencing_stated", "n_later_claims_replicating"
    for name, sub in sets.items():
        for tv in ("true", "false"):
            x = sub[sub.truth == tv]
            r[f"{name}_{tv}_units"] = len(x)
            r[f"B_{name}_share_{tv}_built_on"] = float((x[bcol] > 0).mean()) if len(x) else None
            r[f"B_{name}_mean_refs_per_{tv}_unit"] = float(x[bcol].mean()) if len(x) else None
            r[f"B_{name}_share_{tv}_affirmed"] = float((x[acol] > 0).mean()) if len(x) else None
        f_, t_ = r.get(f"B_{name}_share_false_built_on"), r.get(f"B_{name}_share_true_built_on")
        r[f"B_{name}_false_to_true_ratio_built_on"] = div(f_, t_)
    return r


# ====================================================================================== run
PN, PE, PU, RC, RE, RDp, SUMS, XW, RECS = [], [], [], [], [], [], [], [], []
TS = []
CACHE = {}
for key in RUNS:
    if key[0].startswith("S0"):
        R = paper_run(key)
        CACHE[key] = R
        PN.append(R["nodes"]); PE.append(R["edges"]); PU.append(R["units"]); RECS.append(R["rec"])
        SUMS.append(paper_summary(key, R))
        XW.append(crossworld(key, R, "paper"))
        E = R["edges"]
        for q in range(1, H + 1):
            rr = E[(E.realized_citation) & (E.citation_quarter == q)]
            lv = E[(E.from_live) & (E.from_quarter_started == q)]
            TS.append({"world": key[0], "seed": key[1], "quarter": q,
                       "citations_to_true": int((rr.to_truth == "true").sum()), "citations_to_false": int((rr.to_truth == "false").sum()),
                       "citations_to_unknown": int((rr.to_truth == "unknown").sum()),
                       "refs_to_true": int((lv.to_truth == "true").sum()), "refs_to_false": int((lv.to_truth == "false").sum())})
    else:
        R = repo_run(key)
        CACHE[key] = R
        RC.append(R["claims"]); RE.append(R["edges"]); RDp.append(R["deposits"]); RECS.append(R["rec"])
        SUMS.append(repo_summary(key, R))
        XW.append(crossworld(key, R, "repository"))
        E = R["edges"]
        ce = E[E.edge_type == "claim_references_claim"]
        for q in range(1, H + 1):
            lv = ce[(ce.quarter == q) & (ce.stated_relation_not_first) & (ce.to_kind == "focal")]
            TS.append({"world": key[0], "seed": key[1], "quarter": q, "citations_to_true": None, "citations_to_false": None,
                       "citations_to_unknown": None,
                       "refs_to_true": int((lv.to_truth == "true").sum()), "refs_to_false": int((lv.to_truth == "false").sum())})
    log(f"done {key}")

pd.concat(PN).to_csv(OUT / "paper_nodes.csv", index=False)
pd.concat(PE).to_csv(OUT / "paper_edges.csv", index=False)
pd.concat(PU).to_csv(OUT / "paper_certified_units_independence.csv", index=False)
pd.concat(RC).to_csv(OUT / "repo_claims_lineage.csv", index=False)
pd.concat(RE).to_csv(OUT / "repo_edges.csv", index=False)
pd.concat(RDp).to_csv(OUT / "repo_deposits.csv", index=False)
pd.DataFrame(RECS).to_csv(OUT / "reconciliation.csv", index=False)
TSD = pd.DataFrame(TS)
TSD.to_csv(OUT / "references_by_quarter.csv", index=False)
SUM = pd.DataFrame(SUMS)
SUM.to_csv(OUT / "lineage_by_run.csv", index=False)
XWD = pd.DataFrame(XW)
XWD.to_csv(OUT / "crossworld_by_run.csv", index=False)


def world_summary(df, id_cols=("world", "seed", "regime")):
    out = []
    num = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    for w, sub in df.groupby("world", sort=False):
        for c in num:
            v = [float(x) for x in sub[c] if x is not None and not (isinstance(x, float) and math.isnan(x))]
            out.append({"world": w, "metric": c, "n_seeds": len(v), "mean": float(np.mean(v)) if v else None,
                        "min": min(v) if v else None, "max": max(v) if v else None,
                        "per_seed": "; ".join(f"s{int(s_)}={'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)}"
                                              for s_, x in zip(sub["seed"], sub[c]))})
    return pd.DataFrame(out)


WSUM = world_summary(SUM)
WSUM.to_csv(OUT / "lineage_by_world.csv", index=False)
WXW = world_summary(XWD)
WXW.to_csv(OUT / "crossworld_by_world.csv", index=False)


def cell(df, metric, world, nd=3):
    r = df[(df.metric == metric) & (df.world == world)]
    if r.empty or r["n_seeds"].iloc[0] == 0:
        return "n/a"
    m, lo, hi = r["mean"].iloc[0], r["min"].iloc[0], r["max"].iloc[0]
    f = (lambda x: f"{x:.{nd}f}") if nd else (lambda x: f"{x:.1f}" if not float(x).is_integer() else f"{int(x)}")
    return f"{f(m)} ({f(lo)} to {f(hi)})"


HEAD_X = [
    ("Run-certified focal units (n)", "run_certified_units", 0),
    ("I1 share of run-certified units resting on data from 2+ independent groups", "I1_share_units_2plus_independent_data_groups", 3),
    ("I1 mean independent data groups per run-certified unit", "I1_mean_data_groups_per_unit", 2),
    ("I2 share of run-certified units corroborated at certification", "I2_share_units_corroborated_at_cert", 3),
    ("Legacy-library focal units, false (n)", "legacy_library_false_units", 0),
    ("Legacy-library focal units, true (n)", "legacy_library_true_units", 0),
    ("B share of false legacy units built on by a later live study", "B_legacy_library_share_false_built_on", 3),
    ("B share of true legacy units built on by a later live study", "B_legacy_library_share_true_built_on", 3),
    ("B mean later references per false legacy unit", "B_legacy_library_mean_refs_per_false_unit", 2),
    ("B mean later references per true legacy unit", "B_legacy_library_mean_refs_per_true_unit", 2),
    ("B share of false legacy units affirmed (later study, same direction)", "B_legacy_library_share_false_affirmed", 3),
    ("B share of true legacy units affirmed", "B_legacy_library_share_true_affirmed", 3),
    ("Run-certified false units (n)", "run_certified_false_units", 0),
    ("B share of false run-certified units built on later", "B_run_certified_share_false_built_on", 3),
    ("B share of true run-certified units built on later", "B_run_certified_share_true_built_on", 3),
]
md = ["| Measure | " + " | ".join(WORLD_LABEL[w] for w in WORLDS) + " |", "|---|---|---|---|"]
for lab, m, nd in HEAD_X:
    md.append(f"| {lab} | " + " | ".join(cell(WXW, m, w, nd) for w in WORLDS) + " |")
HEAD_P = [
    ("Paper: citations realized in run (published citing paper, Q1-20)", "citations_in_run", 0),
    ("Paper: builds_on references by live studies (any outcome)", "builds_on_refs_live_studies", 0),
    ("Paper: references per live study", "refs_per_live_study", 2),
    ("Paper: share of in-run citations to inherited work", "share_in_run_citations_to_inherited", 3),
    ("Paper: self-citation share of in-run citations", "self_citation_share_in_run_citations", 3),
    ("Paper: mean in-run citations per true published unit", "mean_citations_in_run_per_true_unit", 3),
    ("Paper: mean in-run citations per false published unit", "mean_citations_in_run_per_false_unit", 3),
    ("Paper: Gini of in-run citations (published nodes)", "gini_in_run_citations_published_nodes", 3),
    ("Paper: share of in-run citations to top 10% of papers", "top10pct_share_in_run_citations", 3),
    ("Paper: share of published papers never cited in run", "share_published_nodes_never_cited_in_run", 3),
    ("Paper: mean in-run citations, earliest finding on question", "mean_in_run_citations_earliest_cohort", 3),
    ("Paper: mean in-run citations, later findings on question", "mean_in_run_citations_later_papers", 3),
    ("Paper: live refs per inherited false unit-quarter, before overturn", "inherited_false_live_refs_per_unit_quarter_before", 4),
    ("Paper: live refs per inherited false unit-quarter, after overturn", "inherited_false_live_refs_per_unit_quarter_after", 4),
    ("Paper: live refs per inherited false unit-quarter, never overturned", "inherited_false_live_refs_per_unit_quarter_never_overturned", 4),
    ("Paper: live refs per inherited true unit-quarter", "inherited_true_live_refs_per_unit_quarter", 4),
    ("Paper: share of live references to false units", "share_live_refs_to_false", 3),
    ("Paper: share of live references to the citing team's own work", "self_citation_share_live_refs", 3),
    ("Paper: max citation chain depth", "max_citation_depth", 0),
    ("Paper: max builds_on chain depth (all studies)", "max_builds_on_depth", 0),
    ("Paper: mean builds_on chain depth, live studies", "mean_builds_on_depth_live_studies", 2),
]
HEAD_R = [
    ("Repository: claims filed (all kinds)", "claims_filed", 0),
    ("Repository: share of analyses reusing a deposit", "share_analyses_reusing_deposit", 3),
    ("Repository: share of analyses using another lab's deposit", "share_analyses_using_other_labs_deposit", 3),
    ("Repository: share of analyses pooling two deposits", "share_analyses_pooled", 3),
    ("Repository: share of analysis deposits used by another lab", "share_analysis_deposits_used_by_other_lab", 3),
    ("Repository: share of certified focal claims on reused data", "share_certified_focal_on_reused_data", 3),
    ("Repository: share of certified (all kinds) on 2+ data labs", "share_certified_all_kinds_2plus_data_labs", 3),
    ("Repository: share of focal claims linked to an earlier claim", "share_focal_claims_linked_to_earlier_claim", 3),
    ("Repository: false focal claims filed (n)", "filed_focal_false", 0),
    ("Repository: share false focal verified", "share_false_focal_verified", 3),
    ("Repository: share false focal reproduced at least once", "share_false_focal_reproduced_at_least_once", 3),
    ("Repository: share false focal ever endorsed", "share_false_focal_ever_endorsed", 3),
    ("Repository: share false focal with an endorse vote", "share_false_focal_with_endorse_vote", 3),
    ("Repository: share false focal referenced by a later claim", "share_false_focal_referenced_by_later_claim", 3),
    ("Repository: share true focal ever endorsed", "share_true_focal_ever_endorsed", 3),
    ("Repository: share true focal referenced by a later claim", "share_true_focal_referenced_by_later_claim", 3),
    ("Repository: stated refs per inherited false unit-quarter, before overturn", "inherited_false_refs_stated_per_unit_quarter_before", 4),
    ("Repository: stated refs per inherited false unit-quarter, after overturn", "inherited_false_refs_stated_per_unit_quarter_after", 4),
    ("Repository: stated refs per inherited false unit-quarter, never overturned", "inherited_false_refs_stated_per_unit_quarter_never_overturned", 4),
    ("Repository: stated refs per inherited true unit-quarter", "inherited_true_refs_stated_per_unit_quarter", 4),
    ("Repository: claim-to-claim references with a stated relation (n)", "claim_references_stated", 0),
    ("Repository: share of those references to inherited claims", "share_stated_refs_to_inherited", 3),
    ("Repository: share of those references to the referencing lab's own claims", "share_stated_refs_to_own_lab", 3),
    ("Repository: share of those references to false claims", "share_stated_refs_to_false", 3),
    ("Repository: max reference chain depth", "max_reference_depth", 0),
    ("Repository: mean reference chain depth, live focal claims", "mean_reference_depth_live_focal", 2),
]
HEAD_BOTH = [
    ("Both: inherited false focal units, false discovery + missed effect (n)", "inherited_false_units", 0),
    ("Both:   of which overturned by Q20", "inherited_false_overturned_by_Q20", 0),
    ("Both: inherited false discoveries only (n; the x20 note's set)", "inherited_false_discovery_units", 0),
    ("Both:   of which overturned by Q20", "inherited_false_discovery_overturned_by_Q20", 0),
]
md2 = ["| Measure | " + " | ".join(WORLD_LABEL[w] for w in WORLDS) + " |", "|---|---|---|---|"]
for lab, m, nd in HEAD_BOTH + HEAD_P + HEAD_R:
    md2.append(f"| {lab} | " + " | ".join(cell(WSUM, m, w, nd) for w in WORLDS) + " |")
(OUT / "headline_table.md").write_text(
    "Mean (min to max) across seeds; S0 n=3, S0P n=2, R n=3. Generated by build_lineage.py.\n\n## Cross-world\n\n" + "\n".join(md)
    + "\n\n## Within-world\n\n" + "\n".join(md2) + "\n", encoding="utf-8")

# ====================================================================================== figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

fam = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
log("figure font:", fam)
FIG.mkdir(parents=True, exist_ok=True)
COLW = {"S0-150": "#2a78d6", "S0P-150": "#eb6834", "R-b-150": "#1baf7a"}
MKW = {"S0-150": "o", "S0P-150": "s", "R-b-150": "^"}
COLT = {"true": "#2a78d6", "false": "#eb6834", "unknown": "#a3a29b"}
MKT = {"true": "o", "false": "X", "unknown": "s"}
GRID = dict(color="#e3e3e0", lw=0.7)

# --- Figure 1: citation network, S0 seed 1 (x = quarter published, y = question; edges = realized citations)
R = CACHE[("S0-150", 1)]
N, E = R["nodes"], R["edges"]
rr = E[E.realized_citation]
inv = set(rr["from"]) | set(rr["to"])
sub = N[N.paper_id.isin(inv)].copy()
sub["x"] = np.where(sub.origin == "legacy_library", -19, sub.quarter_published.fillna(0))
qs = sorted(sub.question_id.unique(), key=lambda q: (sub[sub.question_id == q].x.min(), q))
qy = {q: i for i, q in enumerate(qs)}
sub["y"] = sub.question_id.map(qy)
# jitter papers sharing a cell
sub["y"] = sub["y"] + sub.groupby(["question_id", "x"]).cumcount() * 0.22
pos = dict(zip(sub.paper_id, zip(sub.x, sub.y)))
fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.28 * len(qs) + 1.4)))
for _, e in rr.iterrows():
    if e["from"] in pos and e["to"] in pos:
        (x0, y0), (x1, y1) = pos[e["from"]], pos[e["to"]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#8a8984", lw=0.7, alpha=0.7,
                                                                     shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=0.15"))
for tv in ("true", "false", "unknown"):
    for inh, face in ((True, "white"), (False, None)):
        z = sub[(sub.truth == tv) & (sub.inherited == inh)]
        if len(z):
            ax.scatter(z.x, z.y, s=46, marker=MKT[tv], facecolors=face if face else COLT[tv], edgecolors=COLT[tv], linewidths=1.4, zorder=3)
ax.set_yticks(range(len(qs)))
ax.set_yticklabels(qs, fontsize=7)
ax.set_xticks([-19, -15, -10, -5, 0, 5, 10, 15, 20])
ax.set_xticklabels(["legacy\nlibrary", "-15", "-10", "-5", "0", "5", "10", "15", "20"])
ax.axvline(0.5, color="#c3c2b7", lw=0.8, ls="--")
ax.text(0.7, len(qs) - 0.4, "live run starts", fontsize=7, color="#52514e", va="top")
ax.set_xlabel("Quarter published (warm-start pre-history < 1)")
ax.set_title("Citation network, paper world S0 seed 1: realized citations (arrow: citing -> cited)", loc="left")
ax.grid(axis="x", **GRID)
hand = [Line2D([], [], marker=MKT[t], ls="", markerfacecolor=COLT[t], markeredgecolor=COLT[t], ms=7, label=f"{t} focal finding") for t in ("true", "false", "unknown")]
hand += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#52514e", ms=7, label="inherited (open)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="#52514e", markeredgecolor="#52514e", ms=7, label="run-published (filled)")]
ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig1_citation_network_S0_s1.png", dpi=200)
plt.close(fig)

# --- Figure 2: references to true vs false units by quarter (mean per seed), three worlds
fig, axs = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
for ax, w in zip(axs, WORLDS):
    t = TSD[TSD.world == w].groupby("quarter")[["refs_to_true", "refs_to_false"]].mean()
    ax.plot(t.index, t.refs_to_true, color=COLT["true"], lw=2, marker="o", ms=4, label="to true units")
    ax.plot(t.index, t.refs_to_false, color=COLT["false"], lw=2, marker="X", ms=5, label="to false units")
    ax.set_title(WORLD_LABEL[w], loc="left")
    ax.set_xlabel("Quarter")
    ax.grid(axis="y", **GRID)
    ax.set_xticks([1, 5, 10, 15, 20])
axs[0].set_ylabel("References made that quarter (mean per seed)")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Building on true vs false findings over time. Paper: builds_on references by live studies (by start quarter);\n"
             "repository: later claims linked to an earlier focal claim on the same relation, stated relation not 'first claim'",
             x=0.01, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIG / "fig2_references_true_vs_false_by_quarter.png", dpi=200)
plt.close(fig)

# --- Figure 3: evidence independence by world (I1, I2), dots per seed
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
for ax, (m, title) in zip(axs, (("I1_share_units_2plus_independent_data_groups", "I1: data from 2+ independent groups"),
                                ("I2_share_units_corroborated_at_cert", "I2: corroborated at certification"))):
    for i, w in enumerate(WORLDS):
        v = XWD[XWD.world == w][m].astype(float)
        ax.bar(i, v.mean(), width=0.6, color=COLW[w], alpha=0.35, edgecolor=COLW[w])
        ax.scatter([i] * len(v), v, color=COLW[w], marker=MKW[w], s=40, zorder=3, edgecolors="white", linewidths=1)
        ax.text(i + 0.33, v.mean(), f"{v.mean():.2f}", ha="left", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Paper\n(S0)", "Paper +\npressure (S0P)", "Repository\n(R)"])
    ax.set_ylim(0, 1.1)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", **GRID)
axs[0].set_ylabel("Share of run-certified focal units")
fig.suptitle("Independence of evidence behind certified knowledge (bar = mean across seeds, markers = seeds)", x=0.01, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "fig3_evidence_independence_by_world.png", dpi=200)
plt.close(fig)

# --- Figure 4: share of false vs true legacy-library units built on later, by world
fig, ax = plt.subplots(figsize=(6.4, 3.4))
xs = np.arange(3)
for j, tv in enumerate(("true", "false")):
    m = f"B_legacy_library_share_{tv}_built_on"
    for i, w in enumerate(WORLDS):
        v = XWD[XWD.world == w][m].astype(float)
        xx = i + (j - 0.5) * 0.36
        ax.bar(xx, v.mean(), width=0.32, color=COLT[tv], alpha=0.85 if tv == "true" else 0.85, edgecolor="white", linewidth=2,
               label=f"{tv} units" if i == 0 else None)
        ax.scatter([xx] * len(v), v, color="#0b0b0b", s=10, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels(["Paper (S0)", "Paper + pressure (S0P)", "Repository (R)"])
ax.set_ylabel("Share built on by a later live study")
ax.set_ylim(0, 1)
ax.grid(axis="y", **GRID)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Inherited legacy-library findings built on during Q1-20, by truth (dots = seeds)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig4_legacy_findings_built_on_true_vs_false.png", dpi=200)
plt.close(fig)

# --- Figure 5: repository evidence lineage — certified focal claims by number of data-generating labs, per seed
C = pd.concat(RC)
cc = C[(~C.legacy) & C.focal & (C.status == "endorsed")]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
seeds = sorted(cc.seed.unique())
maxl = int(cc.n_data_labs_supporting.max()) if len(cc) else 1
minl = int(cc.n_data_labs_supporting.min()) if len(cc) else 1
bins = list(range(minl, maxl + 1))
blues = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
left = np.zeros(len(seeds))
for b_i, b in enumerate(bins):
    vals = np.array([((cc.seed == s_) & (cc.n_data_labs_supporting == b)).sum() for s_ in seeds], dtype=float)
    ax.barh([f"R seed {s_}" for s_ in seeds], vals, left=left, color=blues[min(b_i, len(blues) - 1)], edgecolor="white", linewidth=2,
            label=f"{b} lab{'s' if b > 1 else ''}")
    for yi, (l0, v) in enumerate(zip(left, vals)):
        if v >= 1:
            ax.text(l0 + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=8, color="#0b0b0b" if b_i < 2 else "white")
    left += vals
ax.set_xlabel("Certified focal claims (run)")
ax.set_title("Repository: labs whose data stand behind each certified claim\n(claim's deposits + fresh reproducing verifications)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), title="data-generating labs", title_fontsize=8)
ax.grid(axis="x", **GRID)
fig.tight_layout()
fig.savefig(FIG / "fig5_repository_data_labs_per_certified_claim.png", dpi=200)
plt.close(fig)

(OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("written to", OUT)
