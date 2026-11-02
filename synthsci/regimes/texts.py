"""Rule-generated agent-visible texts: record cards, lab cards, menu entries, literature views, registry views,
review digests. Everything here passes the strict leak scan."""
from __future__ import annotations

STAGE_LABEL = {"doctoral": "Doctoral student", "assistant": "Assistant professor", "associate": "Associate professor", "full": "Full professor"}
FAMILY_LABEL = {"agency": "agency theory", "rbv": "the resource-based view", "tce": "transaction cost economics",
                "org_learning": "organizational learning", "institutional": "institutional theory", "networks": "social network theory",
                "upper_echelons": "upper echelons theory", "dynamic_capabilities": "dynamic capabilities"}
REPERTOIRE_LABEL = {"archival_regression": "archival regression (OLS, fixed effects, count models)",
                    "causal_design": "causal designs (instrumental variables, difference-in-differences) and archival regression",
                    "survey": "survey methods (SEM, multilevel models) and regression"}


def record_card(author: dict, record: dict, regime: str, quarter: int, words: int = 200, pressure: dict | None = None) -> str:
    """P: papers led, decisions, criticisms, reviews written. R: analyses led, claims, verifications, challenges, points.
    `pressure` (incentives.career_pressure: visible): the career, department-news and Journal-statistics lines, appended
    after the word cap so the record itself is never cut to make room for them."""
    text = _record_body(author, record, regime, quarter, words)
    if pressure:
        text += "\n\n" + "\n".join(pressure[k] for k in ("career", "news", "journal") if pressure.get(k))
    return text


def _record_body(author: dict, record: dict, regime: str, quarter: int, words: int) -> str:
    lines = [f"{STAGE_LABEL[author['stage']]}, {author['years_in_rank']} year(s) in rank. "
             f"Theories: {FAMILY_LABEL[author['primary_theory']]} (primary), {FAMILY_LABEL[author['secondary_theory']]}. "
             f"Methods: {REPERTOIRE_LABEL[author['repertoire']]}. "
             f"{'Two lead studies may be in progress at once.' if author['slots'] == 2 else 'One lead study at a time.'}"]
    if regime == "paper":
        papers = record.get("papers", [])[-8:]
        if papers:
            lines.append("Papers led: " + "; ".join(f"{p['paper_id']} \"{p['title'][:60]}\" ({p['question_id']}; {p['status']}"
                                                    + (f"; {p['gist']}" if p.get("gist") else "") + ")" for p in papers) + ".")
        else:
            lines.append("Papers led: none yet.")
        if record.get("coauthored"):
            lines.append(f"Coauthored papers: {record['coauthored']}.")
        rv = record.get("reviews_written", 0)
        if rv:
            lines.append(f"Reviews written: {rv}; recommendations matching the editor's decision: {record.get('reviews_matched', 0)} of {rv}.")
        if record.get("citations") is not None:
            lines.append(f"Citations received in the Journal: {record.get('citations', 0)}.")
    else:
        lines.append(f"Group {record.get('lab_id', '?')}. Analyses led: {record.get('analyses', 0)}. "
                     f"Claims proposed: {record.get('claims', 0)} (endorsed {record.get('endorsed', 0)}, contested {record.get('contested', 0)}, "
                     f"retracted {record.get('retracted', 0)}, lapsed {record.get('lapsed', 0)}). "
                     f"Verifications: {record.get('verifications', 0)} (reproduced {record.get('reproduced', 0)}). "
                     f"Data deposits: {record.get('deposits', 0)}, reused {record.get('reused', 0)} time(s). "
                     f"Challenges received: {record.get('challenges_received', 0)}. Reputation points: {record.get('points', 0):.0f}.")
    crit = record.get("criticisms", [])[-4:]
    if crit:
        lines.append("Criticisms received: " + " | ".join(crit) + ".")
    text = " ".join(lines)
    return _truncate_words(text, words)


def _truncate_words(text: str, words: int) -> str:
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " ..."


# ------------------------------------------------------------------ career pressure (stated as consequences only, never as instructions)
NEWS_WORD = {"tenured": "was promoted to associate professor", "tenure_exit": "left the department after the tenure review",
             "promoted_full": "was promoted to full professor", "not_promoted_full": "was not promoted to full professor",
             "retired": "retired", "exit": "left the department"}


def _years(n: int) -> str:
    return f"{n} year" if n == 1 else f"{n} years"


def career_line(author: dict, turnover: dict, n_published: int) -> str:
    """Stage, the clock the department's rule runs on, the record so far, and the bar that rule has applied (its own
    thresholds, stated as facts). Nothing here tells the author what to do."""
    y = int(author.get("years_in_rank", 0))
    T, D, F = int(turnover["tenure_years"]), int(turnover["doctoral_years"]), int(turnover["assoc_to_full_years"])
    thr = int(turnover["tenure_threshold"])
    prior = int(author.get("prior_credit", 0))
    record = (f"Publications before this record began: {prior}. Publications in the Journal since: {n_published} (any authorship).")
    stage = author["stage"]
    if stage == "doctoral":
        return (f"Career: doctoral student, year {y + 1}; the doctorate takes {_years(D)}, after which the department appoints "
                f"assistant professors. {record}")
    if stage == "assistant":
        left = T - y
        when = f"in {_years(left)}" if left > 0 else "at the end of this year"
        return (f"Career: assistant professor, year {y + 1} of a {T}-year tenure clock; the tenure review is {when}. {record} "
                f"Promotion to associate professor in this department has required {thr} publications over the clock; "
                f"promotion to full professor has required {2 * thr} by a review held after {_years(F)} in that rank.")
    if stage == "associate":
        left = F - y
        when = f"in {_years(left)}" if left > 0 else "at the end of this year, and every two years after that"
        return (f"Career: associate professor, year {y + 1}; the review for promotion to full professor is {when}. {record} "
                f"Promotion to full professor in this department has required {2 * thr} publications.")
    return f"Career: full professor, year {y + 1}; no promotion review pending. {record}"


def department_news_line(news: list[dict], quarter: int, window: int) -> str:
    """Promotion and tenure outcomes of colleagues in the last `window` quarters (true outcomes of the simulated population)."""
    recent = [n for n in news if quarter - window < n["quarter"] <= quarter and n.get("what") in NEWS_WORD]
    if not recent:
        return f"Department news (last {window} quarters): no promotion or tenure decisions."
    items = [f"{n['who']} {NEWS_WORD[n['what']]} (publication record {n['credit']})" if n.get("credit") is not None
             else f"{n['who']} {NEWS_WORD[n['what']]}" for n in sorted(recent, key=lambda n: (-n["quarter"], n["who"]))[:8]]
    return f"Department news (last {window} quarters): " + "; ".join(items) + "."


def journal_stats_line(stats: dict) -> str:
    """The Journal's public statistics over its recent record: submissions, decisions, acceptances, and acceptance by the main
    result as the authors themselves reported it."""
    w = stats["window"]
    if stats["decided"] == 0:
        return f"Journal statistics (last {w} quarters): {stats['submitted']} manuscripts submitted; no final decisions yet."
    pct = 100.0 * stats["accepted"] / stats["decided"]
    by = stats["by_result"]
    parts = [f"{label} {by[k][0]} of {by[k][1]}" for k, label in (("supported", "supported"), ("partially", "partially supported"),
                                                                    ("not_supported", "not supported")) if by[k][1] > 0]
    return (f"Journal statistics (last {w} quarters): {stats['submitted']} manuscripts submitted, {stats['decided']} decided, "
            f"{stats['accepted']} accepted ({pct:.0f}%). Accepted by the main result as reported by the authors: " + "; ".join(parts) + ".")


def abstract_entry(p: dict) -> str:
    hyps = "; ".join(f"{h['hyp_id']} {h['iv']}{' x ' + h['moderator'] if h.get('moderator') else ''} -> {h['dv']}: "
                     f"{h['verdict'].replace('_', ' ')} (est {h['estimate']:.3f}, SE {h['se']:.3f}, N {h['n']})"
                     for h in p["hypotheses"] if h.get("estimate") is not None)
    return (f"**{p['paper_id']}** ({'earlier work' if p.get('legacy') else 'published Q' + str(p['quarter_published'])}; "
            f"cited {p.get('citations', 0)} time(s)). *{p['title']}*. {p.get('abstract') or p.get('abstract_template', '')} "
            f"Hypotheses: {hyps}.")


def menu_md(entries: list[dict]) -> str:
    out = [f"# Menu: {len(entries)} research questions you could take up this quarter", "",
           f"This menu lists {len(entries)} questions; each has its own data folder under `inputs\\`. Choose exactly one.", ""]
    for i, e in enumerate(entries, start=1):
        q = e["question"]
        out += [f"## {i}. {q['id']}: {q['iv_label']} and {q['dv_label']}", "",
                f"{q['prediction']}", "",
                f"Data available for this study: {q['unit']} data ({q['design'].replace('_', ' ').replace('+', ' and ')}); "
                f"about {e['n_preview']:,} observations; folder `inputs\\{q['id']}\\` (data.csv, codebook.md, sample_frame.md). "
                f"Access: {'requires an extra quarter of data collection' if q['access_cost'] == 'costly' else 'archival, immediate'}.", "",
                f"Papers published on this question: {e['count']}.", ""]
        if e.get("handbook_entry"):
            out += [f"Latest handbook entry: {e['handbook_entry']}", ""]
        if e["view"]:
            out.append("Published papers on this question (most cited and most recent):")
            out += [f"- {abstract_entry(p)}" for p in e["view"]]
        else:
            out.append("No papers have been published on this question.")
        out.append("")
    return "\n".join(out)


def prior_md(view: list[dict], qid: str) -> str:
    if not view:
        return f"# Prior work on {qid}\n\nNo papers have been published on this question.\n"
    return f"# Prior work on {qid}\n\n" + "\n".join(f"- {abstract_entry(p)}" for p in view) + "\n"


def submission_summary(p: dict, count: int | None, status_line: str | None = None) -> str:
    hyps = "; ".join(f"{h['hyp_id']}: {h['verdict'].replace('_', ' ')}" for h in p["hypotheses"])
    h1 = p["hypotheses"][0]
    s = [f"### {p['paper_id']}: {p['title']}", "", p["abstract"], "", f"Hypotheses: {hyps}.",
         f"Main result: {h1['iv']} -> {h1['dv']}: estimate {h1['estimate']:.3f} (SE {h1['se']:.3f}, p {h1['p']:.3f}), N {h1['n']}, "
         f"{p['design'].replace('_', ' ')}, {p['estimator']}."]
    if count is not None:
        s.append(f"Papers published on this question: {count}.")
    if status_line:
        s.append(f"Authors: {status_line}.")
    s.append(f"Relation to prior work as stated by the authors: {p['relation_to_prior']}.")
    return "\n".join(s) + "\n"


def reviews_md(reviews: list[dict], names: bool = False) -> str:
    out = []
    for i, r in enumerate(reviews, start=1):
        who = f" ({r['reviewer_id']})" if names else ""
        out += [f"## Referee {i}{who}", "", f"Recommendation: {r['recommendation']}. Validity probability: {r['validity_probability']}. "
                f"Relation to prior work: {r['relation_to_prior']}. Confidence: {r['confidence']:.2f}.", "", f"Summary: {r['summary']}", ""]
        if r["issues"]:
            out.append("Issues:")
            out += [f"- [{x['severity']}, {x['kind']}] {x['location']}: {x['description']}" for x in r["issues"]]
        out.append("")
    return "\n".join(out)


def author_status_line(authors: list[dict], records: dict) -> str:
    return "; ".join(f"{a['id']} ({STAGE_LABEL[a['stage']].lower()}, {len(records.get(a['id'], {}).get('papers_published', []))} paper(s) in the Journal, "
                     f"{records.get(a['id'], {}).get('citations', 0)} citations)" for a in authors)


# ------------------------------------------------------------------ repository regime texts
STATUS_WORD = {"proposed": "proposed", "contested": "contested", "endorsed": "endorsed", "superseded": "superseded",
               "retracted": "retracted", "lapsed": "lapsed", "withdrawn": "withdrawn", "endorsed_legacy": "endorsed (earlier work)"}


def claim_entry(c: dict, hidden_tally: bool = False, blinded: bool = False, labs: dict | None = None) -> str:
    ev = c.get("evidence", [{}])[0] if c.get("evidence") else {}
    prop = "" if blinded else f" Proposed by group {c['proposer_lab']}"
    if not blinded and labs and c.get("proposer_lab") in labs:
        L = labs[c["proposer_lab"]]
        prop += f" ({L.get('size', '?')} members, {L.get('reputation', 0):.0f} points, {L.get('endorsed_claims', 0)} endorsed claims)"
    tally = "" if hidden_tally else (f" Endorsements: {len(c.get('endorsements', []))}"
                                     + ("" if blinded else (" (" + ", ".join(e['lab'] for e in c.get('endorsements', [])) + ")" if c.get('endorsements') else ""))
                                     + f". Challenges: {len(c.get('challenges', []))}"
                                     + (" (" + ", ".join(ch['kind'] for ch in c.get('challenges', [])) + ")" if c.get('challenges') else "") + ".")
    ver = c.get("verifications", [])
    vtxt = f" Verifications: {len(ver)} ({', '.join(v['result'] for v in ver)})." if ver else ""
    evtxt = (f" Evidence: estimate {ev.get('estimate', float('nan')):.3f}, SE {ev.get('se', float('nan')):.3f}, N {ev.get('n', '?')}, "
             f"{ev.get('estimator', '')}, data {', '.join(c.get('deposit_ids', [])) or 'earlier study'}.") if ev else ""
    return (f"**{c['claim_id']}** [{STATUS_WORD.get(c['status'], c['status'])}]: {c['statement']}{prop}.{evtxt}{tally}{vtxt}"
            f" Filed Q{c.get('quarter_filed', 0)}.")


def registry_view_md(q: dict, claims: list[dict], catalogue: list[dict], caps: int = 10, hidden_tally: bool = False,
                     blinded: bool = False, labs: dict | None = None, handbook_entry: str | None = None) -> str:
    order = {"endorsed": 0, "endorsed_legacy": 0, "contested": 1, "proposed": 2, "superseded": 3, "retracted": 4, "withdrawn": 5, "lapsed": 6}
    cs = sorted(claims, key=lambda c: (order.get(c["status"], 9), -c.get("quarter_filed", 0), c["claim_id"]))
    counts = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    out = [f"## {q['id']}: {q['iv_label']} and {q['dv_label']}", "", q["prediction"], ""]
    fq = q.get("frame_quantiles") or {}
    if fq.get("p50"):
        # the cost of a new dataset is its size; the typical sizes let the head plan (a card that states the budget but not the cost leaves heads unable to plan)
        out += [f"Typical sample sizes for {q['unit']} data on this question: median {int(fq['p50']):,} observations "
                f"(middle half {int(fq.get('p25', fq['p50'])):,} to {int(fq.get('p75', fq['p50'])):,}; largest seen {int(fq.get('p95', fq['p50'])):,}). "
                f"A new dataset costs its number of observations from your group's data budget.", ""]
    out += ["Claims filed on this question: " + (", ".join(f"{STATUS_WORD.get(k, k)} {v}" for k, v in sorted(counts.items())) or "none") + "."]
    if handbook_entry:
        out.append(f"Latest handbook entry: {handbook_entry}")
    out.append("")
    out += [f"- {claim_entry(c, hidden_tally, blinded, labs)}" for c in cs[:caps]]
    out += ["", "Datasets available on this question:"]
    if catalogue:
        for d in catalogue:
            out.append(f"- {d['deposit_id']} v{d['version']}: N {d['n']:,}, {d['unit']}, deposited Q{d['quarter']}"
                       + ("" if blinded else f" by group {d['lab']}") + f"; analyses run on it: {d['n_analyses']}; claims attached: {d['n_claims']}"
                       + ("" if d.get("available", True) else " (not yet available)"))
    else:
        out.append("- none; a new dataset must be requested")
    out.append("")
    return "\n".join(out)


def lab_card(lab: dict, members: list[dict], record: dict, capacity: dict, words: int | None = None) -> str:
    """The lab head's card. Never truncated: at 150 scientists a lab has 10-16 members, and a 250-word cap cut
    the card inside the member list, so heads never saw this quarter's slots, verification capacity or data budget. The
    operational lines come first, the full member list (needed to assign analyses) last. `words` is kept for callers only."""
    head = next(m for m in members if m["id"] == lab["head"])
    members_line = ("Members: " + "; ".join(f"{m['id']} ({STAGE_LABEL[m['stage']].lower()}, {FAMILY_LABEL[m['primary_theory']]}, "
                                             f"{m['repertoire'].replace('_', ' ')}, {'careful, checks data' if m['style'] == 'VERIFIED' else 'fast, one-pass'}, "
                                             f"{'two analyses at a time' if m['throughput'] == 'T1' else 'one analysis at a time'})" for m in members) + ".")
    lines = [f"Group {lab['id']}. Head: {head['id']} ({STAGE_LABEL[head['stage']].lower()}; {FAMILY_LABEL[head['primary_theory']]}, "
             f"{FAMILY_LABEL[head['secondary_theory']]}). Primary theory of the group: {FAMILY_LABEL[lab['primary_theory']]}.",
             f"This quarter: analysis slots available: {capacity['slots']} ({', '.join(capacity['available_members']) or 'none'}); "
             f"verifications your members can lead: {capacity['verifications']}. Data budget remaining this quarter: about {capacity['data_budget']:,} observations; "
             f"a new dataset costs its number of observations (typical sizes are stated per question in `registry\\`), a catalogued dataset costs nothing to reuse.",
             f"Record: claims proposed {record.get('claims', 0)} (endorsed {record.get('endorsed', 0)}, contested {record.get('contested', 0)}, "
             f"retracted {record.get('retracted', 0)}, lapsed {record.get('lapsed', 0)}); verifications done {record.get('verifications', 0)}; "
             f"deposits made {record.get('deposits', 0)}; challenges won {record.get('challenges_won', 0)}, lost {record.get('challenges_lost', 0)}; "
             f"reputation points {record.get('points', 0):.0f}."]
    if record.get("endorsed_by"):
        lines.append("Groups that endorsed your claims recently: " + ", ".join(record["endorsed_by"][-6:]) + ".")
    lines.append(members_line)
    return " ".join(lines)
