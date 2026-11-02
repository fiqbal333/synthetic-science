"""Extend a finished run to a longer horizon (e.g. the 8-quarter runs to 20 quarters).

    python -m synthsci extend --run runs/S0-150/1/llm --to 20          # -> runs/S0-150-x20/1/llm
    python -m synthsci extend --run runs/R-b-150/1/llm --to 20         # pairs with runs/S0-150-x20/1/llm (or --paired-run)
    python -m synthsci resume --run runs/S0-150-x20/1/llm              # continues at quarter 9 with the normal PENDING loop

The finished run is never modified: it is hashed before and after, and the copy is checked against it file by file. In the
copy, the horizon close-out is undone so that work in progress continues exactly where it stood:

- paper regime: `PaperRegime.finish()` set every paper still in the pipeline to `unfinished`, appended an uncertified
  hypothesis row per hypothesis and an `unfinished_at_horizon` event (with the prior status), rewrote papers_snapshot.json and
  wrote ledger/citations.json. The copy gets each paper's prior status back (its due quarters, review-queue place and author
  slot were never touched by finish), loses the close-out hypothesis rows and citations.json, and its papers snapshot shows
  the prior statuses. One `extension_reopened` event per paper is appended.
- repository regime: `RepositoryRegime.finish()` wrote the whole hypotheses table (one row per claim with its final status,
  one per unfiled draft entry) and re-ran the registry at the horizon, which recomputed lab sizes after the horizon's annual
  step. The copy loses the table (it is written again at the new horizon) and gets the lab sizes the quarter-H registry
  phase had set, which is what the next quarter would have seen. Claims, analyses, drafts, offers, tests, lapse clocks,
  stakes and deferred budget requests were never changed by finish and carry over as they are.
- everything removed is kept verbatim in `extension/removed_at_reopen/` with a NOTE; the horizon audit is kept in place and
  copied to `extension/audit_at_Q<H>/` (the audit at the new horizon overwrites `audit/`).

The copy's config is the source's with `horizon_quarters: <to>`, `long_horizon.pin_short_rules: true` and
`long_horizon.extended_from: [<H>]` (one rulebook: every horizon-gated rule follows the original horizon; see RegimeBase).
It gets a new run id and config hash; its task namespace is the source's (checked against the source's task records).
`extension.json` records the lineage.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import yaml

from . import config as C
from .util import append_jsonl, read_json, read_jsonl, read_text, sha256_file, sha256_text, write_json, write_text

PIPELINE_STATUSES = ("submitted", "waiting", "sent_out", "under_review", "revise", "resubmitted", "review_pending", "accepted")


# ---------------------------------------------------------------------- manifests
def manifest_lines(root: Path, base: Path | None = None) -> list[str]:
    """'<sha256>  <path relative to base, posix>' for every file under root, sorted by path."""
    root = Path(root)
    base = Path(base) if base else root
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            rows.append((p.relative_to(base).as_posix(), p))
    return [f"{sha256_file(p)}  {rel}" for rel, p in sorted(rows)]


def manifest_hash(lines: list[str]) -> str:
    return sha256_text("\n".join(lines) + "\n")


def write_manifest(path: Path, roots: list[Path], base: Path) -> int:
    lines = []
    for r in roots:
        lines += manifest_lines(r, base)
    write_text(path, "\n".join(lines) + "\n")
    return len(lines)


def verify_manifest(path: Path, base: Path, roots: list[Path] | None = None) -> dict:
    """Check every file of a manifest (as written by write_manifest or `Get-FileHash`: '<hash>  <relative path>'); with
    `roots`, also list files under them that the manifest does not name."""
    want = {}
    for line in read_text(path).splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            want[rel.strip().replace("\\", "/")] = h.strip().lower()
    mismatch, missing = [], []
    for rel, h in want.items():
        p = Path(base) / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != h:
            mismatch.append(rel)
    extra = []
    for r in roots or []:
        for p in Path(r).rglob("*"):
            if p.is_file() and p.relative_to(base).as_posix() not in want:
                extra.append(p.relative_to(base).as_posix())
    return {"files": len(want), "mismatch": mismatch, "missing": missing, "extra": sorted(extra),
            "ok": not (mismatch or missing or extra)}


# ---------------------------------------------------------------------- helpers
def _git_commit() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.WS_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "synthsci", "configs"], cwd=C.WS_ROOT, capture_output=True,
                               text=True, timeout=30).stdout.strip()
        return {"commit": head or None, "engine_dirty": bool(dirty)}
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "error": str(e)}


def task_namespaces(rd: Path) -> set[str]:
    """The namespaces (`run_id` in task.json) the run's materialised tasks were named in; empty in fast mode."""
    out = set()
    for tj in (Path(rd) / "tasks").glob("*/task.json"):
        ns = read_json(tj).get("run_id")
        if ns:
            out.add(ns)
    return out


def _load_cfg(rd: Path) -> dict:
    return yaml.safe_load(read_text(rd / "config.resolved.yaml"))


def extended_config(cfg: dict, to: int) -> dict:
    """The source config with the new horizon and the pinned rulebook, round-tripped through YAML as the
    run folder stores it, so that its hash is the one `resume` will compute."""
    new = copy.deepcopy(cfg)
    H = int(cfg["horizon_quarters"])
    new["horizon_quarters"] = int(to)
    lh = new.setdefault("long_horizon", {})
    lh["pin_short_rules"] = True
    lh["extended_from"] = [int(h) for h in (lh.get("extended_from") or [])] + [H]
    return yaml.safe_load(C.dump_yaml(new))


def _rewrite_lines(path: Path, keep: list[bytes]) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        for line in keep:
            f.write(line)
    tmp.replace(path)


def _split_jsonl(path: Path, removed_if) -> tuple[list[bytes], list[bytes], list[dict]]:
    """Raw lines (bytes, verbatim) of a JSONL file split by a predicate on the parsed row."""
    import json
    keep, gone, gone_rows = [], [], []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                keep.append(line)
                continue
            row = json.loads(line.decode("utf-8-sig"))
            if removed_if(row):
                gone.append(line)
                gone_rows.append(row)
            else:
                keep.append(line)
    return keep, gone, gone_rows


# ---------------------------------------------------------------------- reopening the close-out
def reopen_paper(rd: Path, H: int, new_run_id: str, mode: str) -> dict:
    st = read_json(rd / "state" / "state.json")
    removed = rd / "extension" / "removed_at_reopen"
    events = read_jsonl(rd / "ledger" / "events.jsonl")
    closed = [e for e in events if e.get("event") == "unfinished_at_horizon" and int(e.get("quarter", -1)) == H]
    prior = {e["paper_id"]: e["status"] for e in closed}
    unfinished_now = {pid for pid, p in st["papers"].items() if p.get("status") == "unfinished"}
    if unfinished_now != set(prior):
        raise RuntimeError(f"papers marked unfinished ({len(unfinished_now)}) differ from the close-out events ({len(prior)})")
    bad = {pid: s for pid, s in prior.items() if s not in PIPELINE_STATUSES}
    if bad:
        raise RuntimeError(f"close-out events carry statuses outside the pipeline: {bad}")
    for pid, s in prior.items():
        st["papers"][pid]["status"] = s
    # the close-out's hypothesis rows: status 'unfinished' exists only there
    hp = rd / "ledger" / "hypotheses.jsonl"
    keep, gone, gone_rows = _split_jsonl(hp, lambda r: r.get("status") == "unfinished" and not r.get("certified")
                                         and int(r.get("quarter", -1)) == H and r.get("paper_id") in prior)
    expected = sum(len(st["papers"][pid]["hypotheses"]) for pid in prior)
    if len(gone) != expected:
        raise RuntimeError(f"expected {expected} close-out hypothesis rows, found {len(gone)}")
    removed.mkdir(parents=True, exist_ok=True)
    with open(removed / "hypotheses.jsonl", "wb") as f:
        for line in gone:
            f.write(line)
    _rewrite_lines(hp, keep)
    # papers snapshot: the close-out rewrote it with 'unfinished'; the copy shows the prior statuses (every other field is the
    # same projection of the same paper)
    sp = rd / "ledger" / "papers_snapshot.json"
    if sp.exists():
        shutil.copy2(sp, removed / "papers_snapshot.json")
        snap = read_json(sp)
        for row in snap:
            if row.get("paper_id") in prior:
                row["status"] = prior[row["paper_id"]]
        write_json(sp, snap)
    cp = rd / "ledger" / "citations.json"
    if cp.exists():   # written only by finish(); written again at the new horizon
        shutil.move(str(cp), str(removed / "citations.json"))
    # where each reopened paper stands (all untouched by finish)
    slots = {}
    for aid, a in st["authors"].items():
        for si, slot in enumerate(a["slots"]):
            if slot.get("paper_id"):
                slots[slot["paper_id"]] = (aid, si, slot["state"])
    rows = []
    for pid in sorted(prior):
        p = st["papers"][pid]
        sl = slots.get(pid)
        rows.append({"quarter": H, "event": "extension_reopened", "paper_id": pid, "status": p["status"],
                     "review_due": p.get("review_due"), "revision_due": p.get("revision_due"), "publish_due": p.get("publish_due"),
                     "review_round": p.get("review_round"), "in_review_queue": pid in st.get("review_queue", []),
                     "lead_slot": {"author": sl[0], "slot": sl[1], "state": sl[2]} if sl else None,
                     "run_id": new_run_id, "mode": mode, "regime": "paper"})
    append_jsonl(rd / "ledger" / "events.jsonl", rows)
    write_json(rd / "state" / "state.json", st, indent=None)
    by_status = Counter(prior.values())
    return {"regime": "paper", "papers_reopened": len(prior), "by_status": dict(sorted(by_status.items())),
            "hypothesis_rows_removed": len(gone), "in_review_queue": sum(1 for r in rows if r["in_review_queue"]),
            "holding_a_slot": sum(1 for r in rows if r["lead_slot"]),
            "live_authored_reopened": sum(1 for pid in prior if not st["papers"][pid].get("prehistory"))}


def reopen_repository(rd: Path, H: int) -> dict:
    st = read_json(rd / "state" / "state.json")
    removed = rd / "extension" / "removed_at_reopen"
    removed.mkdir(parents=True, exist_ok=True)
    # every hypothesis row of the repository regime is written by finish() (one per claim, one per unfiled draft entry)
    hp = rd / "ledger" / "hypotheses.jsonl"
    n_rows = len(read_jsonl(hp)) if hp.exists() else 0
    expected = len(st["claims"]) + sum(1 for d in st["drafts"].values() for e in d["entries"] if e.get("filed") in (None, "dropped"))
    if n_rows != expected:
        raise RuntimeError(f"expected {expected} close-out hypothesis rows in the repository ledger, found {n_rows}")
    if hp.exists():
        shutil.move(str(hp), str(removed / "hypotheses.jsonl"))
    # lab sizes: finish() re-ran the registry after the horizon's annual step; the registry phase of quarter H had counted the
    # members active before that step (careers move only in the annual step, so the last annual snapshot before H has them)
    pop = rd / "population"
    snaps = []
    for f in pop.glob("scientists_Q*.json"):
        try:
            q = int(f.stem.split("_Q")[1])
        except ValueError:
            continue
        if q < H:
            snaps.append((q, f))
    src = max(snaps)[1] if snaps else pop / "scientists.json"
    active = {s["id"]: s.get("active", True) for s in read_json(src)}
    sizes = {}
    for lid, L in st["labs"].items():
        before = int(L["size"])
        L["size"] = sum(1 for m in L["members"] if active.get(m, True))
        if L["size"] != before:
            sizes[lid] = {"after_finish": before, "restored": L["size"]}
    write_json(rd / "state" / "state.json", st, indent=None)
    cl = [c for c in st["claims"].values() if not c.get("legacy")]
    bstate = read_json(rd / "budget" / "state.json") if (rd / "budget" / "state.json").exists() else {}
    return {"regime": "repository", "hypothesis_rows_removed": n_rows, "lab_sizes_restored": sizes, "lab_sizes_from": src.name,
            "claims_by_status": dict(sorted(Counter(c["status"] for c in cl).items())),
            "claims_open": sum(1 for c in cl if c["status"] in ("proposed", "contested")),
            "analyses_by_status": dict(sorted(Counter(a["status"] for a in st["analyses"].values()).items())),
            "drafts_by_status": dict(sorted(Counter(d["status"] for d in st["drafts"].values()).items())),
            "verification_offers_open": len(st.get("offers", [])),
            "tests_by_status": dict(sorted(Counter(t.get("status") for t in st.get("tests", {}).values()).items())) if isinstance(st.get("tests"), dict) else None,
            "budget_deferred_requests": len(bstate.get("deferred", [])), "budget_carry": bstate.get("carry")}


# ---------------------------------------------------------------------- extend
def extend(run_dir: Path, to: int, out_root: Path | None = None, paired_run: Path | None = None) -> dict:
    from .engine import crn_namespace
    from .regimes.paper import PaperRegime
    from .regimes.repository import RepositoryRegime

    src = Path(run_dir).resolve()
    meta = read_json(src / "run.json")
    cfg = _load_cfg(src)
    H, seed, mode = int(cfg["horizon_quarters"]), int(meta["seed"]), meta["mode"]
    if int(meta["horizon_quarters"]) != H:
        raise RuntimeError(f"run.json and config.resolved.yaml disagree on the horizon in {src}")
    if int(to) <= H:
        raise ValueError(f"--to {to} must exceed the run's horizon {H}")
    if C.config_hash(cfg) != meta["config_hash"]:
        raise RuntimeError(f"config.resolved.yaml of {src} does not hash to run.json's config_hash")
    regime = cfg["regime"]
    phases = (PaperRegime if regime == "paper" else RepositoryRegime).PHASES
    st = read_json(src / "state" / "state.json")
    if int(st["quarter"]) != H or phases[-1] not in st["done"].get(str(H), []) or not (src / "audit" / "audit.json").exists():
        raise RuntimeError(f"{src} is not a finished run (quarter {st['quarter']} of {H}, or no audit)")
    root = Path(out_root).resolve() if out_root else C.WS_ROOT / "runs"
    dest = root / f"{meta['scenario']}-x{int(to)}" / str(seed) / mode
    if dest.exists():
        raise FileExistsError(f"{dest} exists; nothing is written over")
    if src == dest or src in dest.parents:
        raise ValueError("the extension folder must lie outside the source run")

    new_cfg = extended_config(cfg, to)
    new_run_id, new_hash = C.run_id(new_cfg, seed, mode), C.config_hash(new_cfg)
    ns_src, ns_new = crn_namespace(cfg, seed, mode), crn_namespace(new_cfg, seed, mode)
    if ns_src != ns_new:
        raise RuntimeError(f"task namespace would change ({ns_src} -> {ns_new})")
    # the namespace the source's live tasks were actually named in (configs may have moved since the run)
    recorded = task_namespaces(src)
    if recorded and recorded != {ns_src}:
        raise RuntimeError(f"the source's tasks were named in {sorted(recorded)}, the engine now computes {ns_src}; configs changed since the run")

    paired_dir, paired_ext = None, None
    if regime == "repository" and cfg["budget"]["rule"] == "pairing":
        paired_dir = Path(paired_run).resolve() if paired_run else root / f"{cfg['budget']['paired_scenario']}-x{int(to)}" / str(seed) / mode
        if not (paired_dir / "extension.json").exists():
            raise RuntimeError(f"budget pairing: extend the paired paper run first; no extended paper run at {paired_dir}")
        paired_ext = read_json(paired_dir / "extension.json")
        pm = read_json(paired_dir / "run.json")
        if int(paired_ext["target_horizon"]) != int(to) or int(pm["seed"]) != seed or pm["mode"] != mode or pm["regime"] != "paper":
            raise RuntimeError(f"{paired_dir} is not the {mode} paper run at seed {seed} extended to {to}")
        if meta.get("paired_run_dir") and Path(paired_ext["source"]).resolve() != Path(meta["paired_run_dir"]).resolve():
            raise RuntimeError(f"{paired_dir} extends {paired_ext['source']}, but this run was paired with {meta['paired_run_dir']}")
        if pm.get("scenario") != cfg["budget"]["paired_scenario"]:
            raise RuntimeError(f"{paired_dir} is scenario {pm.get('scenario')}, not {cfg['budget']['paired_scenario']}")

    t0 = time.time()
    lines_before = manifest_lines(src)
    h_before = manifest_hash(lines_before)
    shutil.copytree(src, dest)
    lines_copy = manifest_lines(dest)
    if lines_copy != lines_before:
        raise RuntimeError(f"the copy at {dest} differs from the source; left in place for inspection")

    ext_dir = dest / "extension"
    if ext_dir.exists():
        raise RuntimeError(f"{src} already holds an extension/ folder")
    ext_dir.mkdir()
    write_text(ext_dir / "source_manifest.sha256", "\n".join(lines_before) + "\n")
    shutil.copy2(src / "run.json", ext_dir / "source_run.json")
    shutil.copy2(src / "config.resolved.yaml", ext_dir / "source_config.resolved.yaml")
    if (dest / "audit").exists():
        shutil.copytree(dest / "audit", ext_dir / f"audit_at_Q{H}")

    report = reopen_paper(dest, H, new_run_id, mode) if regime == "paper" else reopen_repository(dest, H)
    write_text(ext_dir / "removed_at_reopen" / "NOTE.md",
               f"# Removed when the run was reopened at quarter {H}\n\n"
               f"These files are the verbatim rows and files that the quarter-{H} close-out (`{regime}` finish) wrote and that the "
               f"extension took out of this copy, so that the audit at quarter {int(to)} does not count them. The source run "
               f"`{src}` still holds them unchanged.\n\n"
               + ("- `hypotheses.jsonl`: the uncertified rows of papers marked `unfinished` at the horizon (status restored in state).\n"
                  "- `papers_snapshot.json`: the snapshot as finish() wrote it (statuses `unfinished`).\n"
                  "- `citations.json`: written only by finish(); written again at the new horizon.\n"
                  if regime == "paper" else
                  "- `hypotheses.jsonl`: finish()'s one row per claim and per unfiled draft entry (the table is written again at the new horizon).\n"
                  f"- lab sizes recomputed by finish()'s registry call after the quarter-{H} annual step were set back to the quarter-{H} "
                  f"registry phase's values (see extension.json: reopen.lab_sizes_restored).\n"))

    # the copy's identity: the extended config, a new run id and hash; run() rewrites run.json on the next resume
    write_text(dest / "config.resolved.yaml", C.dump_yaml(new_cfg))
    new_meta = dict(meta)
    new_meta.update({"run_id": new_run_id, "config_hash": new_hash, "horizon_quarters": int(to),
                     "paired_run_dir": str(paired_dir) if paired_dir else meta.get("paired_run_dir")})
    write_json(dest / "run.json", new_meta)

    lines_after = manifest_lines(src)
    unchanged = lines_after == lines_before
    if not unchanged:
        raise RuntimeError(f"the source run {src} changed during the extension")
    git = _git_commit()
    ext = {"source": str(src), "source_run_id": meta["run_id"], "source_config_hash": meta["config_hash"],
           "source_manifest_sha256": h_before, "source_manifest_files": len(lines_before), "source_unchanged": unchanged,
           "source_horizon": H, "target_horizon": int(to), "run_dir": str(dest), "run_id": new_run_id, "config_hash": new_hash,
           "task_namespace": ns_new, "pin": {"pin_short_rules": True, "extended_from": new_cfg["long_horizon"]["extended_from"]},
           "paired_run_dir": str(paired_dir) if paired_dir else None,
           "paired_extension_source": paired_ext["source"] if paired_ext else None,
           "git_commit": git.get("commit"), "engine_dirty": git.get("engine_dirty"),
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1), "reopen": report}
    write_json(dest / "extension.json", ext)
    with open(dest / "log" / "run.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} extended from {src} (Q{H}) to Q{int(to)}: run id {new_run_id}; reopen {report}\n")
    return ext
