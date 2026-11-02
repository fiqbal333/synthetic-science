"""The hidden ledger: JSONL tables streamed to disk, one row per event, never shown to agents.

Tables (all under runs/<scenario>/<seed>/<mode>/ledger/):
  hypotheses.jsonl   one row per hypothesis (P) or claim entry (R analyses), with truth check and flags
  papers.jsonl       one row per paper (P) or analysis (R)
  claims.jsonl       one row per claim status change (R)
  votes.jsonl        every endorsement / challenge / abstention / offer / stake (R)
  verifications.jsonl verification reports with truth (R)
  reviews.jsonl      every review with truth flags (P)
  decisions.jsonl    desk and editor decisions (P)
  datasets.jsonl     dataset meta + hidden truth per dataset/deposit
  budget.jsonl       every debit
  events.jsonl       everything else (assignment, queue, publication, promotion, turnover)
  questions_Q<t>.json per-question roll-up at quarter t, written by the observatory
"""
from __future__ import annotations

from pathlib import Path

from .util import append_jsonl, read_jsonl, write_json

TABLES = ("hypotheses", "papers", "claims", "votes", "verifications", "reviews", "decisions", "datasets", "budget", "events")

HYPOTHESIS_FIELDS = [
    "run_id", "mode", "regime", "unit_id", "hyp_id", "paper_id", "claim_id", "analysis_id", "question_id", "core",
    "quarter", "author_id", "lab_id", "style", "throughput", "tier", "iv", "dv", "moderator", "kind", "predicted_sign",
    "estimate", "se", "p", "n", "n_correct", "estimator", "se_type", "controls", "verdict", "relation_to_prior",
    "theta", "theta_fe", "theta_naive", "truth_class", "truth_check", "truth_check_lenient", "magnitude_ratio", "ci_covers",
    "traps", "flags", "certified", "certified_quarter", "status", "attention", "dataset_id", "deposit_ids",
]


class Ledger:
    def __init__(self, run_dir: Path, run_id: str, mode: str, regime: str):
        self.dir = Path(run_dir) / "ledger"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.mode, self.regime = run_id, mode, regime
        self._buffers: dict[str, list[dict]] = {t: [] for t in TABLES}

    def add(self, table: str, row: dict) -> None:
        row = dict(row)
        row.setdefault("run_id", self.run_id)
        row.setdefault("mode", self.mode)
        row.setdefault("regime", self.regime)
        self._buffers[table].append(row)
        if len(self._buffers[table]) >= 200:
            self.flush(table)

    def add_many(self, table: str, rows) -> None:
        for r in rows:
            self.add(table, r)

    def flush(self, table: str | None = None) -> None:
        for t in ([table] if table else TABLES):
            if self._buffers[t]:
                append_jsonl(self.dir / f"{t}.jsonl", self._buffers[t])
                self._buffers[t] = []

    def read(self, table: str) -> list[dict]:
        self.flush(table)
        return read_jsonl(self.dir / f"{table}.jsonl")

    def write_rollup(self, quarter: int, rollup: dict) -> None:
        write_json(self.dir / f"questions_Q{quarter}.json", rollup, indent=None)

    def path(self, table: str) -> Path:
        return self.dir / f"{table}.jsonl"
