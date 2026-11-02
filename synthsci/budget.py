"""Equal budgets: two resource ledgers (data = observations, compute = tariff tokens), the pairing rule,
the fixed rule, admission by priority with service-debt rotation, and deferral.

Pairing balances COMPUTE only. The paired repository world's compute allotment each
quarter is the paper world's realized compute spend; its data allotment is `budget.data_ceiling_multiple` (2.0 by default; 4.0 in R-b-150) times the
paper world's realized data spend, a safety ceiling rather than a constraint, and data used is a reported outcome (data
per certified unit, per correct handbook answer). Every deferral names the resource that bound it.

The ledger table `budget.jsonl` is the source of truth. The in-memory tallies are rebuilt from it when a run resumes,
so a run that pauses at every dispatch boundary keeps its tallies across resumes instead of restarting them from zero. Every call row carries the tariff debit and, when the backend
reports it, the realized tokens of the agent call, so the tariff's accuracy can be audited after the run.
"""
from __future__ import annotations

from pathlib import Path

from .config import CONFIG_DIR, load_yaml
from .util import read_json, read_jsonl, write_json


class PairingNotReady(RuntimeError):
    """The paired paper run has not closed the quarter this repository run is about to budget (a paired run may lag its paper run only by closed quarters)."""

    def __init__(self, quarter: int, closed_through: int, paired_run_dir: Path):
        self.quarter, self.closed_through, self.paired_run_dir = quarter, closed_through, paired_run_dir
        super().__init__(f"WAITING FOR PAIRED RUN: quarter {quarter} needs the paired paper run's realized spend for that quarter, "
                         f"but {paired_run_dir} has closed only through quarter {closed_through}. Run the paper run until it has "
                         f"closed quarter {quarter}, then resume this run.")


def paired_closed_through(paired_run_dir: Path | None) -> int | None:
    """The last quarter the paired paper run has closed (every phase run and the quarter's budget closed), from its saved
    state; None when no state file exists (old runs: the spend file alone is used, as before)."""
    if not paired_run_dir:
        return None
    sp = Path(paired_run_dir) / "state" / "state.json"
    if not sp.exists():
        return None
    from .regimes.paper import PaperRegime
    st = read_json(sp)
    q = int(st.get("quarter", 0))
    # run_all only moves to quarter q after closing q - 1; q itself is closed once its last phase is done
    return q if PaperRegime.PHASES[-1] in (st.get("done") or {}).get(str(q), []) else q - 1


class Budget:
    def __init__(self, cfg: dict, run_dir: Path, ledger, paired_run_dir: Path | None = None, allow_unpaired: bool = False):
        self.cfg, self.run_dir, self.ledger = cfg, Path(run_dir), ledger
        bc = cfg["budget"]
        tpath = Path(bc["tariffs"])
        self.tariffs = load_yaml(tpath if tpath.exists() else CONFIG_DIR / bc["tariffs"])
        self.rule = bc["rule"]
        self.spend: dict[int, dict[str, float]] = {}
        self.realized: dict[int, float] = {}          # realized agent tokens per quarter (reported by the backend; 0 for rule agents)
        self.allot: dict[int, dict[str, float]] = {}
        self.carry = {"data": 0.0, "compute": 0.0}
        self.deferred: list[dict] = []
        self.service_debt: dict[str, int] = {}
        self.paired_spend = None
        self.pairing_fallback = None
        self.data_ceiling_multiple = float(bc.get("data_ceiling_multiple", 2.0))   # data floats under k x the paired run's data
        self.last_admission: dict = {}                # what the last admit() saw and did (binding resource per deferral)
        self.suspended = False                        # True during the warm-start pre-history: rows are recorded, nothing is charged
        if cfg.get("regime") == "paper" and self.rule == "pairing":
            self.rule = "uncapped"   # the paper regime's own spending is never capped under pairing
        self.paired_run_dir = Path(paired_run_dir) if paired_run_dir else None
        self.paired_closed = None
        if self.rule == "pairing":
            p = Path(paired_run_dir) / "budget" / "spend_by_quarter.json" if paired_run_dir else None
            if p and p.exists():
                self.paired_spend = {int(k): v for k, v in read_json(p).items()}
                self.paired_closed = paired_closed_through(paired_run_dir)
            elif allow_unpaired:
                self.pairing_fallback = f"paired run spend not found ({p}); using fixed budget (explicitly allowed with --allow-unpaired)"
                self.rule = "fixed"
                self.ledger.add("events", {"quarter": 0, "event": "pairing_fallback", "detail": self.pairing_fallback})
            else:
                raise RuntimeError(f"budget pairing: scenario {cfg.get('scenario')} declares budget.rule pairing with {bc.get('paired_scenario')}, "
                                   f"but the paired run's spend file is missing: {p}. Run the paper scenario at this seed and mode first, "
                                   f"or pass --allow-unpaired to use the fixed budget (logged).")

    # ------------------------------------------------------------ tariffs
    def tariff(self, call_type: str) -> int:
        return int(self.tariffs["calls"][call_type]["tokens"])

    def is_debited(self, call_type: str) -> bool:
        return call_type not in set(self.tariffs.get("not_debited", []))

    # ------------------------------------------------------------ debits
    def debit(self, quarter: int, resource: str, amount: float, cls: str, ref: str, who: str | None,
              admitted: str = "admitted", extra: dict | None = None) -> None:
        if self.suspended:
            admitted = "prehistory"
        if admitted == "admitted":
            q = self.spend.setdefault(int(quarter), {"data": 0.0, "compute": 0.0})
            q[resource] += float(amount)
        self.ledger.add("budget", {"quarter": int(quarter), "resource": resource, "amount": float(amount), "class": cls,
                                   "ref": ref, "who": who, "admitted": admitted, **(extra or {})})

    def call_scale(self, batch: int = 1, lean: bool = False) -> float:
        """Planning multiplier for a research/analyst call carrying `batch` studies and/or the lean paper contract
        (tariffs.yaml: batch_tariff_multiplier, lean_paper_multiplier; design defaults until measured from live runs)."""
        bm = self.tariffs.get("batch_tariff_multiplier", {1: 1.0})
        s = float(bm.get(batch, bm.get(str(batch), 1.0)))
        if lean:
            s *= float(self.tariffs.get("lean_paper_multiplier", 1.0))
        return s

    def debit_call(self, quarter: int, call_type: str, ref: str, who: str | None = None, realized_tokens: int | None = None,
                   scale: float = 1.0, failed: bool = False):
        """One agent call at tariff x scale. `realized_tokens` (usage.json of a live task; 0 for rule agents) travels in the
        same row as the tariff so both can be compared per call. A call that ended FAILED after its retries is recorded
        (admitted 'failed', amount 0): it is not part of the field's paired spend, but its realized tokens are kept."""
        extra = {"tariff": self.tariff(call_type), "scale": float(scale), "realized_tokens": realized_tokens}
        if realized_tokens is not None and not self.suspended:
            self.realized[int(quarter)] = self.realized.get(int(quarter), 0.0) + float(realized_tokens)
        if failed:
            self.debit(quarter, "compute", 0.0, call_type, ref, who, admitted="failed", extra=extra)
            return
        if not self.is_debited(call_type):
            self.debit(quarter, "compute", 0.0, call_type, ref, who, admitted="not_debited", extra=extra)
            return
        self.debit(quarter, "compute", self.tariff(call_type) * scale, call_type, ref, who, extra=extra)

    def debit_data(self, quarter: int, n_obs: int, ref: str, who: str | None = None):
        self.debit(quarter, "data", int(n_obs), "dataset", ref, who)

    # ------------------------------------------------------------ allotments (R under pairing / fixed)
    def require_paired(self, quarter: int) -> None:
        """Refuse to budget a quarter the paired paper run has not closed yet (so an extended or resumed pair stays in step)."""
        if self.rule != "pairing" or self.paired_spend is None or self.paired_closed is None:
            return
        if int(quarter) > self.paired_closed:
            raise PairingNotReady(int(quarter), self.paired_closed, self.paired_run_dir)

    def allotment(self, quarter: int) -> dict[str, float]:
        if quarter in self.allot:
            return self.allot[quarter]
        if self.rule == "pairing" and self.paired_spend is not None:
            ps = self.paired_spend.get(quarter, {"data": 0.0, "compute": 0.0})
            # compute is balanced one to one; data is a ceiling of k x the paired world's data (k = data_ceiling_multiple)
            base = {"data": float(ps.get("data", 0.0)) * self.data_ceiling_multiple, "compute": float(ps.get("compute", 0.0))}
        else:
            f = self.cfg["budget"]["fixed"]
            base = {"data": float(f["data_per_quarter"]), "compute": float(f["compute_per_quarter"])}
        a = {"data": max(0.0, float(base.get("data", 0.0)) + self.carry["data"]),
             "compute": max(0.0, float(base.get("compute", 0.0)) + self.carry["compute"])}
        self.allot[quarter] = a
        return a

    def remaining(self, quarter: int) -> dict[str, float]:
        a = self.allotment(quarter)
        s = self.spend.get(quarter, {"data": 0.0, "compute": 0.0})
        return {"data": a["data"] - s["data"], "compute": a["compute"] - s["compute"]}

    def close_quarter(self, quarter: int) -> None:
        """Unspent allotment carries forward; an overspend (a realized draw larger than its admitted estimate, e.g. duplicated rows
        or an unbalanced panel) is charged against the next quarter, so the paired cap holds cumulatively."""
        rem = self.remaining(quarter)
        self.carry = {"data": rem["data"], "compute": rem["compute"]}

    def admit(self, quarter: int, requests: list[dict]) -> tuple[list[dict], list[dict]]:
        """Requests: {'id', 'class' in priority list, 'lab', 'compute', 'data'}. Returns (admitted, deferred).
        Priority classes first; within a class labs are served by service debt (quarters since last deferral)."""
        order = {c: i for i, c in enumerate(self.cfg["budget"]["priority"])}
        pending = sorted(self.deferred + requests,
                         key=lambda r: (0 if r.get("deferred_from") is not None else 1, order.get(r["class"], 99),
                                        -self.service_debt.get(r.get("lab", ""), 0), str(r["id"])))
        if self.cfg["budget"].get("allocation") == "proportional_size":
            pending.sort(key=lambda r: (order.get(r["class"], 99), -float(r.get("lab_size", 1)), str(r["id"])))
        rem = self.remaining(quarter)
        before = dict(rem)
        admitted, deferred = [], []
        for req in pending:
            c, d = float(req.get("compute", 0)), float(req.get("data", 0))
            if c <= rem["compute"] and d <= rem["data"]:
                rem["compute"] -= c
                rem["data"] -= d
                admitted.append(req)
                self.service_debt[req.get("lab", "")] = 0
                self.ledger.add("budget", {"quarter": quarter, "resource": "request", "amount": c, "class": req["class"],
                                           "ref": req["id"], "who": req.get("lab"), "admitted": "admitted", "data": d,
                                           "deferred_from": req.get("deferred_from")})
            else:
                req = dict(req)
                req["deferred_from"] = req.get("deferred_from", quarter)
                # the resource that bound this request (every deferral names it)
                req["binding"] = "both" if (c > rem["compute"] and d > rem["data"]) else ("compute" if c > rem["compute"] else "data")
                req["short_by"] = {"compute": max(0.0, c - rem["compute"]), "data": max(0.0, d - rem["data"])}
                deferred.append(req)
                self.service_debt[req.get("lab", "")] = self.service_debt.get(req.get("lab", ""), 0) + 1
                self.ledger.add("budget", {"quarter": quarter, "resource": "request", "amount": c, "class": req["class"],
                                           "ref": req["id"], "who": req.get("lab"), "admitted": "deferred", "data": d,
                                           "binding": req["binding"], "short_by": req["short_by"]})
        self.deferred = deferred
        self.last_admission = {"quarter": quarter, "remaining_before": before, "remaining_after_admission": dict(rem),
                               "deferred": [{"id": r["id"], "class": r["class"], "lab": r.get("lab"), "compute": float(r.get("compute", 0)),
                                             "data": float(r.get("data", 0)), "binding": r["binding"], "short_by": r["short_by"]} for r in deferred],
                               "deferred_by_binding": {k: sum(1 for r in deferred if r["binding"] == k) for k in ("compute", "data", "both")}}
        return admitted, deferred

    # ------------------------------------------------------------ persistence (a run resumes from its files)
    def restore(self) -> dict:
        """Rebuild the tallies from `ledger/budget.jsonl` (spend and realized tokens) and `budget/state.json` (carry-over,
        deferred requests, service debt, allotments). Called when a run resumes; idempotent."""
        self.spend, self.realized = {}, {}
        rows = self.ledger.read("budget") if self.ledger.path("budget").exists() else []
        for b in rows:
            q = int(b.get("quarter", 0))
            if b.get("resource") in ("data", "compute") and b.get("admitted") == "admitted":
                self.spend.setdefault(q, {"data": 0.0, "compute": 0.0})[b["resource"]] += float(b.get("amount", 0.0))
            if b.get("resource") == "compute" and b.get("realized_tokens") is not None and b.get("admitted") != "prehistory":
                self.realized[q] = self.realized.get(q, 0.0) + float(b["realized_tokens"])
        sp = self.run_dir / "budget" / "state.json"
        if sp.exists():
            st = read_json(sp)
            self.carry = {"data": float(st.get("carry", {}).get("data", 0.0)), "compute": float(st.get("carry", {}).get("compute", 0.0))}
            self.deferred = list(st.get("deferred", []))
            self.service_debt = dict(st.get("service_debt", {}))
            self.allot = {int(k): v for k, v in st.get("allotments", {}).items()}
        return {"rows": len(rows), "quarters": sorted(self.spend)}

    def totals(self) -> dict:
        return {"data": sum(v["data"] for v in self.spend.values()), "compute": sum(v["compute"] for v in self.spend.values()),
                "realized_tokens": sum(self.realized.values())}

    # ------------------------------------------------------------ outputs
    def write(self) -> None:
        out = self.run_dir / "budget"
        write_json(out / "spend_by_quarter.json", {str(k): v for k, v in sorted(self.spend.items())})
        write_json(out / "summary.json", {"rule": self.rule, "pairing_fallback": self.pairing_fallback,
                                          "data_ceiling_multiple": self.data_ceiling_multiple if self.rule == "pairing" else None,
                                          "paired_spend": ({str(k): v for k, v in sorted(self.paired_spend.items())} if self.paired_spend else None),
                                          "allotments": {str(k): v for k, v in sorted(self.allot.items())},
                                          "total": self.totals(),
                                          "realized_tokens_by_quarter": {str(k): v for k, v in sorted(self.realized.items())},
                                          "deferred_outstanding": len(self.deferred), "tariff_status": self.tariffs.get("status")})
        write_json(out / "state.json", {"carry": self.carry, "deferred": self.deferred, "service_debt": self.service_debt,
                                        "allotments": {str(k): v for k, v in sorted(self.allot.items())}})


def totals_from_ledger(run_dir: Path) -> dict:
    """Audit-side recomputation (no engine objects): paired spend, realized tokens and call counts from `budget.jsonl`."""
    rows = read_jsonl(Path(run_dir) / "ledger" / "budget.jsonl") if (Path(run_dir) / "ledger" / "budget.jsonl").exists() else []
    out = {"data": 0.0, "compute": 0.0, "realized_tokens": 0.0, "calls": 0, "calls_failed": 0, "calls_not_debited": 0,
           "prehistory_calls": 0, "prehistory_data": 0.0, "by_class": {}}
    for b in rows:
        res, adm = b.get("resource"), b.get("admitted")
        if res == "compute":
            if adm == "admitted":
                out["compute"] += float(b.get("amount", 0.0)); out["calls"] += 1
            elif adm == "not_debited":
                out["calls_not_debited"] += 1
            elif adm == "failed":
                out["calls_failed"] += 1
            elif adm == "prehistory":
                out["prehistory_calls"] += 1
            if adm != "prehistory" and b.get("realized_tokens") is not None:
                out["realized_tokens"] += float(b["realized_tokens"])
            if adm in ("admitted", "not_debited", "failed"):
                c = out["by_class"].setdefault(b.get("class"), {"calls": 0, "tariff_tokens": 0.0, "realized_tokens": 0.0, "n_realized": 0})
                c["calls"] += 1
                c["tariff_tokens"] += float(b.get("amount", 0.0))
                if b.get("realized_tokens") is not None:
                    c["realized_tokens"] += float(b["realized_tokens"]); c["n_realized"] += 1
        elif res == "data":
            if adm == "admitted":
                out["data"] += float(b.get("amount", 0.0))
            elif adm == "prehistory":
                out["prehistory_data"] += float(b.get("amount", 0.0))
        elif res == "request" and adm == "deferred":
            out.setdefault("deferrals_by_binding", {"compute": 0, "data": 0, "both": 0, "unrecorded": 0})
            out["deferrals_by_binding"][b.get("binding") or "unrecorded"] += 1
    out.setdefault("deferrals_by_binding", {"compute": 0, "data": 0, "both": 0, "unrecorded": 0})
    # runs made with an early engine version logged the realized tokens of debited calls as `realized_tokens` events instead
    seen = {b.get("ref") for b in rows if b.get("realized_tokens") is not None}
    ev_path = Path(run_dir) / "ledger" / "events.jsonl"
    if ev_path.exists():
        for e in read_jsonl(ev_path):
            if e.get("event") == "realized_tokens" and e.get("ref") not in seen and e.get("tokens") is not None:
                out["realized_tokens"] += float(e["tokens"])
                seen.add(e.get("ref"))
    return out


def paired_totals(run_dir: Path) -> dict | None:
    """The paired paper world's realized spend (for reporting): read from the paired run's spend_by_quarter.json named in
    run.json, else from the engine summary's copy. None for an unpaired run."""
    rd = Path(run_dir)
    meta = read_json(rd / "run.json") if (rd / "run.json").exists() else {}
    sp = None
    p = Path(meta["paired_run_dir"]) / "budget" / "spend_by_quarter.json" if meta.get("paired_run_dir") else None
    if p and p.exists():
        sp = read_json(p)
    else:
        summ = read_json(rd / "budget" / "summary.json") if (rd / "budget" / "summary.json").exists() else {}
        sp = summ.get("paired_spend")
    if not sp:
        return None
    return {"data": sum(float(v.get("data", 0.0)) for v in sp.values()), "compute": sum(float(v.get("compute", 0.0)) for v in sp.values()),
            "quarters": len(sp), "source": str(p) if (p and p.exists()) else "budget/summary.json"}
