"""The one refit allowed by the recalibration protocol (pass criteria fixed in tools/fastcal_validate.py before fitting).

The second validation round (runs/fastcal-2026-10-15-b, 30 seeds) failed these pass criteria (ids as in
tools/fastcal_validate.py): P25 (paper compute), R1 (repository supported
share), R2/R4 (vote mix) and R6 (verification power). Two changes, each aimed at a diagnosed cause:

1. Editor intercepts (cause of P25). The fast engine realised a round-1 reject share of .482 and a round-2 accept share of
   .140 against the live .541 and .208 that the composite simulation was fitted to: the composite (live papers' conditions)
   and the fast engine's papers differ. More revisions mean more re-reviews and revision calls (fast 19 round-2 decisions
   and 204 reviews per run against live 11-16 and 156-201), so compute runs high. The intercepts are moved by the shift
   that takes the composite from the live share to the live share plus the engine's shortfall (the composite-to-engine
   offset is assumed constant near the solution); validation seeds are not used to fit.
2. `fast_honours_deposit_n: true` (cause of R1, R2, R4, R6). The fast analytic study ignored the size a lab
   head requested and the claim's N set for a verification draw: fast focal N median 722 against live 3,038, so few
   significant entries, low endorsement (endorsement rises with N in both modes) and low verification power.

Writes configs/fastmode/params-stage1.yaml (the first fit is kept as configs/fastmode/params-stage1-fit1.yaml).
    python tools/fastcal_refit.py [--sims 40]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# realised in the second validation round (median over 30 fast S0-150 seeds; runs/fastcal-2026-10-15-b/criteria_results.md) and live pooled
ENGINE_R1_REJECT, LIVE_R1_REJECT = 0.482, 125 / 231
ENGINE_R2_ACCEPT, LIVE_R2_ACCEPT = 0.140, 15 / 72


def main() -> int:
    import yaml
    from calibrate_fastmode_stage1 import load_decisions, load_reviews, paper_conditions, simulate
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40)
    a = ap.parse_args()
    src = ROOT / "configs" / "fastmode" / "params-stage1.yaml"
    keep = ROOT / "configs" / "fastmode" / "params-stage1-fit1.yaml"
    if not keep.exists():
        shutil.copyfile(src, keep)
    text = keep.read_text(encoding="utf-8")
    p = yaml.safe_load(text)
    conds = paper_conditions(load_decisions(), load_reviews())
    p_sim = copy.deepcopy(p)
    p_sim["editor"]["accept_cut_round1"] = 50.0

    def bisect(fun, lo, hi, target, increasing, it=16):
        for _ in range(it):
            mid = (lo + hi) / 2
            if (fun(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    t1 = LIVE_R1_REJECT + (LIVE_R1_REJECT - ENGINE_R1_REJECT)
    i1 = bisect(lambda v: simulate(p_sim | {"editor": p_sim["editor"] | {"intercept_round1": v}}, conds, a.sims)["r1"].get("reject", 0.0),
                p["editor"]["intercept_round1"] - 4, p["editor"]["intercept_round1"] + 4, t1, increasing=False)
    p_sim["editor"]["intercept_round1"] = i1
    base2 = simulate(p_sim, conds, a.sims)["r2_accept"]
    t2 = base2 + (LIVE_R2_ACCEPT - ENGINE_R2_ACCEPT)
    i2 = bisect(lambda v: simulate(p_sim | {"editor": p_sim["editor"] | {"intercept_round2": v}}, conds, a.sims)["r2_accept"] or 0.0,
                p["editor"]["intercept_round2"] - 4, p["editor"]["intercept_round2"] + 4, t2, increasing=True)
    # the accept threshold keeps its distance above the round-1 intercept (the expected round-1 acceptances stay ~1/2)
    cut = p["editor"]["accept_cut_round1"] + (i1 - p["editor"]["intercept_round1"])
    old = (p["editor"]["intercept_round1"], p["editor"]["intercept_round2"], p["editor"]["accept_cut_round1"])
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("intercept_round1:"):
            ln = f"  intercept_round1: {round(i1, 3)}   # refit (tools/fastcal_refit.py): first fit {old[0]}; composite target {t1:.3f} = live .541 + engine shortfall {LIVE_R1_REJECT - ENGINE_R1_REJECT:.3f}"
        elif s.startswith("intercept_round2:"):
            ln = f"  intercept_round2: {round(i2, 3)}   # refit: first fit {old[1]}; composite target {t2:.3f} = composite at the new i1 {base2:.3f} + engine shortfall {LIVE_R2_ACCEPT - ENGINE_R2_ACCEPT:.3f}"
        elif s.startswith("accept_cut_round1:"):
            ln = f"  accept_cut_round1: {round(cut, 3)}   # refit: first fit {old[2]}, moved with intercept_round1 (0/231 live round-1 acceptances)"
        elif s.startswith("status:"):
            ln = "status: calibrated-stage1-2026-10-15-refit1"
        lines.append(ln)
    lines.insert(next(i for i, l in enumerate(lines) if l.startswith("status:")),
                 "# REFIT 1 (tools/fastcal_refit.py, after the second validation round failed some pass criteria): editor intercepts moved by\n"
                 "# the composite-to-engine offset; fast analyses and verifications keep a deposit's requested size. The refit saves the first\n"
                 "# fit next to this file as params-stage1-fit1.yaml.")
    lines.append("fast_honours_deposit_n: true   # the fast analytic study keeps a requested deposit size and a verification draw's claim N")
    out = "\n".join(lines) + "\n"
    yaml.safe_load(out)
    src.write_text(out, encoding="utf-8")
    print(f"intercept_round1 {old[0]} -> {i1:.3f}; intercept_round2 {old[1]} -> {i2:.3f}; accept_cut_round1 {old[2]} -> {cut:.3f}")
    print(f"wrote {src} ({hashlib.sha256(src.read_bytes()).hexdigest().upper()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
