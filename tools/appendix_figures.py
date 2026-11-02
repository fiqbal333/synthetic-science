"""Two summary figures of the 20-quarter pilot results for the technical appendix.

Reads analysis/x20/ (results_by_run.csv, handbook_trajectory.csv, planted_false_literature.csv)
and writes docs/figures/appendix_fig1_certification_JR.png and appendix_fig2_tradeoffs_JR.png.
Descriptive only: 3 seeds per world (2 for S0P); every dot is one run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

W = Path(__file__).resolve().parents[1]
A = W / "analysis" / "x20"
OUT = W / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

WORLDS = [("S0-150", "Journal (S0)"), ("R-b-150", "Repository (R)")]
COLOR = {"S0-150": "#2a78d6", "S0P-150": "#eb6834", "R-b-150": "#1baf7a"}  # reference slots 1-3
INK, SOFT, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": SOFT,
    "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

runs = pd.read_csv(A / "results_by_run.csv")
traj = pd.read_csv(A / "handbook_trajectory.csv")
lit = pd.read_csv(A / "planted_false_literature.csv")
lit = lit[lit.snapshot == "Q20"].copy()
lit["still_share"] = lit.still_believed_B1 / lit.inherited_false_units
runs = runs.merge(lit[["world", "seed", "still_share"]], on=["world", "seed"])
keep = [w for w, _ in WORLDS]
runs, traj = runs[runs.world.isin(keep)], traj[traj.world.isin(keep)]


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color=MUTED, ha="left", va="bottom")


# ---------- Figure 1: what each world certified, and the handbook over time ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [0.9, 1]})
cats = [("units_true", "True", "#3d3d3a", None),
        ("units_missed_effect", "Missed effect (certified as null)", "#b9b8b2", None),
        ("units_false_discovery", "False discovery", "#e34948", None),
        ("units_unknown", "Not scorable", "#ffffff", "////")]
x, ticks, labels, groups = 0, [], [], []
for w, name in WORLDS:
    start = x
    for _, r in runs[runs.world == w].sort_values("seed").iterrows():
        bottom = 0
        for col, lab, c, h in cats:
            v = r[col]
            if v:
                a.bar(x, v, bottom=bottom, width=0.72, color=c, edgecolor=SURF if not h else MUTED,
                      linewidth=1.5 if not h else 0.6, hatch=h)
            bottom += v
        a.text(x, bottom + 0.4, f"{int(bottom)}", ha="center", va="bottom", fontsize=9, color=SOFT)
        ticks.append(x); labels.append(f"s{int(r.seed)}")
        x += 1
    groups.append(((start + x - 1) / 2, name, w))
    x += 0.8
a.set_xticks(ticks, labels, fontsize=9)
for gx, name, w in groups:
    a.text(gx, -2.9, name, ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
a.set_ylabel("Certified knowledge units, quarters 1-20")
a.set_ylim(0, 28)
a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
a.set_title("A. Certified units by truth, per run")
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=MUTED if h else c, hatch=h, lw=0.6) for _, _, c, h in cats]
a.legend(handles, [l for _, l, _, _ in cats], loc="upper left", frameon=False, fontsize=8.5)

for w, name in WORLDS:
    t = traj[traj.world == w]
    for s, g in t.groupby("seed"):
        g = g.sort_values("quarter")
        b.plot(g.quarter, g.accuracy, color=COLOR[w], lw=1, alpha=0.35)
    m = t.groupby("quarter").accuracy.mean()
    b.plot(m.index, m.values, color=COLOR[w], lw=2.2, marker="o", ms=5, mec=SURF, mew=1.2)
    b.text(20.6, m.values[-1], f"{name.replace(chr(10), ' ')}  {m.values[-1]:.2f}", va="center", fontsize=8.5, color=INK)
b.set_xlim(-0.5, 20.4); b.set_xticks([0, 4, 8, 12, 16, 20])
b.set_xlabel("Quarter"); b.set_ylabel("Handbook accuracy (share of answered\ncore questions with the right sign)")
b.set_ylim(0.55, 0.85)
b.yaxis.grid(True, color=GRID, lw=0.8); b.set_axisbelow(True)
b.set_title("B. Handbook accuracy over 20 quarters")
b.text(0.2, 0.565, "Thick line: mean across seeds; thin lines: single runs", fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.80, bottom=0.25, top=0.9, wspace=0.28)
note(fig, f"Six pilot runs, 150 scientists each, 20 quarters. {int(runs.certified_units.sum())} units certified in all; "
          f"{int(runs.units_false_discovery.sum())} was a false discovery. "
          "Descriptive only: 3 seeds per world, no tests. Source: analysis/x20/.")
fig.savefig(OUT / "appendix_fig1_certification_JR.png", dpi=220)
plt.close(fig)

# ---------- Figure 2: trade-offs between the worlds ----------
panels = [("focal_analyses_live_true_share", "A. True share of live estimates,\nbefore certification", "{:.2f}", (0.4, 0.8), 1),
          ("start_to_certification_median_q_live", "B. Median quarters to certification", "{:.1f}", (0, 10), 1),
          ("data_per_correct_handbook_answer", "C. Data per correct handbook answer\n(thousand observations)", "{:.0f}", (0, 160), 1e-3),
          ("still_share", "D. Inherited false units still\nbelieved at quarter 20", "{:.0%}", (0.5, 1.0), 1)]
fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
ys = {w: i for i, (w, _) in enumerate(reversed(WORLDS))}
for ax, (col, title, fmt, lim, k) in zip(axs, panels):
    for w, name in WORLDS:
        v = runs.loc[runs.world == w, col] * k
        ax.scatter(v, [ys[w]] * len(v), s=60, color=COLOR[w], edgecolor=SURF, lw=1.2, zorder=3)
        ax.plot([v.mean()] * 2, [ys[w] - 0.25, ys[w] + 0.25], color=INK, lw=1.6, zorder=4)
        ax.text(v.mean(), ys[w] + 0.3, fmt.format(v.mean()), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlim(*lim); ax.set_ylim(-0.6, len(WORLDS) - 0.25)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    if col == "still_share":
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[0].set_yticks(list(ys.values()), [n for w, n in reversed(WORLDS)], fontsize=9.5)
fig.subplots_adjust(left=0.12, right=0.975, bottom=0.2, top=0.8, wspace=0.2)
note(fig, "Dots: single runs; black tick and number: mean. B counts from the study's start in the journal world "
          "and from the analysis in the repository, so it is indicative only.\nD uses the vote-count belief rule. "
          "Descriptive only: 3 seeds per world. Source: analysis/x20/.")
fig.savefig(OUT / "appendix_fig2_tradeoffs_JR.png", dpi=220)
plt.close(fig)
print("written:", *sorted(p.name for p in OUT.glob("appendix_fig*.png")))
