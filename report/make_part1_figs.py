"""Part I (pendant drop) slide figure: surface tension per image with error bars.
Values transcribed from the "Summary of Results" tables (Table 1: per-image fits, Table 2: outliers).

    .venv/bin/python report/make_part1_figs.py      # -> images/gamma_errorbars.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(exist_ok=True)

# (ethanol % v/v, gamma [N/m], dgamma [N/m])
IMAGES = [(0, 0.06290, 0.00966), (0, 0.07178, 0.01183), (0, 0.07226, 0.01141), (0, 0.06583, 0.01053), (0, 0.06557, 0.01058),
          (10, 0.05143, 0.00899), (10, 0.05246, 0.00814), (10, 0.05246, 0.00843), (10, 0.05181, 0.00791),
          (20, 0.04769, 0.00747), (20, 0.04158, 0.00688), (20, 0.04431, 0.00692),
          (40, 0.02563, 0.00439), (40, 0.02556, 0.00455), (40, 0.01865, 0.00322), (40, 0.02398, 0.00382),
          (40, 0.02728, 0.00453), (40, 0.02554, 0.00446),
          (70, 0.02524, 0.00372), (70, 0.02686, 0.00397), (70, 0.02610, 0.00408)]
OUTLIERS = [(0, 0.03059, 0.00494), (0, 0.03070, 0.00501)]           # water S04, S05 (excluded)
LIT = [(0, 72.0), (10, 50.7), (20, 41.4), (40, 32.1), (70, 25.8)]   # literature, mN/m (as quoted on the slide)

BG, INK, INK2, GRID = "#1c2030", "#e6e6ea", "#b4b5c0", "#353b50"
BLUE, GOLD, CORAL = "#7fa8d1", "#d9a441", "#e5645c"                 # accent2 / accent / outlier of the deck
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "legend.frameon": False, "legend.labelcolor": INK})


def jitter(rows, step=1.1):
    """spread images at the same concentration side by side"""
    out, seen = [], {}
    n = {c: sum(1 for r in rows if r[0] == c) for c, *_ in rows}
    for c, g, dg in rows:
        k = seen.get(c, 0); seen[c] = k + 1
        out.append((c + (k - (n[c] - 1) / 2) * step, 1e3 * g, 1e3 * dg))
    return out


fig, ax = plt.subplots(figsize=(3.1, 1.85))
x, g, dg = zip(*jitter(IMAGES))
ax.errorbar(x, g, yerr=dg, fmt="o", ms=3, color=BLUE, ecolor=BLUE, elinewidth=0.7, capsize=1.5, capthick=0.7,
            mec=BG, mew=0.4, label="fitted, one point per image ($\\pm\\delta\\gamma$)")
xo, go, dgo = zip(*[(c + 3.2 + 1.1 * k, 1e3 * a, 1e3 * b) for k, (c, a, b) in enumerate(OUTLIERS)])
ax.errorbar(xo, go, yerr=dgo, fmt="o", ms=3, mfc="none", mec=CORAL, ecolor=CORAL, elinewidth=0.7, capsize=1.5,
            capthick=0.7, mew=0.8, label="outliers, water S04/S05 (excluded)")
for c in sorted({r[0] for r in IMAGES}):
    m = 1e3 * np.mean([r[1] for r in IMAGES if r[0] == c])
    ax.hlines(m, c - 3.6, c + 3.6, color=INK, lw=1.0, zorder=3)
ax.plot([], [], color=INK, lw=1.0, label="mean of the images")
lx, ly = zip(*LIT)
ax.plot(lx, ly, "--D", color=GOLD, lw=0.9, ms=3, label="literature")
ax.set_xlabel("ethanol [% v/v]")
ax.set_ylabel("$\\gamma$ [mN/m]")
ax.set_xticks([0, 10, 20, 40, 70])
ax.set_xlim(-6, 76)
ax.set_ylim(10, 90)
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "gamma_errorbars.pdf")
fig.savefig(OUT / "gamma_errorbars.png", dpi=250)
print("written", OUT / "gamma_errorbars.pdf")
