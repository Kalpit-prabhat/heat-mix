"""Cross-section just before the split (left) next to every branch outlet (right).

    .venv/bin/python scripts/plot_presplit.py FIELDS.json [OUT.png]

FIELDS.json is a branched sine_split_2d field export (``python -m comsol_opt fields ...``),
e.g. runs/final_b2d_serp/fields/run0150_fine.json; the plot goes next to it as
<name>_presplit.png.  Both temperature axes span the same number of mK (different
offsets); the speed axes are identical.
Note: outlet samples are read in grid_y_mm order (branch by branch); grid_eta in the
export is interleaved and is not used.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_presplit.png")
d = json.loads(src.read_text())
F = d["fields"]
Tn = F["T_need"]
x0 = F["slice_x"][-1]
y = np.array(F["slice_y"][-1])
Ts = (np.array(F["slice_T"][-1]) - Tn) * 1000
Us = np.array(F["slice_U"][-1]) * 1000
yc = y - (y[0] - (y[1] - y[0]) / 2)                      # 0 = lower wall
e = np.concatenate([[0], np.cumsum(d["branch_w_in"])])  # branch inlet edges
N = d["n_branch"]
ny = len(d["T"]) // N
To = (np.array(d["T"]).reshape(N, ny) - Tn) * 1000
Uo = np.array(d["un"]).reshape(N, ny) * 1000
W = float(np.sum(d["branch_w_in"]))                     # main channel width [mm]
wo = W / N                                              # outlet width of each branch
eta = (np.arange(ny) + 0.5) / ny

span = 10 * np.ceil(max(Ts.max() - Ts.min(), To.max() - To.min()) * 1.25 / 10)
lo_s = 10 * np.floor(((Ts.min() + Ts.max()) / 2 - span / 2) / 10)
lo_o = 10 * np.floor(((To.min() + To.max()) / 2 - span / 2) / 10)
umax = 10 * np.ceil(max(Uo.max(), Us.max()) * 1.1 / 10)

BLUE, INK, INK2, GRID, SURF, BAND = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeeea"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(2, 2, figsize=(14, 6.8), sharex="col", gridspec_kw={"hspace": 0.12, "wspace": 0.16})


def deco(a, edges):
    for i in range(N):
        if i % 2 == 0:
            a.axvspan(edges[i], edges[i + 1], color=BAND, lw=0, zorder=0)
    for v in edges[1:-1]:
        a.axvline(v, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


eo = np.arange(N + 1) * wo
for a in ax[:, 0]:
    deco(a, e)
for a in ax[:, 1]:
    deco(a, eo)
ax[0, 0].plot(yc, Ts, color=BLUE, lw=2, zorder=3)
ax[1, 0].plot(yc, Us, color=BLUE, lw=2, zorder=3)
for i in range(N):
    xx = i * wo + eta * wo
    ax[0, 1].plot(xx, To[i], color=BLUE, lw=2, zorder=3)
    ax[1, 1].plot(xx, Uo[i], color=BLUE, lw=2, zorder=3)
ax[0, 1].axhline(0, color=INK2, lw=0.8, zorder=2)
ax[0, 0].set_ylim(lo_s, lo_s + span)
ax[0, 1].set_ylim(lo_o, lo_o + span)
ax[1, 0].set_ylim(0, umax)
ax[1, 1].set_ylim(0, umax)
for i in range(N):
    m = (yc >= e[i]) & (yc < e[i + 1])
    Tb = np.sum(Ts[m] * Us[m]) / np.sum(Us[m])
    xm, xo = (e[i] + e[i + 1]) / 2, (i + 0.5) * wo
    ax[0, 0].text(xm, lo_s + span, f"branch {i + 1}", ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
    ax[0, 1].text(xo, lo_o + span, f"branch {i + 1}", ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
    ax[0, 0].text(xm, lo_s + 0.03 * span, f"bulk {Tb:+.0f}", ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax[0, 1].text(xo, lo_o + 0.03 * span, f"bulk {d['branch_Tb_dev'][i] * 1000:+.1f}", ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax[1, 0].text(xm, 0.035 * umax, f"{d['branch_w_in'][i]:.2f} mm", ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax[1, 1].text(xo, 0.035 * umax, f"ΔQ {d['branch_Q_dev'][i] * 100:+.2f} %", ha="center", va="bottom", color=INK2, fontsize=8.5)
ax[0, 0].set_ylabel("T − T_need  [mK]")
ax[1, 0].set_ylabel("speed  [mm/s]")
ax[1, 0].set_xlabel("position across the main channel  [mm]   (0 = lower wall)")
ax[1, 1].set_xlabel(f"outlet of each branch, side by side (each {wo:.2f} mm wide, lower wall left)  [mm]")
ax[1, 0].set_xlim(0, W)
ax[1, 1].set_xlim(0, W)
ax[0, 0].set_title(f"Just before the split, x = {x0:.2f} mm", loc="left", color=INK, fontsize=10.5, pad=18)
ax[0, 1].set_title(f"At the branch outlets, x = {F['L']:.0f} mm", loc="left", color=INK, fontsize=10.5, pad=18)
fig.suptitle(f"{src.stem}: temperature axes span {span:.0f} mK in both columns (offsets {lo_s:.0f}…{lo_s + span:.0f} "
             f"vs {lo_o:.0f}…{lo_o + span:.0f} mK); speed axes identical", x=0.07, y=1.0, ha="left", color=INK2, fontsize=9)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
