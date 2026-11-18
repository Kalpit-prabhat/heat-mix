"""Figures for the channel slides -> images/ch_*.png
Crops the temperature panel out of the fine-mesh field maps (title text removed).

    .venv/bin/python report/make_slide_figs.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SRC = HERE / "figs" / "runs"
OUT = HERE.parent / "images"
OUT.mkdir(exist_ok=True)

# name: (source, title rows to blank, bottom of the temperature panel incl. colorbar)
CROPS = {
    "ch_u2d_straight_T": ("final_u2d_straight_ref_single_maps.png", 112, 352),
    "ch_u2d_straight_win_T": ("final_u2d_straight_ref_maps.png", 112, 352),
    "ch_u2d_sine_T": ("final_u2d_sine_grid_maps.png", 152, 628),
    "ch_u2d_serp_T": ("final_u2d_serp_grid_maps.png", 152, 628),
    "ch_u2d_serp_bo_T": ("final_u2d_serp_maps.png", 129, 645),
    "ch_u3d_serp_T": ("final_u3d_serp_maps.png", 127, 642),
    "ch_b2d_ref_T": ("final_b2d_straight_ref_maps.png", 102, 648),
    "ch_b2d_straight_T": ("final_b2d_straight_maps.png", 102, 648),
    "ch_b2d_sine_T": ("final_b2d_sine_maps.png", 102, 648),
    "ch_b2d_serp_T": ("final_b2d_serp_maps.png", 102, 648),
}
for name, (src, title_end, bottom) in CROPS.items():
    im = Image.open(SRC / src).convert("RGB")
    ImageDraw.Draw(im).rectangle([0, 0, 1655, title_end], fill="white")
    # tight crop of what is left
    box = Image.eval(im.crop((0, 0, im.width, bottom)), lambda v: 255 - v).getbbox()
    im.crop((max(box[0] - 8, 0), max(box[1] - 8, 0), min(box[2] + 8, im.width), bottom)).save(OUT / f"{name}.png")

# ------------------------------------------------------------------ dark charts at slide size
# Slide text area is 5.5 x ~2.9 in; figures are drawn at the size they are shown, text ~6.5 pt.
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.ticker import NullFormatter

R = HERE.parent / "runs"
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]  # dark steps of the report hues
INK, INK2, GRID, BG = "#e6e6ea", "#b4b5c0", "#353b50", "#1c2030"       # beamer colours of the deck
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
                     "lines.linewidth": 1.4, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": INK, "legend.labelcolor": INK, "legend.facecolor": BG, "legend.edgecolor": INK2})


def fj(p):
    return json.load(open(R / p))


def best(run):
    return json.load(open(R / run / "best.json"))


def recs(run):
    return [json.loads(l) for l in open(R / run / "history.jsonl")]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# 1. summary
un = [("2D straight", fj("final_u2d_straight_ref_single/fields/design_fine.json")["T_rms_flow"]),
      ("2D straight, best heater", best("final_u2d_straight_ref")["metrics"]["l2_flow"]),
      ("2D sine, BO", best("final_u2d_sine")["metrics"]["l2_flow"]),
      ("2D sine, grid", best("final_u2d_sine_grid")["metrics"]["l2_flow"]),
      ("2D serpentine, BO", best("final_u2d_serp")["metrics"]["l2_flow"]),
      ("2D serpentine, grid", best("final_u2d_serp_grid")["metrics"]["l2_flow"]),
      ("3D straight", fj("final_u3d_straight_ref_single/fields/design_fine.json")["T_rms_flow"]),
      ("3D straight, best heater", best("final_u3d_straight_ref")["metrics"]["l2_flow"]),
      ("3D serpentine, BO", best("final_u3d_serp")["metrics"]["l2_flow"])]
br = [("straight, analytic openings", best("final_b2d_straight_ref")),
      ("straight, BO", best("final_b2d_straight")),
      ("sine + straight, BO", best("final_b2d_sine")),
      ("serpentine + straight, BO", best("final_b2d_serp"))]
fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.75), layout="constrained")
for ax, rows, title in ((axs[0], [(n, 1e3 * v) for n, v in un], "one outlet: RMS $\\ell_{2,\\mathrm{flow}}$ [mK]"),
                        (axs[1], [(n, 1e3 * b["metrics"]["branch_T"]) for n, b in br],
                         "branches: $T$ spread [mK]")):
    y = np.arange(len(rows))[::-1]
    v = [r[1] for r in rows]
    ax.hlines(y, 1, v, color=GRID, lw=1.5)
    ax.plot(v, y, "o", color=C[0], ms=4.5, mec=BG, mew=0.8)
    for yi, vi in zip(y, v):
        ax.text(vi * 1.35, yi, f"{vi:.3g}", va="center", color=INK, fontsize=6)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(3, 8000)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
save(fig, "ch_summary")


# 2. outlet profiles
def prof(p):
    d = fj(p)
    T = np.array(d["T"], float) - d["T_need"]
    if T.ndim == 2:
        T = np.nanmean(T, axis=1)
    return np.array(d["grid_eta"]) + 0.5, 1e3 * T


u2 = [("straight, fully heated", "final_u2d_straight_ref_single/fields/design_fine.json", C[0]),
      ("straight, best heater", "final_u2d_straight_ref/fields/run0001_fine.json", C[1]),
      ("sine (λ 11.1, a 1)", "final_u2d_sine_grid/fields/run0112_fine.json", C[2]),
      ("serpentine BO (λ 8.7, A 3.1)", "final_u2d_serp/fields/run0051_fine.json", C[3]),
      ("serpentine grid (λ 8.3, A 2.1)", "final_u2d_serp_grid/fields/run0108_fine.json", C[4])]
u3 = [("straight, fully heated", "final_u3d_straight_ref_single/fields/design_fine.json", C[0]),
      ("straight, best heater", "final_u3d_straight_ref/fields/run0001_fine.json", C[1]),
      ("serpentine (λ 7.3, A 3.5)", "final_u3d_serp/fields/run0029_fine.json", C[5])]

# 2a. the problem: outlet profile of the fully heated straight channel, in K above the inlet
d = fj(u2[0][1])
fig, ax = plt.subplots(figsize=(2.15, 1.35))
ax.plot(np.array(d["grid_eta"]) + 0.5, np.array(d["T"], float) - d["T_in"], color=C[0], label="outlet")
ax.axhline(d["T_need"] - d["T_in"], color=INK2, lw=0.8, ls="--")
ax.text(0.5, d["T_need"] - d["T_in"] + 0.12, "target  $\\Delta T$ = 2 K", ha="center", va="bottom", fontsize=6, color=INK)
ax.set_xticks([0, 0.5, 1])
ax.set_xlabel("y / w across the outlet")
ax.set_ylabel("$T - T_\\mathrm{in}$ [K]")
ax.set_title("outlet, straight channel fully heated", loc="left")
fig.tight_layout(pad=0.3)
save(fig, "ch_problem_profile")

# 2b. heater only: outlet profile, fully heated vs best window
fig, ax = plt.subplots(figsize=(2.6, 1.3))
for lab, p, col in u2[:2]:
    ax.plot(*prof(p), color=col, label=lab)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks([0, 0.5, 1])
ax.set_xlabel("y / w across the outlet")
ax.set_ylabel("$T - T_\\mathrm{need}$ [mK]")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
fig.tight_layout(pad=0.3)
save(fig, "ch_heater_profile")

# 2c. 2D results: sine and serpentine on separate plots
fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.75))
for ax, rows, title in ((axs[0], u2[2:3], "sine"), (axs[1], u2[3:], "serpentine")):
    for lab, p, col in rows:
        ax.plot(*prof(p), color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("y / w across the outlet")
    ax.set_ylabel("$T - T_\\mathrm{need}$ [mK]")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left" if title == "sine" else "lower center", handlelength=1.2)
axs[0].set_ylim(-260, 230)
axs[1].set_ylim(-85, 30)
fig.tight_layout(pad=0.3, w_pad=1.5)
save(fig, "ch_outlet_2d")

# 2e. 3D serpentine: depth-averaged outlet profile
fig, ax = plt.subplots(figsize=(2.6, 1.35))
lab, p, col = u3[2]
ax.plot(*prof(p), color=col, label=lab)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks([0, 0.5, 1])
ax.set_xlabel("y / w across the outlet")
ax.set_ylabel("$T - T_\\mathrm{need}$ [mK]")
ax.set_title("3D serpentine, depth-averaged outlet", loc="left")
fig.tight_layout(pad=0.3)
save(fig, "ch_outlet_3d")

# 3. spread along x (2D), sine and serpentines
fig, ax = plt.subplots(figsize=(2.65, 1.5))
for lab, p, col in u2[2:]:
    d = fj(p)["profile"]
    ax.plot(d["x_stat"], 1e3 * np.array(d["T_std_x"], float), color=col,
            label=lab.replace("serpentine ", "serp. ").split(" (")[0])
ax.set_yscale("log")
ax.set_xlabel("x [mm]")
ax.set_ylabel("std of T across [mK]")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
fig.tight_layout(pad=0.3)
save(fig, "ch_along_x_2d")

# 4. grid landscape (coarse grid) with BO points
fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.8))
cmap = ListedColormap(plt.cm.Blues(np.linspace(0.1, 0.75, 256)))  # pale = better, readable on dark
for ax, (grid, bo, yv, ylab, name) in zip(axs, (("final_u2d_sine_grid", "final_u2d_sine", "a", "a = A / A$_\\max$", "sine"),
                                                ("final_u2d_serp_grid", "final_u2d_serp", "A", "A [mm]", "serpentine"))):
    g = [r for r in recs(grid) if r["fidelity"] == "coarse" and r["status"] == "ok"]
    z = 1e3 * np.array([r["metrics"]["l2_flow"] for r in g])
    sc = ax.scatter([r["params"]["lam"] for r in g], [r["params"][yv] for r in g], c=z, cmap=cmap, s=11,
                    edgecolor=BG, linewidth=0.3, norm=LogNorm(z.min(), z.max()))
    fl = [r for r in recs(grid) if r["status"] != "ok"]
    ax.scatter([r["params"]["lam"] for r in fl], [r["params"][yv] for r in fl], marker="x", color="#e66767", s=10,
               linewidth=0.8, label="failed")
    b = [r for r in recs(bo) if r["fidelity"] == "coarse" and r["status"] == "ok"]
    ax.scatter([r["params"]["lam"] for r in b], [r["params"][yv] for r in b], marker="+", color=C[1], s=14,
               linewidth=0.8, label="BO run")
    gb, bb = best(grid), best(bo)
    ax.plot(gb["params"]["lam"], gb["params"][yv], "s", mfc="none", mec=INK, ms=7, mew=1, label="grid best")
    ax.plot(bb["params"]["lam"], bb["params"][yv], "D", mfc="none", mec=C[1], ms=6, mew=1, label="BO best")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 8, 12, 20], ["2", "3", "5", "8", "12", "20"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("wavelength λ [mm]")
    ax.set_ylabel(ylab)
    ax.grid(False)
    ax.set_title(name, loc="left")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
    t = [v for v in (5, 10, 20, 50, 100, 150, 200, 300, 400, 1000) if z.min() <= v <= z.max()]
    cb.set_ticks(t, labels=[str(v) for v in t])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label("$\\ell_{2,\\mathrm{flow}}$ [mK]", fontsize=6)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.89))
save(fig, "ch_grid_landscape")


# 5. 3D exit plane: temperature and speed maps (half depth solved, mirrored about mid-depth)
DIV = plt.cm.RdBu_r
SPD = ListedColormap(plt.cm.Blues(np.linspace(0.95, 0.08, 256)))  # dark = slow, pale = fast


def exit_plane(p):
    d = fj(p)
    y = np.array(d["grid_y_mm"]); z = np.array(d["grid_z_mm"])
    T = 1e3 * (np.array(d["T"], float) - d["T_need"]); u = 1e3 * np.array(d["un"], float)
    if d.get("symmetric"):
        z = np.r_[z, 2 * z.max() + (z[1] - z[0]) - z[::-1]]
        T = np.c_[T, T[:, ::-1]]; u = np.c_[u, u[:, ::-1]]
    return y, z, T, u


def edges(c):
    h = np.diff(c).mean()
    return np.r_[c - h / 2, c[-1] + h / 2]


def plane(ax, fig, y, z, V, cmap, vmin, vmax, title, cblab, ticks=None):
    m = ax.pcolormesh(edges(y), edges(z), V.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlabel("y [mm]")
    ax.set_title(title, loc="left")
    ax.grid(False)
    ax.set_xticks([-1.5, 0, 1.5]); ax.set_yticks([0, 1.5, 3])
    cb = fig.colorbar(m, ax=ax, fraction=0.05, pad=0.03, ticks=ticks)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label(cblab, fontsize=6)


fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.65))
for k, (ax, (lab, p, _)) in enumerate(zip(axs[:2], u3[:2])):
    y, z, T, u = exit_plane(p)
    plane(ax, fig, y, z, T, DIV, -1200, 1200, lab, "" if k == 0 else "$T - T_\\mathrm{need}$ [mK]")
y, z, T, u = exit_plane(u3[0][1])
plane(axs[2], fig, y, z, u, SPD, 0, np.nanmax(u), "speed (both)", "$u_n$ [mm/s]")
axs[0].set_ylabel("z [mm]")
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "ch_u3d_exit_straight")

fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.45))
y, z, T, u = exit_plane(u3[2][1])
lim = np.nanmax(np.abs(T))
plane(axs[0], fig, y, z, T, DIV, -lim, lim, "temperature", "$T - T_\\mathrm{need}$ [mK]")
plane(axs[1], fig, y, z, u, SPD, 0, np.nanmax(u), "speed", "$u_n$ [mm/s]")
axs[0].set_ylabel("z [mm]")
fig.tight_layout(pad=0.3, w_pad=0.6)
save(fig, "ch_u3d_exit_serp")

# 6. branched runs: balance of the four branches
BR = [("straight, analytic\nopenings", "final_b2d_straight_ref/fields/run0001_fine.json"),
      ("straight, BO", "final_b2d_straight/fields/run0016_fine.json"),
      ("sine + straight, BO", "final_b2d_sine/fields/run0150_fine.json"),
      ("serpentine +\nstraight, BO", "final_b2d_serp/fields/run0150_fine.json")]
fig, axs = plt.subplots(2, 4, figsize=(5.4, 2.3), sharex=True)
for j, (lab, p) in enumerate(BR):
    d = fj(p)
    dT = 1e3 * np.array(d["branch_Tb_dev"], float); dQ = 100 * np.array(d["branch_Q_dev"], float)
    x = np.arange(1, len(dT) + 1)
    axs[0, j].bar(x, dT, color=C[0], width=0.6)
    axs[1, j].bar(x, dQ, color=C[2], width=0.6)
    for ax, v, f in ((axs[0, j], dT, "{:+.0f}"), (axs[1, j], dQ, "{:+.1f}")):
        ax.axhline(0, color=INK2, lw=0.6)
        lim = max(np.abs(v).max() * 1.45, 1e-9)
        ax.set_ylim(-lim, lim)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + (0.06 * lim if vi >= 0 else -0.06 * lim), f.format(vi), ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=5, color=INK)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axs[0, j].set_title(lab.replace("\n", " ") if len(lab) < 22 else lab, loc="left", fontsize=6)
    axs[1, j].set_xticks(x)
    axs[1, j].set_xlabel("branch")
axs[0, 0].set_ylabel("$T_b - T_\\mathrm{need}$ [mK]")
axs[1, 0].set_ylabel("$Q_i/\\bar Q - 1$ [%]")
fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.6)
save(fig, "ch_branch_balance")

# 7. how BO works: serpentine BO points in order, and best-so-far vs grid
bo = recs("final_u2d_serp"); gr = recs("final_u2d_serp_grid")
fig, axs = plt.subplots(2, 1, figsize=(2.7, 2.35), gridspec_kw={"height_ratios": [1.15, 1]})
ax = axs[0]
co = [r for r in bo if r["fidelity"] == "coarse"]
ini = [r for r in co if r["stage"] == "init"]; nxt = [r for r in co if r["stage"] != "init"]
ax.scatter([r["params"]["lam"] for r in ini], [r["params"]["A"] for r in ini], marker="s", s=12, facecolor="none",
           edgecolor=INK2, linewidth=0.7, label="start (12)")
sc = ax.scatter([r["params"]["lam"] for r in nxt], [r["params"]["A"] for r in nxt], c=np.arange(len(nxt)), s=10,
                cmap=ListedColormap(plt.cm.Oranges(np.linspace(0.35, 0.95, 256))), edgecolor=BG, linewidth=0.3,
                label="BO picks (38),\ndarker = later")
bb = best("final_u2d_serp")
ax.plot(bb["params"]["lam"], bb["params"]["A"], "D", mfc="none", mec=INK, ms=5.5, mew=0.9, label="best (fine)")
ax.set_xscale("log"); ax.set_xticks([7, 10, 14, 20], ["7", "10", "14", "20"]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("λ [mm]", labelpad=1); ax.set_ylabel("A [mm]")
ax.set_title("serpentine BO: where it sampled", loc="left")
ax.grid(False)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2, fontsize=5.5, labelspacing=0.3)
ax = axs[1]
for rr, col, lab in ((co, C[1], "BO"), ([r for r in gr if r["fidelity"] == "coarse"], C[0], "grid")):
    v = 1e3 * np.array([r["metrics"]["l2_flow"] if r["status"] == "ok" else np.nan for r in rr])
    ax.step(np.arange(1, len(v) + 1), np.fmin.accumulate(np.nan_to_num(v, nan=np.inf)), where="post", color=col, label=lab)
ax.axvline(12.5, color=INK2, lw=0.5, ls=":")
ax.text(13.5, 300, "end of start", fontsize=5, color=INK2)
ax.set_yscale("log"); ax.set_xlabel("coarse runs", labelpad=1); ax.set_ylabel("best so far [mK]")
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
ax.legend(loc="upper right")
fig.tight_layout(pad=0.3, h_pad=0.6)
save(fig, "ch_bo_explain")

# 8. how the grid works: serpentine grid stages
fig, ax = plt.subplots(figsize=(2.7, 2.2))
sty = {"grid_L0": ("o", C[0], 9, "level 0: 10 λ × 8 A"), "grid_Z1": ("o", C[2], 9, "zoom 1: ±Δ/2 around 3 best"),
       "grid_Z2": ("o", C[1], 9, "zoom 2: ±Δ/4 around best"), "fine_top": ("s", INK, 28, "fine mesh: best 3")}
for st, (mk, col, sz, lab) in sty.items():
    rr = [r for r in gr if r["stage"] == st]
    kw = dict(facecolor="none", edgecolor=col, linewidth=0.9) if mk == "s" else dict(color=col, edgecolor=BG, linewidth=0.3)
    ax.scatter([r["params"]["lam"] for r in rr], [r["params"]["A"] for r in rr], marker=mk, s=sz, label=f"{lab} ({len(rr)})", **kw)
fl = [r for r in gr if r["status"] != "ok"]
ax.set_xscale("log"); ax.set_xticks([7, 10, 14, 20], ["7", "10", "14", "20"]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("λ [mm]"); ax.set_ylabel("A [mm]")
ax.grid(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=1, fontsize=5.5, handletextpad=0.2, labelspacing=0.25)
ax.set_title("serpentine grid: the three stages", loc="left")
fig.tight_layout(pad=0.3)
save(fig, "ch_grid_explain")
print("charts written")

# 9. branched runs: temperature across the main channel just before the split (last exported slice)
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.85))
for ax, (lab, p) in zip(axs, BR):
    d = fj(p); F = d["fields"]
    y = np.array(F["slice_y"][-1]); y = y - (y[0] - (y[1] - y[0]) / 2)  # 0 = lower wall
    T = 1e3 * (np.array(F["slice_T"][-1], float) - F["T_need"])
    e = np.r_[0, np.cumsum(d["branch_w_in"])]
    for i in range(len(e) - 1):
        if i % 2 == 0:
            ax.axvspan(e[i], e[i + 1], color=GRID, alpha=0.45, lw=0)
        ax.text((e[i] + e[i + 1]) / 2, 1.0, str(i + 1), transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=5.5, color=INK2)
    ax.plot(y, T, color=C[0])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(0, e[-1]); ax.set_xticks([0, 1.5, 3])
    ax.set_xlabel("y across [mm]")
    ax.set_title(lab, loc="left", fontsize=6, pad=9)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("$T - T_\\mathrm{need}$ [mK]")
fig.tight_layout(pad=0.3, w_pad=0.6)
save(fig, "ch_presplit_all")
print("pre-split written")
