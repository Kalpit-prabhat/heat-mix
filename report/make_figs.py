"""Summary figures for the final report (report/figs/).

    .venv/bin/python report/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2})


def fj(p):
    return json.load(open(R / p))


def best(run):
    return json.load(open(R / run / "best.json"))


# ---------------------------------------------------------------- 1. summary chart
un = [("2D straight, fully heated", fj("final_u2d_straight_ref_single/fields/design_fine.json")["T_rms_flow"]),
      ("2D straight, best heater window", best("final_u2d_straight_ref")["metrics"]["l2_flow"]),
      ("2D sine, BO", best("final_u2d_sine")["metrics"]["l2_flow"]),
      ("2D sine, grid", best("final_u2d_sine_grid")["metrics"]["l2_flow"]),
      ("2D serpentine, BO", best("final_u2d_serp")["metrics"]["l2_flow"]),
      ("2D serpentine, grid", best("final_u2d_serp_grid")["metrics"]["l2_flow"]),
      ("3D straight, fully heated", fj("final_u3d_straight_ref_single/fields/design_fine.json")["T_rms_flow"]),
      ("3D straight, best heater window", best("final_u3d_straight_ref")["metrics"]["l2_flow"]),
      ("3D serpentine, BO", best("final_u3d_serp")["metrics"]["l2_flow"])]
br = [("straight, analytic openings, fully heated", best("final_b2d_straight_ref")),
      ("straight, BO (opening + heater window)", best("final_b2d_straight")),
      ("sine + straight, BO", best("final_b2d_sine")),
      ("serpentine + straight, BO", best("final_b2d_serp"))]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
for ax, rows, title in ((axs[0], [(n, 1e3 * v) for n, v in un], "one outlet: outlet RMS $\\ell_{2,\\mathrm{flow}}$ [mK]"),
                        (axs[1], [(n, 1e3 * b["metrics"]["branch_T"]) for n, b in br],
                         "four branches: branch temperature spread [mK]")):
    y = np.arange(len(rows))[::-1]
    v = [r[1] for r in rows]
    ax.hlines(y, 1, v, color=GRID, lw=2)
    ax.plot(v, y, "o", color=C[0], ms=8, mec="white", mew=1.5)
    for yi, vi in zip(y, v):
        ax.text(vi * 1.18, yi, f"{vi:.3g}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. outlet profiles, 2D and 3D
def prof(p, depth_avg=False):
    d = fj(p)
    T = np.array(d["T"], float) - d["T_need"]
    eta = np.array(d["grid_eta"])
    if T.ndim == 2:
        T = np.nanmean(T, axis=1)
    return eta + 0.5, 1e3 * T


sets = {
    "2D, one outlet (fine mesh)": [
        ("straight, fully heated", "final_u2d_straight_ref_single/fields/design_fine.json"),
        ("straight, best heater (10 %)", "final_u2d_straight_ref/fields/run0001_fine.json"),
        ("sine, best (λ 11.1, a 1)", "final_u2d_sine_grid/fields/run0112_fine.json"),
        ("serpentine, BO (λ 8.7, A 3.1)", "final_u2d_serp/fields/run0051_fine.json"),
        ("serpentine, grid (λ 8.3, A 2.1)", "final_u2d_serp_grid/fields/run0108_fine.json")],
    "3D, one outlet, depth-averaged (fine mesh)": [
        ("straight, fully heated", "final_u3d_straight_ref_single/fields/design_fine.json"),
        ("straight, best heater (1.2 mm)", "final_u3d_straight_ref/fields/run0001_fine.json"),
        ("serpentine, BO (λ 7.3, A 3.5)", "final_u3d_serp/fields/run0029_fine.json")]}
fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
panels = [("2D, one outlet (fine mesh)", sets["2D, one outlet (fine mesh)"], None, 0),
          ("2D, zoom on the wavy designs", sets["2D, one outlet (fine mesh)"][2:], (-200, 200), 2),
          ("3D, one outlet, depth-averaged (fine mesh)", sets["3D, one outlet, depth-averaged (fine mesh)"], None, 0)]
for ax, (title, rows, yl, c0) in zip(axs, panels):
    for k, (lab, p) in enumerate(rows):
        x, t = prof(p)
        ax.plot(x, t, color=C[k + c0] if "3D" not in title else [C[0], C[1], C[5]][k], label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("position across the outlet, y / w")
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
axs[0].set_ylabel("$T - T_\\mathrm{need}$ at the outlet [mK]")
fig.tight_layout()
fig.savefig(OUT / "outlet_profiles.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. non-uniformity along x (2D)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for k, (lab, p) in enumerate(sets["2D, one outlet (fine mesh)"]):
    d = fj(p)["profile"]
    ax.plot(d["x_stat"], 1e3 * np.array(d["T_std_x"], float), color=C[k], label=lab)
ax.set_yscale("log")
ax.set_xlabel("x [mm]")
ax.set_ylabel("std of T across the channel [mK]")
ax.set_title("2D, one outlet: how fast the cross-section evens out", fontsize=9.5, loc="left", color=INK)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "along_x_2d.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. grid landscape with BO points
def recs(run):
    return [json.loads(l) for l in open(R / run / "history.jsonl")]


fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, (grid, bo, yv, ylab) in zip(axs, (("final_u2d_sine_grid", "final_u2d_sine", "a", "a = A / A$_\\max$(λ)"),
                                          ("final_u2d_serp_grid", "final_u2d_serp", "A", "A [mm]"))):
    g = [r for r in recs(grid) if r["fidelity"] == "coarse" and r["status"] == "ok"]
    lam = np.array([r["params"]["lam"] for r in g])
    yy = np.array([r["params"][yv] for r in g])
    z = 1e3 * np.array([r["metrics"]["l2_flow"] for r in g])
    sc = ax.scatter(lam, yy, c=z, cmap="Blues_r", s=46, edgecolor="white", linewidth=0.6, norm=LogNorm(z.min(), z.max()))
    fl = [r for r in recs(grid) if r["status"] != "ok"]
    if fl:
        ax.scatter([r["params"]["lam"] for r in fl], [r["params"][yv] for r in fl], marker="x", color="#d03b3b",
                   s=30, label="failed (grid)")
    b = [r for r in recs(bo) if r["fidelity"] == "coarse" and r["status"] == "ok"]
    ax.scatter([r["params"]["lam"] for r in b], [r["params"][yv] for r in b], marker="+", color=C[1], s=40,
               linewidth=1.2, label="BO evaluations")
    gb, bb = best(grid), best(bo)
    ax.plot(gb["params"]["lam"], gb["params"][yv], "s", mfc="none", mec=INK, ms=13, mew=1.5, label="grid best (fine)")
    ax.plot(bb["params"]["lam"], bb["params"][yv], "D", mfc="none", mec=C[1], ms=11, mew=1.5, label="BO best (fine)")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 8, 12, 20], ["2", "3", "5", "8", "12", "20"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("wavelength λ [mm]")
    ax.set_ylabel(ylab)
    ax.grid(False)
    ax.set_title(("sine" if yv == "a" else "serpentine") + ": coarse-mesh grid (colour) and BO points",
                 fontsize=9.5, loc="left", color=INK)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    t = [v for v in (5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 1000) if z.min() <= v <= z.max()]
    cb.set_ticks(t, labels=[str(v) for v in t])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.set_label("$\\ell_{2,\\mathrm{flow}}$ [mK] (best heater window)")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85, frameon=True)
fig.tight_layout()
fig.savefig(OUT / "grid_landscape.png", dpi=200)
plt.close(fig)

print("written to", OUT)
