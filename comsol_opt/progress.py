"""How the optimization progressed: loss, design and temperature profiles over the runs.

    python -m comsol_opt report runs/<name>      (also written during a run, every
                                                  output.progress_every evaluations)

Files in <run dir>/report/:

    progress_loss.png     loss of every evaluation + best so far, and below it each
                          design parameter (all evaluated values, and the best design's
                          value step by step)
    progress_outlet.png   outlet temperature across the channel for every new best
                          design (light = early, dark = late) and, below, the outlet
                          profile of every evaluation in order (one column per run)
    progress_along_x.png  bulk temperature and cross-section spread along the channel
                          for every new best design (needs res.profile, stored by the
                          case scripts with every run)

"New best" is tracked on the coarse mesh (the fine-mesh re-runs are shown as their own
dashed curve) because the two meshes give slightly different losses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fields import C_AQUA, C_BLUE, C_ORANGE, C_YELLOW, GRID, HEATER, INK, INK2, _arr, _style

STAGES = (("init", "initial design", C_BLUE), ("coarse_bo", "Bayesian opt. (coarse)", C_ORANGE),
          ("grid_L0", "grid, level 0", C_BLUE), ("grid_Z1", "grid, zoom 1", C_ORANGE),
          ("grid_Z2", "grid, zoom 2", C_YELLOW), ("fine_top", "fine-mesh check", C_AQUA))
MAX_SNAPSHOTS = 10


def _records(run_dir: Path) -> list[dict]:
    p = run_dir / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _cfg(run_dir: Path) -> dict:
    import yaml
    p = run_dir / "config_used.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def _improvements(recs: list[dict], primary: str) -> list[dict]:
    out, best = [], np.inf
    for r in recs:
        v = r["metrics"][primary]
        if v < best:
            best = v
            out.append(r)
    if len(out) > MAX_SNAPSHOTS:      # first, last and evenly spaced in between
        idx = np.unique(np.round(np.linspace(0, len(out) - 1, MAX_SNAPSHOTS)).astype(int))
        out = [out[i] for i in idx]
    return out


def _raw(run_dir: Path, rec: dict, cache: dict) -> dict:
    if rec["id"] not in cache:
        try:
            cache[rec["id"]] = json.loads((run_dir / rec["raw"]).read_text())
        except Exception:
            cache[rec["id"]] = {}
    return cache[rec["id"]]


def _width_profile(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """T - T_need across the width at the outlet (3D: averaged over the depth)"""
    T = _arr(raw["T"])
    T = T.reshape(-1) if T.ndim == 1 else np.nanmean(T, axis=1)
    return _arr(raw["grid_eta"]).reshape(-1), T - raw["T_need"]


def _depth_profile(raw: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """T - T_need over the depth at the outlet, averaged over the width (3D only)"""
    T = _arr(raw["T"])
    if T.ndim == 1 or T.shape[1] == 1:
        return None
    z = np.atleast_1d(_arr(raw["grid_z_mm"]))
    t = np.nanmean(T, axis=0)
    if raw.get("symmetric"):
        zt = 2 * (z[-1] + 0.5 * (z[1] - z[0])) if len(z) > 1 else 2 * z[-1]
        z = np.concatenate([z, zt - z[::-1]])
        t = np.concatenate([t, t[::-1]])
    return z, t - raw["T_need"]


def _seq(n: int):
    """sequential blues, light (early) -> dark (late); lightest step still visible on white"""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("Blues")
    return [cmap(0.38 + 0.62 * i / max(n - 1, 1)) for i in range(n)], cmap


def _colorbar(fig, axes, snaps, cmap):
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cols, _ = _seq(len(snaps))
    norm = BoundaryNorm(np.arange(len(snaps) + 1), len(snaps))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=ListedColormap(cols))
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, ticks=np.arange(len(snaps)) + 0.5)
    cb.ax.set_yticklabels([f"#{r['id']}" for r in snaps], fontsize=7, color=INK2)
    cb.set_label("new best design at evaluation", color=INK2, fontsize=9)
    cb.outline.set_edgecolor(GRID)


def plot_progress(run_dir: str | Path, primary: str | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    recs = _records(run_dir)
    cfg = _cfg(run_dir)
    primary = primary or (cfg.get("loss") or {}).get("primary", "l2_area")
    ok = [r for r in recs if r["status"] == "ok" and r.get("metrics")]
    if not ok:
        return []
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    files: list[Path] = []
    cache: dict = {}
    name = (cfg.get("output") or {}).get("name", run_dir.name)
    coarse = [r for r in ok if r["fidelity"] == "coarse"]
    fine = [r for r in ok if r["fidelity"] == "fine"]
    snaps = _improvements(coarse or fine, primary)
    best_fine = min(fine, key=lambda r: r["metrics"][primary]) if fine and coarse else None
    cols, cmap = _seq(len(snaps))

    # ---- 1. loss and design parameters over the evaluations -------------------
    var = cfg.get("variables") or {}
    names = list(ok[0]["params"].keys())
    fig, axes = plt.subplots(1 + len(names), 1, figsize=(10, 4.2 + 1.15 * len(names)), sharex=True,
                             gridspec_kw={"height_ratios": [3.2] + [1] * len(names)},
                             layout="constrained")
    ax = axes[0]
    for st, lab, c in STAGES:
        rs = [r for r in ok if r["stage"] == st]
        if rs:
            ax.scatter([r["id"] for r in rs], [r["metrics"][primary] for r in rs], s=26, color=c,
                       edgecolor="white", linewidth=0.6, label=lab, zorder=3)
    cl = [r for r in ok if r["diag"].get("power_clamped")]
    if cl:
        ax.scatter([r["id"] for r in cl], [r["metrics"][primary] for r in cl], s=70,
                   facecolor="none", edgecolor=INK2, linewidth=1, label="heater power at its limit",
                   zorder=4)
    for fid, ls in (("coarse", "-"), ("fine", "--")):
        rs = coarse if fid == "coarse" else fine
        if rs:
            ids = [r["id"] for r in rs] + [max(r["id"] for r in recs)]
            v = np.minimum.accumulate([r["metrics"][primary] for r in rs])
            ax.step(ids, np.append(v, v[-1]), where="post", color=INK, lw=1.6, ls=ls,
                    label=f"best so far ({fid} mesh)", zorder=2)
    fails = [r["id"] for r in recs if r["status"] != "ok"]
    ax.set_yscale("log")
    lv = [r["metrics"][primary] for r in ok]
    ax.set_ylim(min(lv) / 1.8, max(lv) * 1.5)
    if fails:
        y0 = ax.get_ylim()[0]
        ax.scatter(fails, [y0 * 1.15] * len(fails), marker="x", color=HEATER, s=30, label="failed")
    best = min(coarse or fine, key=lambda r: r["metrics"][primary])
    ax.annotate(f"#{best['id']}: {best['metrics'][primary]:.3g} K",
                (best["id"], best["metrics"][primary]), xytext=(6, -14),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_ylabel(f"loss {primary} [K]")
    ax.grid(color=GRID, lw=0.6, which="major")
    leg = ax.legend(fontsize=8, ncol=3, loc="upper right", frameon=True, framealpha=0.92,
                    facecolor="white", edgecolor="none")
    leg.set_zorder(10)
    ax.set_title(f"{name}: optimization progress, {len(recs)} evaluations "
                 f"({len(ok)} ok, {len(fails)} failed)\nbelow: every evaluated value (dots) and the "
                 f"best design's value (line) per parameter", fontsize=10, loc="left", color=INK)
    _style(ax)
    for a, nm in zip(axes[1:], names):
        a.scatter([r["id"] for r in ok], [r["params"][nm] for r in ok], s=9, color="#b8b7b1", lw=0)
        inc, v = [], np.inf
        for r in (coarse or fine):
            if r["metrics"][primary] < v:
                v = r["metrics"][primary]
                inc.append(r)
        ids = [r["id"] for r in inc] + [max(r["id"] for r in recs)]
        vals = [r["params"][nm] for r in inc]
        a.step(ids, vals + vals[-1:], where="post", color=C_BLUE, lw=2)
        lo_hi = var.get(nm) or {}
        if "min" in lo_hi and "max" in lo_hi:
            pad = 0.06 * (lo_hi["max"] - lo_hi["min"])
            a.set_ylim(lo_hi["min"] - pad, lo_hi["max"] + pad)
        unit = lo_hi.get("unit", "")
        a.set_ylabel(f"{nm}" + (f"\n[{unit}]" if unit else ""), fontsize=8, rotation=0,
                     ha="right", va="center")
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a.grid(color=GRID, lw=0.6)
        _style(a)
    axes[-1].set_xlabel("evaluation #")
    files.append(out / "progress_loss.png")
    fig.savefig(files[-1], dpi=140)
    plt.close(fig)

    # ---- 2. outlet profiles ---------------------------------------------------
    raws = {r["id"]: _raw(run_dir, r, cache) for r in snaps}
    is3d = any(_depth_profile(x) is not None for x in raws.values() if x)
    order = sorted(ok, key=lambda r: r["id"])
    prof = [_width_profile(_raw(run_dir, r, cache)) if _raw(run_dir, r, cache) else None for r in order]
    fig = plt.figure(figsize=(11 if is3d else 9, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, 2 if is3d else 1, height_ratios=[1.25, 1], width_ratios=[1.6, 1] if is3d else [1])
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1]) if is3d else None
    for r, c in zip(snaps, cols):
        raw = raws[r["id"]]
        if not raw:
            continue
        last = r is snaps[-1]
        eta, t = _width_profile(raw)
        a1.plot(eta, t, color=c, lw=2.4 if last else 1.4, zorder=3 if last else 2)
        if a2 is not None and (dp := _depth_profile(raw)) is not None:
            a2.plot(dp[1], dp[0], color=c, lw=2.4 if last else 1.4, zorder=3 if last else 2)
    if best_fine:
        raw = _raw(run_dir, best_fine, cache)
        if raw:
            eta, t = _width_profile(raw)
            a1.plot(eta, t, color=C_ORANGE, lw=1.8, ls="--", label=f"fine mesh best #{best_fine['id']}")
            a1.legend(fontsize=8, frameon=False, loc="lower center")
            if a2 is not None and (dp := _depth_profile(raw)) is not None:
                a2.plot(dp[1], dp[0], color=C_ORANGE, lw=1.8, ls="--")
    a1.axhline(0, color=INK, lw=1, ls=":")
    a1.set_xlabel("position across the channel, y / w  (-0.5 and 0.5 = side walls)")
    a1.set_ylabel("T - T_need [K]" + ("  (averaged over the depth)" if is3d else ""))
    a1.set_title("outlet (x = L) temperature of each new best design", fontsize=10, loc="left", color=INK)
    a1.grid(color=GRID, lw=0.6)
    _style(a1)
    if a2 is not None:
        a2.axvline(0, color=INK, lw=1, ls=":")
        a2.set_xlabel("T - T_need [K]  (averaged over the width)")
        a2.set_ylabel("z [mm]")
        a2.set_title("... over the depth", fontsize=10, loc="left", color=INK)
        a2.grid(color=GRID, lw=0.6)
        _style(a2)
    _colorbar(fig, [a1, a2] if a2 is not None else [a1], snaps, cmap)
    # every evaluation as one column: profile across the width
    a3 = fig.add_subplot(gs[1, :])
    ids = [r["id"] for r, p in zip(order, prof) if p is not None]
    eta_ref = np.linspace(-0.5, 0.5, 60)
    M = np.array([np.interp(eta_ref, p[0], p[1], left=np.nan, right=np.nan) for p in prof if p is not None])
    # symmetric log colour scale: the good designs (~0.01-0.1 K) and the bad early ones
    # (several K) are both readable
    lim = max(np.nanmax(np.abs(M)), 1e-3)
    lin = float(np.clip(0.5 * best["metrics"][primary], 1e-3, lim / 10))
    from matplotlib.colors import SymLogNorm
    im = a3.pcolormesh(np.arange(len(ids)), eta_ref, M.T, cmap="RdBu_r", shading="nearest",
                       norm=SymLogNorm(linthresh=lin, vmin=-lim, vmax=lim, base=10), rasterized=True)
    step = max(1, len(ids) // 12)
    a3.set_xticks(np.arange(0, len(ids), step))
    a3.set_xticklabels([str(ids[i]) for i in range(0, len(ids), step)])
    for r in snaps:                       # mark the new-best columns
        if r["id"] in ids:
            a3.plot(ids.index(r["id"]), 0.53, marker="v", color=INK, ms=4, clip_on=False)
    a3.set_ylim(-0.5, 0.5)
    a3.set_xlabel("evaluation # (successful runs, in order; triangles: new best)")
    a3.set_ylabel("y / w")
    a3.set_title("outlet temperature across the channel for every evaluation", fontsize=10,
                 loc="left", color=INK, pad=12)
    dec = 10.0 ** np.arange(np.ceil(np.log10(lin * 1.5)), np.floor(np.log10(lim)) + 1)
    cb = fig.colorbar(im, ax=a3, fraction=0.03, pad=0.02, ticks=np.concatenate([-dec[::-1], [0], dec]))
    cb.ax.set_yticklabels([f"{t:g}" for t in np.concatenate([-dec[::-1], [0], dec])])
    cb.set_label(f"T - T_need [K] (log scale beyond ±{lin:.2g} K)", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    _style(a3)
    fig.suptitle(f"{name}: outlet temperature over the optimization", fontsize=11, x=0.01, ha="left",
                 color=INK)
    files.append(out / "progress_outlet.png")
    fig.savefig(files[-1], dpi=140)
    plt.close(fig)

    # ---- 3. along the channel -------------------------------------------------
    have = [(r, c) for r, c in zip(snaps, cols) if raws.get(r["id"], {}).get("profile")]
    if have:
        raw0 = raws[have[-1][0]["id"]]
        Tin, Tn = raw0["T_in"], raw0["T_need"]
        fig, (b1, b2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, layout="constrained")
        Lx = ((cfg.get("fixed") or {}).get("L") or {}).get("value")   # exit plane [mm]
        for r, c in have:
            raw = raws[r["id"]]
            P = raw["profile"]
            x = _arr(P["x_stat"])
            last = r is have[-1][0]
            kw = dict(color=c, lw=2.4 if last else 1.4, zorder=3 if last else 2)
            b1.plot(x, _arr(P["T_bulk_x"]) - Tin, **kw)
            b2.plot(x, _arr(P["T_std_x"]), **kw)
            hs = r["params"].get("heat_start")
            hl = r["params"].get("heat_len")
            if last and hs is not None and hl is not None:
                for b in (b1, b2):
                    b.axvspan(hs, hs + hl, color=HEATER, alpha=0.08, lw=0)
        if best_fine and (rf := _raw(run_dir, best_fine, cache)).get("profile"):
            P = rf["profile"]
            x = _arr(P["x_stat"])
            b1.plot(x, _arr(P["T_bulk_x"]) - Tin, color=C_ORANGE, lw=1.8, ls="--",
                    label=f"fine mesh best #{best_fine['id']}")
            b2.plot(x, _arr(P["T_std_x"]), color=C_ORANGE, lw=1.8, ls="--")
            b1.legend(fontsize=8, frameon=False, loc="lower right")
        b1.axhline(Tn - Tin, color=INK, lw=1, ls=":")
        b1.text(0.0, Tn - Tin, " target rise", va="bottom", fontsize=8, color=INK2)
        for b in (b1, b2):
            if Lx:
                b.axvline(Lx, color=INK2, lw=0.8, ls="--")
            b.grid(color=GRID, lw=0.6)
            _style(b)
        b1.set_ylabel("bulk T - T_in [K]")
        b1.set_title(f"{name}: temperature along the channel for each new best design\n"
                     f"(shaded: heater of the final best design, dashed: exit plane x = L)",
                     fontsize=10, loc="left", color=INK)
        b2.set_yscale("log")
        b2.set_ylabel("std of T over the\ncross-section [K]")
        b2.set_xlabel("x [mm]")
        _colorbar(fig, [b1, b2], [r for r, _ in have], cmap)
        files.append(out / "progress_along_x.png")
        fig.savefig(files[-1], dpi=140)
        plt.close(fig)
    return files

