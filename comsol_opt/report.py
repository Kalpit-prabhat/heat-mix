"""Plots for a finished (or running) optimization: python -m comsol_opt report <run dir>"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def make_report(run_dir: str | Path, primary: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    recs = _load(run_dir)
    if not recs:
        raise SystemExit("no evaluations yet")
    outdir = run_dir / "report"
    outdir.mkdir(exist_ok=True)
    names = list(recs[0]["params"].keys())
    ok = [r for r in recs if r["status"] == "ok" and r["metrics"]]
    if primary is None:
        cfg_text = (run_dir / "config_used.yaml").read_text() if (run_dir / "config_used.yaml").exists() else ""
        import yaml
        primary = ((yaml.safe_load(cfg_text) or {}).get("loss") or {}).get("primary", "l2_area")

    # 1. convergence -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fid, color, marker in (("coarse", "#3b6ea8", "o"), ("fine", "#c8553d", "s")):
        rs = [r for r in ok if r["fidelity"] == fid]
        if not rs:
            continue
        ids = [r["id"] for r in rs]
        vals = [r["metrics"][primary] for r in rs]
        ax.scatter(ids, vals, s=22, color=color, marker=marker, label=f"{fid} mesh", alpha=0.8)
        best = np.minimum.accumulate(vals)
        ax.step(ids, best, where="post", color=color, lw=1.5)
    fails = [r["id"] for r in recs if r["status"] != "ok"]
    if fails:
        ymin = min(r["metrics"][primary] for r in ok) if ok else 1
        ax.scatter(fails, [ymin] * len(fails), marker="x", color="0.4", label="failed")
    ax.set_yscale("log")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel(f"{primary} [K]")
    ax.set_title("Optimization history (line = best so far)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=130)
    plt.close(fig)

    # 2. parameter vs loss --------------------------------------------------
    n = len(names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for i, nm in enumerate(names):
        ax = axes[i // cols][i % cols]
        for fid, color in (("coarse", "#3b6ea8"), ("fine", "#c8553d")):
            rs = [r for r in ok if r["fidelity"] == fid]
            ax.scatter([r["params"][nm] for r in rs], [r["metrics"][primary] for r in rs],
                       s=16, color=color, alpha=0.75, label=fid)
        ax.set_yscale("log")
        ax.set_xlabel(nm)
        ax.set_ylabel(primary)
        ax.grid(alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "parameters.png", dpi=120)
    plt.close(fig)

    # 3. outlet temperature maps of the best coarse / fine runs ---------------
    for fid in ("coarse", "fine"):
        rs = [r for r in ok if r["fidelity"] == fid]
        if not rs:
            continue
        best = min(rs, key=lambda r: r["metrics"][primary])
        raw = json.loads((run_dir / best["raw"]).read_text())
        _outlet_plot(raw, best, primary, outdir / f"outlet_best_{fid}.png")

    # 4. text summary -------------------------------------------------------
    lines = [f"{len(recs)} evaluations ({len(ok)} ok, {len(fails)} failed)", ""]
    for fid in ("coarse", "fine"):
        rs = sorted([r for r in ok if r["fidelity"] == fid], key=lambda r: r["metrics"][primary])
        if not rs:
            continue
        lines.append(f"top {fid}-mesh designs ({primary} in K):")
        for r in rs[:5]:
            p = ", ".join(f"{k}={v:.4g}" for k, v in r["params"].items())
            lines.append(f"  #{r['id']:<4d} {r['metrics'][primary]:.4g}  linf={r['metrics']['linf']:.3g}  | {p}")
        lines.append("")
    (outdir / "summary.txt").write_text("\n".join(lines))
    from .progress import plot_progress
    for f in plot_progress(run_dir, primary):
        lines.append(f"progress plot: {f}")
    print("\n".join(lines))
    print(f"plots written to {outdir}")
    return outdir


def outlet_figure(raw: dict, title: str):
    """T - T_need on the outlet: map over width x depth (3D) or profile (2D)"""
    import matplotlib.pyplot as plt

    T = np.asarray(raw["T"], dtype=float)
    y = np.asarray(raw["grid_y_mm"], dtype=float)
    Tn = raw["T_need"]
    if T.ndim == 1 or T.shape[1] == 1:          # 2D model: profile across the width
        T = T.reshape(-1)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(y, T - Tn, color="#2a78d6", lw=2)
        ax.axhline(0, color="#0b0b0b", lw=1, ls=":")
        ax.set_xlabel("y across channel [mm]")
        ax.set_ylabel("T - T_need [K]")
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return fig
    z = np.atleast_1d(np.asarray(raw["grid_z_mm"], dtype=float))
    if raw.get("symmetric"):
        # mirror the half model about the symmetry plane z = z_max
        zt = 2 * (z[-1] + 0.5 * (z[1] - z[0]) if len(z) > 1 else z[-1])
        z = np.concatenate([z, zt - z[::-1]])
        T = np.concatenate([T, T[:, ::-1]], axis=1)
    dev = np.nanmax(np.abs(T - Tn))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.pcolormesh(y, z, (T - Tn).T, cmap="RdBu_r", vmin=-dev, vmax=dev, shading="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("y across channel [mm]")
    ax.set_ylabel("z depth [mm]")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="K")
    fig.tight_layout()
    return fig


def _outlet_plot(raw: dict, rec: dict, primary: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = outlet_figure(raw, f"run #{rec['id']} ({rec['fidelity']} mesh): T - T_need at outlet\n"
                             f"{primary} = {rec['metrics'][primary]:.3g} K, "
                             f"max |dev| = {rec['metrics']['linf']:.3g} K")
    fig.savefig(path, dpi=130)
    plt.close(fig)
