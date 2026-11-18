"""Temperature-field plots of one design (after an optimization or for any parameters).

    python -m comsol_opt fields runs/<name>                  best design of a run
    python -m comsol_opt fields runs/<name> --id 17          a given run
    python -m comsol_opt fields runs/<name> --replot         only redraw from saved data
    python -m comsol_opt fields CONFIG.yaml --set lam=3 A=1  any design of a case

The design is re-solved once in COMSOL with the case option ``fields`` (field
samples are not stored during the optimization to keep runs small).  The raw
samples go to <dir>/fields/<name>.json, the plots next to it:

    <name>_maps.png      T, |grad T| and |u| at mid-depth over the whole channel
    <name>_along_x.png   cross-section bulk / min / max temperature and the spread
                         (std, RMS deviation from T_need) along the channel
    <name>_slices.png    cross-section temperature at evenly spaced stations
                         (3D: maps over width x depth; 2D: profiles across the width)
    <name>_outlet.png    outlet temperature deviation from T_need
    <name>_profile.png   flow profile: speed and T - T_need across the channel at the
                         stations, plus the score table (all loss metrics, heater, losses)
    <name>_slices_u.png  (3D) speed over the cross-section at the stations
    <name>_summary.json  parameters, scores and key results of the design
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# reference categorical slots (fixed order) and neutral inks
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
HEATER = "#e34948"


def _arr(v) -> np.ndarray:
    """JSON lists (MATLAB NaN -> null) to float arrays"""
    a = np.array(v, dtype=object)
    return np.where(a == None, np.nan, a).astype(float)  # noqa: E711


def _style(ax):
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _heaters(F) -> list[tuple[float, float]]:
    """heated x windows: F['heaters'] (n x 2, several heaters) or heat_start/heat_end"""
    if "heaters" in F:
        return [(float(a), float(b)) for a, b in np.atleast_2d(_arr(F["heaters"]))]
    return [(F["heat_start"], F["heat_end"])]


def _densify(x, y, ds: float = 0.02):
    """resample a polyline so that long straight pieces can be masked by x"""
    xs, ys = [x[:1]], [y[:1]]
    for i in range(len(x) - 1):
        n = max(int(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / ds), 1)
        t = np.arange(1, n + 1) / n
        xs.append(x[i] + t * (x[i + 1] - x[i]))
        ys.append(y[i] + t * (y[i + 1] - y[i]))
    return np.concatenate(xs), np.concatenate(ys)


def _walls(ax, F, heater=True):
    if "walls" in F:                       # list of polylines (2D model: walls and branch walls)
        lines = [(_arr(w["x"]), _arr(w["y"])) for w in F["walls"]]
        heated = lines[:2]                 # the first two are the heated main-channel walls
    else:
        lines = [(_arr(F[f"wall_{s}_x"]), _arr(F[f"wall_{s}_y"])) for s in ("upper", "lower")]
        heated = lines
    for x, y in lines:
        ax.plot(x, y, color=INK, lw=0.8)
    if heater:
        first = True
        for x, y in heated:
            x, y = _densify(x, y)
            for a, b in _heaters(F):
                on = (x >= a) & (x <= b)
                ax.plot(np.where(on, x, np.nan), np.where(on, y, np.nan), color=HEATER, lw=2.4,
                        solid_capstyle="butt", label="heater" if first else None)
                first = False
    ax.axvline(F["L"], color=INK2, lw=0.8, ls="--")


def _mirror(F, z, A):
    """mirror half-depth slice data (ns x ny x nz) about the symmetry plane"""
    if F.get("symmetric"):
        ztop = 2 * F["depth_modelled"]
        return np.concatenate([z, ztop - z[::-1]]), np.concatenate([A, A[:, :, ::-1]], axis=2)
    return z, A


def _slice_maps(F, A, z, cmap, norm, label, title):
    """3D: one map over width x depth per station"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    sx, eta = _arr(F["slice_x"]), _arr(F["slice_eta"])
    ns = len(sx)
    cols = min(ns, 3)
    rows = int(np.ceil(ns / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows + 0.8), squeeze=False,
                             layout="constrained")
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= ns:
            ax.axis("off")
            continue
        im = ax.pcolormesh(eta * F["w"], z, A[i].T, cmap=cmap, norm=norm, shading="nearest")
        ax.set_aspect("equal")
        ax.set_title(f"x = {sx[i]:.1f} mm", fontsize=9)
        ax.set_xlabel("across the channel [mm]")
        ax.set_ylabel("z [mm]")
        ax.xaxis.set_major_locator(MaxNLocator(5 if F["w"] > 1.5 else 3))
        _style(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(label, color=INK2)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    return fig


def score_rows(res: dict, primary: str | None = None) -> list[tuple[str, str]]:
    """(label, value) rows of the score table"""
    from .loss import outlet_metrics
    m = outlet_metrics(res)
    g = lambda k: float("nan") if res.get(k) is None else res.get(k)  # noqa: E731
    rows = []
    for k, lab in (("l2_area", "RMS(T - T_need), area"), ("l2_flow", "RMS(T - T_need), flow-weighted"),
                   ("linf", "max |T - T_need|"), ("std_area", "std of T over the outlet")):
        rows.append((lab + ("   <- optimized" if k == primary else ""), f"{m[k]:.4f} K"))
    rows += [
        ("bulk T - T_need at x = L", f"{g('T_bulk') - g('T_need'):+.4f} K"),
        ("T_in / T_need", f"{g('T_in'):.2f} / {g('T_need'):.2f} K"),
        ("heater power", f"{g('P_heater_W'):.3f} W  (x{g('P_factor'):.3f} of rho*Q*cp*dT)"
                         + ("  [at limit]" if res.get("power_clamped") else "")),
        ("heater wall temperature", f"{g('T_heater') - 273.15:.2f} C"),
    ]
    if res.get("T_heater2") is not None:
        rows.append(("heater 2 wall temperature", f"{g('T_heater2') - 273.15:.2f} C"))
    if np.isfinite(m.get("branch", np.nan)):
        Q, Tb = np.atleast_1d(_arr(res["branch_Q_ml_min"])), np.atleast_1d(_arr(res["branch_Tb"]))
        rows += [
            ("branch loss" + ("   <- optimized" if primary == "branch" else ""),
             f"{m['branch']:.4f}  (T spread {m['branch_T'] * 1e3:.1f} mK, flow spread {100 * m['branch_Q']:.2f} %)"),
            ("branch flow Q_i / mean - 1", "  ".join(f"{100 * (q / Q.mean() - 1):+.2f}%" for q in Q)),
            ("branch bulk T_i - T_need", "  ".join(f"{1e3 * (t - g('T_need')):+.1f} mK" for t in Tb)),
        ]
        if res.get("branch_w_in") is not None:
            rows.append(("branch inlet widths", "  ".join(f"{v:.3f}" for v in np.atleast_1d(_arr(res["branch_w_in"]))) + " mm"))
    rows += [
        ("heated area / mean flux", f"{g('A_heat_mm2'):.1f} mm2 / {g('P_heater_W') / max(g('A_heat_mm2'), 1e-9) * 100:.2f} W/cm2"),
        ("wall losses", f"{g('P_loss_W'):.4f} W"),
        ("max wall T - T_in", f"{g('T_wall_max') - g('T_in'):.2f} K"),
        ("Re / pressure drop", f"{g('Re'):.0f} / {g('dp_Pa'):.1f} Pa"),
        ("max speed", f"{g('u_max') * 1e3:.1f} mm/s"),
        ("mesh elements", f"{int(g('mesh_elements')) if np.isfinite(g('mesh_elements')) else 'n/a'}"),
    ]
    return rows


def plot_fields(res: dict, out_dir: Path, name: str, title: str = "",
                primary: str | None = None, params: dict | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    F = res["fields"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    is3d = F.get("model_dim", 3) == 3
    Tin, Tn = F["T_in"], F["T_need"]
    x, y = _arr(F["map_x"]), _arr(F["map_y"])
    T, gT, U = _arr(F["map_T"]), _arr(F["map_gradT"]), _arr(F["map_U"])
    where = "mid-depth plane" if is3d else "2D model"
    head = f"{title}\n" if title else ""
    info = (f"heater {res.get('P_heater_W', float('nan')):.3f} W "
            f"(x{res.get('P_factor', float('nan')):.3f} of rho*Q*cp*dT)"
            + (f", heater wall {res['T_heater'] - 273.15:.1f} C" if res.get('T_heater') is not None and np.isfinite(res['T_heater']) else "")
            + f"{'  [heater at its limit]' if res.get('power_clamped') else ''}, "
            f"wall losses {res.get('P_loss_W', float('nan')):.3f} W, "
            f"outlet RMS(T - T_need) {res.get('T_rms_area', float('nan')):.3f} K")

    # ---- 1. maps ---------------------------------------------------------
    span = (x[-1] - x[0]) / (y[-1] - y[0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 3 * 3.6 / max(span / 4.5, 1) + 1.6), sharex=True)
    dT = T - Tin
    vmax = np.nanpercentile(dT, 99.5)
    panels = [
        (dT, dict(cmap="OrRd", vmin=0, vmax=vmax), f"T - T_in [K] (scale clipped at {vmax:.1f} K)"),
        (gT, dict(cmap="Purples", norm=LogNorm(vmin=max(np.nanpercentile(gT, 2), 1e-3),
                                              vmax=np.nanpercentile(gT, 99.8))), "|grad T| [K/mm]"),
        (U * 1e3, dict(cmap="Blues", vmin=0), "|u| [mm/s]"),
    ]
    for ax, (Z, kw, lab) in zip(axes, panels):
        im = ax.pcolormesh(x, y, Z.T, shading="nearest", rasterized=True, **kw)
        _walls(ax, F)
        ax.set_aspect("equal")
        ax.set_ylabel("y [mm]")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(lab, color=INK2, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("x [mm]   (dashed: exit plane x = L, where the loss is measured)")
    axes[0].set_title(f"{head}{where}: temperature rise, temperature gradient, speed\n{info}",
                      fontsize=10, loc="left")
    fig.tight_layout()
    files.append(out_dir / f"{name}_maps.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # ---- 2. along x ------------------------------------------------------
    xs = _arr(F["x_stat"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for a in (a1, a2):
        for h0, h1 in _heaters(F):
            a.axvspan(h0, h1, color=HEATER, alpha=0.08, lw=0)
        a.axvline(F["L"], color=INK2, lw=0.8, ls="--")
        a.grid(color=GRID, lw=0.6)
    a1.fill_between(xs, _arr(F["T_min_x"]) - Tin, _arr(F["T_max_x"]) - Tin, color=C_ORANGE, alpha=0.18,
                    lw=0, label="min-max over the cross-section")
    a1.plot(xs, _arr(F["T_bulk_x"]) - Tin, color=C_BLUE, lw=2, label="bulk (flow-weighted)")
    a1.plot(xs, _arr(F["T_mean_x"]) - Tin, color=C_AQUA, lw=2, label="area mean")
    a1.axhline(Tn - Tin, color=INK, lw=1, ls=":", label="target rise")
    top = np.nanpercentile(_arr(F["T_max_x"]) - Tin, 98)
    a1.set_ylim(min(-0.1, np.nanmin(_arr(F["T_min_x"]) - Tin)), max(top, 1.2 * (Tn - Tin)))
    a1.set_ylabel("T - T_in [K]")
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a1.set_title(f"{head}cross-sections x = const (shaded: heated zone, dashed: exit plane)",
                 fontsize=10, loc="left")
    a2.plot(xs, _arr(F["T_std_x"]), color=C_BLUE, lw=2, label="std of T over the cross-section")
    a2.plot(xs, _arr(F["T_rms_x"]), color=C_ORANGE, lw=2, label="RMS(T - T_need)")
    a2.set_yscale("log")
    a2.set_ylabel("non-uniformity [K]")
    a2.set_xlabel("x [mm]")
    a2.legend(fontsize=8, frameon=False)
    for a in (a1, a2):
        _style(a)
    fig.tight_layout()
    files.append(out_dir / f"{name}_along_x.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # ---- 3. slices -------------------------------------------------------
    sx = _arr(F["slice_x"])
    ST = _arr(F["slice_T"]) - Tn
    ns = len(sx)
    lim = np.nanmax(np.abs(ST))
    if is3d:
        z, ST = _mirror(F, _arr(F["slice_z"]), ST)
        fig = _slice_maps(F, ST, z, "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                          "T - T_need [K]",
                          f"{head}temperature over the cross-section (vertical cuts x = const)")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cmap = plt.get_cmap("Blues")
        eta = _arr(F["slice_eta"])
        for i in range(ns):
            ax.plot(eta * F["w"], ST[i], color=cmap(0.35 + 0.65 * i / max(ns - 1, 1)), lw=2,
                    label=f"x = {sx[i]:.1f} mm")
        ax.axhline(0, color=INK, lw=1, ls=":")
        ax.set_xlabel("across the channel [mm]")
        ax.set_ylabel("T - T_need [K]")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{head}temperature profiles across the channel", fontsize=10, loc="left")
        _style(ax)
    files.append(out_dir / f"{name}_slices.png")
    fig.savefig(files[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. outlet ---------------------------------------------------------
    from .report import outlet_figure
    fig = outlet_figure(res, f"{head}outlet (x = L): T - T_need")
    files.append(out_dir / f"{name}_outlet.png")
    fig.savefig(files[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. flow profile + scores -----------------------------------------
    files += _profile_plots(res, F, out_dir, name, head, primary, params)
    return files


def _profile_plots(res, F, out_dir, name, head, primary, params) -> list[Path]:
    import matplotlib.pyplot as plt
    files = []
    is3d = F.get("model_dim", 3) == 3
    sx, eta = _arr(F["slice_x"]), _arr(F["slice_eta"])
    ST, SU = _arr(F["slice_T"]) - F["T_need"], _arr(F["slice_U"]) * 1e3
    ns = len(sx)
    if is3d:
        z, SU3 = _mirror(F, _arr(F["slice_z"]), SU)
        fig = _slice_maps(F, SU3, z, "Blues", None, "|u| [mm/s]",
                          f"{head}speed over the cross-section (vertical cuts x = const)")
        files.append(out_dir / f"{name}_slices_u.png")
        fig.savefig(files[-1], dpi=150, bbox_inches="tight")
        plt.close(fig)
        k = SU.shape[2] - 1 if F.get("symmetric") else SU.shape[2] // 2     # mid-depth line
        ST, SU, where = ST[:, :, k], SU[:, :, k], " (mid-depth)"
    else:
        where = ""
    yw = eta * F["w"]
    n_rows = len(score_rows(res, primary)) + bool(params)
    grow = max(1.0, n_rows / 15)               # longer score tables (e.g. branched outlets)
    fig = plt.figure(figsize=(13, 4.3 + 3.9 * grow))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95 * grow], hspace=0.35, wspace=0.22)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("Blues")
    for i in range(ns):
        c = cmap(0.35 + 0.65 * i / max(ns - 1, 1))
        lab = f"x = {sx[i]:.1f} mm" + ("  (exit)" if i == ns - 1 else "")
        a1.plot(yw, SU[i], color=c, lw=2, label=lab)
        a2.plot(yw, ST[i], color=c, lw=2, label=lab)
    a2.axhline(0, color=INK, lw=1, ls=":")
    a1.set_ylabel("|u| [mm/s]")
    a2.set_ylabel("T - T_need [K]")
    a1.set_title(f"speed across the channel{where}", fontsize=10, loc="left")
    a2.set_title(f"temperature across the channel{where}", fontsize=10, loc="left")
    for a in (a1, a2):
        a.set_xlabel("across the channel [mm]")
        a.grid(color=GRID, lw=0.6)
        _style(a)
    a1.legend(fontsize=8, frameon=False)
    a3.axis("off")
    rows = score_rows(res, primary)
    if params:
        prow = ", ".join(f"{k} = {v:.3g}" for k, v in params.items())
        rows = [("design", prow)] + rows
    tab = a3.table(cellText=[[v] for _, v in rows], rowLabels=[k for k, _ in rows],
                   colLabels=["value"], loc="center", cellLoc="left", colWidths=[0.5])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK if r == 0 or c == -1 else INK2)
    fig.suptitle(f"{head}flow profile and score", fontsize=11, x=0.01, ha="left", color=INK)
    files.append(out_dir / f"{name}_profile.png")
    fig.savefig(files[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    from .loss import outlet_metrics
    summ = {"name": name, "params": params, "primary": primary, "metrics": outlet_metrics(res),
            **{k: res.get(k) for k in ("T_in", "T_need", "T_bulk", "P_heater_W", "P_factor", "T_heater",
                                       "power_clamped", "P_loss_W", "A_heat_mm2", "T_wall_max",
                                       "Re", "dp_Pa", "u_max", "mesh_elements")}}
    files.append(out_dir / f"{name}_summary.json")
    files[-1].write_text(json.dumps(summ, indent=2, default=float))
    return files


# --------------------------------------------------------------------------
def fields_for_record(cfg, run_dir: Path, rec: dict, mesh: str | None = None,
                      field_opts: dict | None = None) -> list[Path]:
    """re-solve one stored design with field export and plot it"""
    from .runner import ComsolRunner
    from .study import Study

    fid = mesh or rec["fidelity"]
    name = f"run{rec['id']:04d}_{fid}" if "id" in rec and rec["id"] is not None else f"design_{fid}"
    out = run_dir / "fields"
    out.mkdir(parents=True, exist_ok=True)
    st = Study.__new__(Study)
    st.cfg, st.out = cfg, run_dir
    opts = st._comsol_opts(fid, 0)
    opts["save_mph"] = str((out / f"{name}.mph").resolve())
    opts["fields"] = field_opts or {}
    opts["reuse_model"] = False
    opts.pop("heat_list", None)                # plot the chosen heater window only
    params = dict(rec["params"])
    for k, v in (rec.get("sweep_best") or {}).items():
        if k in ("heat_start", "heat_len", "heat1_s", "heat1_l"):
            params[k] = v
    r = ComsolRunner(cfg.section("comsol"), cfg.case_dir, cfg.case["function"], out / "work")
    try:
        r.start()
        res = r.evaluate(cfg.space.comsol_params(params), opts, tag=name)
    finally:
        r.close()
    if res.get("status") != "ok":
        raise SystemExit(f"COMSOL run failed:\n{res.get('message')}")
    (out / f"{name}.json").write_text(json.dumps(res))
    p = ", ".join(f"{k}={v:.3g}" for k, v in rec["params"].items())
    files = plot_fields(res, out, name, f"{cfg.case['function']} {name}: {p}",
                        primary=cfg.section("loss").get("primary"), params=rec["params"])
    print(f"field data: {out / (name + '.json')}\nsolved model: {opts['save_mph']}")
    return files
