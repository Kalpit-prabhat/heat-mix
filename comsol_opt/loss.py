"""Loss functions computed from the raw outlet data returned by COMSOL.

The COMSOL side only exports raw data (outlet temperature T and normal
velocity un sampled on a uniform ny x nz grid of cell centres, plus some exact
integrals); every loss is computed here from that data.

All grid cells have equal area, so the discrete sums below are the midpoint
rule for the surface integrals:

    l2_area :  sqrt( (1/A) * int (T - T_need)^2 dA )            [K]
               == sqrt(mean_ij (T_ij - T_need)^2)
               i.e. the RMS form of  sum_ij (T_ij - T_need)^2.
    l2_flow :  same, weighted by the outflowing velocity max(un, 0)
               (what the fluid actually carries out of the device).
    linf    :  max_ij |T_ij - T_need|                            [K]
    std_area:  area standard deviation of T (ignores any mean offset)  [K]

Branched outlets (cases that return ``branch_Q_ml_min`` and ``branch_Tb``, one
entry per outlet branch, e.g. sine_split_2d):

    branch_T:  RMS over the branches of (bulk T_i - T_need)                 [K]
    branch_Q:  RMS over the branches of (Q_i / mean(Q) - 1)                 [-]
    branch  :  branch_T / T_scale + flow_weight * branch_Q                  [-]
               T_scale defaults to dT_target, flow_weight to 1 (YAML loss:
               section), i.e. a 1 % flow imbalance counts like a spread of
               0.01*dT_target in temperature.
    (NaN for cases without branches.)

All metrics are always computed and stored; ``primary`` selects the one that
is optimized.  The optimizer works on log10(primary), which is much easier to
model with a Gaussian process than the raw value (losses span decades).
"""
from __future__ import annotations

import math

import numpy as np

METRICS = ("l2_area", "l2_flow", "linf", "std_area", "branch", "branch_T", "branch_Q")


def branch_metrics(res: dict, target: float, opts: dict | None = None) -> dict[str, float]:
    """balance of flow rate and bulk temperature between outlet branches"""
    opts = opts or {}
    nan = {"branch": math.nan, "branch_T": math.nan, "branch_Q": math.nan}
    if "branch_Q_ml_min" not in res or "branch_Tb" not in res:
        return nan
    Q = np.atleast_1d(np.asarray(res["branch_Q_ml_min"], dtype=float))
    Tb = np.atleast_1d(np.asarray(res["branch_Tb"], dtype=float))
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(Tb))) or Q.mean() <= 0:
        return nan
    t_scale = float(opts.get("T_scale") or res.get("dT_target") or 1.0)
    w_q = float(opts.get("flow_weight", 1.0))
    bT = float(np.sqrt(np.mean((Tb - target) ** 2)))
    bQ = float(np.sqrt(np.mean((Q / Q.mean() - 1.0) ** 2)))
    return {"branch": bT / t_scale + w_q * bQ, "branch_T": bT, "branch_Q": bQ}


def outlet_metrics(res: dict, target: float | None = None, opts: dict | None = None) -> dict[str, float]:
    T = np.asarray(res["T"], dtype=float)
    un = np.asarray(res["un"], dtype=float)
    if target is None:
        target = float(res["T_need"])
    ok = np.isfinite(T)
    nan_frac = 1.0 - ok.mean()
    if ok.sum() == 0:
        raise ValueError("outlet temperature grid is all NaN")
    dT = T[ok] - target
    w = np.clip(un[ok], 0.0, None)
    w = np.nan_to_num(w)
    out = {
        "l2_area": float(np.sqrt(np.mean(dT ** 2))),
        "l2_flow": float(np.sqrt(np.sum(w * dT ** 2) / np.sum(w))) if w.sum() > 0 else math.nan,
        "linf": float(np.max(np.abs(dT))),
        "std_area": float(np.std(T[ok])),
        "T_mean_grid": float(np.mean(T[ok])),
        "grid_nan_frac": float(nan_frac),
    }
    out.update(branch_metrics(res, target, opts))
    return out


def objective_value(metrics: dict[str, float], primary: str, floor: float = 1e-4) -> float:
    """value the GP models (lower is better)"""
    v = metrics[primary]
    if not np.isfinite(v):
        return math.nan
    return math.log10(max(v, floor))
