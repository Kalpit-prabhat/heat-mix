"""Optimization driver: coarse-mesh Bayesian optimization, then a fine-mesh check.

Stages (all recorded in <out>/history.jsonl, so a run can be stopped and
resumed at any time with the same command):

  1. init       n_init space-filling points, coarse mesh
  2. coarse_bo  GP/EI proposals on the coarse mesh until n_coarse evaluations
  3. fine_top   the fine_top_k best distinct coarse designs re-run on the fine mesh

The best fine-mesh design is written to <out>/best.json and re-solved once with
field export for the plots (<out>/fields/).  The staged grid search (gridcase.py)
uses the same evaluation machinery.
"""
from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import bo
from .config import Config
from .loss import METRICS, objective_value, outlet_metrics
from .runner import ComsolRunner


def best_of_sweep(res: dict, m_main: dict, primary: str, target, loss_cfg: dict):
    """A run with a heater sweep (res['sweep'], heat-only re-solves on the same flow)
    is scored by its best heater window.  Returns (metrics, choice)."""
    best_m, choice = m_main, {"main": True}
    for e in res["sweep"]:
        if e.get("T") is None:
            continue
        sub = dict(res)
        sub["T"] = e["T"]
        for k in ("T_bulk", "branch_Tb"):
            if e.get(k) is not None:
                sub[k] = e[k]
        try:
            m = outlet_metrics(sub, target, loss_cfg)
        except Exception:
            continue
        if np.isfinite(m.get(primary, np.nan)) and m[primary] < best_m.get(primary, np.inf):
            best_m = m
            choice = {k: e[k] for k in ("heat_start", "heat_len", "heat1_s", "heat1_l",
                                        "T_heater", "T_bulk", "power_clamped") if k in e}
    return best_m, choice


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


class Study:
    def __init__(self, cfg: Config, out_dir: Path):
        self.cfg = cfg
        self.space = cfg.space
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "evals").mkdir(exist_ok=True)
        (self.out / "mph").mkdir(exist_ok=True)
        self.hist_path = self.out / "history.jsonl"
        self.records: list[dict] = []
        if self.hist_path.exists():
            self.records = [json.loads(l) for l in self.hist_path.read_text().splitlines() if l.strip()]
            log(f"resuming: {len(self.records)} evaluations found in {self.hist_path}")
        self.opt = cfg.section("optimizer")
        self.loss_cfg = cfg.section("loss")
        self.primary = self.loss_cfg.get("primary", "l2_area")
        if self.primary not in METRICS:
            raise ValueError(f"loss.primary must be one of {METRICS}")
        self.seed = int(self.opt.get("seed", 0))
        self.runners: list[ComsolRunner] = []
        # keep a copy of the config next to the results
        (self.out / "config_used.yaml").write_text(cfg.path.read_text())
        root = Path(__file__).resolve().parent.parent
        rel = cfg.path.relative_to(root) if cfg.path.is_relative_to(root) else cfg.path
        (self.out / "run_info.json").write_text(json.dumps({"config": str(rel)}, indent=1))

    # ------------------------------------------------------------------ data
    def _data(self, fidelity: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        recs = [r for r in self.records if r["fidelity"] == fidelity]
        if not recs:
            return np.zeros((0, self.space.dim)), np.zeros(0), []
        X = np.array([self.space.to_unit(r["params"]) for r in recs])
        y = np.array([r["objective"] if r["status"] == "ok" and r["objective"] is not None
                      and math.isfinite(r["objective"]) else math.nan for r in recs])
        ok = np.isfinite(y)
        if ok.any() and not ok.all():
            # failed simulations: impute a bad value so the GP steers away
            y[~ok] = np.max(y[ok]) + float(self.opt.get("failure_penalty", 0.5))
        return X, y, recs

    def _count(self, stage: str) -> int:
        return sum(r["stage"] == stage for r in self.records)

    def best(self, fidelity: str) -> dict | None:
        ok = [r for r in self.records if r["fidelity"] == fidelity and r["status"] == "ok"
              and r["objective"] is not None and math.isfinite(r["objective"])]
        return min(ok, key=lambda r: r["objective"]) if ok else None

    # ------------------------------------------------------------ evaluation
    def _comsol_opts(self, fidelity: str, rid: int) -> dict:
        c = self.cfg.section("comsol_opts")
        opts = {k: v for k, v in c.items() if k not in ("mesh",)}
        opts["mesh"] = (c.get("mesh") or {}).get(fidelity, {})
        opts["fidelity"] = fidelity
        if fidelity == "fine" and (c.get("mesh") or {}).get("coarse"):
            # solve the flow on the coarse mesh first as Newton initial guess
            opts["mesh_init"] = c["mesh"]["coarse"]
        save = self.cfg.section("output").get(f"save_mph_{fidelity}", fidelity == "fine")
        if save:
            opts["save_mph"] = str((self.out / "mph" / f"{rid:04d}_{fidelity}.mph").resolve())
        return opts

    def _start_runners(self) -> None:
        if self.runners:
            return
        nw = int(self.opt.get("n_workers", 1))
        case = self.cfg.case
        for w in range(nw):
            r = ComsolRunner(self.cfg.section("comsol"), self.cfg.case_dir, case["function"],
                             self.out / "work" / f"w{w}", worker=w)
            log(f"starting worker {w} (COMSOL port {r.port})")
            r.start()
            self.runners.append(r)

    def close(self) -> None:
        for r in self.runners:
            r.close()
        self.runners = []

    def evaluate(self, U: np.ndarray, fidelity: str, stage: str) -> list[dict]:
        self._start_runners()
        U = np.atleast_2d(U)
        jobs = []
        for u in U:
            rid = len(self.records) + len(jobs)
            params = self.space.from_unit(u)
            jobs.append((rid, params))

        def run(i_job):
            i, (rid, params) = i_job
            runner = self.runners[i % len(self.runners)]
            cp = self.space.comsol_params(params)
            res = runner.evaluate(cp, self._comsol_opts(fidelity, rid), tag=f"{rid:04d}")
            return rid, params, res

        results = []
        n = len(self.runners)
        for k in range(0, len(jobs), n):
            chunk = list(enumerate(jobs[k:k + n]))
            with ThreadPoolExecutor(max_workers=n) as ex:
                results.extend(ex.map(run, chunk))

        new = []
        for rid, params, res in sorted(results, key=lambda t: t[0]):
            rec = self._record(rid, params, res, fidelity, stage)
            self.records.append(rec)
            with open(self.hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            new.append(rec)
            self._log_record(rec)
        self.write_csv()
        every = int(self.cfg.section("output").get("progress_every", 5))
        if every and (len(self.records) // every) != ((len(self.records) - len(new)) // every):
            self.plot_progress()
        return new

    def plot_progress(self) -> None:
        """report/progress_*.png: loss, design and temperature profiles over the runs"""
        try:
            from .progress import plot_progress
            plot_progress(self.out, self.primary)
        except Exception as e:          # plotting must never stop the optimization
            log(f"progress plots failed: {type(e).__name__}: {e}")

    def _record(self, rid, params, res, fidelity, stage) -> dict:
        raw_file = self.out / "evals" / f"{rid:04d}_{fidelity}.json"
        raw_file.write_text(json.dumps(res))
        rec = {"id": rid, "stage": stage, "fidelity": fidelity, "params": params,
               "status": res.get("status", "failed"), "objective": None, "metrics": {},
               "diag": {}, "raw": str(raw_file.relative_to(self.out)),
               "time": res.get("wall_time_python", res.get("wall_time"))}
        if rec["status"] == "ok":
            try:
                target = self.loss_cfg.get("target")
                m = outlet_metrics(res, None if target is None else float(target), self.loss_cfg)
                if res.get("sweep") and self.loss_cfg.get("sweep_best", True):
                    m, choice = best_of_sweep(res, m, self.primary,
                                              None if target is None else float(target), self.loss_cfg)
                    rec["sweep_best"] = choice
                rec["metrics"] = m
                rec["objective"] = objective_value(m, self.primary)
                if not np.isfinite(rec["objective"]):
                    rec["status"] = "failed"
                    rec["message"] = f"loss '{self.primary}' is not finite for this result"
                elif m["grid_nan_frac"] > 0.05:
                    rec["status"] = "failed"
                    rec["message"] = f"{100 * m['grid_nan_frac']:.0f}% of outlet grid is NaN"
            except Exception as e:
                rec["status"] = "failed"
                rec["message"] = f"loss evaluation failed: {e}"
            for k in ("T_bulk", "T_mean_area", "T_min", "T_max", "T_rms_area", "T_rms_flow",
                      "dp_Pa", "P_tot_W", "P_heater_W", "P_factor", "power_clamped",
                      "T_heater", "A_heat_mm2", "T_wall_max", "P_loss_W", "u_max", "Re", "path_length_mm",
                      "min_gap_mm", "min_curv_radius_mm", "mesh_elements", "t_solve", "t_total",
                      "T_heater2", "branch_Q_ml_min", "branch_Tb", "branch_T_rms", "branch_w_in"):
                if k in res:
                    rec["diag"][k] = res[k]
            sb = rec.get("sweep_best") or {}
            for k in ("T_heater", "T_bulk", "power_clamped"):
                if k in sb:                     # report the chosen heater window, not the main one
                    rec["diag"][k] = sb[k]
        else:
            rec["message"] = (res.get("message") or "")[:4000]
        return rec

    def _log_record(self, rec: dict) -> None:
        p = ", ".join(f"{k}={v:.4g}" for k, v in rec["params"].items())
        sb = {k: v for k, v in (rec.get("sweep_best") or {}).items() if k in ("heat_start", "heat_len", "heat1_s", "heat1_l")}
        if sb:
            p += "  | best heater: " + ", ".join(f"{k}={v:.3g}" for k, v in sb.items())
        if rec["status"] == "ok":
            m = rec["metrics"]
            log(f"#{rec['id']:<4d} {rec['stage']:<9s} {rec['fidelity']:<6s} "
                f"{self.primary}={m[self.primary]:.4g}  (linf={m['linf']:.3g} K"
                + (f", branch dT={m['branch_T']:.3g} K, dQ={100 * m['branch_Q']:.2f}%"
                   if np.isfinite(m.get('branch', float('nan'))) else "") + ")  "
                f"P x{rec['diag'].get('P_factor', float('nan')):.2f}"
                + (f" T_heat={rec['diag']['T_heater'] - 273.15:.1f}C"
                   if np.isfinite(rec['diag'].get('T_heater') or float('nan')) else "")
                + f"{'(limit)' if rec['diag'].get('power_clamped') else ''}  "
                f"t={rec['time']:.0f}s  | {p}")
        else:
            msg = (rec.get("message") or "").strip().splitlines()
            first = next((l for l in msg if l.strip()), "")
            log(f"#{rec['id']:<4d} {rec['stage']:<9s} {rec['fidelity']:<6s} FAILED: {first[:200]}  | {p}")

    def write_csv(self) -> None:
        names = self.space.names
        cols = ["id", "stage", "fidelity", "status", "objective", *names,
                *METRICS, "T_bulk", "T_min", "T_max", "P_factor", "power_clamped", "P_loss_W",
                "T_heater", "dp_Pa", "min_gap_mm", "mesh_elements", "time"]
        with open(self.out / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.records:
                row = [r["id"], r["stage"], r["fidelity"], r["status"], r["objective"]]
                row += [r["params"][n] for n in names]
                row += [r["metrics"].get(k) for k in METRICS]
                row += [r["diag"].get(k) for k in ("T_bulk", "T_min", "T_max", "P_factor",
                                                  "power_clamped", "P_loss_W", "T_heater", "dp_Pa",
                                                  "min_gap_mm", "mesh_elements")]
                row += [r["time"]]
                w.writerow(row)

    # ----------------------------------------------------------------- stages
    def run(self) -> dict | None:
        o = self.opt
        nw = int(o.get("n_workers", 1))
        n_init = int(o.get("n_init", 5 * self.space.dim))
        n_coarse = int(o.get("n_coarse", 60))
        top_k = int(o.get("fine_top_k", 4))
        ncand = int(o.get("n_candidates", 20000))
        xi = float(o.get("xi", 0.01))
        try:
            # 1. initial design ------------------------------------------------
            done = self._count("init")
            if done < n_init:
                rng = np.random.default_rng(self.seed)
                U0 = bo.initial_design(self.space, n_init, rng)
                log(f"stage 1/3: initial design, {n_init - done} coarse runs left")
                for k in range(done, n_init, nw):
                    self.evaluate(U0[k:min(k + nw, n_init)], "coarse", "init")

            # 2. coarse BO -----------------------------------------------------
            n_c = sum(r["fidelity"] == "coarse" for r in self.records)
            if n_c < n_coarse:
                log(f"stage 2/3: coarse-mesh Bayesian optimization, {n_coarse - n_c} runs left")
            while n_c < n_coarse:
                X, y, _ = self._data("coarse")
                if not np.isfinite(y).any():
                    raise RuntimeError("all coarse simulations failed - fix the model first "
                                       "(see 'message' in history.jsonl)")
                model = bo.GPModel.fit(X, y, self.seed)
                rng = np.random.default_rng(self.seed + len(self.records))
                U = bo.propose(model, self.space, X, float(np.min(y)), min(nw, n_coarse - n_c),
                               rng, ncand, xi)
                self.evaluate(U, "coarse", "coarse_bo")
                n_c = sum(r["fidelity"] == "coarse" for r in self.records)
            self._save_gp_summary()

            # 3. fine-mesh check of the best coarse designs ------------------------
            done = self._count("fine_top")
            if done < top_k:
                X, y, recs = self._data("coarse")
                order = np.argsort(y)
                picks, min_sep = [], float(o.get("fine_top_min_sep", 0.1))
                already = [self.space.to_unit(r["params"]) for r in self.records if r["stage"] == "fine_top"]
                for i in order:
                    if recs[i]["status"] != "ok":
                        continue
                    if all(np.linalg.norm(X[i] - p) > min_sep for p in picks):
                        picks.append(X[i])
                    if len(picks) == top_k:
                        break
                todo = [p for p in picks if all(np.linalg.norm(p - a) > 1e-9 for a in already)]
                todo = todo[:top_k - done]
                log(f"stage 3/3: re-running {len(todo)} best coarse designs on the fine mesh")
                for k in range(0, len(todo), nw):
                    self.evaluate(np.array(todo[k:k + nw]), "fine", "fine_top")

        finally:
            self.close()

        best = self.best("fine") or self.best("coarse")
        self.plot_progress()
        if best:
            (self.out / "best.json").write_text(json.dumps(best, indent=2))
            log(f"best ({best['fidelity']} mesh, run #{best['id']}): "
                f"{self.primary} = {best['metrics'][self.primary]:.4g} K")
            for k, v in best["params"].items():
                unit = next(var.unit for var in self.space.variables if var.name == k)
                log(f"    {k:12s} = {v:.4g} {unit}")
            if self.cfg.section("output").get("fields_best", True) and self.runners == []:
                try:
                    from .fields import fields_for_record
                    log("re-solving the best design with field export for the plots ...")
                    files = fields_for_record(self.cfg, self.out, best)
                    log("field plots: " + ", ".join(str(f) for f in files))
                except BaseException as e:   # plots must never lose the optimization result
                    log(f"field plots failed: {e}")
        return best

    def _save_gp_summary(self) -> None:
        """length scales of the coarse GP: short length scale = influential parameter"""
        X, y, _ = self._data("coarse")
        if len(y) < 3 or not np.isfinite(y).any():
            return
        gp = bo.fit_gp(X, y, self.seed)
        k = gp.kernel_
        ls = np.atleast_1d(k.k1.k2.length_scale)
        summary = {"kernel": str(k),
                   "length_scales": dict(zip(self.space.names, map(float, ls))),
                   "note": "length scales in unit-cube coordinates; smaller = objective more "
                           "sensitive to that parameter; > ~3 means practically irrelevant"}
        (self.out / "gp_summary.json").write_text(json.dumps(summary, indent=2))
        log("GP length scales (smaller = more influential): "
            + ", ".join(f"{n}={v:.2f}" for n, v in summary["length_scales"].items()))
