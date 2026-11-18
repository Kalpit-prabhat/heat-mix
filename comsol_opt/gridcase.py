"""Staged grid search over any case YAML (the same file the BO uses).

    .venv/bin/python -m comsol_opt.gridcase CASE.yaml [--out DIR] [--levels L0,Z1,Z2,fine]

It shares the BO's machinery (workers, history.jsonl, resume, loss, heater sweeps):
every grid point is a design in the YAML's variable box, infeasible points are skipped,
and points already in the history are not run again.  The grid lives in the unit cube,
so log-scaled variables are spaced logarithmically.

YAML section ``grid``::

    grid:
      L0:   {lam: 8, a: 5, f_sine: 4}   # points per variable (ends included); variables
                                        # not listed are held at 'hold'
      hold: {heat2_l: 0.5}              # YAML units; default: middle of the box
      zoom: [[3, 0.5], [1, 0.25]]       # per zoom level: [centres, half-width as a fraction
                                        # of the L0 step]; 3 points per variable around each of
                                        # the best distinct designs (variables not in L0: the
                                        # step is 1/4 of the box)
      fine_top: 3                       # best distinct coarse designs re-run on the fine mesh

Output: <out>/history.jsonl, best.json, grid_summary.txt and the field plots of the best
design (as for the BO).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .study import Study, log


def _key(u: np.ndarray) -> tuple:
    return tuple(np.round(np.asarray(u, dtype=float), 6))


class GridCase:
    def __init__(self, cfg, out: Path):
        self.cfg = cfg
        self.sp = cfg.space
        self.g = cfg.section("grid") or {}
        if not self.g.get("L0"):
            raise SystemExit("the YAML needs a grid: section with L0 (points per variable)")
        self.st = Study(cfg, out)
        self.names = self.sp.names
        self.n0 = {k: int(v) for k, v in self.g["L0"].items()}
        unknown = set(self.n0) - set(self.names)
        if unknown:
            raise SystemExit(f"grid.L0 lists unknown variables {sorted(unknown)}")
        self.hold = np.full(self.sp.dim, 0.5)
        for k, v in (self.g.get("hold") or {}).items():
            i = self.names.index(k)
            self.hold[i] = float(self.sp.variables[i].to_unit(float(v)))
        # L0 step per variable (unit cube); variables not scanned: 1/4 of the box
        self.step = np.array([1.0 / (self.n0[n] - 1) if self.n0.get(n, 1) > 1 else 0.25 for n in self.names])

    def _done(self, fidelity: str) -> set:
        return {_key(self.sp.to_unit(r["params"])) for r in self.st.records if r["fidelity"] == fidelity}

    def _run(self, U: list, fidelity: str, stage: str) -> None:
        U = [np.clip(u, 0, 1) for u in U]
        seen, todo = self._done(fidelity), []
        for u in U:
            k = _key(u)
            if k in seen:
                continue
            seen.add(k)
            todo.append(u)
        if not todo:
            log(f"{stage}: nothing left to run")
            return
        feas = self.sp.feasible_unit(np.array(todo))
        todo = [u for u, f in zip(todo, feas) if f]
        log(f"{stage}: {len(todo)} feasible {fidelity} runs ({int((~feas).sum())} infeasible skipped)")
        nw = int(self.cfg.section("optimizer").get("n_workers", 1))
        batch = 4 * nw                      # write history in small batches
        for k in range(0, len(todo), batch):
            self.st.evaluate(np.array(todo[k:k + batch]), fidelity, stage)

    def _best_distinct(self, n: int, fidelity: str = "coarse", sep: float | None = None) -> list:
        ok = [r for r in self.st.records if r["fidelity"] == fidelity and r["status"] == "ok"
              and r["objective"] is not None and np.isfinite(r["objective"])]
        ok.sort(key=lambda r: r["objective"])
        sep = sep if sep is not None else 0.5 * float(np.min(self.step))
        picks = []
        for r in ok:
            u = self.sp.to_unit(r["params"])
            if all(np.linalg.norm(u - p) > sep for p in picks):
                picks.append(u)
            if len(picks) == n:
                break
        return picks

    def level0(self) -> None:
        axes = []
        for i, n in enumerate(self.names):
            m = self.n0.get(n, 1)
            axes.append(np.linspace(0, 1, m) if m > 1 else np.array([self.hold[i]]))
        U = [np.array(p) for p in itertools.product(*axes)]
        self._run(U, "coarse", "grid_L0")

    def zoom(self, level: int, n_centres: int, frac: float) -> None:
        centres = self._best_distinct(n_centres)
        U = []
        for c in centres:
            for d in itertools.product((-1, 0, 1), repeat=self.sp.dim):
                U.append(c + np.array(d) * frac * self.step)
        self._run(U, "coarse", f"grid_Z{level}")

    def fine(self, n: int) -> None:
        self._run(self._best_distinct(n, sep=0.25 * float(np.min(self.step))), "fine", "fine_top")

    def finish(self) -> None:
        self.st.close()
        st = self.st
        best = st.best("fine") or st.best("coarse")
        lines = [f"grid search on {self.cfg.path.name}: {len(st.records)} runs, "
                 f"{sum(r['status'] != 'ok' for r in st.records)} failed, loss {st.primary}"]
        for stage in ("grid_L0", "grid_Z1", "grid_Z2", "grid_Z3", "fine_top"):
            rs = [r for r in st.records if r["stage"] == stage and r["status"] == "ok"]
            if rs:
                b = min(rs, key=lambda r: r["objective"])
                lines.append(f"  {stage:9s} {len(rs):5d} ok   best #{b['id']} {st.primary}="
                             f"{b['metrics'][st.primary]:.4g}  "
                             + ", ".join(f"{k}={v:.4g}" for k, v in b["params"].items())
                             + (f"  heater {b['sweep_best']}" if b.get("sweep_best") else ""))
        (st.out / "grid_summary.txt").write_text("\n".join(lines) + "\n")
        log("\n".join(lines))
        st.plot_progress()
        if best:
            (st.out / "best.json").write_text(json.dumps(best, indent=2))
            if self.cfg.section("output").get("fields_best", True):
                try:
                    from .fields import fields_for_record
                    log("re-solving the best design with field export for the plots ...")
                    files = fields_for_record(self.cfg, st.out, best)
                    log("field plots: " + ", ".join(str(f) for f in files))
                except BaseException as e:
                    log(f"field plots failed: {e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="results folder (default: runs/<output.name>_grid)")
    ap.add_argument("--levels", default="L0,Z,fine", help="L0, Z (all zoom levels) or Z1/Z2/..., fine")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    root = Path(__file__).resolve().parent.parent
    out = Path(a.out) if a.out else root / "runs" / ((cfg.section("output").get("name") or "grid") + "_grid")
    g = GridCase(cfg, out)
    lev = [s.strip() for s in a.levels.split(",")]
    try:
        if "L0" in lev:
            g.level0()
        for i, (n, frac) in enumerate(g.g.get("zoom") or [], start=1):
            if "Z" in lev or f"Z{i}" in lev:
                g.zoom(i, int(n), float(frac))
        if "fine" in lev:
            g.fine(int(g.g.get("fine_top", 3)))
    finally:
        g.finish()


if __name__ == "__main__":
    main()
