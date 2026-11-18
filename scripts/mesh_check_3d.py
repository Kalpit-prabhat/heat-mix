"""3D mesh check: one serpentine design (lam 8, A 3; heater 2.4, 12 and 24 mm long) on
the coarse, the fine and a finer reference mesh, solved in parallel.  Results go to
runs/final_u3d_meshcheck/<mesh>/, and a summary table is printed at the end.

    PYTHONPATH=. .venv/bin/python scripts/mesh_check_3d.py
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from comsol_opt.config import load_config
from comsol_opt.loss import outlet_metrics
from comsol_opt.study import Study

ROOT = Path(__file__).resolve().parent.parent
MESHES = {"coarse": None, "fine": None,
          "ref": {"type": "mapped", "n_w": 36, "w_ratio": 3, "hxy": 0.06, "nz": 16}}
PORTS = {"coarse": 2072, "fine": 2074, "ref": 2076}


def one(mesh):
    cfg = load_config(ROOT / "cases/final/u3d_serp.yaml")
    cfg.raw["comsol"]["port"] = PORTS[mesh]
    cfg.raw["comsol"]["server_args"] = ["-np", "6" if mesh == "ref" else "4"]
    cfg.raw["comsol"]["timeout_s"] = 10800
    cfg.raw["optimizer"]["n_workers"] = 1
    cfg.raw["output"]["progress_every"] = 0
    cfg.raw["comsol_opts"]["heat_list"] = [[2, 2.4], [2, 12], [2, 24]]
    fid = "coarse" if mesh == "coarse" else "fine"
    if MESHES[mesh]:
        cfg.raw["comsol_opts"]["mesh"]["fine"] = MESHES[mesh]
    st = Study(cfg, ROOT / "runs/final_u3d_meshcheck" / mesh)
    t = time.time()
    rec = st.evaluate(np.array([cfg.space.to_unit({"lam": 8.0, "A": 3.0})]), fid, "meshcheck")[0]
    st.close()
    return mesh, rec, time.time() - t


with ThreadPoolExecutor(3) as ex:
    out = list(ex.map(one, MESHES))
print("\n3D serpentine mesh check (lam 8, A 3):")
for mesh, rec, dt in out:
    if rec["status"] != "ok":
        print(f"  {mesh:6s} FAILED after {dt:.0f} s: {rec.get('message', '')[:300]}")
        continue
    raw = json.loads((ROOT / "runs/final_u3d_meshcheck" / mesh / rec["raw"]).read_text())
    per = []
    for e in raw.get("sweep", []):
        sub = dict(raw, T=e["T"], T_bulk=e.get("T_bulk", raw["T_bulk"]))
        per.append(f"heater {e['heat_len']:.1f}: {1000 * outlet_metrics(sub)['l2_flow']:.1f} mK")
    print(f"  {mesh:6s} {raw.get('mesh_elements')} el, {dt:.0f} s, best l2_flow "
          f"{1000 * rec['metrics']['l2_flow']:.1f} mK;  " + ";  ".join(per))
