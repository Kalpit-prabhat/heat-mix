# Uniform heating of a microfluidic channel: COMSOL + Bayesian optimization

Water flows through a 3 mm wide channel, and a heater on the channel walls raises it by
2 K. The goal is an outlet where **every point** is close to the target temperature
`T_need = T_in + 2 K`, not just the average. The channel shape (straight, sine,
serpentine), the heater window and, for the four-outlet case, the branch openings are
optimized with COMSOL models driven from Python.

This folder holds everything used for the **final study**: the COMSOL models (MATLAB
LiveLink), the case files, the optimizer, the run results, the report and the slides.

## Contents

```
cases/
  final/make_cases.py        writes the 9 case YAMLs below (edit this, not the YAMLs)
  final/*.yaml               one file per case: variables + ranges, constraints, fixed values,
                             loss, mesh, optimizer settings, COMSOL paths
  sine_split_2d/
    sine_split_2d.m          2D model (x-y plane): wavy -> straight -> optional fan into N branches
    split_geometry.m         its outline (pure MATLAB)
  sine_channel/
    sine_channel.m           3D model: wavy channel, rectangular section, half depth modelled
matlab/
  comsol_eval.m              job.json -> case function -> result.json (errors are returned, not raised)
  channel_walls.m            wall shapes: sine (wall_mode 0) or serpentine (wall_mode 1), constant width
  add_channel_curves.m       draws those walls in a COMSOL geometry sequence
comsol_opt/                  Python package
  __main__.py                command line: run, single, report, fields
  config.py                  YAML -> parameter space (unit cube, log scales, constraints)
  runner.py                  one COMSOL server + one MATLAB engine per worker, one evaluation
  loss.py                    losses from the raw outlet data (l2_flow, l2_area, branch, ...)
  bo.py                      Gaussian process + expected improvement
  study.py                   BO stages, heater-sweep scoring, history/resume, parallel workers
  gridcase.py                staged grid search on the same YAML (python -m comsol_opt.gridcase)
  fields.py, progress.py, report.py   plots of one design / of a run
scripts/
  run_final_study.sh         runs every case of the final study, stage by stage
  mesh_check_3d.py           3D coarse / fine / reference mesh comparison
  plot_presplit.py           branched designs: profile just before the split vs each outlet
runs/                        results of the final study (see "Results" below)
report/
  report.tex, report.pdf     the written report (appendix: every field plot)
  make_figs.py               report summary figures       -> report/figs/
  make_appendix.py           copies the field plots, writes appendix_fields.tex
  make_slide_figs.py         slide figures of this project -> images/ch_*.png
  make_part1_figs.py         pendant-drop results figure (Part I of the talk) -> images/gamma_errorbars.*
images/                      slide figures (the slides themselves are not in this repository)
requirements.txt             Python packages
```

## Requirements and setup

| what | version used |
|---|---|
| COMSOL Multiphysics + LiveLink for MATLAB | 6.4 |
| MATLAB | R2026b (26.2) |
| Python | 3.14 with `requirements.txt`, including the MATLAB Engine API (`matlabengine==26.2.*`, must match MATLAB) |
| License | a COMSOL licence with LiveLink for MATLAB; run `comsol mphserver` once by hand to store the login |

```bash
python3 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```

Tell the code where COMSOL is installed: the folder that holds `bin/comsol` and the
LiveLink folder `mli/` (the case files use `${COMSOL_ROOT}/bin/comsol` and
`${COMSOL_ROOT}/mli`):

```bash
export COMSOL_ROOT=/path/to/comsol64/multiphysics
```

The MATLAB Engine package finds its MATLAB by itself. Run everything from the project
root. The case YAMLs are written by `cases/final/make_cases.py`; run it again after
changing a setting there:

```bash
.venv/bin/python cases/final/make_cases.py
```

## How it works

```
 case YAML  (variables, ranges, constraints, fixed values, loss, mesh)
     |
 optimizer (BO or grid) --params--> runner --job.json--> MATLAB comsol_eval.m --> <case>.m --> COMSOL
     ^                                                                                        |
     '---- loss.py <------------- raw outlet data (T, u on the exit plane, integrals) <------'
```

1. **The model is a MATLAB function** `res = sine_split_2d(P, opts)` or
   `res = sine_channel(P, opts)`. `P` holds COMSOL parameter expressions (`lam = '8[mm]'`),
   so the `.m` file is never edited. The function builds the geometry, mesh and physics,
   solves (flow first, then heat), and returns **raw** data: temperature and normal
   velocity sampled on the exit plane, exact integrals and diagnostics. With
   `reuse_model` the COMSOL model stays alive between runs and only geometry and mesh
   are rebuilt.
2. **Heater sweep on one flow solve.** The heater is a smooth window on the wall, not a
   geometric edge, so the flow and the mesh do not depend on it. Each design is solved
   once, then only the heat step is re-solved for every heater window in `heat_list`
   (10 lengths for the wavy designs; 105–209 start/length pairs for the straight
   references). The design is scored at its best window.
3. **Heater control.** The heated walls are held at `T_heat` (flux
   `h_heat·(T_heat − T)`, `h_heat = 1e5 W/m²K`, a thin metal wall). A COMSOL global
   equation solves for `T_heat` so the outlet bulk temperature is exactly `T_need`,
   limited to 99 °C. A design that would need more is flagged `power_clamped` and misses
   the target, which shows in its loss.
4. **The loss is computed in Python** (`comsol_opt/loss.py`), on `log10` scale for the
   optimizer:
   - `l2_flow` (one outlet): `sqrt( Σ u⁺ (T − T_need)² / Σ u⁺ )`, `u⁺ = max(u, 0)`, the RMS
     deviation weighted by the outflowing velocity, i.e. by what the fluid actually
     carries out. The plain area RMS `l2_area` was also tried; it rewards hot, slow layers
     at the walls that carry almost no fluid, so `l2_flow` was used for every final run.
   - `branch` (four outlets): `RMS_i(T_b,i − T_need)/ΔT + 0.25·RMS_i(Q_i/Q̄ − 1)`, the
     spread of the branch bulk temperatures plus a quarter of the flow imbalance.
5. **Bayesian optimization** (`python -m comsol_opt run`), all on the coarse mesh:
   `n_init` Latin-hypercube points (greedy maximin, constraints checked first), then a
   Gaussian process (Matérn-5/2 with one length scale per variable, plus white noise)
   on `log10(loss)` and expected improvement (ξ = 0.01, 20 000 random + local candidates,
   polished with L-BFGS-B, batches of `n_workers` with the constant-liar trick) up to
   `n_coarse` runs. Failed runs count as the worst value + 0.5 decades. Finally the
   `fine_top_k` best distinct designs are re-run on the fine mesh; the best of those is
   the result and is re-solved once with field export for the plots.
6. **Staged grid search** (`python -m comsol_opt.gridcase`), same YAML and machinery:
   a full grid (`grid.L0` points per variable, log-spaced where the variable is),
   then zooms of ±½ and ±¼ of the grid step around the best designs, then the best 3 on
   the fine mesh. Run for the two 2D one-outlet cases to check the BO.

Every evaluation is appended to `runs/<name>/history.jsonl` (and `history.csv`); running
the same command again **resumes** where it stopped.

## Physics and geometry (all final runs)

| | |
|---|---|
| Channel | width 3 mm (normal to the flow everywhere), depth 3 mm; heated/wavy length 24 mm |
| Flow | water, 6 mL/min (mean 11 mm/s, Re ≈ 91), constant properties at ~79 °C, steady laminar, fully developed inlet |
| Temperatures | T_in = 343.15 K (70 °C), target rise 2 K, heater ≤ 99 °C |
| Losses | non-heated walls lose `h_loss·(T_amb − T)`, h_loss 2 W/m²K, T_amb 20 °C (branches: 10 W/m²K); in 2D the top/bottom walls enter as a volume loss `2·h·(T_amb − T)/depth` |
| Wall shapes | sine: centre line `A sin(2π(x − x_a)/λ)`, walls offset ±w/2 along the normal; amplitude given as `a = A/A_max(λ)` with the bend-radius limit `A_max = min(3.5, λ²/(4π²(w/2 + 0.05)))`. Serpentine: triangle wave with corners rounded to a 1.55 mm centre-line radius, `A ≤ 3.5 mm` |
| One outlet | 2 mm inlet port, 24 mm wavy section heated from its start, 3 mm straight to the exit plane, then an adiabatic extension (2D 3 mm, 3D 5 mm) |
| Four outlets (2D) | 2 mm port, wavy section (fraction `f_sine` of 24 mm, heater 1) + straight section (heater 2), 6 mm knife-edge fan into four branches with inlet widths 1 : ρ : ρ : 1, 0.75 mm outlet ports at 4 mm pitch |
| 3D | heater on all four walls; half the depth is modelled (symmetry plane at mid-depth) |

### Cases (`cases/final/`)

| case | model | variables (range) | search |
|---|---|---|---|
| `u2d_straight_ref` | 2D | heater start × length (209 windows) | reference |
| `u2d_sine` | 2D | λ 1.5–20 mm (log), a 0–1; heater length | BO 50 + grid 12×8 |
| `u2d_serp` | 2D | λ 6.5–20 mm (log), A 0–3.5 mm; heater length | BO 50 + grid 10×8 |
| `u3d_straight_ref` | 3D | heater start × length (105 windows) | reference |
| `u3d_serp` | 3D | λ 6.5–20 mm (log), A 0–3.5 mm; heater length | BO 28 |
| `b2d_straight_ref` | 2D, 4 outlets | none: analytic equal-flow openings ρ = 0.532, fully heated | reference |
| `b2d_straight` | 2D, 4 outlets | ρ 0.35–1 (log); heater start × length (209 windows) | BO 16 |
| `b2d_sine` | 2D, 4 outlets | λ, a, f_sine 0.3–0.89, heater-2 length 0–1, ρ; heater-1 length | BO 150 |
| `b2d_serp` | 2D, 4 outlets | λ, A, f_sine, heater-2 length, ρ; heater-1 length | BO 150 |

### Meshes

Structured (mapped) meshes, finer towards the walls (size ratio 3). The searches ran on
the coarse mesh; the best 3–4 designs and every reported number and field plot are from
the fine mesh. Velocity, pressure and temperature use linear elements.

| | coarse | fine |
|---|---|---|
| 2D, one outlet | 24 elements across, 0.07 mm along the flow (11–23k) | 48 across, 0.035 mm (44–73k) |
| 2D, four outlets | 12 per branch (48 across the main channel), 0.07 mm (26–49k) | 24 per branch, 0.035 mm (105–165k) |
| 3D | 20 across, 0.12 mm, 8 layers in the half depth (45–94k hexahedra) | 28 across, 0.08 mm, 12 layers (143–289k) |

3D mesh check (`scripts/mesh_check_3d.py`, serpentine λ 8, A 3): the fine mesh is within
4 % of a 535k-element reference; the coarse mesh reads up to 30 % low but ranks the
designs the same way.

## Running

```bash
setsid nohup bash scripts/run_final_study.sh > runs/final_driver.log 2>&1 < /dev/null &
```

runs the whole study (3 h 24 min on the server; stages and per-stage times are in the
script). Single pieces:

```bash
.venv/bin/python -m comsol_opt run cases/final/u2d_serp.yaml
```
```bash
.venv/bin/python -m comsol_opt.gridcase cases/final/u2d_serp.yaml --levels L0,Z,fine
```
```bash
.venv/bin/python -m comsol_opt single cases/final/u3d_serp.yaml --mesh coarse --set lam=8 A=3
```
```bash
.venv/bin/python -m comsol_opt fields runs/final_u2d_serp
```
```bash
.venv/bin/python -m comsol_opt report runs/final_u2d_serp
```

- `run` / `gridcase`: results go to `runs/<output.name>` (grid: `..._grid`).
- `single`: one design, prints the scores (`--save-mph` keeps the model).
- `fields`: re-solves the best design (or `--id N`, or a YAML with `--set ...`) with
  field export and draws the maps, profiles and score table; `--replot` redraws from
  the saved data without COMSOL.
- `report`: convergence, parameter and outlet plots of a run, into `<run>/report/`.

Figures for the report and the slides (no COMSOL needed):

```bash
.venv/bin/python report/make_figs.py
```
```bash
.venv/bin/python report/make_appendix.py
```
```bash
PYTHONPATH=. .venv/bin/python report/make_slide_figs.py
```
```bash
.venv/bin/python report/make_part1_figs.py
```

Then build `report/report.tex` with a LaTeX engine (it was built with tectonic).

### Parallel runs on the server

- The server is an i7-14700F (20 cores / 28 threads, 125 GB RAM) shared with other
  users; check `uptime` before starting a big batch.
- Each worker gets its own COMSOL server on port `comsol.port + worker index`, started
  with `comsol.server_args` (`-np 4`). Without `-np` every server tries to use all 28
  threads. One job gains nothing beyond about 4 cores, and parallel jobs slow each other
  down: 4–6 workers give about 2–2.5× the throughput of one.
- If a server is already listening on a port, the runner uses it, so one can also be
  started by hand: `comsol mphserver -port 2070 -np 4 -multi on`.
- Run long jobs under `setsid nohup` or `tmux` so they survive a dropped SSH session.
- To stop a COMSOL server, kill the PID listening on its port
  (`ss -ltnp | grep ':2070 '`). Do not use `pkill -f comsol`: it also matches and kills
  the shell that runs it.
- Measured cost per design: 2D one outlet ≈ 30 s coarse / 65–70 s fine; 3D ≈ 136 s
  coarse / 355 s fine (including the 10-window heater sweep).

## Results

Outlet non-uniformity of the best design of each case (fine mesh). One outlet:
`l2_flow`; four outlets: spread of the branch bulk temperatures and of the branch flows.

| case | best design | result |
|---|---|---|
| 2D straight, fully heated | — | 980 mK |
| 2D straight, best heater window | heater 0–2.4 mm (10 %) | 388 mK |
| 2D sine (BO / grid) | λ 11.1, a = 1, heater 30 % | 120 / 120 mK |
| 2D serpentine, BO | λ 8.72, A 3.10, heater 60 % | 11.9 mK |
| 2D serpentine, grid | λ 8.34, A 2.13, heater 40 % | **6.3 mK** |
| 3D straight, fully heated | — | 756 mK |
| 3D straight, best heater window | heater 1.2 mm at the start | 378 mK |
| 3D serpentine, BO | λ 7.26, A 3.5, heater 19.2 mm | **27.4 mK** |
| 4 outlets, straight, analytic openings | ρ 0.532, fully heated | 970 mK, flow 0.87 % |
| 4 outlets, straight, BO | ρ 0.540, heater 2.4 mm (at 99 °C) | 515 mK, 0.16 % |
| 4 outlets, sine + straight, BO | λ 10.0, a 1, f 0.68, heater-2 97 %, ρ 0.535 | 116 mK, 0.82 % |
| 4 outlets, serpentine + straight, BO | λ 7.89, A 3.5, f 0.78, no heater 2, ρ 0.469 | **12.3 mK**, 3.9 % |

The serpentine wins everywhere: its bends mix the cross-section, so the heater can be
long and gentle and the outlet still comes out even. Details, field plots and caveats are
in `report/report.pdf`.

### What is in a run folder (`runs/final_<case>[_grid]/`)

| file | content |
|---|---|
| `history.jsonl` / `history.csv` | every evaluation: parameters, all losses, best heater window, diagnostics |
| `best.json` | the result (best fine-mesh design) |
| `evals/NNNN_<mesh>.json` | raw COMSOL output of every evaluation |
| `fields/` | field export of the best design (`*.json`, `*.mph`) and its plots |
| `report/` | progress plots of the search |
| `gp_summary.json` | GP length scales (BO runs): smaller = more influential variable |
| `config_used.yaml` | the case file as it was when the run started |

Also: `runs/final_u{2,3}d_straight_ref_single/` (fully heated references),
`runs/final_u3d_meshcheck/` (3D mesh check), `runs/final_logs/` and
`runs/final_driver.log` (logs of the driver run).

`config_used.yaml` records the case files as they were when the study ran. They still
contain a few keys that the cleaned-up code no longer reads (pillar and fixed-flux heater
settings, all at their off values); the current `cases/final/*.yaml` give the same models.
