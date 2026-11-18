#!/bin/bash
# Runs the whole final study: every case in cases/final/, one stage after another;
# the jobs inside a stage run side by side on separate COMSOL ports (see the port and
# n_workers of each YAML).  Every run resumes from its history.jsonl, so running this
# script again continues where it stopped.  The study took 3 h 24 min on the i7-14700F server.
#
#   setsid nohup bash scripts/run_final_study.sh > runs/final_driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
C=cases/final
L=runs/final_logs
mkdir -p $L
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
presplit() {   # cross-section just before the split next to each branch outlet, for every fine field export
    for j in runs/$1/fields/*_fine.json; do
        [ -f "$j" ] && $PY scripts/plot_presplit.py "$j" > /dev/null 2>&1 && stamp "  presplit plot: ${j%.json}_presplit.png"
    done
}

stamp "stage A: 2D references (ETA ~45 min)"
$PY -m comsol_opt run $C/u2d_straight_ref.yaml > $L/u2d_straight_ref.log 2>&1 &
$PY -m comsol_opt run $C/b2d_straight_ref.yaml > $L/b2d_straight_ref.log 2>&1 &
$PY -m comsol_opt run $C/b2d_straight.yaml     > $L/b2d_straight.log 2>&1 &
wait
# the fully heated straight channel itself (the 'best' of a reference run is its best heater window)
$PY -m comsol_opt fields $C/u2d_straight_ref.yaml --mesh fine --set lam=5.5 > $L/u2d_straight_ref_fullheat.log 2>&1
presplit final_b2d_straight_ref; presplit final_b2d_straight
stamp "stage A done"

stamp "stage B: 2D one-outlet BO (ETA ~25 min)"
$PY -m comsol_opt run $C/u2d_sine.yaml > $L/u2d_sine.log 2>&1 &
$PY -m comsol_opt run $C/u2d_serp.yaml > $L/u2d_serp.log 2>&1 &
wait
stamp "stage B done"

stamp "stage C: 2D one-outlet grid searches (ETA ~30 min)"
$PY -m comsol_opt.gridcase $C/u2d_sine.yaml --levels L0,Z,fine > $L/u2d_sine_grid.log 2>&1 &
$PY -m comsol_opt.gridcase $C/u2d_serp.yaml --levels L0,Z,fine > $L/u2d_serp_grid.log 2>&1 &
wait
stamp "stage C done"

stamp "stage D: 2D branched BO (ETA ~1 h)"
$PY -m comsol_opt run $C/b2d_sine.yaml > $L/b2d_sine.log 2>&1 &
$PY -m comsol_opt run $C/b2d_serp.yaml > $L/b2d_serp.log 2>&1 &
wait
presplit final_b2d_sine; presplit final_b2d_serp
stamp "stage D done"

stamp "stage E: 3D straight reference + 3D mesh check (ETA ~1.5 h)"
$PY -m comsol_opt run $C/u3d_straight_ref.yaml > $L/u3d_straight_ref.log 2>&1 &
PYTHONPATH=$PWD $PY scripts/mesh_check_3d.py > $L/u3d_meshcheck.log 2>&1 &
wait
$PY -m comsol_opt fields $C/u3d_straight_ref.yaml --mesh fine --set lam=5.5 > $L/u3d_straight_ref_fullheat.log 2>&1
stamp "stage E done"

stamp "stage F: 3D serpentine BO (ETA ~2-3 h)"
$PY -m comsol_opt run $C/u3d_serp.yaml > $L/u3d_serp.log 2>&1
stamp "stage F done - all runs finished"
