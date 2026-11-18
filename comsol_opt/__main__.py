"""Command line interface.

    python -m comsol_opt run    CONFIG.yaml [--out DIR]
    python -m comsol_opt single CONFIG.yaml [--mesh coarse|fine] [--set name=value ...] [--save-mph]
    python -m comsol_opt report DIR
    python -m comsol_opt fields DIR [--id N] [--mesh coarse|fine] [--replot]
    python -m comsol_opt fields CONFIG.yaml [--set name=value ...] [--mesh coarse|fine]

The staged grid search is a separate entry point: python -m comsol_opt.gridcase CONFIG.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _default_out(cfg) -> Path:
    name = cfg.raw.get("output", {}).get("name") or cfg.case["function"]
    return (cfg.path.parent.parent.parent / "runs" / name).resolve() if "cases" in cfg.path.parts \
        else (Path.cwd() / "runs" / name)


def cmd_run(a):
    from .study import Study
    cfg = load_config(a.config)
    out = Path(a.out).resolve() if a.out else _default_out(cfg)
    print(f"results -> {out}")
    Study(cfg, out).run()


def cmd_single(a):
    from .loss import outlet_metrics
    from .runner import ComsolRunner
    from .study import Study
    cfg = load_config(a.config)
    sp = cfg.space
    params = {v.name: float(v.from_unit(0.5)) for v in sp.variables}   # mid-range default
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        if k not in params:
            sys.exit(f"unknown variable '{k}' (variables: {', '.join(params)})")
        params[k] = float(v)
    if not sp.feasible(params):
        print("WARNING: these parameters violate a constraint in the YAML file")
    out = Path(a.out).resolve() if a.out else _default_out(cfg).with_name(_default_out(cfg).name + "_single")
    out.mkdir(parents=True, exist_ok=True)
    st = Study.__new__(Study)          # reuse the option builder without a history
    st.cfg, st.out = cfg, out
    (out / "mph").mkdir(exist_ok=True)
    opts = st._comsol_opts(a.mesh, 0)
    if a.save_mph:
        opts["save_mph"] = str((out / "mph" / f"single_{a.mesh}.mph").resolve())
    else:
        opts.pop("save_mph", None)
    r = ComsolRunner(cfg.section("comsol"), cfg.case_dir, cfg.case["function"], out / "work")
    print("parameters:", json.dumps(sp.comsol_params(params), indent=1))
    try:
        r.start()
        res = r.evaluate(sp.comsol_params(params), opts, tag="single")
    finally:
        r.close()
    (out / f"single_{a.mesh}.json").write_text(json.dumps(res))
    if res.get("status") != "ok":
        print(res.get("message"))
        sys.exit(1)
    m = outlet_metrics(res)
    print(json.dumps({"metrics": m, **{k: res[k] for k in res if not isinstance(res[k], list)}}, indent=1))


def cmd_report(a):
    from .report import make_report
    make_report(a.dir, a.primary)


def cmd_fields(a):
    """temperature / gradient plots of one design (re-solved once with field export)"""
    from .fields import fields_for_record, plot_fields
    src = Path(a.target).resolve()
    if src.is_dir():
        run_dir = src
        info = run_dir / "run_info.json"
        cfg_path = a.config or (json.loads(info.read_text())["config"] if info.exists() else None)
        if not cfg_path:
            sys.exit("pass --config CASE.yaml (this run has no run_info.json)")
        if not Path(cfg_path).is_absolute():       # stored relative to the project root
            cfg_path = Path(__file__).resolve().parent.parent / cfg_path
        cfg = load_config(cfg_path)
        if a.replot:
            done = sorted((run_dir / "fields").glob("*.json"))
            if a.id is not None:
                done = [p for p in done if p.name.startswith(f"run{a.id:04d}_")]
            if not done:
                sys.exit("nothing to replot in fields/")
            for p in done:
                print(*plot_fields(json.loads(p.read_text()), p.parent, p.stem, p.stem), sep="\n")
            return
        recs = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
        ok = [r for r in recs if r["status"] == "ok" and r.get("objective") is not None]
        if a.id is not None:
            sel = [r for r in recs if r["id"] == a.id]
            if not sel:
                sys.exit(f"no run #{a.id}")
            rec = sel[0]
        else:
            fine = [r for r in ok if r["fidelity"] == "fine"]
            rec = min(fine or ok, key=lambda r: r["objective"])
            print(f"best design: run #{rec['id']} ({rec['fidelity']} mesh)")
    else:
        cfg = load_config(src)
        params = {v.name: float(v.from_unit(0.5)) for v in cfg.space.variables}
        for kv in a.set or []:
            k, v = kv.split("=", 1)
            if k not in params:
                sys.exit(f"unknown variable '{k}' (variables: {', '.join(params)})")
            params[k] = float(v)
        run_dir = _default_out(cfg).with_name(_default_out(cfg).name + "_single")
        rec = {"id": None, "fidelity": a.mesh or "coarse", "params": params}
    files = fields_for_record(cfg, run_dir, rec, a.mesh)
    print(*files, sep="\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m comsol_opt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run / resume the optimization")
    s.add_argument("config")
    s.add_argument("--out", help="result directory (default runs/<name>)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("single", help="run one simulation (for testing / timing)")
    s.add_argument("config")
    s.add_argument("--mesh", default="coarse", choices=["coarse", "fine"])
    s.add_argument("--set", nargs="*", metavar="NAME=VALUE",
                   help="variable values in the YAML units (others: mid-range)")
    s.add_argument("--save-mph", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_single)

    s = sub.add_parser("report", help="plots + summary of a run directory")
    s.add_argument("dir")
    s.add_argument("--primary", default=None)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("fields", help="temperature / gradient field plots of one design")
    s.add_argument("target", help="run directory (best or --id design) or a case YAML (with --set)")
    s.add_argument("--id", type=int, help="run id in the history (default: best)")
    s.add_argument("--mesh", choices=["coarse", "fine"], help="mesh (default: the run's own)")
    s.add_argument("--set", nargs="*", metavar="NAME=VALUE", help="with a case YAML: variable values")
    s.add_argument("--config", help="case YAML (if the run directory has no run_info.json)")
    s.add_argument("--replot", action="store_true", help="redraw from saved field data, no COMSOL")
    s.set_defaults(func=cmd_fields)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
