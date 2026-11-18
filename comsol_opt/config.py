"""Configuration loading and the parameter space.

A case is described by one YAML file (see cases/final/*.yaml).
Only the ``variables`` section is optimized; ``fixed`` values are passed to
COMSOL unchanged.  Every value carries a COMSOL unit string so the MATLAB side
can set it as a COMSOL parameter expression such as ``3[mm]``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class Variable:
    name: str
    lo: float
    hi: float
    unit: str = ""
    log: bool = False

    def to_unit(self, x: np.ndarray | float) -> np.ndarray | float:
        """physical value -> [0, 1]"""
        if self.log:
            return (np.log(x) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (x - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: np.ndarray | float) -> np.ndarray | float:
        """[0, 1] -> physical value"""
        if self.log:
            return np.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + u * (self.hi - self.lo)


def comsol_expr(value: Any, unit: str) -> str:
    if isinstance(value, str):
        return value
    v = f"{float(value):.12g}"
    return f"{v}[{unit}]" if unit else v


# names usable inside constraint expressions
_SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE.update(abs=abs, min=min, max=max)


@dataclass
class Space:
    variables: list[Variable]
    fixed: dict[str, dict]              # name -> {"value": ..., "unit": ...}
    constraints: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    # ---- conversions -------------------------------------------------------
    def to_unit(self, params: dict[str, float]) -> np.ndarray:
        return np.array([v.to_unit(params[v.name]) for v in self.variables], dtype=float)

    def from_unit(self, u: np.ndarray) -> dict[str, float]:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return {v.name: float(v.from_unit(ui)) for v, ui in zip(self.variables, u)}

    def namespace(self, params: dict[str, float]) -> dict[str, float]:
        """fixed numeric values + variable values, in the units of the YAML file"""
        ns = {k: f["value"] for k, f in self.fixed.items() if not isinstance(f["value"], str)}
        ns.update(params)
        return ns

    def comsol_params(self, params: dict[str, float]) -> dict[str, str]:
        out = {k: comsol_expr(f["value"], f.get("unit", "")) for k, f in self.fixed.items()}
        for v in self.variables:
            out[v.name] = comsol_expr(params[v.name], v.unit)
        for k, val in params.items():          # overrides of fixed values (e.g. a heater window
            if k in self.fixed and not isinstance(val, str):   # chosen by a heat-only sweep)
                out[k] = comsol_expr(val, self.fixed[k].get("unit", ""))
        return out

    # ---- constraints -------------------------------------------------------
    def feasible(self, params: dict[str, float]) -> bool:
        ns = dict(_SAFE)
        ns.update(self.namespace(params))
        for c in self.constraints:
            try:
                if not eval(c, {"__builtins__": {}}, ns):
                    return False
            except Exception as e:  # a constraint that cannot be evaluated is a config error
                raise ValueError(f"constraint '{c}' failed to evaluate: {e}") from e
        return True

    def feasible_unit(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        return np.array([self.feasible(self.from_unit(u)) for u in U], dtype=bool)


@dataclass
class Config:
    path: Path
    raw: dict
    space: Space

    @property
    def case(self) -> dict:
        return self.raw["case"]

    @property
    def case_dir(self) -> Path:
        return (self.path.parent / self.case.get("matlab_dir", ".")).resolve()

    def section(self, name: str) -> dict:
        return self.raw.get(name, {}) or {}


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    variables = []
    for name, spec in (raw.get("variables") or {}).items():
        lo, hi = float(spec["min"]), float(spec["max"])
        if not hi > lo:
            raise ValueError(f"variable {name}: max must be > min")
        log = bool(spec.get("log", False))
        if log and lo <= 0:
            raise ValueError(f"variable {name}: log scale needs min > 0")
        variables.append(Variable(name, lo, hi, spec.get("unit", ""), log))
    fixed = {}
    for name, spec in (raw.get("fixed") or {}).items():
        if isinstance(spec, dict):
            fixed[name] = {"value": spec["value"], "unit": spec.get("unit", "")}
        else:
            fixed[name] = {"value": spec, "unit": ""}
    overlap = set(fixed) & {v.name for v in variables}
    if overlap:
        raise ValueError(f"parameters both fixed and variable: {sorted(overlap)}")
    space = Space(variables, fixed, list(raw.get("constraints") or []))
    return Config(path, raw, space)
