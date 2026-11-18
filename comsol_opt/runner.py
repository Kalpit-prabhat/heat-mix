"""Running one COMSOL evaluation through MATLAB LiveLink.

Flow of one evaluation:

    Python  --job.json-->  MATLAB comsol_eval.m  -->  <case>.m  -->  COMSOL server
    Python  <--result.json--------------------------------------------'

Each worker keeps one MATLAB session alive through the MATLAB Engine API for
Python (pip install matlabengine), so there is no MATLAB start-up cost per
evaluation, and talks to its own ``comsol mphserver``.  The install paths in the
YAML (comsol_cmd, mli_path) may use environment variables such as ${COMSOL_ROOT}.  By default the runner
starts that server (port = base port + worker index, extra arguments from
``comsol.server_args``, e.g. ["-np", "4"]) and stops it at the end; a server
already listening on the port is used as it is.  The server is started with
``-multi on`` so it survives MATLAB disconnecting.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATLAB_DIR = (HERE.parent / "matlab").resolve()


def install_path(value: str) -> str:
    """expand environment variables in an install path from the YAML, e.g.
    ${COMSOL_ROOT}/bin/comsol, and fail clearly if one is not set"""
    out = os.path.expandvars(str(value))
    if "$" in out:
        raise RuntimeError(f"'{value}': set the environment variable first, e.g. "
                           "export COMSOL_ROOT=/path/to/comsol64/multiphysics")
    return out


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


class ComsolServer:
    def __init__(self, port: int, comsol_cmd: str = "comsol", start: bool = True,
                 log_dir: Path | None = None, extra_args: list[str] | None = None):
        self.port = port
        self.comsol_cmd = comsol_cmd
        self.start_server = start
        self.log_dir = log_dir
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None

    def ensure(self, timeout: float = 180.0) -> None:
        if _port_open(self.port):
            return
        if not self.start_server:
            raise RuntimeError(f"no COMSOL server on port {self.port} and start_server is false; "
                               f"start one with:  comsol mphserver -port {self.port} -multi on")
        if shutil.which(self.comsol_cmd) is None:
            raise RuntimeError(f"'{self.comsol_cmd}' not found on PATH")
        log = open(self.log_dir / f"mphserver_{self.port}.log", "ab") if self.log_dir else subprocess.DEVNULL
        cmd = [self.comsol_cmd, "mphserver", "-port", str(self.port), "-multi", "on", "-silent", *self.extra_args]
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if _port_open(self.port):
                time.sleep(2.0)  # let it finish initializing
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"COMSOL server exited with code {self.proc.returncode} "
                                   f"(first time? run 'comsol mphserver' once by hand to store a login)")
            time.sleep(1.0)
        raise RuntimeError(f"COMSOL server did not open port {self.port} within {timeout:.0f} s")

    def stop(self) -> None:
        """stop a server this object started: the 'comsol' launcher script and the Java
        server it spawns share one process group (start_new_session), so signal the group"""
        if self.proc is None:
            return

        def signal_group(sig) -> None:
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        signal_group(signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < 30 and (self.proc.poll() is None or _port_open(self.port)):
            time.sleep(1.0)
        if self.proc.poll() is None or _port_open(self.port):
            signal_group(signal.SIGKILL)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None


class ComsolRunner:
    """One worker: one COMSOL server + one MATLAB session."""

    def __init__(self, comsol_cfg: dict, case_dir: Path, case_function: str,
                 work_dir: Path, worker: int = 0):
        self.cfg = comsol_cfg
        self.case_dir = Path(case_dir)
        self.case_function = case_function
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.port = int(comsol_cfg.get("port", 2036)) + worker
        self.mli = install_path(comsol_cfg["mli_path"])
        self.timeout = float(comsol_cfg.get("timeout_s", 3600))
        self.server = ComsolServer(self.port, install_path(comsol_cfg.get("comsol_cmd", "comsol")),
                                   bool(comsol_cfg.get("start_server", True)), self.work_dir,
                                   [str(x) for x in comsol_cfg.get("server_args") or []])
        self.eng = None

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.server.ensure()
        self._start_engine()

    def _start_engine(self) -> None:
        try:
            import matlab.engine  # type: ignore
        except ImportError as e:
            raise RuntimeError("the MATLAB Engine API for Python is missing "
                               "(pip install matlabengine, version matching MATLAB)") from e
        self.eng = matlab.engine.start_matlab("-nodesktop -nosplash")
        self.eng.addpath(self.mli, nargout=0)
        self.eng.mphstart(float(self.port), nargout=0)
        self.eng.addpath(str(MATLAB_DIR), nargout=0)
        self.eng.addpath(str(self.case_dir), nargout=0)

    def close(self) -> None:
        if self.eng is not None:
            try:
                self.eng.eval("com.comsol.model.util.ModelUtil.clear;", nargout=0)  # free server memory
            except Exception:
                pass
            try:
                self.eng.quit()
            except Exception:
                pass
            self.eng = None
        self.server.stop()

    def restart(self) -> None:
        self.close()
        self.start()

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, params: dict[str, str], opts: dict, tag: str | None = None) -> dict:
        """Run one simulation; never raises for simulation errors."""
        tag = tag or uuid.uuid4().hex[:8]
        # keep the COMSOL model alive between runs, only updating geometry and mesh
        opts = dict(opts)
        opts.setdefault("reuse_model", bool(self.cfg.get("reuse_model", True)))
        job = self.work_dir / f"job_{tag}.json"
        out = self.work_dir / f"res_{tag}.json"
        job.write_text(json.dumps({"case_function": self.case_function,
                                   "params": params, "opts": opts}, indent=1))
        if out.exists():
            out.unlink()
        t0 = time.time()
        try:
            self._run_engine(job, out)
            res = json.loads(out.read_text())
        except Exception as e:  # MATLAB crash, timeout, ...
            res = {"status": "failed", "message": f"{type(e).__name__}: {e}"}
            try:
                self.restart()
            except Exception as e2:
                res["message"] += f" | restart failed: {e2}"
        res.setdefault("wall_time", time.time() - t0)
        res["wall_time_python"] = time.time() - t0
        for p in (job, out):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        return res

    def _run_engine(self, job: Path, out: Path) -> None:
        if self.eng is None:
            self._start_engine()
        fut = self.eng.comsol_eval(str(job), str(out), nargout=0, background=True)
        try:
            fut.result(timeout=self.timeout)
        except TimeoutError:
            fut.cancel()
            raise TimeoutError(f"evaluation exceeded {self.timeout:.0f} s")
