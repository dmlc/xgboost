"""Run pytest with targeted diagnostics for the slow macOS CI tests.

Temporary investigation of Intel/Apple Silicon differences in process startup,
collectives and training. Remove this wrapper and its workflow wiring once the
bottleneck is explained and a fix has been measured. Profiling adds overhead;
compare these runs with each other, not directly with uninstrumented timings.
"""

import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest
from threadpoolctl import threadpool_info

TARGETS = (
    "python/test_tracker.py::",
    "python/test_ordinal.py::test_training_continuation",
    "test_distributed/test_with_dask/test_ranking.py::test_dask_ranking",
)
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_WAIT_POLICY",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "KMP_BLOCKTIME",
    "KMP_AFFINITY",
    "NUMEXPR_NUM_THREADS",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
)


def command(args):
    """Unavailable OS probes are evidence, not reasons to fail a test."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=8, check=False
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


class Diagnostics:
    def __init__(self):
        self.root = Path("macos-diagnostics") / str(os.getpid())
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"

    def emit(self, event, **data):
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **data}, default=str) + "\n")

    def pytest_sessionstart(self, session):
        packages = {}
        for name in (
            "xgboost",
            "numpy",
            "scipy",
            "scikit-learn",
            "dask",
            "distributed",
            "loky",
            "hypothesis",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        # Conda runtime packages (notably llvm-openmp) have no Python metadata.
        conda_packages = []
        for path in sorted((Path(sys.prefix) / "conda-meta").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            conda_packages.append(
                {key: record.get(key) for key in ("name", "version", "build")}
            )
        hardware = {}
        if sys.platform == "darwin":
            for key in (
                "machdep.cpu.brand_string",
                "hw.logicalcpu",
                "hw.physicalcpu",
                "hw.memsize",
                "hw.cpufrequency",
            ):
                hardware[key] = command(["sysctl", "-n", key])
        self.emit(
            "session",
            platform=platform.platform(),
            machine=platform.machine(),
            python=sys.version,
            cpu_count=os.cpu_count(),
            hardware=hardware,
            packages=packages,
            conda_packages=conda_packages,
            thread_environment={key: os.environ.get(key) for key in THREAD_ENV},
            pytest_args=session.config.invocation_params.args,
        )

    def sample(self, stop, path):
        # CPU counters include each surviving child separately. Children can exit
        # between samples, so this is a timeline, not an exact total CPU charge.
        parent = psutil.Process()
        started = time.perf_counter()
        stacks_taken = False
        with path.open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                processes = []
                try:
                    children = parent.children(recursive=True)
                except psutil.Error:
                    children = []
                for process in [parent, *children]:
                    try:
                        with process.oneshot():
                            processes.append(
                                {
                                    "pid": process.pid,
                                    "created": process.create_time(),
                                    "name": process.name(),
                                    "threads": process.num_threads(),
                                    "cpu": process.cpu_times()._asdict(),
                                    "memory": process.memory_info()._asdict(),
                                    "status": process.status(),
                                }
                            )
                    except psutil.Error:
                        pass
                elapsed = time.perf_counter() - started
                stream.write(
                    json.dumps(
                        {
                            "elapsed": elapsed,
                            "processes": processes,
                            "system_memory": psutil.virtual_memory()._asdict(),
                            "swap": psutil.swap_memory()._asdict(),
                        }
                    )
                    + "\n"
                )
                stream.flush()
                if sys.platform == "darwin" and elapsed >= 10 and not stacks_taken:
                    # One parent and one worker sample expose native waits and
                    # loaded dylibs that cProfile cannot see. Never attach a debugger.
                    stacks_taken = True
                    # Resource-tracker helpers are also children; prefer the
                    # worker with the most observed CPU time over the first PID.
                    cpu_by_pid = {
                        row["pid"]: row["cpu"]["user"] + row["cpu"]["system"]
                        for row in processes
                    }
                    workers = sorted(
                        children,
                        key=lambda child: cpu_by_pid.get(child.pid, 0),
                        reverse=True,
                    )
                    for process in [parent, *workers[:1]]:
                        output = path.with_name(f"{path.stem}-{process.pid}.sample.txt")
                        result = command(
                            [
                                "sample",
                                str(process.pid),
                                "1",
                                "10",
                                "-file",
                                str(output),
                            ]
                        )
                        stream.write(
                            json.dumps({"native_sample_pid": process.pid, **result})
                            + "\n"
                        )
                stop.wait(1)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        if not any(target in item.nodeid for target in TARGETS):
            yield
            return
        key = hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
        stop = threading.Event()
        sampler = threading.Thread(
            target=self.sample, args=(stop, self.root / f"{key}.jsonl"), daemon=True
        )
        profiler = cProfile.Profile()
        self.emit(
            "test_start", nodeid=item.nodeid, key=key, threadpools=threadpool_info()
        )
        sampler.start()
        started = time.perf_counter()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - started
            stop.set()
            sampler.join(timeout=20)
            profiler.dump_stats(str(self.root / f"{key}.prof"))
            with (self.root / f"{key}.profile.txt").open(
                "w", encoding="utf-8"
            ) as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(40)
            self.emit(
                "test_end",
                nodeid=item.nodeid,
                key=key,
                wall_seconds=elapsed,
                sampler_finished=not sampler.is_alive(),
                threadpools=threadpool_info(),
            )
            print(f"\n[macOS diagnostics] {item.nodeid}: {elapsed:.2f}s; profile={key}")

    def pytest_runtest_logreport(self, report):
        if any(target in report.nodeid for target in TARGETS):
            self.emit(
                "phase",
                nodeid=report.nodeid,
                phase=report.when,
                duration=report.duration,
                outcome=report.outcome,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[Diagnostics()]))
