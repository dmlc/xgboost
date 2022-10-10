import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

from pylint import epylint
from test_utils import DirectoryExcursion

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURDIR, os.path.pardir, os.path.pardir))


def run_formatter(rel_path: str) -> bool:
    path = os.path.join(PROJECT_ROOT, rel_path)
    isort_ret = subprocess.run(["isort", "--check", "--profile=black", path]).returncode
    black_ret = subprocess.run(["black", "--check", rel_path]).returncode
    if isort_ret != 0 or black_ret != 0:
        msg = (
            "Please run the following command on your machine to address the format"
            f" errors:\n isort --profile=black {rel_path}\n black {rel_path}\n"
        )
        print(msg, file=sys.stdout)
        return False
    return True


def run_mypy(rel_path: str) -> bool:
    with DirectoryExcursion(os.path.join(PROJECT_ROOT, "python-package")):
        path = os.path.join(PROJECT_ROOT, rel_path)
        ret = subprocess.run(["mypy", path])
        if ret.returncode != 0:
            return False
        return True


class PyLint:
    """A helper for running pylint, mostly copied from dmlc-core/scripts."""

    def __init__(self) -> None:
        self.pypackage_root = os.path.join(PROJECT_ROOT, "python-package/")
        self.pylint_cats = set(["error", "warning", "convention", "refactor"])
        self.pylint_opts = [
            "--extension-pkg-whitelist=numpy",
            "--rcfile=" + os.path.join(self.pypackage_root, ".pylintrc"),
        ]

    def run(self, path: str) -> Tuple[Dict, str, str]:
        (pylint_stdout, pylint_stderr) = epylint.py_run(
            " ".join([str(path)] + self.pylint_opts), return_std=True
        )
        emap = {}
        err = pylint_stderr.read()

        out = []
        for line in pylint_stdout:
            out.append(line)
            key = line.split(":")[-1].split("(")[0].strip()
            if key not in self.pylint_cats:
                continue
            if key not in emap:
                emap[key] = 1
            else:
                emap[key] += 1

        return {path: emap}, err, "\n".join(out)

    def __call__(self) -> bool:
        all_errors: Dict[str, Dict[str, int]] = {}

        def print_summary_map(result_map: Dict[str, Dict[str, int]]) -> int:
            """Print summary of certain result map."""
            if len(result_map) == 0:
                return 0
            ftype = "Python"
            npass = sum(1 for x in result_map.values() if len(x) == 0)
            print(f"====={npass}/{len(result_map)} {ftype} files passed check=====")
            for fname, emap in result_map.items():
                if len(emap) == 0:
                    continue
                print(
                    f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={str(emap)}"
                )
            return len(result_map) - npass

        all_scripts = []
        for root, dirs, files in os.walk(self.pypackage_root):
            for f in files:
                if f.endswith(".py"):
                    all_scripts.append(os.path.join(root, f))

        with Pool(cpu_count()) as pool:
            error_maps = pool.map(self.run, all_scripts)
            for emap, err, out in error_maps:
                print(out)
                if len(err) != 0:
                    print(err)
                all_errors.update(emap)

        nerr = print_summary_map(all_errors)
        return nerr == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=int, choices=[0, 1], default=1)
    parser.add_argument("--type-check", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pylint", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if args.format == 1:
        if not all(
            run_formatter(path)
            for path in [
                # core
                "python-package/xgboost/__init__.py",
                "python-package/xgboost/_typing.py",
                "python-package/xgboost/compat.py",
                "python-package/xgboost/config.py",
                "python-package/xgboost/dask.py",
                "python-package/xgboost/sklearn.py",
                "python-package/xgboost/spark",
                "python-package/xgboost/federated.py",
                "python-package/xgboost/testing.py",
                # tests
                "tests/python/test_config.py",
                "tests/python/test_spark/",
                "tests/python/test_quantile_dmatrix.py",
                "tests/python-gpu/test_gpu_spark/",
                "tests/ci_build/lint_python.py",
                # demo
                "demo/guide-python/cat_in_the_dat.py",
                "demo/guide-python/categorical.py",
                "demo/guide-python/spark_estimator_examples.py",
            ]
        ):
            sys.exit(-1)

    if args.type_check == 1:
        if not all(
            run_mypy(path)
            for path in [
                "python-package/xgboost/",
                "demo/guide-python/external_memory.py",
                "demo/guide-python/cat_in_the_dat.py",
                "tests/python/test_data_iterator.py",
                "tests/python/test_spark/test_data.py",
                "tests/python-gpu/test_gpu_with_dask/test_gpu_with_dask.py",
                "tests/python-gpu/test_gpu_data_iterator.py",
                "tests/python-gpu/test_gpu_spark/test_data.py",
                "tests/ci_build/lint_python.py",
            ]
        ):
            sys.exit(-1)

    if args.pylint == 1:
        if not PyLint()():
            sys.exit(-1)
