import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

from pylint import epylint
from test_utils import PY_PACKAGE, ROOT, cd, print_time, record_time

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))


@record_time
def run_black(rel_path: str) -> bool:
    cmd = ["black", "-q", "--check", rel_path]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        subprocess.run(["black", "--version"])
        msg = """
Please run the following command on your machine to address the formatting error:

        """
        msg += " ".join(cmd)
        print(msg, file=sys.stderr)
        return False
    return True


@record_time
def run_isort(rel_path: str) -> bool:
    cmd = ["isort", "--check", "--profile=black", rel_path]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        subprocess.run(["isort", "--version"])
        msg = """
Please run the following command on your machine to address the formatting error:

        """
        msg += " ".join(cmd)
        print(msg, file=sys.stderr)
        return False
    return True


@record_time
@cd(PY_PACKAGE)
def run_mypy(rel_path: str) -> bool:
    path = os.path.join(ROOT, rel_path)
    ret = subprocess.run(["mypy", path])
    if ret.returncode != 0:
        return False
    return True


class PyLint:
    """A helper for running pylint, mostly copied from dmlc-core/scripts."""

    def __init__(self) -> None:
        self.pypackage_root = os.path.join(ROOT, "python-package/")
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


@record_time
def run_pylint() -> bool:
    return PyLint()()


@record_time
def main(args: argparse.Namespace) -> None:
    if args.format == 1:
        black_results = [
            run_black(path)
            for path in [
                # core
                "python-package/",
                # tests
                "tests/python/test_config.py",
                "tests/python/test_data_iterator.py",
                "tests/python/test_dt.py",
                "tests/python/test_quantile_dmatrix.py",
                "tests/python/test_tree_regularization.py",
                "tests/python-gpu/test_gpu_data_iterator.py",
                "tests/ci_build/lint_python.py",
                "tests/test_distributed/test_with_spark/",
                "tests/test_distributed/test_gpu_with_spark/",
                # demo
                "demo/json-model/json_parser.py",
                "demo/guide-python/cat_in_the_dat.py",
                "demo/guide-python/categorical.py",
                "demo/guide-python/feature_weights.py",
                "demo/guide-python/sklearn_parallel.py",
                "demo/guide-python/spark_estimator_examples.py",
                "demo/guide-python/individual_trees.py",
                # CI
                "tests/ci_build/lint_python.py",
                "tests/ci_build/test_r_package.py",
                "tests/ci_build/test_utils.py",
                "tests/ci_build/change_version.py",
            ]
        ]
        if not all(black_results):
            sys.exit(-1)

        isort_results = [
            run_isort(path)
            for path in [
                # core
                "python-package/",
                # tests
                "tests/test_distributed/",
                "tests/python/",
                "tests/python-gpu/",
                "tests/ci_build/",
                # demo
                "demo/",
                # misc
                "dev/",
                "doc/",
            ]
        ]
        if not all(isort_results):
            sys.exit(-1)

    if args.type_check == 1:
        if not all(
            run_mypy(path)
            for path in [
                # core
                "python-package/xgboost/",
                # demo
                "demo/json-model/json_parser.py",
                "demo/guide-python/external_memory.py",
                "demo/guide-python/cat_in_the_dat.py",
                "demo/guide-python/feature_weights.py",
                "demo/guide-python/individual_trees.py",
                # tests
                "tests/python/test_dt.py",
                "tests/python/test_data_iterator.py",
                "tests/python-gpu/test_gpu_data_iterator.py",
                "tests/test_distributed/test_with_spark/test_data.py",
                "tests/test_distributed/test_gpu_with_spark/test_data.py",
                "tests/test_distributed/test_gpu_with_dask/test_gpu_with_dask.py",
                # CI
                "tests/ci_build/lint_python.py",
                "tests/ci_build/test_r_package.py",
                "tests/ci_build/test_utils.py",
                "tests/ci_build/change_version.py",
            ]
        ):
            subprocess.check_call(["mypy", "--version"])
            sys.exit(-1)

    if args.pylint == 1:
        if not run_pylint():
            sys.exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run static checkers for XGBoost, see `python_lint.yml' "
            "conda env file for a list of dependencies."
        )
    )
    parser.add_argument("--format", type=int, choices=[0, 1], default=1)
    parser.add_argument("--type-check", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pylint", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    try:
        main(args)
    finally:
        print_time()
