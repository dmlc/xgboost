import argparse
import os
import pathlib
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

from test_utils import PY_PACKAGE, ROOT, cd, print_time, record_time


class LintersPaths:
    """The paths each linter run on."""

    BLACK = (
        # core
        "python-package/",
        # tests
        "tests/python/test_config.py",
        "tests/python/test_data_iterator.py",
        "tests/python/test_dt.py",
        "tests/python/test_predict.py",
        "tests/python/test_quantile_dmatrix.py",
        "tests/python/test_tree_regularization.py",
        "tests/python-gpu/test_gpu_data_iterator.py",
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
        "demo/guide-python/quantile_regression.py",
        "demo/guide-python/multioutput_regression.py",
        # CI
        "tests/ci_build/lint_python.py",
        "tests/ci_build/test_r_package.py",
        "tests/ci_build/test_utils.py",
        "tests/ci_build/change_version.py",
    )

    ISORT = (
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
    )

    MYPY = (
        # core
        "python-package/",
        # tests
        "tests/python/test_dt.py",
        "tests/python/test_data_iterator.py",
        "tests/python-gpu/test_gpu_data_iterator.py",
        "tests/test_distributed/test_with_spark/test_data.py",
        "tests/test_distributed/test_gpu_with_spark/test_data.py",
        "tests/test_distributed/test_gpu_with_dask/test_gpu_with_dask.py",
        # demo
        "demo/json-model/json_parser.py",
        "demo/guide-python/external_memory.py",
        "demo/guide-python/cat_in_the_dat.py",
        "demo/guide-python/feature_weights.py",
        "demo/guide-python/individual_trees.py",
        "demo/guide-python/quantile_regression.py",
        "demo/guide-python/multioutput_regression.py",
        # CI
        "tests/ci_build/lint_python.py",
        "tests/ci_build/test_r_package.py",
        "tests/ci_build/test_utils.py",
        "tests/ci_build/change_version.py",
    )


def check_cmd_print_failure_assistance(cmd: List[str]) -> bool:
    if subprocess.run(cmd).returncode == 0:
        return True

    subprocess.run([cmd[0], "--version"])
    msg = """
Please run the following command on your machine to address the formatting error:

    """
    msg += " ".join(cmd)
    print(msg, file=sys.stderr)
    return False


@record_time
@cd(PY_PACKAGE)
def run_black(rel_path: str, fix: bool) -> bool:
    cmd = ["black", "-q", os.path.join(ROOT, rel_path)]
    if not fix:
        cmd += ["--check"]

    return check_cmd_print_failure_assistance(cmd)


@record_time
@cd(PY_PACKAGE)
def run_isort(rel_path: str, fix: bool) -> bool:
    # Isort gets confused when trying to find the config file, so specified explicitly.
    cmd = ["isort", "--settings-path", PY_PACKAGE, os.path.join(ROOT, rel_path)]
    if not fix:
        cmd += ["--check"]

    return check_cmd_print_failure_assistance(cmd)


@record_time
@cd(PY_PACKAGE)
def run_mypy(rel_path: str) -> bool:
    cmd = ["mypy", os.path.join(ROOT, rel_path)]

    return check_cmd_print_failure_assistance(cmd)


class PyLint:
    """A helper for running pylint, mostly copied from dmlc-core/scripts."""

    MESSAGE_CATEGORIES = {
        "Fatal",
        "Error",
        "Warning",
        "Convention",
        "Refactor",
        "Information",
    }
    MESSAGE_PREFIX_TO_CATEGORY = {
        category[0]: category for category in MESSAGE_CATEGORIES
    }

    @classmethod
    @cd(PY_PACKAGE)
    def get_summary(cls, path: str) -> Tuple[str, Dict[str, int], str, str, bool]:
        """Get the summary of pylint's errors, warnings, etc."""
        ret = subprocess.run(["pylint", path], capture_output=True)
        stdout = ret.stdout.decode("utf-8")

        emap: Dict[str, int] = Counter()
        for line in stdout.splitlines():
            if ":" in line and (
                category := cls.MESSAGE_PREFIX_TO_CATEGORY.get(
                    line.split(":")[-2].strip()[0]
                )
            ):
                emap[category] += 1

        return path, emap, stdout, ret.stderr.decode("utf-8"), ret.returncode == 0

    @staticmethod
    def print_summary_map(result_map: Dict[str, Dict[str, int]]) -> int:
        """Print summary of certain result map."""
        if len(result_map) == 0:
            return 0

        ftype = "Python"
        nfail = sum(map(bool, result_map.values()))
        print(
            f"====={len(result_map) - nfail}/{len(result_map)} {ftype} files passed check====="
        )
        for fname, emap in result_map.items():
            if emap:
                print(
                    f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={emap}"
                )
        return nfail

    @classmethod
    def run(cls) -> bool:
        """Run pylint with parallelization on a batch of paths."""
        all_errors: Dict[str, Dict[str, int]] = {}

        with Pool(cpu_count()) as pool:
            error_maps = pool.map(
                cls.get_summary,
                (os.fspath(file) for file in pathlib.Path(PY_PACKAGE).glob("**/*.py")),
            )
            for path, emap, out, err, succeeded in error_maps:
                all_errors[path] = emap
                if succeeded:
                    continue

                print(out)
                if len(err) != 0:
                    print(err)

        nerr = cls.print_summary_map(all_errors)
        return nerr == 0


@record_time
def run_pylint() -> bool:
    return PyLint.run()


@record_time
def main(args: argparse.Namespace) -> None:
    if args.format == 1:
        black_results = [run_black(path, args.fix) for path in LintersPaths.BLACK]
        if not all(black_results):
            sys.exit(-1)

        isort_results = [run_isort(path, args.fix) for path in LintersPaths.ISORT]
        if not all(isort_results):
            sys.exit(-1)

    if args.type_check == 1:
        mypy_results = [run_mypy(path) for path in LintersPaths.MYPY]
        if not all(mypy_results):
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
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix the formatting issues instead of emitting an error.",
    )
    args = parser.parse_args()
    try:
        main(args)
    finally:
        print_time()
