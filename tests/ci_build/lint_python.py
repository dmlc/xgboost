import argparse
import os
import subprocess
import sys

from test_utils import DirectoryExcursion

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURDIR, os.path.pardir, os.path.pardir))


def run_formatter(rel_path: str):
    path = os.path.join(PROJECT_ROOT, rel_path)
    isort_ret = subprocess.run(["isort", "--check", "--profile=black", path]).returncode
    black_ret = subprocess.run(
        ["black", "--check", "./python-package/xgboost/dask.py"]
    ).returncode
    if isort_ret != 0 or black_ret != 0:
        msg = (
            "Please run the following command on your machine to address the format"
            f" errors:\n isort --check --profile=black {rel_path}\n black {rel_path}\n"
        )
        print(msg, file=sys.stdout)
        return False
    return True


def run_mypy(rel_path: str):
    with DirectoryExcursion(os.path.join(PROJECT_ROOT, "python-package")):
        path = os.path.join(PROJECT_ROOT, rel_path)
        ret = subprocess.run(["mypy", path])
        if ret.returncode != 0:
            return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=int, choices=[0, 1], default=1)
    parser.add_argument("--type-check", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if args.format == 1:
        if not all(
            [
                run_formatter("python-package/xgboost/dask.py"),
                run_formatter("python-package/xgboost/spark"),
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
                "tests/python-gpu/test_gpu_with_dask.py",
                "tests/python-gpu/test_gpu_data_iterator.py",
            ]
        ):
            sys.exit(-1)
