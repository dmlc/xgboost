import os
import subprocess
import sys
from typing import List

from test_utils import PY_PACKAGE, ROOT, cd, print_time, record_time


class TypeCheckPaths:
    """The paths mypy runs on."""

    MYPY = (
        # core
        "python-package/",
        # tests
        "tests/python/generate_models.py",
        "tests/python/test_model_compatibility.py",
        "tests/python/test_collective.py",
        "tests/python/test_demos.py",
        "tests/python/test_data_iterator.py",
        "tests/python/test_multi_target.py",
        "tests/python/test_intercept.py",
        "tests/python/test_model_io.py",
        "tests/python/test_ordinal.py",
        "tests/python/test_interaction_constraints.py",
        "tests/python-gpu/",
        "tests/test_distributed/test_federated/",
        "tests/test_distributed/test_gpu_federated/",
        "tests/test_distributed/test_with_dask/",
        "tests/test_distributed/test_with_spark/test_data.py",
        "tests/test_distributed/test_gpu_with_dask/",
        # demo
        "demo/dask/",
        "demo/guide-python/",
        "demo/aft_survival/aft_survival_viz_demo.py",
        # CI
        "ops/",
    )


def check_cmd_print_failure_assistance(cmd: List[str]) -> bool:
    if subprocess.run(cmd).returncode == 0:
        return True

    subprocess.run([cmd[0], "--version"])
    msg = """
Please run the following command on your machine to address the error:

    """
    msg += " ".join(cmd)
    print(msg, file=sys.stderr)
    return False


@record_time
@cd(PY_PACKAGE)
def run_mypy(rel_path: str) -> bool:
    cmd = ["mypy", os.path.join(ROOT, rel_path)]
    return check_cmd_print_failure_assistance(cmd)


@record_time
def main() -> None:
    mypy_results = [run_mypy(path) for path in TypeCheckPaths.MYPY]
    if not all(mypy_results):
        sys.exit(-1)


if __name__ == "__main__":
    try:
        main()
    finally:
        print_time()
