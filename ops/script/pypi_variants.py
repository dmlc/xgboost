"""Create Package variants for PyPI distribution."""

import argparse
import os

from test_utils import PY_PACKAGE

IN_PATH = os.path.join(PY_PACKAGE, "pyproject.toml.in")
OUT_PATH = os.path.join(PY_PACKAGE, "pyproject.toml")

NCCL_WHL = """    \"nvidia-nccl-cu12 ; platform_system == 'Linux' and platform_machine != 'aarch64'\","""

NAME = "{{ name }}"
NCCL = "{{ nccl }}"


def copyfile(src: str, dst: str) -> None:
    with open(src, "rb") as fd:
        content = fd.read()
    with open(dst, "wb") as fd:
        fd.write(content)


def make_pyproject(*, use_cpu_suffix: int, require_nccl_dep: int) -> None:
    if use_cpu_suffix == 1 and require_nccl_dep == 1:
        raise ValueError(
            "xgboost-cpu cannot require NCCL dependency. "
            "If --use-cpu-suffix=1, you must set --require-nccl-dep=0."
        )

    with open(IN_PATH) as fd:
        pyproject = fd.read()

    readme_dft = os.path.join(PY_PACKAGE, "README.dft.rst")
    readme_cpu = os.path.join(PY_PACKAGE, "README.cpu.rst")
    readme = os.path.join(PY_PACKAGE, "README.rst")
    pyproject = pyproject.replace(NAME, "xgboost-cpu" if use_cpu_suffix else "xgboost")
    copyfile(readme_cpu if use_cpu_suffix else readme_dft, readme)
    pyproject = pyproject.replace(NCCL, NCCL_WHL if require_nccl_dep else "")
    pyproject = (
        f"# Generated by `{os.path.basename(__file__)}`, don't edit.\n" + pyproject
    )

    with open(OUT_PATH, "w") as fd:
        fd.write(pyproject)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cpu-suffix",
        type=int,
        choices=[0, 1],
        required=True,
        help="Whether to rename the package name to xgboost-cpu",
    )
    parser.add_argument(
        "--require-nccl-dep",
        type=int,
        choices=[0, 1],
        required=True,
        help="Whether to require the NCCL dependency",
    )
    args = parser.parse_args()
    make_pyproject(
        use_cpu_suffix=args.use_cpu_suffix,
        require_nccl_dep=args.require_nccl_dep,
    )
