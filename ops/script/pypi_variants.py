"""Create Package variants for PyPI distribution."""

import argparse
import os

from test_utils import PY_PACKAGE, ROOT

in_path = os.path.join(PY_PACKAGE, "pyproject.toml.in")
out_path = os.path.join(PY_PACKAGE, "pyproject.toml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        choices=["default", "cpu", "manylinux2014"],
        default="default",
    )
    args = parser.parse_args()

    with open(in_path) as fd:
        pyproject = fd.read()

    if args.variant == "cpu":
        pyproject = pyproject.replace("{{ nccl }}", "").replace(
            "{{ README.rst }}", '"README.cpu.rst"'
        )
    elif args.variant == "manylinux2014":
        pyproject = pyproject.replace("{{ nccl }}", "").replace(
            "{{ README.rst }}", '"README.rst"'
        )
    else:
        pyproject = pyproject.replace(
            "{{ nccl }}",
            """    \"nvidia-nccl-cu12 ; platform_system == 'Linux' and platform_machine != 'aarch64'\",""",
        ).replace("{{ README.rst }}", '"README.rst"')
    pyproject = (
        f"# Generated by `{os.path.basename(__file__)}`, don't not edit.\n" + pyproject
    )

    with open(out_path, "w") as fd:
        fd.write(pyproject)
