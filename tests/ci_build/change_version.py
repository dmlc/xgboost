"""
1. Modify ``CMakeLists.txt`` in source tree and ``python-package/xgboost/VERSION`` if
needed, run CMake .
    If this is a RC release, the Python version has the form <major>.<minor>.<patch>rc1
2. Modify ``DESCRIPTION`` and ``configure.ac`` in R-package. Run ``autoreconf``.
3. Run ``mvn`` in ``jvm-packages``
    If this is a RC release, the version for JVM packages has the form
    <major>.<minor>.<patch>-RC1
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile

from test_utils import JVM_PACKAGES, PY_PACKAGE, R_PACKAGE, ROOT, cd


@cd(ROOT)
def cmake(major: int, minor: int, patch: int) -> None:
    version = f"{major}.{minor}.{patch}"
    with open("CMakeLists.txt", "r") as fd:
        cmakelist = fd.read()
    pattern = r"project\(xgboost LANGUAGES .* VERSION ([0-9]+\.[0-9]+\.[0-9]+)\)"
    matched = re.search(pattern, cmakelist)
    assert matched, "Couldn't find the version string in CMakeLists.txt."
    print(matched.start(1), matched.end(1))
    cmakelist = cmakelist[: matched.start(1)] + version + cmakelist[matched.end(1) :]
    with open("CMakeLists.txt", "w") as fd:
        fd.write(cmakelist)

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.call(["cmake", "-S", ".", "-B", tmpdir])


@cd(PY_PACKAGE)
def pypkg(
    major: int, minor: int, patch: int, rc: int, is_rc: bool, is_dev: bool
) -> None:
    version = f"{major}.{minor}.{patch}"
    pyver = version
    if is_rc:
        pyver = pyver + f"rc{rc}"
    if is_dev:
        pyver = pyver + "-dev"

    pyver_path = os.path.join("xgboost", "VERSION")
    with open(pyver_path, "w") as fd:
        fd.write(pyver + "\n")

    pyprj_path = os.path.join("pyproject.toml")
    with open(pyprj_path, "r") as fd:
        pyprj = fd.read()
    matched = re.search('version = "' + r"([0-9]+\.[0-9]+\.[0-9]+.*)" + '"', pyprj)
    assert matched, "Couldn't find version string in pyproject.toml."
    pyprj = pyprj[: matched.start(1)] + pyver + pyprj[matched.end(1) :]
    with open(pyprj_path, "w") as fd:
        fd.write(pyprj)


@cd(R_PACKAGE)
def rpkg(major: int, minor: int, patch: int) -> None:
    version = f"{major}.{minor}.{patch}.1"
    # Version: 2.0.0.1
    desc_path = "DESCRIPTION"
    with open(desc_path, "r") as fd:
        description = fd.read()
        pattern = r"Version:\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)"
        matched = re.search(pattern, description)
        assert matched, "Couldn't find version string in DESCRIPTION."
        description = (
            description[: matched.start(1)] + version + description[matched.end(1) :]
        )
        pattern = r"Date:\ ([0-9]+\-[0-9]+\-[0-9]+)"
        today = datetime.date.today()
        matched = re.search(pattern, description)
        assert matched, "Couldn't find date string in DESCRIPTION."
        description = (
            description[: matched.start(1)] + str(today) + description[matched.end(1) :]
        )
    with open(desc_path, "w") as fd:
        fd.write(description)

    config_path = "configure.ac"
    # AC_INIT([xgboost],[2.0.0],[],[xgboost],[])
    version = f"{major}.{minor}.{patch}"
    with open(config_path, "r") as fd:
        config = fd.read()
        pattern = (
            r"AC_INIT\(\[xgboost\],\[([0-9]+\.[0-9]+\.[0-9]+)\],\[\],\[xgboost\],\[\]\)"
        )
        matched = re.search(pattern, config)
        assert matched, "Couldn't find version string in configure.ac"
        config = config[: matched.start(1)] + version + config[matched.end(1) :]

    with open(config_path, "w") as fd:
        fd.write(config)

    subprocess.check_call(["autoreconf"])


@cd(JVM_PACKAGES)
def jvmpkgs(
    major: int, minor: int, patch: int, rc: int, is_rc: bool, is_dev: bool
) -> None:
    version = f"{major}.{minor}.{patch}"
    if is_dev:
        version += "-SNAPSHOT"
    if is_rc:
        version += f"-RC{rc}"
    subprocess.check_call(["mvn", "versions:set", f"-DnewVersion={version}"])


@cd(ROOT)
def main(args: argparse.Namespace) -> None:
    major = args.major
    minor = args.minor
    patch = args.patch
    rc = args.rc
    is_rc = args.is_rc == 1
    is_dev = args.is_dev == 1
    if is_rc and is_dev:
        raise ValueError("It cannot be both a rc and a dev branch.")
    if is_rc:
        assert rc >= 1, "RC version starts from 1."
    else:
        assert rc == 0, "RC is not used."

    cmake(major, minor, patch)
    pypkg(major, minor, patch, rc, is_rc, is_dev)
    rpkg(major, minor, patch)
    jvmpkgs(major, minor, patch, rc, is_rc, is_dev)

    print(
        """

Please examine the changes and commit. Be aware that mvn might leave backup files in the
source tree.

"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--major", type=int)
    parser.add_argument("--minor", type=int)
    parser.add_argument("--patch", type=int)
    parser.add_argument("--rc", type=int, default=0)
    parser.add_argument("--is-rc", type=int, choices=[0, 1])
    parser.add_argument("--is-dev", type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        exit(-1)
