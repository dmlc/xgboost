"""Utilities for packaging R code and running tests."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from platform import system

from test_utils import DirectoryExcursion, cd, print_time, record_time

ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir
    )
)
r_package = os.path.join(ROOT, "R-package")


def get_mingw_bin() -> str:
    return os.path.join("c:/rtools40/mingw64/", "bin")


@cd(ROOT)
@record_time
def pack_rpackage() -> Path:
    """Compose the directory used for creating R package tar ball."""
    dest = Path("xgboost")

    def pkgroot(path: str) -> None:
        with open(Path("R-package") / "src" / path, "r") as fd:
            makefile = fd.read()
            makefile = makefile.replace("PKGROOT=../../", "PKGROOT=.", 1)
        with open(dest / "src" / path, "w") as fd:
            fd.write(makefile)

    output = subprocess.run(["git", "clean", "-xdf", "--dry-run"], capture_output=True)
    if output.returncode != 0:
        raise ValueError("Failed to check git repository status.", output)
    would_remove = output.stdout.decode("utf-8").strip().split("\n")

    if would_remove and not all(f.find("tests/ci_build") != -1 for f in would_remove):
        raise ValueError(
            "\n".join(would_remove) + "\nPlease cleanup the working git repository."
        )

    shutil.copytree("R-package", dest)
    os.remove(dest / "demo" / "runall.R")
    # core
    shutil.copytree("src", dest / "src" / "src")
    shutil.copytree("include", dest / "src" / "include")
    shutil.copytree("amalgamation", dest / "src" / "amalgamation")
    # rabit
    os.mkdir(dest / "src" / "rabit")
    shutil.copytree(Path("rabit") / "src", dest / "src" / "rabit" / "src")
    shutil.copytree(Path("rabit") / "include", dest / "src" / "rabit" / "include")
    # dmlc-core
    os.mkdir(dest / "src" / "dmlc-core")
    shutil.copytree(
        Path("dmlc-core") / "include", dest / "src" / "dmlc-core" / "include"
    )
    shutil.copytree(Path("dmlc-core") / "src", dest / "src" / "dmlc-core" / "src")
    # makefile & license
    shutil.copyfile("LICENSE", dest / "LICENSE")
    osxmakef = dest / "src" / "Makevars.win-e"
    if os.path.exists(osxmakef):
        os.remove(osxmakef)
    pkgroot("Makevars.in")
    pkgroot("Makevars.win")
    # misc
    rwsp = Path("R-package") / "remove_warning_suppression_pragma.sh"
    if system() != "Windows":
        subprocess.check_call(rwsp)
    rwsp = dest / "remove_warning_suppression_pragma.sh"
    if system() != "Windows":
        subprocess.check_call(rwsp)
    os.remove(rwsp)
    os.remove(dest / "CMakeLists.txt")
    shutil.rmtree(dest / "tests" / "helper_scripts")
    return dest


@cd(ROOT)
@record_time
def build_rpackage(path: str) -> str:
    def find_tarbal() -> str:
        found = []
        for root, subdir, files in os.walk("."):
            for f in files:
                if f.endswith(".tar.gz") and f.startswith("xgboost"):
                    found.append(os.path.join(root, f))
        if not found:
            raise ValueError("Failed to find output tar ball.")
        if len(found) > 1:
            raise ValueError("Found more than one packages:", found)
        return found[0]

    env = os.environ.copy()
    print("Ncpus:", f"{os.cpu_count()}")
    env.update({"MAKEFLAGS": f"-j{os.cpu_count()}"})
    subprocess.check_call([R, "CMD", "build", path], env=env)

    tarball = find_tarbal()
    return tarball


@cd(ROOT)
@record_time
def check_rpackage(path: str) -> None:
    env = os.environ.copy()
    print("Ncpus:", f"{os.cpu_count()}")
    env.update(
        {
            "MAKEFLAGS": f"-j{os.cpu_count()}",
            # cran specific environment variables
            "_R_CHECK_EXAMPLE_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(2.5),
        }
    )

    if system() == "Windows":
        # make sure compiler from rtools is used.
        mingw_bin = get_mingw_bin()
        CXX = os.path.join(mingw_bin, "g++.exe")
        CC = os.path.join(mingw_bin, "gcc.exe")
        env.update({"CC": CC, "CXX": CXX})

    status = subprocess.run([R, "CMD", "check", "--as-cran", path], env=env)
    with open(Path("xgboost.Rcheck") / "00check.log", "r") as fd:
        check_log = fd.read()

    with open(Path("xgboost.Rcheck") / "00install.out", "r") as fd:
        install_log = fd.read()

    msg = f"""
----------------------- Install ----------------------
{install_log}

-----------------------  Check -----------------------
{check_log}

    """

    if status.returncode != 0:
        print(msg)
        raise ValueError("Failed r package check.")

    if check_log.find("WARNING") != -1:
        print(msg)
        raise ValueError("Has unresolved warnings.")
    if check_log.find("Examples with CPU time") != -1:
        print(msg)
        raise ValueError("Suspicious NOTE.")

    # fixme(jiamingy): More checks for NOTE.


@record_time
def test_with_cmake(args: argparse.Namespace) -> None:
    os.mkdir("build")
    with DirectoryExcursion("build"):
        if args.compiler == "mingw":
            mingw_bin = get_mingw_bin()
            CXX = os.path.join(mingw_bin, "g++.exe")
            CC = os.path.join(mingw_bin, "gcc.exe")
            env = os.environ.copy()
            env.update({"CC": CC, "CXX": CXX})
            subprocess.check_call(
                [
                    "cmake",
                    os.path.pardir,
                    "-DUSE_OPENMP=ON",
                    "-DR_LIB=ON",
                    "-DCMAKE_CONFIGURATION_TYPES=Release",
                    "-G",
                    "Unix Makefiles",
                ],
                env=env,
            )
            subprocess.check_call(["make", "-j", "install"])
        elif args.compiler == "msvc":
            subprocess.check_call(
                [
                    "cmake",
                    os.path.pardir,
                    "-DUSE_OPENMP=ON",
                    "-DR_LIB=ON",
                    "-DCMAKE_CONFIGURATION_TYPES=Release",
                    "-A",
                    "x64",
                ]
            )
            subprocess.check_call(
                [
                    "cmake",
                    "--build",
                    os.path.curdir,
                    "--target",
                    "install",
                    "--config",
                    "Release",
                ]
            )
        else:
            raise ValueError("Wrong compiler")
    with DirectoryExcursion(r_package):
        subprocess.check_call(
            [
                R,
                "-q",
                "-e",
                "library(testthat); setwd('tests'); source('testthat.R')",
            ]
        )
        subprocess.check_call([R, "-q", "-e", "demo(runall, package = 'xgboost')"])


@record_time
def main(args: argparse.Namespace) -> None:
    if args.task == "build":
        src_dir = pack_rpackage()
        build_rpackage(src_dir)
        return
    elif args.task == "check":
        if args.build_tool == "autotools":
            src_dir = pack_rpackage()
            tarball = build_rpackage(src_dir)
            check_rpackage(tarball)
        else:
            test_with_cmake(args)
    else:
        raise ValueError("Unexpected task.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Helper script for making R package and running R tests on CI. There are"
            " also other helper scripts in the R tests directory for installing"
            " dependencies and running linter."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["build", "check"],
        default="check",
        required=False,
    )
    parser.add_argument(
        "--compiler",
        type=str,
        choices=["mingw", "msvc"],
        help="Compiler used for compiling CXX code.",
    )
    parser.add_argument(
        "--build-tool",
        type=str,
        choices=["cmake", "autotools"],
        help="Build tool for compiling CXX code and install R package.",
    )
    parser.add_argument(
        "--r",
        type=str,
        default="R" if system() != "Windows" else "R.exe",
        help="Path to the R executable.",
    )
    args = parser.parse_args()
    R = args.r

    try:
        main(args)
    finally:
        print_time()
