"""Utilities for packaging R code and running tests."""

import argparse
import os
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from platform import system

try:
    import pandas as pd
except ImportError:
    pd = None

from test_utils import R_PACKAGE, ROOT, DirectoryExcursion, cd, print_time, record_time


def get_mingw_bin() -> str:
    return os.path.join("c:/rtools40/mingw64/", "bin")


@cd(ROOT)
@record_time
def pack_rpackage() -> Path:
    """Compose the directory used for creating R package tar ball."""
    dest = Path("xgboost")

    def pkgroot(path: str) -> None:
        """Change makefiles according to the package layout."""
        with open(Path("R-package") / "src" / path, "r") as fd:
            makefile = fd.read()
            makefile = makefile.replace("PKGROOT=../../", "PKGROOT=.", 1)
        with open(dest / "src" / path, "w") as fd:
            fd.write(makefile)

    output = subprocess.run(["git", "clean", "-xdf", "--dry-run"], capture_output=True)
    if output.returncode != 0:
        raise ValueError("Failed to check git repository status.", output)
    if len(output.stdout) == 0:
        would_remove = None
    else:
        would_remove = output.stdout.decode("utf-8").strip().split("\n")

    if would_remove and not all(f.find("ops") != -1 for f in would_remove):
        raise ValueError(
            "\n".join(would_remove) + "\nPlease cleanup the working git repository."
        )

    shutil.copytree("R-package", dest)
    # core
    shutil.copytree("src", dest / "src" / "src")
    shutil.copytree("include", dest / "src" / "include")
    shutil.copytree("amalgamation", dest / "src" / "amalgamation")
    # dmlc-core
    dmlc_core = Path("dmlc-core")
    os.mkdir(dest / "src" / dmlc_core)
    shutil.copytree(dmlc_core / "include", dest / "src" / "dmlc-core" / "include")
    shutil.copytree(dmlc_core / "src", dest / "src" / "dmlc-core" / "src")
    # makefile & license
    shutil.copyfile("LICENSE", dest / "LICENSE")
    osxmakef = dest / "src" / "Makevars.win-e"
    if os.path.exists(osxmakef):
        os.remove(osxmakef)
    pkgroot("Makevars.in")
    pkgroot("Makevars.win.in")
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
    def find_tarball() -> str:
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

    tarball = find_tarball()
    return tarball


def check_example_timing(rcheck_dir: Path, threshold: float) -> None:
    with open(rcheck_dir / "xgboost-Ex.timings", "r") as fd:
        timings = fd.readlines()
        newlines = []
        for line in timings:
            line = line.strip()
            newlines.append(line)
        con_timings = "\n".join(newlines)
        df = pd.read_csv(StringIO(con_timings), delimiter="\t")
        ratio_n = "user/elapsed"
        df[ratio_n] = df["user"] / df["elapsed"]
        offending = df[df[ratio_n] > threshold]

    try:
        # requires the tabulate package
        df.to_markdown("timings.md")
        offending.to_markdown("offending.md")
    except ImportError:
        print("failed to export markdown files.")
        pass

    if offending.shape[0] == 0:
        return

    print(offending)
    raise ValueError("There are examples using too many threads")


@cd(ROOT)
@record_time
def check_rpackage(path: str) -> None:
    env = os.environ.copy()
    print("Ncpus:", f"{os.cpu_count()}")
    threshold = 2.5
    env.update(
        {
            "MAKEFLAGS": f"-j{os.cpu_count()}",
            # cran specific environment variables
            "_R_CHECK_EXAMPLE_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
            "_R_CHECK_TEST_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
            "_R_CHECK_VIGNETTE_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
        }
    )

    # Actually we don't run this check on windows due to dependency issue.
    if system() == "Windows":
        # make sure compiler from rtools is used.
        mingw_bin = get_mingw_bin()
        CXX = os.path.join(mingw_bin, "g++.exe")
        CC = os.path.join(mingw_bin, "gcc.exe")
        env.update({"CC": CC, "CXX": CXX})

    status = subprocess.run(
        [R, "CMD", "check", "--as-cran", "--timings", path], env=env
    )
    rcheck_dir = Path("xgboost.Rcheck")
    with open(rcheck_dir / "00check.log", "r") as fd:
        check_log = fd.read()

    with open(rcheck_dir / "00install.out", "r") as fd:
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
    if pd is not None:
        check_example_timing(rcheck_dir, threshold)


@cd(R_PACKAGE)
@record_time
def check_rmarkdown() -> None:
    assert system() != "Windows", "Document test doesn't support Windows."
    env = os.environ.copy()
    env.update({"MAKEFLAGS": f"-j{os.cpu_count()}"})
    print("Checking R documentation.")
    bin_dir = os.path.dirname(R)
    rscript = os.path.join(bin_dir, "Rscript")
    subprocess.check_call([rscript, "-e", "roxygen2::roxygenize()"], env=env)
    output = subprocess.run(["git", "diff", "--name-only"], capture_output=True)
    if len(output.stdout.decode("utf-8").strip()) != 0:
        output = subprocess.run(["git", "diff"], capture_output=True)
        raise ValueError(
            "Please run `roxygen2::roxygenize()`. Diff:\n",
            output.stdout.decode("utf-8"),
        )


@cd(R_PACKAGE)
@record_time
def test_with_autotools() -> None:
    """Windows only test. No `--as-cran` check, only unittests. We don't want to manage
    the dependencies on Windows machine.

    """
    assert system() == "Windows"
    mingw_bin = get_mingw_bin()
    CXX = os.path.join(mingw_bin, "g++.exe")
    CC = os.path.join(mingw_bin, "gcc.exe")
    cmd = [R, "CMD", "INSTALL", str(os.path.curdir)]
    env = os.environ.copy()
    env.update({"CC": CC, "CXX": CXX, "MAKEFLAGS": f"-j{os.cpu_count()}"})
    subprocess.check_call(cmd, env=env)
    subprocess.check_call(
        ["R.exe", "-q", "-e", "library(testthat); setwd('tests'); source('testthat.R')"]
    )


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
                    "-G",
                    "Visual Studio 17 2022",
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
        elif args.compiler == "none":
            subprocess.check_call(
                [
                    "cmake",
                    os.path.pardir,
                    "-DUSE_OPENMP=ON",
                    "-DR_LIB=ON",
                    "-DCMAKE_CONFIGURATION_TYPES=Release",
                    "-G",
                    "Unix Makefiles",
                ]
            )
            subprocess.check_call(["make", "-j", "install"])
        else:
            raise ValueError("Wrong compiler")
    with DirectoryExcursion(R_PACKAGE):
        subprocess.check_call(
            [
                R,
                "-q",
                "-e",
                "library(testthat); setwd('tests'); source('testthat.R')",
            ]
        )


@record_time
def main(args: argparse.Namespace) -> None:
    if args.task == "pack":
        pack_rpackage()
    elif args.task == "build":
        src_dir = pack_rpackage()
        build_rpackage(src_dir)
    elif args.task == "doc":
        check_rmarkdown()
    elif args.task == "check":
        if args.build_tool == "autotools" and system() != "Windows":
            src_dir = pack_rpackage()
            tarball = build_rpackage(src_dir)
            check_rpackage(tarball)
        elif args.build_tool == "autotools":
            test_with_autotools()
        else:
            test_with_cmake(args)
    elif args.task == "timings":
        check_example_timing(Path("xgboost.Rcheck"), 2.5)
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
        choices=["pack", "build", "check", "doc", "timings"],
        default="check",
        required=False,
    )
    parser.add_argument(
        "--compiler",
        type=str,
        choices=["mingw", "msvc", "none"],
        help="Compiler used for compiling CXX code. Only relevant for windows build",
        default="none",
        required=False,
    )
    parser.add_argument(
        "--build-tool",
        type=str,
        choices=["cmake", "autotools"],
        help="Build tool for compiling CXX code and install R package.",
        default="autotools",
        required=False,
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
