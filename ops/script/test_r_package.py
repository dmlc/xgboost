"""Utilities for packaging R code and running tests."""

import argparse
import os
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from platform import system

from test_utils import R_PACKAGE, ROOT, DirectoryExcursion, cd, print_time, record_time


def read_cmake_source_manifest() -> list[Path]:
    """Read the build-only source payload embedded in an R source package."""
    manifest = Path(R_PACKAGE) / "tools" / "cmake-source-files"
    entries = []
    with open(manifest, encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(Path(line))
    if not entries:
        raise ValueError(f"Empty CMake source manifest: {manifest}")
    return entries


def copy_cmake_source_tree(dest: Path) -> None:
    """Copy the minimal top-level CMake graph into an assembled R package."""
    embedded_root = dest / "src"
    for relative_path in read_cmake_source_manifest():
        source = Path(ROOT) / relative_path
        target = embedded_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"Missing CMake source manifest entry: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"{source} -> {target}")
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)


@cd(ROOT)
@record_time
def pack_rpackage() -> Path:
    """Compose the directory used for creating R package tar ball."""
    dest = Path("xgboost")

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
    os.remove(dest / "bootstrap.R")
    copy_cmake_source_tree(dest)
    shutil.copyfile("LICENSE", dest / "LICENSE")
    rwsp = Path("R-package") / "remove_warning_suppression_pragma.sh"
    if system() != "Windows":
        subprocess.check_call([rwsp, dest])
    os.remove(dest / "remove_warning_suppression_pragma.sh")
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
    env.update({"CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 1)})
    subprocess.check_call([R, "CMD", "build", path], env=env)

    tarball = find_tarball()
    return tarball


def emit_r_log(name: str, log: str) -> None:
    """Print a log produced by R CMD check."""
    print(f"\n----------------------- {name} -----------------------")
    print(log, end="" if log.endswith("\n") else "\n")


def check_example_timing(rcheck_dir: Path, threshold: float) -> None:
    try:
        import pandas as pd
    except ImportError:
        return

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
def check_rpackage(path: str, *, print_install_log: bool = False) -> None:
    env = os.environ.copy()
    print("Ncpus:", f"{os.cpu_count()}")
    threshold = 2.5
    env.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 1),
            # cran specific environment variables
            "_R_CHECK_EXAMPLE_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
            "_R_CHECK_TEST_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
            "_R_CHECK_VIGNETTE_TIMING_CPU_TO_ELAPSED_THRESHOLD_": str(threshold),
        }
    )

    status = subprocess.run(
        [R, "CMD", "check", "--as-cran", "--timings", path], env=env
    )
    rcheck_dir = Path("xgboost.Rcheck")
    with open(rcheck_dir / "00check.log", "r") as fd:
        check_log = fd.read()

    with open(rcheck_dir / "00install.out", "r") as fd:
        install_log = fd.read()

    error = None
    if status.returncode != 0:
        error = "Failed r package check."
    elif check_log.find("WARNING") != -1:
        error = "Has unresolved warnings."
    elif check_log.find("Examples with CPU time") != -1:
        error = "Suspicious NOTE."

    if print_install_log or error is not None:
        emit_r_log("Install", install_log)
    if error is not None:
        emit_r_log("Check", check_log)
        raise ValueError(error)

    check_example_timing(rcheck_dir, threshold)


@cd(R_PACKAGE)
@record_time
def check_rmarkdown() -> None:
    assert system() != "Windows", "Document test doesn't support Windows."
    env = os.environ.copy()
    env.update({"CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 1)})
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
def test_rpackage_on_windows() -> None:
    """Windows only test. No `--as-cran` check, only unittests. We don't want to manage
    the dependencies on Windows machine.

    """
    assert system() == "Windows"
    with DirectoryExcursion(ROOT):
        package_dir = Path(ROOT) / pack_rpackage()
    cmd = [R, "CMD", "INSTALL", str(package_dir)]
    env = os.environ.copy()
    env.update({"CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 1)})
    subprocess.check_call(cmd, env=env)
    subprocess.check_call(
        [R, "-q", "-e", "library(testthat); setwd('tests'); source('testthat.R')"]
    )


@record_time
def test_with_rchk() -> None:
    """Test with rchk, which is one of the additional checks in CRAN.

    See https://github.com/kalibera/rchk/blob/master/doc/DOCKER.md for reference.

    """
    results_dir = os.path.join(ROOT, "rchk_results")
    if os.path.exists(results_dir):
        raise ValueError(f"{results_dir} exists, please remove it first.")
    src_dir = pack_rpackage()
    tarball = build_rpackage(src_dir)

    os.mkdir(results_dir)
    shutil.copyfile(tarball, os.path.join(results_dir, tarball))

    tarball = os.path.basename(tarball)
    pkgpath = os.path.join("/rchk/packages/", tarball)
    image = "kalibera/rchk:latest"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--mount",
        f"type=bind,src={results_dir},dst=/rchk/packages",
        image,
        pkgpath,
    ]
    subprocess.check_call(cmd)


@record_time
def main(args: argparse.Namespace) -> None:
    match args.task:
        case "pack":
            pack_rpackage()
        case "build":
            src_dir = pack_rpackage()
            build_rpackage(src_dir)
        case "doc":
            check_rmarkdown()
        case "check":
            if system() != "Windows":
                src_dir = pack_rpackage()
                tarball = build_rpackage(src_dir)
                check_rpackage(tarball, print_install_log=args.print_install_log)
            else:
                test_rpackage_on_windows()
        case "rchk":
            test_with_rchk()
        case "timings":
            check_example_timing(Path("xgboost.Rcheck"), 2.5)
        case _:
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
        choices=["pack", "build", "check", "doc", "timings", "rchk"],
        default="check",
        required=False,
    )
    parser.add_argument(
        "--r",
        type=str,
        default="R" if system() != "Windows" else "R.exe",
        help="Path to the R executable.",
    )
    parser.add_argument(
        "--print-install-log",
        action="store_true",
        help="Print 00install.out after R CMD check on Unix-like systems.",
    )
    args = parser.parse_args()
    R = args.r

    try:
        main(args)
    finally:
        print_time()
