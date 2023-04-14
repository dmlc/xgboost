"""
Functions for building libxgboost
"""
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from platform import system

from .build_config import BuildConfiguration


def _lib_name() -> str:
    """Return platform dependent shared object name."""
    if system() in ["Linux", "OS400"] or system().upper().endswith("BSD"):
        name = "libxgboost.so"
    elif system() == "Darwin":
        name = "libxgboost.dylib"
    elif system() == "Windows":
        name = "xgboost.dll"
    else:
        raise NotImplementedError(f"System {system()} not supported")
    return name


def build_libxgboost(
    cpp_src_dir: pathlib.Path,
    *,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Build libxgboost in a temporary directory and obtain the path to built libxgboost"""
    logger = logging.getLogger("xgboost.packager.build_libxgboost")

    if not cpp_src_dir.is_dir():
        raise RuntimeError(f"Expected {cpp_src_dir} to be a directory")
    logger.info(
        "Building %s from the C++ source files in %s...", _lib_name(), str(cpp_src_dir)
    )

    if shutil.which("ninja"):
        build_tool = "ninja"
    else:
        build_tool = "make"

    if system() == "Windows":
        raise NotImplementedError(
            "Installing from sdist is not supported on Windows. You have two alternatives:\n"
            "1. Install XGBoost from the official wheel (recommended): pip install xgboost\n"
            "2. Build XGBoost from the source by running CMake at the project root folder. See "
            "https://xgboost.readthedocs.io/en/latest/build.html for details."
        )

    generator = "-GNinja" if build_tool == "ninja" else "-GUnix Makefiles"
    cmake_cmd = [
        "cmake",
        str(cpp_src_dir),
        generator,
        "-DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON",
    ]
    cmake_cmd.extend(build_config.get_cmake_args())

    logger.info("CMake args: %s", str(cmake_cmd))
    subprocess.check_call(cmake_cmd, cwd=build_dir)

    nproc = os.cpu_count()
    assert build_tool is not None
    subprocess.check_call([build_tool, f"-j{nproc}"], cwd=build_dir)

    return build_dir / "lib" / _lib_name()


def locate_or_build_libxgboost(
    toplevel_dir: pathlib.Path,
    *,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Locate libxgboost; if not exist, build it"""
    logger = logging.getLogger("xgboost.packager.locate_or_build_libxgboost")

    libxgboost = toplevel_dir.parent / "lib" / _lib_name()
    if libxgboost.exists():
        logger.info("Found %s at %s", libxgboost.name, str(libxgboost.parent))
        return libxgboost
    if build_config.use_system_libxgboost:
        # Find libxgboost from system prefix
        sys_prefix = pathlib.Path(sys.prefix).absolute().resolve()
        libxgboost = sys_prefix / "lib" / _lib_name()
        if not libxgboost.exists():
            raise AssertionError(
                f"use_system_libxgboost was specified but {_lib_name()} is "
                f"not found in {libxgboost.parent}"
            )

        logger.info("Using system XGBoost: %s", str(libxgboost))
        return libxgboost

    if toplevel_dir.joinpath("cpp_src").exists():
        # Source distribution; all C++ source files to be found in cpp_src/
        cpp_src_dir = toplevel_dir.joinpath("cpp_src")
    else:
        # Probably running "pip install ." from python-package/
        cpp_src_dir = toplevel_dir.parent
        if not cpp_src_dir.joinpath("CMakeLists.txt").exists():
            raise RuntimeError(f"Did not find CMakeLists.txt from {cpp_src_dir}")
    return build_libxgboost(cpp_src_dir, build_dir=build_dir, build_config=build_config)
