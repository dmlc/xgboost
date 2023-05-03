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
from typing import Optional

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

    def _build(*, generator: str) -> None:
        cmake_cmd = [
            "cmake",
            str(cpp_src_dir),
            generator,
            "-DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON",
        ]
        cmake_cmd.extend(build_config.get_cmake_args())

        # Flag for cross-compiling for Apple Silicon
        # We use environment variable because it's the only way to pass down custom flags
        # through the cibuildwheel package, which calls `pip wheel` command.
        if "CIBW_TARGET_OSX_ARM64" in os.environ:
            cmake_cmd.append("-DCMAKE_OSX_ARCHITECTURES=arm64")

        logger.info("CMake args: %s", str(cmake_cmd))
        subprocess.check_call(cmake_cmd, cwd=build_dir)

        if system() == "Windows":
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release"], cwd=build_dir
            )
        else:
            nproc = os.cpu_count()
            assert build_tool is not None
            subprocess.check_call([build_tool, f"-j{nproc}"], cwd=build_dir)

    if system() == "Windows":
        supported_generators = (
            "-GVisual Studio 17 2022",
            "-GVisual Studio 16 2019",
            "-GVisual Studio 15 2017",
            "-GMinGW Makefiles",
        )
        for generator in supported_generators:
            try:
                _build(generator=generator)
                logger.info(
                    "Successfully built %s using generator %s", _lib_name(), generator
                )
                break
            except subprocess.CalledProcessError as e:
                logger.info(
                    "Tried building with generator %s but failed with exception %s",
                    generator,
                    str(e),
                )
                # Empty build directory
                shutil.rmtree(build_dir)
                build_dir.mkdir()
        else:
            raise RuntimeError(
                "None of the supported generators produced a successful build!"
                f"Supported generators: {supported_generators}"
            )
    else:
        build_tool = "ninja" if shutil.which("ninja") else "make"
        generator = "-GNinja" if build_tool == "ninja" else "-GUnix Makefiles"
        try:
            _build(generator=generator)
        except subprocess.CalledProcessError as e:
            logger.info("Failed to build with OpenMP. Exception: %s", str(e))
            build_config.use_openmp = False
            _build(generator=generator)

    return build_dir / "lib" / _lib_name()


def locate_local_libxgboost(
    toplevel_dir: pathlib.Path,
    logger: logging.Logger,
) -> Optional[pathlib.Path]:
    """
    Locate libxgboost from the local project directory's lib/ subdirectory.
    """
    libxgboost = toplevel_dir.parent / "lib" / _lib_name()
    if libxgboost.exists():
        logger.info("Found %s at %s", libxgboost.name, str(libxgboost.parent))
        return libxgboost
    return None


def locate_or_build_libxgboost(
    toplevel_dir: pathlib.Path,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Locate libxgboost; if not exist, build it"""
    logger = logging.getLogger("xgboost.packager.locate_or_build_libxgboost")

    if build_config.use_system_libxgboost:
        # Find libxgboost from system prefix
        sys_prefix = pathlib.Path(sys.prefix).absolute().resolve()
        libxgboost_sys = sys_prefix / "lib" / _lib_name()
        if not libxgboost_sys.exists():
            raise RuntimeError(
                f"use_system_libxgboost was specified but {_lib_name()} is "
                f"not found in {libxgboost_sys.parent}"
            )

        logger.info("Using system XGBoost: %s", str(libxgboost_sys))
        return libxgboost_sys

    libxgboost = locate_local_libxgboost(toplevel_dir, logger=logger)
    if libxgboost is not None:
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
