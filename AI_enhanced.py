"""
Functions for building libxgboost with AI-driven optimizations
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
from .ai_optimizer import AIBuildOptimizer  # New AI module for optimization

# Initialize AI optimizer
ai_optimizer = AIBuildOptimizer()

def _lib_name() -> str:
    """Return platform dependent shared object name."""
    platform = system()
    if platform in ["Linux", "OS400"] or platform.upper().endswith("BSD"):
        return "libxgboost.so"
    elif platform == "Darwin":
        return "libxgboost.dylib"
    elif platform == "Windows":
        return "xgboost.dll"
    else:
        raise NotImplementedError(f"System {platform} not supported")

def build_libxgboost(
    cpp_src_dir: pathlib.Path,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Build libxgboost in a temporary directory and obtain the path to built
    libxgboost with AI-driven optimizations.
    """
    logger = logging.getLogger("xgboost.packager.build_libxgboost")

    if not cpp_src_dir.is_dir():
        raise RuntimeError(f"Expected {cpp_src_dir} to be a directory")
    logger.info(
        "Building %s from the C++ source files in %s...", _lib_name(), str(cpp_src_dir)
    )

    # Use AI to optimize build parameters
    build_tool, nproc = ai_optimizer.optimize_build_tool_and_cores(system(), build_config)

    def _build(generator: str) -> None:
        cmake_cmd = [
            "cmake",
            str(cpp_src_dir),
            generator,
            "-DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON",
        ]
        cmake_cmd.extend(build_config.get_cmake_args())

        logger.info("CMake args: %s", str(cmake_cmd))
        
        # AI-Driven Predictive Error Handling
        try:
            ai_optimizer.preemptive_error_check(cmake_cmd, build_dir)
            subprocess.check_call(cmake_cmd, cwd=build_dir)

            if system() == "Windows":
                subprocess.check_call(
                    ["cmake", "--build", ".", "--config", "Release"], cwd=build_dir
                )
            else:
                subprocess.check_call([build_tool, f"-j{nproc}"], cwd=build_dir)

        except subprocess.CalledProcessError as e:
            logger.error("Build failed: %s", str(e))
            ai_optimizer.handle_build_failure(e, cmake_cmd, build_dir)

    if system() == "Windows":
        supported_generators = ai_optimizer.get_supported_generators()
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
        generator = ai_optimizer.select_generator(build_tool)
        try:
            _build(generator=generator)
        except subprocess.CalledProcessError as e:
            logger.info("Failed to build with OpenMP. Exception: %s", str(e))
            build_config.use_openmp = False
            _build(generator=generator)

    # Generate AI-enhanced build report
    ai_optimizer.generate_build_report(build_dir)

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
    """Locate libxgboost; if not exist, build it with AI-driven optimizations"""
    logger = logging.getLogger("xgboost.packager.locate_or_build_libxgboost")

    if build_config.use_system_libxgboost:
        # Find libxgboost from system prefix
        sys_prefix = pathlib.Path(sys.base_prefix)
        sys_prefix_candidates = [
            sys_prefix / "lib",
            # Paths possibly used on Windows
            sys_prefix / "bin",
            sys_prefix / "Library",
            sys_prefix / "Library" / "bin",
            sys_prefix / "Library" / "lib",
            sys_prefix / "Library" / "mingw-w64",
            sys_prefix / "Library" / "mingw-w64" / "bin",
            sys_prefix / "Library" / "mingw-w64" / "lib",
        ]
        sys_prefix_candidates = [
            p.expanduser().resolve() for p in sys_prefix_candidates
        ]
        for candidate_dir in sys_prefix_candidates:
            libxgboost_sys = candidate_dir / _lib_name()
            if libxgboost_sys.exists():
                logger.info("Using system XGBoost: %s", str(libxgboost_sys))
                return libxgboost_sys
        raise RuntimeError(
            f"use_system_libxgboost was specified but {_lib_name()} is "
            f"not found. Paths searched (in order): \n"
            + "\n".join([f"* {str(p)}" for p in sys_prefix_candidates])
        )

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
