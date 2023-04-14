import logging
import os
import shutil
import subprocess
from platform import system


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


def build_libxgboost(cpp_src_dir, *, build_dir):
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
            f"Installing from sdist is not supported on Windows. You have two alternatives:\n"
            "1. Install XGBoost from the official wheel (recommended): pip install xgboost\n"
            "2. Build XGBoost from the source by running CMake at the project root folder. See "
            "documentation for details."
        )

    generator = "-GNinja" if build_tool == "ninja" else "-GUnix Makefiles"
    cmake_cmd = ["cmake", str(cpp_src_dir), generator]
    cmake_cmd.append("-DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON")

    # TODO(hcho3): handle CMake args
    logger.info("CMake args: %s", str(cmake_cmd))
    subprocess.check_call(cmake_cmd, cwd=build_dir)

    nproc = os.cpu_count()
    assert build_tool is not None
    subprocess.check_call([build_tool, f"-j{nproc}"], cwd=build_dir)

    return build_dir / "lib" / _lib_name()


def locate_or_build_libxgboost(toplevel_dir, *, build_dir):
    logger = logging.getLogger("xgboost.packager.locate_or_build_libxgboost")

    libxgboost = toplevel_dir.parent / "lib" / _lib_name()
    if libxgboost.exists():
        logger.info("Found %s at %s", libxgboost.name, str(libxgboost.parent))
        return libxgboost
    if toplevel_dir.joinpath("cpp_src").exists():
        # Source distribution; all C++ source files to be found in cpp_src/
        cpp_src_dir = toplevel_dir.joinpath("cpp_src")
    else:
        # Probably running "pip install ." from python-package/
        cpp_src_dir = toplevel_dir.parent
        if not cpp_src_dir.joinpath("CMakeLists.txt").exists():
            raise RuntimeError(f"Did not find CMakeLists.txt from {cpp_src_dir}")
    return build_libxgboost(cpp_src_dir, build_dir=build_dir)
