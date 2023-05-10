"""
Custom build backend for XGBoost Python package.
Builds source distribution and binary wheels, following PEP 517 / PEP 660.
Reuses components of Hatchling (https://github.com/pypa/hatch/tree/master/backend) for the sake
of brevity.
"""
import dataclasses
import logging
import os
import pathlib
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union

import hatchling.build

from .build_config import BuildConfiguration
from .nativelib import locate_local_libxgboost, locate_or_build_libxgboost
from .sdist import copy_cpp_src_tree
from .util import copy_with_logging, copytree_with_logging


@contextmanager
def cd(path: Union[str, pathlib.Path]) -> Iterator[str]:  # pylint: disable=C0103
    """
    Temporarily change working directory.
    TODO(hcho3): Remove this once we adopt Python 3.11, which implements contextlib.chdir.
    """
    path = str(path)
    path = os.path.realpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


TOPLEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute().resolve()
logging.basicConfig(level=logging.INFO)


# Aliases
get_requires_for_build_sdist = hatchling.build.get_requires_for_build_sdist
get_requires_for_build_wheel = hatchling.build.get_requires_for_build_wheel
get_requires_for_build_editable = hatchling.build.get_requires_for_build_editable


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel"""
    logger = logging.getLogger("xgboost.packager.build_wheel")

    build_config = BuildConfiguration()
    build_config.update(config_settings)
    logger.info("Parsed build configuration: %s", dataclasses.asdict(build_config))

    # Create tempdir with Python package + libxgboost
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        build_dir = td_path / "libbuild"
        build_dir.mkdir()

        workspace = td_path / "whl_workspace"
        workspace.mkdir()
        logger.info("Copying project files to temporary directory %s", str(workspace))

        copy_with_logging(TOPLEVEL_DIR / "pyproject.toml", workspace, logger=logger)
        copy_with_logging(TOPLEVEL_DIR / "hatch_build.py", workspace, logger=logger)
        copy_with_logging(TOPLEVEL_DIR / "README.rst", workspace, logger=logger)

        pkg_path = workspace / "xgboost"
        copytree_with_logging(TOPLEVEL_DIR / "xgboost", pkg_path, logger=logger)
        lib_path = pkg_path / "lib"
        lib_path.mkdir()
        libxgboost = locate_or_build_libxgboost(
            TOPLEVEL_DIR, build_dir=build_dir, build_config=build_config
        )
        if not build_config.use_system_libxgboost:
            copy_with_logging(libxgboost, lib_path, logger=logger)

        with cd(workspace):
            wheel_name = hatchling.build.build_wheel(
                wheel_directory, config_settings, metadata_directory
            )
    return wheel_name


def build_sdist(
    sdist_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a source distribution"""
    logger = logging.getLogger("xgboost.packager.build_sdist")

    if config_settings:
        raise NotImplementedError(
            "XGBoost's custom build backend doesn't support config_settings option "
            f"when building sdist. {config_settings=}"
        )

    cpp_src_dir = TOPLEVEL_DIR.parent
    if not cpp_src_dir.joinpath("CMakeLists.txt").exists():
        raise RuntimeError(f"Did not find CMakeLists.txt from {cpp_src_dir}")

    # Create tempdir with Python package + C++ sources
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)

        workspace = td_path / "sdist_workspace"
        workspace.mkdir()
        logger.info("Copying project files to temporary directory %s", str(workspace))

        copy_with_logging(TOPLEVEL_DIR / "pyproject.toml", workspace, logger=logger)
        copy_with_logging(TOPLEVEL_DIR / "hatch_build.py", workspace, logger=logger)
        copy_with_logging(TOPLEVEL_DIR / "README.rst", workspace, logger=logger)

        copytree_with_logging(
            TOPLEVEL_DIR / "xgboost", workspace / "xgboost", logger=logger
        )
        copytree_with_logging(
            TOPLEVEL_DIR / "packager", workspace / "packager", logger=logger
        )

        temp_cpp_src_dir = workspace / "cpp_src"
        copy_cpp_src_tree(cpp_src_dir, target_dir=temp_cpp_src_dir, logger=logger)

        with cd(workspace):
            sdist_name = hatchling.build.build_sdist(sdist_directory, config_settings)
    return sdist_name


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Build an editable installation. We mostly delegate to Hatchling."""
    logger = logging.getLogger("xgboost.packager.build_editable")

    if config_settings:
        raise NotImplementedError(
            "XGBoost's custom build backend doesn't support config_settings option "
            f"when building editable installation. {config_settings=}"
        )

    if locate_local_libxgboost(TOPLEVEL_DIR, logger=logger) is None:
        raise RuntimeError(
            "To use the editable installation, first build libxgboost with CMake. "
            "See https://xgboost.readthedocs.io/en/latest/build.html for detailed instructions."
        )

    return hatchling.build.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
