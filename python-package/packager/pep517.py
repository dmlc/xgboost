"""
Custom build backend for XGBoost Python package.
Builds source distribution and binary wheels, following PEP 517 / PEP 660.
Re-uses components of Hatchling (https://github.com/pypa/hatch/tree/master/backend) for the sake
of brevity.
"""
import logging
import os
import pathlib
import sysconfig
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import hatchling.build

from .nativelib import locate_or_build_libxgboost
from .sdist import copy_cpp_src_tree
from .util import copy_with_logging, copytree_with_logging


@contextmanager
def cd(path: Union[str, pathlib.Path]) -> Iterator[str]:  # pylint: disable=C0103
    """Temporarily change working directory"""
    if isinstance(path, pathlib.Path):
        path = str(path)
    path = os.path.realpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def get_tag() -> str:
    """Get appropate wheel tag, according to system"""
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


TOPLEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute().resolve()
logging.basicConfig(level=logging.INFO)


def get_requires_for_build_wheel(
    config_settings: Optional[Dict[str, Any]] = None
) -> List[str]:
    """A PEP 517 method. Delegate to Hatchling"""
    return hatchling.build.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(
    config_settings: Optional[Dict[str, Any]] = None
) -> List[str]:
    """A PEP 517 method. Delegate to Hatchling"""
    return hatchling.build.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(
    config_settings: Optional[Dict[str, Any]] = None
) -> List[str]:
    """A PEP 517 method. Delegate to Hatchling"""
    return hatchling.build.get_requires_for_build_editable(config_settings)


def write_hatch_config(dest_dir: pathlib.Path, *, logger: logging.Logger) -> None:
    """Write a small custom hook for Hatch, to set a custom tag."""
    template = (
        "from hatchling.builders.hooks.plugin.interface import BuildHookInterface\n"
        "class CustomBuildHook(BuildHookInterface):\n"
        "    def initialize(self, version, build_data):\n"
        "        build_data['tag'] = '{tag}'\n"
    )
    hatch_build_file = dest_dir / "hatch_build.py"
    logger.info("Writing %s", str(hatch_build_file))
    with open(hatch_build_file, "w", encoding="utf-8") as f:
        f.write(template.format(tag=get_tag()))


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel"""
    logger = logging.getLogger("xgboost.packager.build_wheel")

    if config_settings:
        raise NotImplementedError(
            f"XGBoost's custom build backend doesn't support config_settings option."
            f"{config_settings=}"
        )

    # Create tempdir with Python package + libxgboost
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        build_dir = td_path / "libbuild"
        build_dir.mkdir()

        workspace = td_path / "whl_workspace"
        workspace.mkdir()
        logger.info("Copying project files to temporary directory %s", str(workspace))

        copy_with_logging(TOPLEVEL_DIR / "pyproject.toml", workspace, logger=logger)
        copy_with_logging(TOPLEVEL_DIR / "README.rst", workspace, logger=logger)
        write_hatch_config(workspace, logger=logger)

        pkg_path = workspace / "xgboost"
        copytree_with_logging(TOPLEVEL_DIR / "xgboost", pkg_path, logger=logger)
        lib_path = pkg_path / "lib"
        lib_path.mkdir()
        libxgboost = locate_or_build_libxgboost(TOPLEVEL_DIR, build_dir=build_dir)
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
            f"XGBoost's custom build backend doesn't support config_settings option."
            f"{config_settings=}"
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

    write_hatch_config(TOPLEVEL_DIR, logger=logger)
    return hatchling.build.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
