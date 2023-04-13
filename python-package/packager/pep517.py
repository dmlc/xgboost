"""
Custom build backend for XGBoost Python package.
Builds source distribution and binary wheels
Follows PEP 517
"""
import os
import pathlib
import shutil
import subprocess
import sysconfig
import tempfile
from contextlib import contextmanager

import hatchling.build

from .nativelib import locate_or_build_libxgboost
from .sdist import copy_cpp_src_tree


@contextmanager
def cd(path):
    if isinstance(path, pathlib.Path):
        path = str(path)
    path = os.path.realpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def get_tag():
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


TOPLEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute().resolve()


def get_requires_for_build_wheel(config_settings=None):
    return hatchling.build.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return hatchling.build.get_requires_for_build_sdist(config_settings)


def write_hatch_config(dest_dir):
    """Write a small custom hook for Hatch, to set a custom tag."""
    template = (
        "from hatchling.builders.hooks.plugin.interface import BuildHookInterface\n"
        "class CustomBuildHook(BuildHookInterface):\n"
        "    def initialize(self, version, build_data):\n"
        "        build_data['tag'] = '{tag}'\n"
    )
    with open(dest_dir / "hatch_build.py", "w") as f:
        f.write(template.format(tag=get_tag()))


def build_wheel(
    wheel_directory,
    config_settings=None,
    metadata_directory=None,
):
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

        whl_workspace_path = td_path / "whl_workspace"
        whl_workspace_path.mkdir()
        shutil.copy(TOPLEVEL_DIR / "pyproject.toml", whl_workspace_path)
        shutil.copy(TOPLEVEL_DIR / "README.rst", whl_workspace_path)
        write_hatch_config(whl_workspace_path)

        pkg_path = whl_workspace_path / "xgboost"
        shutil.copytree(TOPLEVEL_DIR / "xgboost", pkg_path)
        lib_path = pkg_path / "lib"
        lib_path.mkdir()
        libxgboost = locate_or_build_libxgboost(TOPLEVEL_DIR, build_dir=build_dir)
        shutil.copy(libxgboost, lib_path)

        subprocess.check_call(["find", str(whl_workspace_path)])

        with cd(whl_workspace_path):
            wheel_name = hatchling.build.build_wheel(
                wheel_directory, config_settings, metadata_directory
            )
    return wheel_name


def build_sdist(sdist_directory, config_settings=None):
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

        sdist_workspace_path = td_path / "sdist_workspace"
        sdist_workspace_path.mkdir()
        shutil.copy(TOPLEVEL_DIR / "pyproject.toml", sdist_workspace_path)
        shutil.copy(TOPLEVEL_DIR / "README.rst", sdist_workspace_path)
        write_hatch_config(sdist_workspace_path)

        shutil.copytree(TOPLEVEL_DIR / "xgboost", sdist_workspace_path / "xgboost")
        shutil.copytree(TOPLEVEL_DIR / "packager", sdist_workspace_path / "packager")

        temp_cpp_src_dir = sdist_workspace_path / "cpp_src"
        copy_cpp_src_tree(cpp_src_dir, target_dir=temp_cpp_src_dir)

        subprocess.check_call(["find", str(sdist_workspace_path)])

        with cd(sdist_workspace_path):
            sdist_name = hatchling.build.build_sdist(sdist_directory, config_settings)
    return sdist_name
