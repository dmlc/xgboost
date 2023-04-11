"""
Custom build backend for XGBoost Python package.
Builds source distribution and binary wheels
Follows PEP 517
"""
import pathlib
import sysconfig
import tarfile
import tempfile

import packaging.version

from .distinfo import iter_files
from .nativelib import locate_libxgboost
from .wheel import create_dist_info, create_wheel


def get_tag():
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


def get_version(toplevel_dir):
    version = (
        open(toplevel_dir / "xgboost" / "VERSION", "r", encoding="utf-8").read().strip()
    )
    return str(packaging.version.Version(version))


TOPLEVEL_DIR = pathlib.Path(__file__).parent.parent.parent.absolute().resolve()
NAME = "xgboost"
VERSION = get_version(TOPLEVEL_DIR)
TAG = get_tag()
PACKAGE = pathlib.Path(TOPLEVEL_DIR / "xgboost")


def build_wheel(
    wheel_directory,
    config_settings=None,
    metadata_directory=None,
):
    print("build_wheel()")
    with tempfile.TemporaryDirectory() as td:
        if config_settings is not None:
            raise NotImplementedError(
                f"XGBoost's custom build backend doesn't support config_settings option"
            )
        if metadata_directory is None:
            td_path = pathlib.Path(td)
            dist_info = create_dist_info(
                NAME,
                VERSION,
                TAG,
                PACKAGE,
                td_path,
            )
        else:
            raise NotImplementedError(
                f"XGBoost's custom build backend doesn't support metadata_directory option"
            )

        wheel_path = create_wheel(
            NAME,
            VERSION,
            TAG,
            PACKAGE,
            dist_info=dist_info,
            libxgboost=locate_libxgboost(TOPLEVEL_DIR),
            output_dir=pathlib.Path(wheel_directory).absolute().resolve(),
        )

    return wheel_path.name


def build_sdist(sdist_directory, config_settings=None):
    if config_settings:
        raise NotImplementedError(
            f"XGBoost's custom build backend doesn't support config_settings option."
            f"{config_settings=}"
        )
    sdist_path = pathlib.Path(sdist_directory, f"{NAME}-{VERSION}.tar.gz")
    with tarfile.open(sdist_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for path, relative in iter_files((PACKAGE,)):
            tf.add(path, relative.as_posix())
        pyproject_path = TOPLEVEL_DIR / "pyproject.toml"
        tf.add(pyproject_path, pyproject_path.relative_to(TOPLEVEL_DIR))
    return sdist_path.name
