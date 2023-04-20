"""
Functions for building sdist
"""
import logging
import pathlib

from .util import copy_with_logging, copytree_with_logging


def copy_cpp_src_tree(
    cpp_src_dir: pathlib.Path, target_dir: pathlib.Path, logger: logging.Logger
) -> None:
    """Copy C++ source tree into build directory"""

    for subdir in [
        "src",
        "include",
        "dmlc-core",
        "gputreeshap",
        "rabit",
        "cmake",
        "plugin",
    ]:
        copytree_with_logging(cpp_src_dir / subdir, target_dir / subdir, logger=logger)

    for filename in ["CMakeLists.txt", "LICENSE"]:
        copy_with_logging(cpp_src_dir.joinpath(filename), target_dir, logger=logger)
