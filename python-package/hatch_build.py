"""
Custom hook to customize the behavior of Hatchling.
Here, we customize the tag of the generated wheels.
"""
import sysconfig
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_tag() -> str:
    """Get appropriate wheel tag according to system"""
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


class CustomBuildHook(BuildHookInterface):
    """A custom build hook"""

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """This step ccurs immediately before each build."""
        build_data["tag"] = get_tag()
