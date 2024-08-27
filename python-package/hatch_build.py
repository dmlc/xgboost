"""
Custom hook to customize the behavior of Hatchling.
Here, we customize the tag of the generated wheels.
"""

from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import platform_tags


def get_tag() -> str:
    """Get appropriate wheel tag according to system"""
    platform_tag = next(platform_tags())
    return f"py3-none-{platform_tag}"


class CustomBuildHook(BuildHookInterface):
    """A custom build hook"""

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """This step ccurs immediately before each build."""
        build_data["tag"] = get_tag()
