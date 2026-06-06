"""
Custom hook to customize the behavior of Hatchling.
Here, we customize the tag of the generated wheels.
"""

from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_tag() -> str:
    """Get appropriate wheel tag according to system"""
    return f"py3-none-any"


class CustomBuildHook(BuildHookInterface):
    """A custom build hook"""

    # pylint: disable=unused-argument
    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """This step ccurs immediately before each build."""
        build_data["tag"] = get_tag()
