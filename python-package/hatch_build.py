import sysconfig
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_tag() -> str:
    """Get appropriate wheel tag according to system"""
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        build_data["tag"] = get_tag()
