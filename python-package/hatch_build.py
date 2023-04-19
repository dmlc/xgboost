import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_tag() -> str:
    """Get appropriate wheel tag according to system"""
    tag_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{tag_platform}"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["tag"] = get_tag()
