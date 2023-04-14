"""Build configuration"""
import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class BuildConfiguration:  # pylint: disable=R0902
    """Configurations use when building libxgboost"""

    # Whether to hide C++ symbols in libxgboost.so
    hide_cxx_symbols: bool = True
    # Whether to enable OpenMP
    use_openmp: bool = True
    # Whether to enable CUDA
    use_cuda: bool = False
    # Whether to enable NCCL
    use_nccl: bool = False
    # Whether to enable HDFS
    use_hdfs: bool = False
    # Whether to enable Azure Storage
    use_azure: bool = False
    # Whether to enable AWS S3
    use_s3: bool = False
    # Whether to enable the dense parser plugin
    plugin_dense_parser: bool = False
    # Whether to bundle vcomp140.dll, OpenMP library from Microsoft
    bundle_vcomp140_dll: bool = False
    # Special option: See explanation below
    use_system_libxgboost: bool = False

    def _set_config_setting(
        self, config_settings: Dict[str, Any], field_name: str
    ) -> None:
        if field_name in config_settings:
            setattr(
                self,
                field_name,
                (
                    config_settings[field_name]
                    in ["TRUE", "True", "true", "1", "On", "ON", "on"]
                ),
            )

    def update(self, config_settings: Optional[Dict[str, Any]]) -> None:
        """Parse config_settings from Pip (or other PEP 517 frontend)"""
        if config_settings is not None:
            for field_name in [x.name for x in dataclasses.fields(self)]:
                self._set_config_setting(config_settings, field_name)

    def get_cmake_args(self) -> List[str]:
        """Convert build configuration to CMake args"""
        cmake_args = []
        for field_name in [x.name for x in dataclasses.fields(self)]:
            if field_name == "use_system_libxgboost":
                continue
            cmake_option = field_name.upper()
            cmake_value = "ON" if getattr(self, field_name) else "OFF"
            cmake_args.append(f"-D{cmake_option}={cmake_value}")
        return cmake_args
