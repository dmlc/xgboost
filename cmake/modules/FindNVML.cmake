if (NVML_LIBRARY)
  unset(NVML_LIBRARY CACHE)
endif(NVML_LIBRARY)

set(NVML_LIB_NAME nvml)

find_path(NVML_INCLUDE_DIR
  NAMES nvml.h
  PATHS ${CUDA_HOME}/include ${CUDA_INCLUDE} /usr/local/cuda/include)

find_library(NVML_LIBRARY
  NAMES nvidia-ml)

message(STATUS "Using nvml library: ${NVML_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG
                                  NVML_INCLUDE_DIR NVML_LIBRARY)

mark_as_advanced(
  NVML_INCLUDE_DIR
  NVML_LIBRARY
)
