if (NVTX_LIBRARY)
  unset(NVTX_LIBRARY CACHE)
endif (NVTX_LIBRARY)

set(NVTX_LIB_NAME nvToolsExt)


find_path(NVTX_INCLUDE_DIR
  NAMES nvToolsExt.h
  PATHS ${CUDA_HOME}/include ${CUDA_INCLUDE} /usr/local/cuda/include)


find_library(NVTX_LIBRARY
  NAMES nvToolsExt
  PATHS ${CUDA_HOME}/lib64 /usr/local/cuda/lib64)

message(STATUS "Using nvtx library: ${NVTX_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG
                                  NVTX_INCLUDE_DIR NVTX_LIBRARY)

mark_as_advanced(
  NVTX_INCLUDE_DIR
  NVTX_LIBRARY
)
