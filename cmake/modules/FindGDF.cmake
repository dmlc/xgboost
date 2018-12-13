# Tries to find GDF headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(GDF)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  GDF_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the GDF installation.
#              The environment variable GDF_ROOT overrides this variable.
#
# This module defines
#  GDF_FOUND, whether nccl has been found
#  GDF_INCLUDE_DIR, directory containing header
#  GDF_LIBRARY, directory containing nccl library
#  GDF_LIB_NAME, nccl library name
#
# This module assumes that the user has already called find_package(CUDA)


set(GDF_LIB_NAME cudf)

find_path(GDF_INCLUDE_DIR
  NAMES cudf.h
  PATHS $ENV{GDF_ROOT}/include ${GDF_ROOT}/include ${CUDA_INCLUDE_DIRS} /usr/include)

find_library(GDF_LIBRARY
  NAMES ${GDF_LIB_NAME}
  PATHS $ENV{GDF_ROOT}/lib ${GDF_ROOT}/lib ${CUDA_INCLUDE_DIRS}/../lib /usr/lib)

if (GDF_INCLUDE_DIR AND GDF_LIBRARY)
  get_filename_component(GDF_LIBRARY ${GDF_LIBRARY} PATH)
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDF DEFAULT_MSG
                                  GDF_INCLUDE_DIR GDF_LIBRARY)

mark_as_advanced(
  GDF_INCLUDE_DIR
  GDF_LIBRARY
  GDF_LIB_NAME
)
