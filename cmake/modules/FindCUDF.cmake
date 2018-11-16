# Tries to find cuDF headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(CUDF)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  CUDF_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the CUDF installation.
#              The environment variable CUDF_ROOT overrides this variable.
#
# This module defines
#  CUDF_FOUND, whether cuDF has been found
#  CUDF_INCLUDE_DIR, directory containing header
#  CUDF_LIBRARY, directory containing cuDF library
#  CUDF_LIB_NAME, cuDF library name
#
# This module assumes that the user has already called find_package(CUDA)


set(CUDF_LIB_NAME gdf)

find_path(CUDF_INCLUDE_DIR
  NAMES gdf/gdf.h
  PATHS $ENV{CUDF_ROOT}/include ${CUDF_ROOT}/include ${CUDA_INCLUDE_DIRS} /usr/include)

find_library(CUDF_LIBRARY
  NAMES ${CUDF_LIB_NAME}
  PATHS $ENV{CUDF_ROOT}/lib ${CUDF_ROOT}/lib ${CUDA_INCLUDE_DIRS}/../lib /usr/lib)

if (CUDF_INCLUDE_DIR AND CUDF_LIBRARY)
  get_filename_component(CUDF_LIBRARY ${CUDF_LIBRARY} PATH)
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDF DEFAULT_MSG
                                  CUDF_INCLUDE_DIR CUDF_LIBRARY)

mark_as_advanced(
  CUDF_INCLUDE_DIR
  CUDF_LIBRARY
  CUDF_LIB_NAME
)
