set(ASan_LIB_NAME ASan)

find_library(ASan_LIBRARY
  NAMES libasan.so libasan.so.4
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ASan DEFAULT_MSG
  ASan_LIBRARY)

mark_as_advanced(
  ASan_LIBRARY
  ASan_LIB_NAME)
