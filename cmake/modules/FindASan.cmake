set(ASan_LIB_NAME ASan)

find_library(ASan_LIBRARY
  NAMES libasan.so libasan.so.4 libasan.so.3 libasan.so.2 libasan.so.1 libasan.so.0
  PATHS ${SANITIZER_PATH} /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib ${CMAKE_PREFIX_PATH}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ASan DEFAULT_MSG
  ASan_LIBRARY)

mark_as_advanced(
  ASan_LIBRARY
  ASan_LIB_NAME)
