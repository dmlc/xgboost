set(UBSan_LIB_NAME UBSan)

find_library(UBSan_LIBRARY
  NAMES libubsan.so libubsan.so.5 libubsan.so.4 libubsan.so.3 libubsan.so.2 libubsan.so.1 libubsan.so.0
  PATHS ${SANITIZER_PATH} /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib ${CMAKE_PREFIX_PATH}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UBSan DEFAULT_MSG
  UBSan_LIBRARY)

mark_as_advanced(
  UBSan_LIBRARY
  UBSan_LIB_NAME)
