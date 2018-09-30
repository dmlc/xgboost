set(LSan_LIB_NAME lsan)

find_library(LSan_LIBRARY
  NAMES liblsan.so liblsan.so.0 liblsan.so.0.0.0
  PATHS ${SANITIZER_PATH} /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib ${CMAKE_PREFIX_PATH}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LSan DEFAULT_MSG
  LSan_LIBRARY)

mark_as_advanced(
  LSan_LIBRARY
  LSan_LIB_NAME)
