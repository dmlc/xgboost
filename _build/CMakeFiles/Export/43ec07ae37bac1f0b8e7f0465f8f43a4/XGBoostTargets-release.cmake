#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "xgboost::xgboost" for configuration "Release"
set_property(TARGET xgboost::xgboost APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xgboost::xgboost PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxgboost.so"
  IMPORTED_SONAME_RELEASE "libxgboost.so"
  )

list(APPEND _cmake_import_check_targets xgboost::xgboost )
list(APPEND _cmake_import_check_files_for_xgboost::xgboost "${_IMPORT_PREFIX}/lib/libxgboost.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
