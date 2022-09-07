#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "xgboost::xgboost" for configuration "Release"
set_property(TARGET xgboost::xgboost APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xgboost::xgboost PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxgboost.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libxgboost.dylib"
  )

list(APPEND _cmake_import_check_targets xgboost::xgboost )
list(APPEND _cmake_import_check_files_for_xgboost::xgboost "${_IMPORT_PREFIX}/lib/libxgboost.dylib" )

# Import target "xgboost::runxgboost" for configuration "Release"
set_property(TARGET xgboost::runxgboost APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xgboost::runxgboost PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xgboost"
  )

list(APPEND _cmake_import_check_targets xgboost::runxgboost )
list(APPEND _cmake_import_check_files_for_xgboost::runxgboost "${_IMPORT_PREFIX}/bin/xgboost" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
