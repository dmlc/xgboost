#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dmlc::dmlc" for configuration "Release"
set_property(TARGET dmlc::dmlc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dmlc::dmlc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdmlc.a"
  )

list(APPEND _cmake_import_check_targets dmlc::dmlc )
list(APPEND _cmake_import_check_files_for_dmlc::dmlc "${_IMPORT_PREFIX}/lib/libdmlc.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
