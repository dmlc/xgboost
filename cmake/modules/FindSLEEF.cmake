find_path(SLEEF_INCLUDE_DIR
  NAMES sleef.h
  HINTS
    ${SLEEF_ROOT}
    $ENV{SLEEF_DIR}
  PATH_SUFFIXES include
)

find_library(SLEEF_LIBRARY
  NAMES sleef
  HINTS
    ${SLEEF_ROOT}
    $ENV{SLEEF_DIR}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEEF
  REQUIRED_VARS SLEEF_LIBRARY SLEEF_INCLUDE_DIR
)

mark_as_advanced(SLEEF_INCLUDE_DIR SLEEF_LIBRARY)

if(SLEEF_FOUND AND NOT TARGET SLEEF::sleef)
  add_library(SLEEF::sleef UNKNOWN IMPORTED)
  set_target_properties(SLEEF::sleef PROPERTIES
    IMPORTED_LOCATION "${SLEEF_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SLEEF_INCLUDE_DIR}"
  )
endif()
