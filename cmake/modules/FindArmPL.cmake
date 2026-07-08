find_path(ARMPL_INCLUDE_DIR
  NAMES armpl.h
  HINTS
    ${ArmPL_ROOT}
    $ENV{ARMPL_DIR}
  PATH_SUFFIXES include
)

find_library(ARMPL_LIBRARY
  NAMES armpl armpl_lp64
  HINTS
    ${ArmPL_ROOT}
    $ENV{ARMPL_DIR}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ArmPL
  REQUIRED_VARS ARMPL_LIBRARY ARMPL_INCLUDE_DIR
)

mark_as_advanced(ARMPL_INCLUDE_DIR ARMPL_LIBRARY)

if(ArmPL_FOUND AND NOT TARGET ArmPL::armpl)
  add_library(ArmPL::armpl UNKNOWN IMPORTED)
  set_target_properties(ArmPL::armpl PROPERTIES
    IMPORTED_LOCATION "${ARMPL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
  )
endif()
