# Description:
#
#   Find / build GTest. Prefers a system install if one is found,
#   otherwise downloads sources with CPM (https://github.com/cpm-cmake/CPM.cmake)
#   and builds GTest.
#
# Targets:
#
#   GTest::gtest
#   Gtest::mock
#
# Usage:
#
#   include(CPMFindGTest.cmake)
#

function(_xgboost_find_gtest gtest_version)

  # Prefer a system-installed GTest if one is found
  find_package(GTest ${gtest_version})
  if(GTEST_FOUND)
    return()
  endif()

  # not found, use CPM to download sources and build it
  message(STATUS "GTest not found, fetching GTest via CPM.")

  # For MSVC, ensure GTest uses the same C runtime (/MD vs. /MT) as XGBoost.
  #
  # As of v1.14.0, googletest did literal string replacements on CMAKE_CXX_FLAGS,
  # and doesn't respect CMAKE_MSVC_RUNTIME_LIBRARY.
  #
  # TODO(jameslamb): remove this once XGBoost uses at least GTest v1.18.0
  # That version has this fix: https://github.com/google/googletest/pull/4877
  #
  set(gtest_force_shared_crt ${FORCE_SHARED_CRT} CACHE BOOL "" FORCE)

  # populate local CPM cache with googletest source (if not already populated), build targets
  include(${xgboost_SOURCE_DIR}/cmake/CPMSetup.cmake)
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v${gtest_version}
    VERSION ${gtest_version}
    OPTIONS
      "BUILD_GMOCK ON"
      "INSTALL_GTEST OFF"
  )

  foreach(target gtest gmock)
    # For MSVC, ensure GTest targets propagate the same MSVC C Runtime
    # to anything linking against them.
    if(MSVC)
      set_target_properties(${target} PROPERTIES
        MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif()
  endforeach()
endfunction()

_xgboost_find_gtest(1.14.0)
