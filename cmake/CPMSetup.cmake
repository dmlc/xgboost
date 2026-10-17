# Ensure CPM ("CMake Package Manager") is available.
#
# ref: https://github.com/cpm-cmake/cpm.cmake

function(_setup_cpm cpm_version)
  # look for already-downloaded CPM source in this order:
  #
  #   1. already-defined CMake variable 'CPM_SOURCE_CACHE'
  #   2. environment variable 'CPM_SOURCE_CACHE'
  #   3. path relative to the top-level build directory
  #
  if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${cpm_version}.cmake")
  elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${cpm_version}.cmake")
  else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${cpm_version}.cmake")
  endif()

  # download source if necessary
  if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v${cpm_version}/CPM.cmake
      ${CPM_DOWNLOAD_LOCATION}
    )
  endif()

  set(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} PARENT_SCOPE)
endfunction()

_setup_cpm(0.43.1)
include(${CPM_DOWNLOAD_LOCATION})
