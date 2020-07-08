# RMM
if (USE_RMM)
  # Use Conda env if available
  if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
  else()
    message(STATUS "No Conda environment detected")
  endif()

  find_path(RMM_INCLUDE "rmm"
    HINTS "$ENV{RMM_ROOT}/include")

  find_library(RMM_LIBRARY "rmm"
    HINTS "$ENV{RMM_ROOT}/lib" "$ENV{RMM_ROOT}/build")

  if ((NOT RMM_LIBRARY) OR (NOT RMM_INCLUDE))
    message(FATAL_ERROR "Could not locate RMM library")
  endif ()

  message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
  message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

  target_include_directories(objxgboost PUBLIC ${RMM_INCLUDE})
  target_link_libraries(objxgboost PUBLIC ${RMM_LIBRARY} cuda)
  target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_RMM=1)
endif ()
