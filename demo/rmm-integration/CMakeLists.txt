cmake_minimum_required (VERSION 3.13)
project(XGBoostRMMIntegrationDemo LANGUAGES CXX C CUDA)

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

find_package(CUDA REQUIRED)

add_library(rmm_bridge SHARED rmm_bridge.h rmm_bridge.cu)
target_include_directories(rmm_bridge PUBLIC ${RMM_INCLUDE} ${CUDA_INCLUDE_DIRS})
target_link_libraries(rmm_bridge PUBLIC ${RMM_LIBRARY})
set_target_properties(rmm_bridge
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)
