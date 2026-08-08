set(XGBOOST_R_INCLUDE_DIR "" CACHE PATH "R include directory provided by R")
set(XGBOOST_R_LINK_FLAGS "" CACHE STRING "R package link flags provided by R")
set(XGBOOST_R_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src" CACHE PATH "R wrapper source directory")
message(STATUS "R include directory: ${XGBOOST_R_INCLUDE_DIR}")
message(STATUS "R link flags: ${XGBOOST_R_LINK_FLAGS}")
message(STATUS "R CUDA flags: ${CMAKE_CUDA_FLAGS}")

separate_arguments(r_link_libraries NATIVE_COMMAND "${XGBOOST_R_LINK_FLAGS}")
if(APPLE)
  # Re-assemble the -framework link flags broken by the `separate_arguments`.
  #
  #   R link flags:    -F/Library/Frameworks/R.framework/.. -framework R
  set(r_link_items)
  while(r_link_libraries)
    list(POP_FRONT r_link_libraries r_link_item)
    if(r_link_item STREQUAL "-framework" OR r_link_item STREQUAL "-weak_framework")
      if(NOT r_link_libraries)
        message(FATAL_ERROR "R provided ${r_link_item} without a framework name")
      endif()
      list(POP_FRONT r_link_libraries r_framework)
      list(APPEND r_link_items "${r_link_item} ${r_framework}")
    else()
      list(APPEND r_link_items "${r_link_item}")
    endif()
  endwhile()
  set(r_link_libraries ${r_link_items})
endif()

set(R_SOURCES
  ${XGBOOST_R_SOURCE_DIR}/xgboost_custom.cc
  ${XGBOOST_R_SOURCE_DIR}/xgboost_R.cc
  ${XGBOOST_R_SOURCE_DIR}/init.c
)

# Use object library to expose symbols
add_library(xgboost-r OBJECT ${R_SOURCES})

if(ENABLE_ALL_WARNINGS)
  target_compile_options(xgboost-r PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(
  xgboost-r PUBLIC
  -DXGBOOST_STRICT_R_MODE=1
  -DDMLC_LOG_BEFORE_THROW=0
  -DDMLC_DISABLE_STDIN=1
  -DDMLC_LOG_CUSTOMIZE=1
)

target_include_directories(
  xgboost-r PRIVATE
  "${XGBOOST_R_INCLUDE_DIR}"
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include
)

if(r_link_libraries)
  target_link_libraries(xgboost-r PUBLIC ${r_link_libraries})
endif()

if(USE_OPENMP)
  target_link_libraries(xgboost-r PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()

set_target_properties(
  xgboost-r PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Get compilation and link flags of xgboost-r and propagate to objxgboost
target_link_libraries(objxgboost PUBLIC xgboost-r)

# Add all objects of xgboost-r to objxgboost
target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:xgboost-r>)
