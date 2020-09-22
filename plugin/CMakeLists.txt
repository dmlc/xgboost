if (PLUGIN_LZ4)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/lz4/sparse_page_lz4_format.cc)
  find_path(LIBLZ4_INCLUDE_DIR lz4.h)
  find_library(LIBLZ4_LIBRARY NAMES lz4)
  message(STATUS "LIBLZ4_INCLUDE_DIR = ${LIBLZ4_INCLUDE_DIR}")
  message(STATUS "LIBLZ4_LIBRARY = ${LIBLZ4_LIBRARY}")
  target_include_directories(objxgboost PUBLIC ${LIBLZ4_INCLUDE_DIR})
  target_link_libraries(objxgboost PUBLIC ${LIBLZ4_LIBRARY})
endif (PLUGIN_LZ4)

if (PLUGIN_DENSE_PARSER)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc)
endif (PLUGIN_DENSE_PARSER)

if (PLUGIN_RMM)
  find_path(RMM_INCLUDE "rmm" HINTS "$ENV{RMM_ROOT}/include")
  if (NOT RMM_INCLUDE)
    message(FATAL_ERROR "Could not locate RMM library")
  endif ()

  message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
  message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

  target_include_directories(objxgboost PUBLIC ${RMM_INCLUDE})
  target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_RMM=1)
endif (PLUGIN_RMM)

if (PLUGIN_UPDATER_ONEAPI)
  add_library(oneapi_plugin OBJECT
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/regression_obj_oneapi.cc
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/predictor_oneapi.cc)
  target_include_directories(oneapi_plugin
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)
  target_compile_definitions(oneapi_plugin PUBLIC -DXGBOOST_USE_ONEAPI=1)
  target_link_libraries(oneapi_plugin PUBLIC -fsycl)
  set_target_properties(oneapi_plugin PROPERTIES
    COMPILE_FLAGS -fsycl
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(oneapi_plugin PUBLIC OpenMP::OpenMP_CXX)
  endif (USE_OPENMP)
  # Get compilation and link flags of oneapi_plugin and propagate to objxgboost
  target_link_libraries(objxgboost PUBLIC oneapi_plugin)
  # Add all objects of oneapi_plugin to objxgboost
  target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:oneapi_plugin>)
endif (PLUGIN_UPDATER_ONEAPI)
