if (PLUGIN_LZ4)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/lz4/sparse_page_lz4_format.cc)
  target_link_libraries(objxgboost PUBLIC lz4)
endif (PLUGIN_LZ4)

if (PLUGIN_DENSE_PARSER)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc)
endif (PLUGIN_DENSE_PARSER)

if (PLUGIN_RMM)
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
endif (PLUGIN_RMM)
