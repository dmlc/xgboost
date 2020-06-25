if (PLUGIN_LZ4)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/lz4/sparse_page_lz4_format.cc)
  target_link_libraries(objxgboost PUBLIC lz4)
endif (PLUGIN_LZ4)

if (PLUGIN_DENSE_PARSER)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc)
endif (PLUGIN_DENSE_PARSER)

if (PLUGIN_UPDATER_ONEAPI)
  set(PLUGINS_SOURCES_ONEAPI 
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/regression_obj_oneapi.cc 
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/predictor_oneapi.cc)

  set(PLUGINS_SOURCES_ONEAPI ${PLUGINS_SOURCES_ONEAPI} PARENT_SCOPE)
  set(PLUGINS_SOURCES ${PLUGINS_SOURCES} ${PLUGINS_SOURCES_ONEAPI} PARENT_SCOPE)
endif (PLUGIN_UPDATER_ONEAPI)
