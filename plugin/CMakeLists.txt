if (PLUGIN_LZ4)
  set(PLUGIN_SOURCES ${PLUGIN_SOURCES}
    ${xgboost_SOURCE_DIR}/plugin/lz4/sparse_page_lz4_format.cc PARENT_SCOPE)
endif (PLUGIN_LZ4)

if (PLUGIN_DENSE_PARSER)
  set(PLUGINS_SOURCES ${PLUGINS_SOURCES}
    ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc PARENT_SCOPE)
endif (PLUGIN_DENSE_PARSER)

if (PLUGIN_UPDATER_ONEAPI)
  set(PLUGINS_SOURCES ${PLUGINS_SOURCES}
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/regression_obj_oneapi.cc
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/hist_util_oneapi.cc
    ${xgboost_SOURCE_DIR}/plugin/updater_oneapi/updater_quantile_hist_oneapi.cc PARENT_SCOPE)
endif (PLUGIN_UPDATER_ONEAPI)
