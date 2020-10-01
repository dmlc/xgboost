if (PLUGIN_LZ4)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/lz4/sparse_page_lz4_format.cc)
  target_link_libraries(objxgboost PUBLIC lz4)
endif (PLUGIN_LZ4)

if (PLUGIN_DENSE_PARSER)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc)
endif (PLUGIN_DENSE_PARSER)
