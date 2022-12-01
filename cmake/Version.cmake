function (write_version)
  message(STATUS "xgboost VERSION: ${xgboost_VERSION}")
  configure_file(
    ${xgboost_SOURCE_DIR}/cmake/version_config.h.in
    ${xgboost_SOURCE_DIR}/include/xgboost/version_config.h @ONLY)
endfunction (write_version)
