function (write_version)
  message(STATUS "xgboost VERSION: ${xgboost_VERSION}")
  configure_file(
    ${xgboost_SOURCE_DIR}/cmake/build_config.h.in
    ${xgboost_SOURCE_DIR}/include/xgboost/build_config.h @ONLY)
  configure_file(
    ${xgboost_SOURCE_DIR}/cmake/Python_version.in
    ${xgboost_SOURCE_DIR}/python-package/xgboost/VERSION
    )
endfunction (write_version)
