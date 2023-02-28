if (PLUGIN_DENSE_PARSER)
  target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/dense_parser/dense_libsvm.cc)
endif (PLUGIN_DENSE_PARSER)

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
    CXX_STANDARD 17
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

# Add the Federate Learning plugin if enabled.
if (PLUGIN_FEDERATED)
  add_subdirectory(federated)
endif (PLUGIN_FEDERATED)
