file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
# Skip files with factory object
set_source_files_properties(
  predictor/predictor.cc gbm/gbm.cc tree/tree_updater.cc metric/metric.cc objective/objective.cc
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_sources(objxgboost PRIVATE ${RABIT_SOURCES})

if (USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu *.cuh)
  target_sources(objxgboost PRIVATE ${CUDA_SOURCES})
endif (USE_CUDA)

target_include_directories(objxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)

if (LOG_CAPI_INVOCATION)
  target_compile_definitions(objxgboost PRIVATE -DLOG_CAPI_INVOCATION=1)
endif (LOG_CAPI_INVOCATION)

# For MSVC: Call msvc_use_static_runtime() once again to completely
# replace /MD with /MT. See https://github.com/dmlc/xgboost/issues/4462
# for issues caused by mixing of /MD and /MT flags
msvc_use_static_runtime()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
