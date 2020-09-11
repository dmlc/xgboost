if (USE_DMLC_GTEST)
  if (NOT TARGET gtest)
    message(FATAL_ERROR "USE_DMLC_GTEST=ON but dmlc-core didn't bundle gtest")
  endif (NOT TARGET gtest)
  set(GTEST_LIBRARIES gtest)
else (USE_DMLC_GTEST)
  find_package(GTest REQUIRED)
endif (USE_DMLC_GTEST)
file(GLOB_RECURSE TEST_SOURCES "*.cc")

if (USE_CUDA)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif (USE_CUDA)

file(GLOB_RECURSE ONEAPI_TEST_SOURCES "plugin/*_oneapi.cc")
if (NOT PLUGIN_UPDATER_ONEAPI)
  list(REMOVE_ITEM TEST_SOURCES ${ONEAPI_TEST_SOURCES})
endif (NOT PLUGIN_UPDATER_ONEAPI)

add_executable(testxgboost ${TEST_SOURCES}
  ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc)
target_link_libraries(testxgboost PRIVATE objxgboost rabit)

if (USE_CUDA)
  xgboost_set_cuda_flags(testxgboost)

  find_package(CUDA)
  target_include_directories(testxgboost PRIVATE ${CUDA_INCLUDE_DIRS})
endif (USE_CUDA)

if (MSVC)
  target_compile_options(testxgboost PRIVATE
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_SECURE_NO_DEPRECATE
  )
endif (MSVC)
if (ENABLE_ALL_WARNINGS)
  target_compile_options(testxgboost PUBLIC
    $<IF:$<COMPILE_LANGUAGE:CUDA>,-Xcompiler=-Wall -Xcompiler=-Wextra,-Wall -Wextra>)
endif (ENABLE_ALL_WARNINGS)

target_include_directories(testxgboost
  PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
set_target_properties(
  testxgboost PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(testxgboost
  PRIVATE
  ${GTEST_LIBRARIES})

set_output_directory(testxgboost ${xgboost_BINARY_DIR})

# This grouping organises source files nicely in visual studio
auto_source_group("${TEST_SOURCES}")
