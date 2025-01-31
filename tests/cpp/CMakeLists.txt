if(USE_DMLC_GTEST)
  if(NOT TARGET gtest)
    message(FATAL_ERROR "USE_DMLC_GTEST=ON but dmlc-core didn't bundle gtest")
  endif()
  set(GTEST_LIBRARIES gtest gmock)
else()
  find_package(GTest REQUIRED)
endif()
file(GLOB_RECURSE TEST_SOURCES "*.cc")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif()

file(GLOB_RECURSE SYCL_TEST_SOURCES "plugin/test_sycl_*.cc")
list(REMOVE_ITEM TEST_SOURCES ${SYCL_TEST_SOURCES})

if(PLUGIN_SYCL)
  set(CMAKE_CXX_COMPILER "icpx")
  file(GLOB_RECURSE SYCL_TEST_SOURCES "plugin/test_sycl_*.cc")
  add_library(plugin_sycl_test OBJECT ${SYCL_TEST_SOURCES})

  target_include_directories(plugin_sycl_test
    PRIVATE
    ${gtest_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)

  target_compile_definitions(plugin_sycl_test PUBLIC -DXGBOOST_USE_SYCL=1)
  target_link_libraries(plugin_sycl_test PUBLIC -fsycl)
  target_link_libraries(plugin_sycl_test PRIVATE ${GTEST_LIBRARIES})

  set_target_properties(plugin_sycl_test PROPERTIES
    COMPILE_FLAGS -fsycl
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set_target_properties(plugin_sycl_test PROPERTIES
    COMPILE_FLAGS "-fsycl -qopenmp")
  endif()
  # Get compilation and link flags of plugin_sycl and propagate to testxgboost
  target_link_libraries(testxgboost PUBLIC plugin_sycl_test)
  # Add all objects of plugin_sycl to testxgboost
  target_sources(testxgboost INTERFACE $<TARGET_OBJECTS:plugin_sycl_test>)
endif()

if(PLUGIN_FEDERATED)
  target_include_directories(testxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/federated)
  target_link_libraries(testxgboost PRIVATE federated_client)
else()
  file(GLOB_RECURSE FEDERATED_TEST_SOURCES "plugin/*_federated_*.*")
  list(REMOVE_ITEM TEST_SOURCES ${FEDERATED_TEST_SOURCES})
endif()

target_sources(testxgboost PRIVATE ${TEST_SOURCES} ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc)

if(USE_CUDA AND PLUGIN_RMM)
  target_include_directories(testxgboost PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

target_include_directories(testxgboost
  PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
target_link_libraries(testxgboost
  PRIVATE
  $<TARGET_NAME_IF_EXISTS:rmm::rmm_logger>
  $<TARGET_NAME_IF_EXISTS:rmm::rmm_logger_impl>
  GTest::gtest GTest::gmock)

set_output_directory(testxgboost ${xgboost_BINARY_DIR})

# This grouping organises source files nicely in visual studio
auto_source_group("${TEST_SOURCES}")
