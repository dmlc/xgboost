find_package(GTest REQUIRED)

add_executable(
  unit_tests
  test_io.cc
  allreduce_robust_test.cc
  allreduce_base_test.cc
  allreduce_mock_test.cc
  test_main.cpp)

target_link_libraries(
  unit_tests PRIVATE
  GTest::GTest GTest::Main
  rabit_base rabit_mock rabit)

target_include_directories(unit_tests PUBLIC
  "$<BUILD_INTERFACE:${rabit_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${DMLC_ROOT}/include>")

set_target_properties(unit_tests
  PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${rabit_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${rabit_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${rabit_BINARY_DIR})

add_test(
  NAME TestRabitLib
  COMMAND unit_tests
  WORKING_DIRECTORY ${rabit_BINARY_DIR})
