cmake_minimum_required(VERSION 3.13)
project(xgboost-c-examples)

add_subdirectory(basic)
add_subdirectory(external-memory)
add_subdirectory(inference)

enable_testing()
add_test(
  NAME test_xgboost_demo_c_basic
  COMMAND api-demo
  WORKING_DIRECTORY ${xgboost-c-examples_BINARY_DIR}
)
add_test(
  NAME test_xgboost_demo_c_external_memory
  COMMAND external-memory-demo
  WORKING_DIRECTORY ${xgboost-c-examples_BINARY_DIR}
)
add_test(
  NAME test_xgboost_demo_c_inference
  COMMAND inference-demo
  WORKING_DIRECTORY ${xgboost-c-examples_BINARY_DIR}
)
