target_sources(
  testxgboost PRIVATE
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_coll.cc
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_comm.cc
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_comm_group.cc
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_tracker.cc
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_learner.cc
  ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_data.cc
)

if(USE_CUDA)
  target_sources(
    testxgboost PRIVATE
    ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_coll.cu
    ${xgboost_SOURCE_DIR}/tests/cpp/plugin/federated/test_federated_comm_group.cu
  )
endif()

target_include_directories(testxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/federated)
target_link_libraries(testxgboost PRIVATE federated_client)
