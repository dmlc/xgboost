/*!
 * Copyright 2022 XGBoost contributors
 */
#ifndef XGBOOST_TESTS_CPP_PLUGIN_HELPERS_H_
#define XGBOOST_TESTS_CPP_PLUGIN_HELPERS_H_

#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <xgboost/json.h>

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"

std::string GetServerAddress();

namespace xgboost {

class BaseFederatedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    server_address_ = GetServerAddress();
    server_thread_.reset(new std::thread([this] {
      grpc::ServerBuilder builder;
      xgboost::federated::FederatedService service{kWorldSize};
      builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      server_->Wait();
    }));
  }

  void TearDown() override {
    server_->Shutdown();
    server_thread_->join();
  }

  void InitCommunicator(int rank) {
    Json config{JsonObject()};
    config["xgboost_communicator"] = String("federated");
    config["federated_server_address"] = String(server_address_);
    config["federated_world_size"] = kWorldSize;
    config["federated_rank"] = rank;
    xgboost::collective::Init(config);
  }

  static int const kWorldSize{3};
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};
}  // namespace xgboost

#endif  // XGBOOST_TESTS_CPP_PLUGIN_HELPERS_H_
