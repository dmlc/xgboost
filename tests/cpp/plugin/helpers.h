/*!
 * Copyright 2022-2023 XGBoost contributors
 */
#pragma once

#include <dmlc/omp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <xgboost/json.h>

#include <random>
#include <thread>  // for thread, sleep_for

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/threading_utils.h"

namespace xgboost {

class ServerForTest {
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;

 public:
  explicit ServerForTest(std::int32_t world_size) {
    server_thread_.reset(new std::thread([this, world_size] {
      grpc::ServerBuilder builder;
      xgboost::federated::FederatedService service{world_size};
      int selected_port;
      builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(), &selected_port);
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      server_address_ = std::string("localhost:") + std::to_string(selected_port);
      server_->Wait();
    }));
  }

  ~ServerForTest() {
    server_->Shutdown();
    server_thread_->join();
  }

  auto Address() const {
    using namespace std::chrono_literals;
    while (server_address_.empty()) {
      std::this_thread::sleep_for(100ms);
    }
    return server_address_;
  }
};

class BaseFederatedTest : public ::testing::Test {
 protected:
  void SetUp() override { server_ = std::make_unique<ServerForTest>(kWorldSize); }

  void TearDown() override { server_.reset(nullptr); }

  static int constexpr kWorldSize{3};
  std::unique_ptr<ServerForTest> server_;
};

template <typename Function, typename... Args>
void RunWithFederatedCommunicator(int32_t world_size, std::string const& server_address,
                                  Function&& function, Args&&... args) {
  auto run = [&](auto rank) {
    Json config{JsonObject()};
    config["xgboost_communicator"] = String("federated");
    config["federated_server_address"] = String(server_address);
    config["federated_world_size"] = world_size;
    config["federated_rank"] = rank;
    xgboost::collective::Init(config);

    std::forward<Function>(function)(std::forward<Args>(args)...);

    xgboost::collective::Finalize();
  };
#if defined(_OPENMP)
  common::ParallelFor(world_size, world_size, run);
#else
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < world_size; rank++) {
    threads.emplace_back(run, rank);
  }
  for (auto& thread : threads) {
    thread.join();
  }
#endif
}

}  // namespace xgboost
