/*!
 * Copyright 2022-2023 XGBoost contributors
 */
#pragma once

#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <xgboost/json.h>

#include <random>

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"

inline int GenerateRandomPort(int low, int high) {
  using namespace std::chrono_literals;
  // Ensure unique timestamp by introducing a small artificial delay
  std::this_thread::sleep_for(100ms);
  auto timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count());
  std::mt19937_64 rng(timestamp);
  std::uniform_int_distribution<int> dist(low, high);
  int port = dist(rng);
  return port;
}

inline std::string GetServerAddress() {
  int port = GenerateRandomPort(50000, 60000);
  std::string address = std::string("localhost:") + std::to_string(port);
  return address;
}

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

  static int const kWorldSize{3};
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

template <typename Function, typename... Args>
void RunWithFederatedCommunicator(int32_t world_size, std::string const& server_address,
                                  Function&& function, Args&&... args) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < world_size; rank++) {
    threads.emplace_back([&, rank]() {
      Json config{JsonObject()};
      config["xgboost_communicator"] = String("federated");
      config["federated_server_address"] = String(server_address);
      config["federated_world_size"] = world_size;
      config["federated_rank"] = rank;
      xgboost::collective::Init(config);

      std::forward<Function>(function)(std::forward<Args>(args)...);

      xgboost::collective::Finalize();
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace xgboost
