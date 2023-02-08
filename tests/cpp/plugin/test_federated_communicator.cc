/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <iostream>
#include <thread>
#include <ctime>

#include "helpers.h"
#include "../../../plugin/federated/federated_communicator.h"
#include "../../../plugin/federated/federated_server.h"

namespace {

std::string GetServerAddress() {
  int port = GenerateRandomPort(50000, 60000);
  std::string address = std::string("localhost:") + std::to_string(port);
  return address;
}

}  // anonymous namespace

namespace xgboost {
namespace collective {

class FederatedCommunicatorTest : public ::testing::Test {
 public:
  static void VerifyAllgather(int rank, const std::string& server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckAllgather(comm, rank);
  }

  static void VerifyAllreduce(int rank, const std::string& server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckAllreduce(comm);
  }

  static void VerifyBroadcast(int rank, const std::string& server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckBroadcast(comm, rank);
  }

 protected:
  void SetUp() override {
    server_address_ = GetServerAddress();
    server_thread_.reset(new std::thread([this] {
      grpc::ServerBuilder builder;
      federated::FederatedService service{kWorldSize};
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

  static void CheckAllgather(FederatedCommunicator &comm, int rank) {
    int buffer[kWorldSize] = {0, 0, 0};
    buffer[rank] = rank;
    comm.AllGather(buffer, sizeof(buffer));
    for (auto i = 0; i < kWorldSize; i++) {
      EXPECT_EQ(buffer[i], i);
    }
  }

  static void CheckAllreduce(FederatedCommunicator &comm) {
    int buffer[] = {1, 2, 3, 4, 5};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kSum);
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void CheckBroadcast(FederatedCommunicator &comm, int rank) {
    if (rank == 0) {
      std::string buffer{"hello"};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    } else {
      std::string buffer{"     "};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    }
  }

  static int const kWorldSize{3};
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

TEST(FederatedCommunicatorSimpleTest, ThrowOnWorldSizeTooSmall) {
  std::string server_address{GetServerAddress()};
  auto construct = [server_address]() {
    FederatedCommunicator comm{0, 0, server_address, "", "", ""};
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooSmall) {
  std::string server_address{GetServerAddress()};
  auto construct = [server_address]() {
    FederatedCommunicator comm{1, -1, server_address, "", "", ""};
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooBig) {
  std::string server_address{GetServerAddress()};
  auto construct = [server_address]() {
    FederatedCommunicator comm{1, 1, server_address, "", "", ""};
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnWorldSizeNotInteger) {
  std::string server_address{GetServerAddress()};
  auto construct = [server_address]() {
    Json config{JsonObject()};
    config["federated_server_address"] = server_address;
    config["federated_world_size"] = std::string("1");
    config["federated_rank"] = Integer(0);
    auto *comm = FederatedCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankNotInteger) {
  std::string server_address{GetServerAddress()};
  auto construct = [server_address]() {
    Json config{JsonObject()};
    config["federated_server_address"] = server_address;
    config["federated_world_size"] = 1;
    config["federated_rank"] = std::string("0");
    auto *comm = FederatedCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, GetWorldSizeAndRank) {
  std::string server_address{GetServerAddress()};
  FederatedCommunicator comm{6, 3, server_address};
  EXPECT_EQ(comm.GetWorldSize(), 6);
  EXPECT_EQ(comm.GetRank(), 3);
}

TEST(FederatedCommunicatorSimpleTest, IsDistributed) {
  std::string server_address{GetServerAddress()};
  FederatedCommunicator comm{2, 1, server_address};
  EXPECT_TRUE(comm.IsDistributed());
}

TEST_F(FederatedCommunicatorTest, Allgather) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(
        std::thread(&FederatedCommunicatorTest::VerifyAllgather, rank, server_address_));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedCommunicatorTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(
        std::thread(&FederatedCommunicatorTest::VerifyAllreduce, rank, server_address_));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedCommunicatorTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(
        std::thread(&FederatedCommunicatorTest::VerifyBroadcast, rank, server_address_));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}
}  // namespace collective
}  // namespace xgboost
