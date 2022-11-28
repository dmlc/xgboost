/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include <thread>

#include "../../../src/collective/in_memory_communicator.h"

namespace xgboost {
namespace collective {

class InMemoryCommunicatorTest : public ::testing::Test {
 public:
  static void VerifyAllreduce(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    CheckAllreduce(comm);
  }

  static void VerifyBroadcast(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    CheckBroadcast(comm, rank);
  }

 protected:
  static void CheckAllreduce(InMemoryCommunicator &comm) {
    int buffer[] = {1, 2, 3, 4, 5};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kSum);
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void CheckBroadcast(InMemoryCommunicator &comm, int rank) {
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
};

TEST(InMemoryCommunicatorSimpleTest, ThrowOnWorldSizeTooSmall) {
  auto construct = []() { InMemoryCommunicator comm{0, 0}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, ThrowOnRankTooSmall) {
  auto construct = []() { InMemoryCommunicator comm{1, -1}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, ThrowOnRankTooBig) {
  auto construct = []() { InMemoryCommunicator comm{1, 1}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, ThrowOnWorldSizeNotInteger) {
  auto construct = []() {
    Json config{JsonObject()};
    config["in_memory_world_size"] = std::string("1");
    config["in_memory_rank"] = Integer(0);
    auto *comm = InMemoryCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, ThrowOnRankNotInteger) {
  auto construct = []() {
    Json config{JsonObject()};
    config["in_memory_world_size"] = 1;
    config["in_memory_rank"] = std::string("0");
    auto *comm = InMemoryCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, GetWorldSizeAndRank) {
  InMemoryCommunicator comm{1, 0};
  EXPECT_EQ(comm.GetWorldSize(), 1);
  EXPECT_EQ(comm.GetRank(), 0);
}

TEST(InMemoryCommunicatorSimpleTest, IsDistributed) {
  InMemoryCommunicator comm{1, 0};
  EXPECT_FALSE(comm.IsDistributed());
}

TEST_F(InMemoryCommunicatorTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&InMemoryCommunicatorTest::VerifyAllreduce, rank));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(InMemoryCommunicatorTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&InMemoryCommunicatorTest::VerifyBroadcast, rank));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

}  // namespace collective
}  // namespace xgboost
