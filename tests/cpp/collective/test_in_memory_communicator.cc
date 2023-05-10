/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include <bitset>
#include <thread>

#include "../../../src/collective/in_memory_communicator.h"

namespace xgboost {
namespace collective {

class InMemoryCommunicatorTest : public ::testing::Test {
 public:
  static void Verify(void (*function)(int)) {
    std::vector<std::thread> threads;
    for (auto rank = 0; rank < kWorldSize; rank++) {
      threads.emplace_back(function, rank);
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

  static void Allgather(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllgather(comm, rank);
  }

  static void AllreduceMax(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceMax(comm, rank);
  }

  static void AllreduceMin(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceMin(comm, rank);
  }

  static void AllreduceSum(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceSum(comm);
  }

  static void AllreduceBitwiseAND(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceBitwiseAND(comm, rank);
  }

  static void AllreduceBitwiseOR(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceBitwiseOR(comm, rank);
  }

  static void AllreduceBitwiseXOR(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyAllreduceBitwiseXOR(comm, rank);
  }

  static void Broadcast(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    VerifyBroadcast(comm, rank);
  }

  static void Mixture(int rank) {
    InMemoryCommunicator comm{kWorldSize, rank};
    for (auto i = 0; i < 5; i++) {
      VerifyAllgather(comm, rank);
      VerifyAllreduceMax(comm, rank);
      VerifyAllreduceMin(comm, rank);
      VerifyAllreduceSum(comm);
      VerifyAllreduceBitwiseAND(comm, rank);
      VerifyAllreduceBitwiseOR(comm, rank);
      VerifyAllreduceBitwiseXOR(comm, rank);
      VerifyBroadcast(comm, rank);
    }
  }

 protected:
  static void VerifyAllgather(InMemoryCommunicator &comm, int rank) {
    char buffer[kWorldSize] = {'a', 'b', 'c'};
    buffer[rank] = '0' + rank;
    comm.AllGather(buffer, kWorldSize);
    for (auto i = 0; i < kWorldSize; i++) {
      EXPECT_EQ(buffer[i], '0' + i);
    }
  }

  static void VerifyAllreduceMax(InMemoryCommunicator &comm, int rank) {
    int buffer[] = {1 + rank, 2 + rank, 3 + rank, 4 + rank, 5 + rank};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kMax);
    int expected[] = {3, 4, 5, 6, 7};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void VerifyAllreduceMin(InMemoryCommunicator &comm, int rank) {
    int buffer[] = {1 + rank, 2 + rank, 3 + rank, 4 + rank, 5 + rank};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kMin);
    int expected[] = {1, 2, 3, 4, 5};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void VerifyAllreduceSum(InMemoryCommunicator &comm) {
    int buffer[] = {1, 2, 3, 4, 5};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kSum);
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void VerifyAllreduceBitwiseAND(InMemoryCommunicator &comm, int rank) {
    std::bitset<2> original(rank);
    auto buffer = original.to_ulong();
    comm.AllReduce(&buffer, 1, DataType::kUInt32, Operation::kBitwiseAND);
    EXPECT_EQ(buffer, 0UL);
  }

  static void VerifyAllreduceBitwiseOR(InMemoryCommunicator &comm, int rank) {
    std::bitset<2> original(rank);
    auto buffer = original.to_ulong();
    comm.AllReduce(&buffer, 1, DataType::kUInt32, Operation::kBitwiseOR);
    std::bitset<2> actual(buffer);
    std::bitset<2> expected{0b11};
    EXPECT_EQ(actual, expected);
  }

  static void VerifyAllreduceBitwiseXOR(InMemoryCommunicator &comm, int rank) {
    std::bitset<3> original(rank * 2);
    auto buffer = original.to_ulong();
    comm.AllReduce(&buffer, 1, DataType::kUInt32, Operation::kBitwiseXOR);
    std::bitset<3> actual(buffer);
    std::bitset<3> expected{0b110};
    EXPECT_EQ(actual, expected);
  }

  static void VerifyBroadcast(InMemoryCommunicator &comm, int rank) {
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
    delete comm;
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(InMemoryCommunicatorSimpleTest, ThrowOnRankNotInteger) {
  auto construct = []() {
    Json config{JsonObject()};
    config["in_memory_world_size"] = 1;
    config["in_memory_rank"] = std::string("0");
    auto *comm = InMemoryCommunicator::Create(config);
    delete comm;
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
  EXPECT_TRUE(comm.IsDistributed());
}

TEST_F(InMemoryCommunicatorTest, Allgather) { Verify(&Allgather); }

TEST_F(InMemoryCommunicatorTest, AllreduceMax) { Verify(&AllreduceMax); }

TEST_F(InMemoryCommunicatorTest, AllreduceMin) { Verify(&AllreduceMin); }

TEST_F(InMemoryCommunicatorTest, AllreduceSum) { Verify(&AllreduceSum); }

TEST_F(InMemoryCommunicatorTest, AllreduceBitwiseAND) { Verify(&AllreduceBitwiseAND); }

TEST_F(InMemoryCommunicatorTest, AllreduceBitwiseOR) { Verify(&AllreduceBitwiseOR); }

TEST_F(InMemoryCommunicatorTest, AllreduceBitwiseXOR) { Verify(&AllreduceBitwiseXOR); }

TEST_F(InMemoryCommunicatorTest, Broadcast) { Verify(&Broadcast); }

TEST_F(InMemoryCommunicatorTest, Mixture) { Verify(&Mixture); }

}  // namespace collective
}  // namespace xgboost
