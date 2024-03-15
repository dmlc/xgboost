/**
 * Copyright 2022-2024, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/rabit_communicator.h"
#include "../helpers.h"

namespace xgboost::collective {
TEST(RabitCommunicatorSimpleTest, ThrowOnWorldSizeTooSmall) {
  auto construct = []() { RabitCommunicator comm{0, 0}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(RabitCommunicatorSimpleTest, ThrowOnRankTooSmall) {
  auto construct = []() { RabitCommunicator comm{1, -1}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(RabitCommunicatorSimpleTest, ThrowOnRankTooBig) {
  auto construct = []() { RabitCommunicator comm{1, 1}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(RabitCommunicatorSimpleTest, GetWorldSizeAndRank) {
  RabitCommunicator comm{6, 3};
  EXPECT_EQ(comm.GetWorldSize(), 6);
  EXPECT_EQ(comm.GetRank(), 3);
}

TEST(RabitCommunicatorSimpleTest, IsNotDistributed) {
  RabitCommunicator comm{2, 1};
  // Rabit is only distributed with a tracker.
  EXPECT_FALSE(comm.IsDistributed());
}

namespace {
void VerifyVectorAllgatherV() {
  auto n_workers = collective::GetWorldSize();
  ASSERT_EQ(n_workers, 3);
  auto rank = collective::GetRank();
  // Construct input that has different length for each worker.
  std::vector<std::vector<char>> inputs;
  for (std::int32_t i = 0; i < rank + 1; ++i) {
    std::vector<char> in;
    for (std::int32_t j = 0; j < rank + 1; ++j) {
      in.push_back(static_cast<char>(j));
    }
    inputs.emplace_back(std::move(in));
  }

  auto outputs = VectorAllgatherV(inputs);

  ASSERT_EQ(outputs.size(), (1 + n_workers) * n_workers / 2);
  auto const& res = outputs;

  for (std::int32_t i = 0; i < n_workers; ++i) {
    std::int32_t k = 0;
    for (auto v : res[i]) {
      ASSERT_EQ(v, k++);
    }
  }
}
}  // namespace

TEST(VectorAllgatherV, Basic) {
  std::int32_t n_workers{3};
  RunWithInMemoryCommunicator(n_workers, VerifyVectorAllgatherV);
}
}  // namespace xgboost::collective
