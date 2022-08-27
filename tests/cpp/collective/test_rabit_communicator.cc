/*!
 * Copyright 2022 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/rabit_communicator.h"

namespace xgboost {
namespace collective {

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

}  // namespace collective
}  // namespace xgboost
