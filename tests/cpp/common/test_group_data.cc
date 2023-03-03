/*!
 * Copyright 2019 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/common/group_data.h"

namespace xgboost {
namespace common {

TEST(GroupData, ParallelGroupBuilder) {
  std::vector<size_t> offsets;
  std::vector<Entry> data;
  ParallelGroupBuilder<Entry, size_t> builder(&offsets, &data);
  builder.InitBudget(0, 1);
  // Add two rows with two elements each
  builder.AddBudget(0, 0, 2);
  builder.AddBudget(1, 0, 2);

  builder.InitStorage();
  builder.Push(0, Entry(0, 0), 0);
  builder.Push(0, Entry(1, 1), 0);
  builder.Push(1, Entry(0, 2), 0);
  builder.Push(1, Entry(1, 3), 0);

  std::vector<Entry> expected_data{
      Entry(0, 0),
      Entry(1, 1),
      Entry(0, 2),
      Entry(1, 3),
  };
  std::vector<size_t> expected_offsets{0, 2, 4};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(offsets, expected_offsets);

  // Create new builder, add one more row given already populated offsets/data
  ParallelGroupBuilder<Entry, size_t> builder2(&offsets, &data,
                                               offsets.size() - 1);
  builder2.InitBudget(0, 1);
  builder2.AddBudget(2, 0, 2);
  builder2.InitStorage();
  builder2.Push(2, Entry(0, 4), 0);
  builder2.Push(2, Entry(1, 5), 0);

  expected_data.emplace_back(0, 4);
  expected_data.emplace_back(1, 5);
  expected_offsets.emplace_back(6);

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(offsets, expected_offsets);
}

}  // namespace common
}  // namespace xgboost
