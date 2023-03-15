/**
 * Copyright 2020-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../src/common/partition_builder.h"
#include "../../../src/common/row_set.h"
#include "../helpers.h"

namespace xgboost::common {
TEST(PartitionBuilder, BasicTest) {
  constexpr size_t kBlockSize = 16;
  constexpr size_t kNodes = 5;
  constexpr size_t kTasks = 3 + 5 + 10 + 1 + 2;

  std::vector<size_t> tasks = { 3, 5, 10, 1, 2 };

  PartitionBuilder<kBlockSize> builder;
  builder.Init(kTasks, kNodes, [&](size_t i) {
    return tasks[i];
  });

  std::vector<size_t> rows_for_left_node = { 2, 12, 0, 16, 8 };

  for(size_t nid = 0; nid < kNodes; ++nid) {
    size_t value_left = 0;
    size_t value_right = 0;

    size_t left_total = tasks[nid] * rows_for_left_node[nid];

    for(size_t j = 0; j < tasks[nid]; ++j) {
      size_t begin = kBlockSize*j;
      size_t end = kBlockSize*(j+1);
      const size_t id = builder.GetTaskIdx(nid, begin);
      builder.AllocateForTask(id);

      auto left  = builder.GetLeftBuffer(nid, begin, end);
      auto right = builder.GetRightBuffer(nid, begin, end);

      size_t n_left   = rows_for_left_node[nid];
      size_t n_right = kBlockSize - rows_for_left_node[nid];

      for(size_t i = 0; i < n_left; i++) {
        left[i] = value_left++;
      }

      for(size_t i = 0; i < n_right; i++) {
        right[i] = left_total + value_right++;
      }

      builder.SetNLeftElems(nid, begin, n_left);
      builder.SetNRightElems(nid, begin, n_right);
    }
  }
  builder.CalculateRowOffsets();

  std::vector<size_t> v(*std::max_element(tasks.begin(), tasks.end()) * kBlockSize);

  for(size_t nid = 0; nid < kNodes; ++nid) {

    for(size_t j = 0; j < tasks[nid]; ++j) {
      builder.MergeToArray(nid, kBlockSize*j, v.data());
    }

    for(size_t j = 0; j < tasks[nid] * kBlockSize; ++j) {
      ASSERT_EQ(v[j], j);
    }
    size_t n_left  = builder.GetNLeftElems(nid);
    size_t n_right = builder.GetNRightElems(nid);

    ASSERT_EQ(n_left, rows_for_left_node[nid] * tasks[nid]);
    ASSERT_EQ(n_right, (kBlockSize - rows_for_left_node[nid]) * tasks[nid]);
  }
}
}  // namespace xgboost::common
