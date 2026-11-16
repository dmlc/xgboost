/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>

#include <cstddef>

#include "../../../src/tree/driver.h"
#include "../../../src/tree/param.h"

namespace xgboost::tree {

template <typename ExpandEntryT, typename MakeEntry>
void TestDriverEmptyPop(MakeEntry make_entry) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"grow_policy", "depthwise"}, {"min_split_loss", "1"}});

  // Exercise a small batch and the batch sizes used by hist and gpu_hist.
  for (std::size_t batch_size : {2, 256, 1024}) {
    SCOPED_TRACE(batch_size);
    Driver<ExpandEntryT> driver{param, batch_size};
    bst_node_t next_nid = 1;
    for (std::size_t i = 0; i < batch_size; ++i) {
      driver.Push(make_entry(next_nid++, 1, 2.0f));
    }
    // This entry has positive gain but cannot pass min_split_loss.
    driver.Push(make_entry(next_nid++, 1, 0.5f));

    auto batch = driver.Pop();
    ASSERT_EQ(batch.size(), batch_size);
    bst_node_t expected_nid = 1;
    for (auto const& entry : batch) {
      EXPECT_EQ(entry.GetNodeId(), expected_nid++);
      EXPECT_EQ(entry.depth, 1);
      ASSERT_TRUE(driver.IsChildValid(entry));
    }

    // Expanding the first batch queues children behind the invalid entry.
    expected_nid = next_nid;
    for (auto const& entry : batch) {
      driver.Push(make_entry(next_nid++, entry.depth + 1, 2.0f));
      driver.Push(make_entry(next_nid++, entry.depth + 1, 2.0f));
    }
    // Pop must skip the invalid entry and return all children in two full batches.
    for (int i = 0; i < 2; ++i) {
      batch = driver.Pop();
      ASSERT_EQ(batch.size(), batch_size);
      for (auto const& entry : batch) {
        EXPECT_EQ(entry.GetNodeId(), expected_nid++);
        EXPECT_EQ(entry.depth, 2);
      }
    }
    EXPECT_TRUE(driver.IsEmpty());
    EXPECT_TRUE(driver.Pop().empty());
  }
}

}  // namespace xgboost::tree
