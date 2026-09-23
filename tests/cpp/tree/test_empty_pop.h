/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../../../src/tree/driver.h"
#include "../../../src/tree/param.h"

namespace xgboost::tree::test_empty_pop {

inline TrainParam DepthwiseGammaParam() {
  TrainParam p{};
  p.grow_policy = TrainParam::kDepthWise;
  p.min_split_loss = 1.0f;
  p.max_depth = 8;
  p.max_leaves = 0;
  return p;
}

template <typename ExpandEntryT>
void ExpectPopReturnsDeeperWork(Driver<ExpandEntryT>* driver, std::size_t batch_size) {
  auto first = driver->Pop();
  ASSERT_EQ(first.size(), batch_size);
  for (auto const& e : first) {
    ASSERT_EQ(e.depth, 1);
  }

  auto second = driver->Pop();
  ASSERT_FALSE(second.empty());
  ASSERT_EQ(second.front().depth, 2);
}

template <typename ExpandEntryT, typename MakeValid>
void ExpectGrowthLoopExpandsChildren(Driver<ExpandEntryT>* driver, MakeValid make_valid,
                                     std::size_t batch_size) {
  std::size_t applied = 0;
  bst_node_t max_depth_applied = 0;
  bst_node_t next_nid = static_cast<bst_node_t>(batch_size) + 1000;

  auto expand_set = driver->Pop();
  while (!expand_set.empty()) {
    applied += expand_set.size();
    for (auto const& e : expand_set) {
      max_depth_applied = std::max(max_depth_applied, e.depth);
      if (driver->IsChildValid(e) && e.depth == 1) {
        driver->Push(make_valid(next_nid++, e.depth + 1));
        driver->Push(make_valid(next_nid++, e.depth + 1));
      }
    }
    expand_set = driver->Pop();
  }

  EXPECT_TRUE(driver->IsEmpty());
  EXPECT_EQ(applied, batch_size + 2 * batch_size);
  EXPECT_EQ(max_depth_applied, 2);
}

}  // namespace xgboost::tree::test_empty_pop
