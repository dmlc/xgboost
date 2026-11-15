/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>

#include <cmath>

#include "../../../src/tree/driver.h"
#include "../../../src/tree/param.h"

namespace xgboost::tree {
template <typename ExpandEntryT, typename MakeEntry>
void TestDriverChildHessian(MakeEntry make_entry) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "1"}, {"max_depth", "3"}});
  for (auto grow_policy : {TrainParam::kDepthWise, TrainParam::kLossGuide}) {
    param.grow_policy = grow_policy;
    Driver<ExpandEntryT> driver{param};
    auto entry = make_entry(0, 0, 1.0f);
    for (double left : {1.0, 1.5, std::nextafter(2.0, 0.0), 2.0, std::nextafter(2.0, 3.0)}) {
      for (double right : {1.0, 1.5, std::nextafter(2.0, 0.0), 2.0, std::nextafter(2.0, 3.0)}) {
        SCOPED_TRACE(left);
        SCOPED_TRACE(right);
        EXPECT_EQ(driver.IsChildValid(entry, left, right), left >= 2.0 || right >= 2.0);
      }
    }
    entry.depth = 2;
    EXPECT_FALSE(driver.IsChildValid(entry, 4.0, 4.0));
  }
  param.max_leaves = 2;
  Driver<ExpandEntryT> driver{param};
  auto entry = make_entry(0, 0, 1.0f);
  driver.Push(entry);
  ASSERT_EQ(driver.Pop().size(), 1);
  EXPECT_FALSE(driver.IsChildValid(entry, 4.0, 4.0));
}
}  // namespace xgboost::tree
