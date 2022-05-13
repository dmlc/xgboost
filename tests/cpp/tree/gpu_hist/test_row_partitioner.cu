/*!
 * Copyright 2019-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../helpers.h"
#include "xgboost/base.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {

void TestUpdatePositionBatch() {
  const int kNumRows = 10;
  RowPartitioner rp(0, kNumRows);
  auto rows = rp.GetRowsHost(0);
  EXPECT_EQ(rows.size(), kNumRows);
  for (auto i = 0ull; i < kNumRows; i++) {
    EXPECT_EQ(rows[i], i);
  }
  std::vector<int> extra_data = {0};
  // Send the first five training instances to the right node
  // and the second 5 to the left node
  rp.UpdatePositionBatch({0}, {1}, {2}, extra_data, [=] __device__(RowPartitioner::RowIndexT ridx, int) {
    if (ridx > 4) {
      return 1;
    }
    else {
      return 2;
    }
  });
  rows = rp.GetRowsHost(1);
  for (auto r : rows) {
    EXPECT_GT(r, 4);
  }
  rows = rp.GetRowsHost(2);
  for (auto r : rows) {
    EXPECT_LT(r, 5);
  }

  // Split the left node again
  rp.UpdatePositionBatch({1}, {3}, {4}, extra_data,[=] __device__(RowPartitioner::RowIndexT ridx, int) {
    if (ridx < 7) {
      return 3;
    }
    return 4;
  });
  EXPECT_EQ(rp.GetRows(3).size(), 2);
  EXPECT_EQ(rp.GetRows(4).size(), 3);
}

TEST(RowPartitioner, Batch) { TestUpdatePositionBatch(); }

void TestFinalise() {
  const int kNumRows = 10;

  ObjInfo task{ObjInfo::kRegression, false, false};
  HostDeviceVector<bst_node_t> position;
  Context ctx;
  ctx.gpu_id = 0;

  {
    RowPartitioner rp(0, kNumRows);
    rp.FinalisePosition(
        &ctx, task, &position,
        [=] __device__(RowPartitioner::RowIndexT ridx, int position) { return 7; },
        [] XGBOOST_DEVICE(size_t idx) { return false; });

    auto position = rp.GetPositionHost();
    for (auto p : position) {
      EXPECT_EQ(p, 7);
    }
  }

  /**
   * Test for sampling.
   */
  dh::device_vector<float> hess(kNumRows);
  for (size_t i = 0; i < hess.size(); ++i) {
    // removed rows, 0, 3, 6, 9
    if (i % 3 == 0) {
      hess[i] = 0;
    } else {
      hess[i] = i;
    }
  }

  auto d_hess = dh::ToSpan(hess);

  RowPartitioner rp(0, kNumRows);
  rp.FinalisePosition(
      &ctx, task, &position,
      [] __device__(RowPartitioner::RowIndexT ridx, bst_node_t position) {
        return ridx % 2 == 0 ? 1 : 2;
      },
      [d_hess] __device__(size_t ridx) { return d_hess[ridx] - 0.f == 0.f; });

  auto const& h_position = position.ConstHostVector();
  for (size_t ridx = 0; ridx < h_position.size(); ++ridx) {
    if (ridx % 3 == 0) {
      ASSERT_LT(h_position[ridx], 0);
    } else {
      ASSERT_EQ(h_position[ridx], ridx % 2 == 0 ? 1 : 2);
    }
  }
}

TEST(RowPartitioner, Finalise) { TestFinalise(); }

}  // namespace tree
}  // namespace xgboost
