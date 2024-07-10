/**
 * Copyright 2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include "../../../../src/collective/aggregator.h"
#include "../../../../src/common/linalg_op.h"
#include "./test_worker.h"

namespace xgboost::collective {
TEST(Collective, BroadcastGrad) {
  std::int32_t n_workers{2};
  TestEncryptedGlobal(n_workers, [&] {
    Context ctx;
    MetaInfo info;
    bst_idx_t n_samples = 16;
    info.data_split_mode = DataSplitMode::kCol;
    info.num_row_ = n_samples;
    ASSERT_TRUE(info.IsVerticalFederated());
    auto out_gpair = linalg::Zeros<GradientPair>(&ctx, n_samples, 1);
    collective::BroadcastGradient(
        &ctx, info,
        [](linalg::Matrix<GradientPair>* out_gpair) {
          out_gpair->Data()->Fill(GradientPair{3.0f, 3.0f});
        },
        &out_gpair);

    std::stringstream ss;
    ss << collective::GetRank() << std::endl;
    auto h_gpair = out_gpair.HostView();
    if (collective::GetRank() == 0) {
      for (auto v : h_gpair) {
        ASSERT_EQ(v.GetGrad(), 3.0f);
        ASSERT_EQ(v.GetHess(), 3.0f);
      }
    } else {
      for (auto v : h_gpair) {
        ASSERT_EQ(v.GetGrad(), 0.0f);
        ASSERT_EQ(v.GetHess(), 0.0f);
      }
    }
  });
}
}  // namespace xgboost::collective
