/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // std::size_t
#include <string>   // std::to_string

#include "../../../../src/tree/hist/sampler.h"  // SampleGradient
#include "../../../../src/tree/param.h"         // TrainParam
#include "xgboost/base.h"                       // GradientPair,bst_target_t
#include "xgboost/context.h"                    // Context
#include "xgboost/data.h"                       // MetaInfo
#include "xgboost/linalg.h"                     // Matrix,Constants

namespace xgboost {
namespace tree {
TEST(Sampler, Basic) {
  std::size_t constexpr kRows = 1024;
  double constexpr kSubsample = .2;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", std::to_string(kSubsample)}});
  Context ctx;

  auto run = [&](bst_target_t n_targets) {
    auto init = GradientPair{1.0f, 1.0f};
    linalg::Matrix<GradientPair> gpair = linalg::Constant(&ctx, init, kRows, n_targets);
    auto h_gpair = gpair.HostView();
    SampleGradient(&ctx, param, h_gpair);
    std::size_t n_sampled{0};
    for (std::size_t i = 0; i < kRows; ++i) {
      bool sampled{false};
      if (h_gpair(i, 0).GetGrad() - .0f != .0f) {
        sampled = true;
        n_sampled++;
      }
      for (bst_target_t t = 1; t < n_targets; ++t) {
        if (sampled) {
          ASSERT_EQ(h_gpair(i, t).GetGrad() - init.GetGrad(), .0f);
          ASSERT_EQ(h_gpair(i, t).GetHess() - init.GetHess(), .0f);

        } else {
          ASSERT_EQ(h_gpair(i, t).GetGrad() - .0f, .0f);
          ASSERT_EQ(h_gpair(i, t).GetHess() - .0f, .0f);
        }
      }
    }
    auto ratio = static_cast<double>(n_sampled) / static_cast<double>(kRows);
    ASSERT_LT(ratio, kSubsample * 1.5);
    ASSERT_GT(ratio, kSubsample * 0.5);
  };

  run(1);
  run(3);
}
}  // namespace tree
}  // namespace xgboost
