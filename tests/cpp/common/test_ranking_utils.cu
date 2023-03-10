/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                          // for Args
#include <xgboost/context.h>                       // for Context
#include <xgboost/linalg.h>                        // for MakeTensorView, Vector

#include <cstddef>                                 // for size_t

#include "../../../src/common/algorithm.cuh"       // for SegmentedSequence
#include "../../../src/common/cuda_context.cuh"    // for CUDAContext
#include "../../../src/common/device_helpers.cuh"  // for device_vector, LaunchN, ToSpan
#include "../../../src/common/ranking_utils.cuh"
#include "../../../src/common/ranking_utils.h"     // for LambdaRankParam
#include "test_ranking_utils.h"

namespace xgboost {
namespace ltr {
void TestCalcQueriesInvIDCG() {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  std::size_t n_groups = 5, n_samples_per_group = 32;

  dh::device_vector<float> scores(n_samples_per_group * n_groups);
  dh::device_vector<bst_group_t> group_ptr(n_groups + 1);
  auto d_group_ptr = dh::ToSpan(group_ptr);
  dh::LaunchN(d_group_ptr.size(), ctx.CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) { d_group_ptr[i] = i * n_samples_per_group; });

  auto d_scores = dh::ToSpan(scores);
  common::SegmentedSequence(&ctx, d_group_ptr, d_scores);

  linalg::Vector<double> inv_IDCG({n_groups}, ctx.gpu_id);

  ltr::LambdaRankParam p;
  p.UpdateAllowUnknown(Args{{"ndcg_exp_gain", "false"}});

  cuda_impl::CalcQueriesInvIDCG(&ctx, linalg::MakeTensorView(&ctx, d_scores, d_scores.size()),
                                dh::ToSpan(group_ptr), inv_IDCG.View(ctx.gpu_id), p);
  for (std::size_t i = 0; i < n_groups; ++i) {
    double inv_idcg = inv_IDCG(i);
    ASSERT_NEAR(inv_idcg, 0.00551782, kRtEps);
  }
}

TEST(RankingUtils, CalcQueriesInvIDCG) { TestCalcQueriesInvIDCG(); }

TEST(RankingCache, InitFromGPU) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  TestRankingCache(&ctx);
}

TEST(NDCGCache, InitFromGPU) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  TestNDCGCache(&ctx);
}
}  // namespace ltr
}  // namespace xgboost
