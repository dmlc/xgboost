/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                          // for Args, XGBOOST_DEVICE, bst_group_t, kRtEps
#include <xgboost/context.h>                       // for Context
#include <xgboost/linalg.h>                        // for MakeTensorView, Vector

#include <cstddef>                                 // for size_t
#include <memory>                                  // for shared_ptr
#include <numeric>                                 // for iota
#include <vector>                                  // for vector

#include "../../../src/common/algorithm.cuh"       // for SegmentedSequence
#include "../../../src/common/cuda_context.cuh"    // for CUDAContext
#include "../../../src/common/device_helpers.cuh"  // for device_vector, ToSpan
#include "../../../src/common/ranking_utils.cuh"   // for CalcQueriesInvIDCG
#include "../../../src/common/ranking_utils.h"     // for LambdaRankParam, RankingCache
#include "../helpers.h"                            // for EmptyDMatrix
#include "test_ranking_utils.h"                    // for TestNDCGCache
#include "xgboost/data.h"                          // for MetaInfo
#include "xgboost/host_device_vector.h"            // for HostDeviceVector

namespace xgboost::ltr {
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

namespace {
void TestRankingCache(Context const* ctx) {
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();

  info.num_row_ = 16;
  info.labels.Reshape(info.num_row_);
  auto& h_label = info.labels.Data()->HostVector();
  for (std::size_t i = 0; i < h_label.size(); ++i) {
    h_label[i] = i % 2;
  }

  LambdaRankParam param;
  param.UpdateAllowUnknown(Args{});

  RankingCache cache{ctx, info, param};

  HostDeviceVector<float> predt(info.num_row_, 0);
  auto& h_predt = predt.HostVector();
  std::iota(h_predt.begin(), h_predt.end(), 0.0f);
  predt.SetDevice(ctx->gpu_id);

  auto rank_idx =
      cache.SortedIdx(ctx, ctx->IsCPU() ? predt.ConstHostSpan() : predt.ConstDeviceSpan());

  std::vector<std::size_t> h_rank_idx(rank_idx.size());
  dh::CopyDeviceSpanToVector(&h_rank_idx, rank_idx);
  for (std::size_t i = 0; i < rank_idx.size(); ++i) {
    ASSERT_EQ(h_rank_idx[i], h_rank_idx.size() - i - 1);
  }
}
}  // namespace

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

TEST(MAPCache, InitFromGPU) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  TestMAPCache(&ctx);
}
}  // namespace xgboost::ltr
