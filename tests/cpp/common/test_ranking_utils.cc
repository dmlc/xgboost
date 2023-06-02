/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "test_ranking_utils.h"

#include <gtest/gtest.h>
#include <xgboost/base.h>                       // for Args, bst_group_t, kRtEps
#include <xgboost/context.h>                    // for Context
#include <xgboost/data.h>                       // for MetaInfo, DMatrix
#include <xgboost/host_device_vector.h>         // for HostDeviceVector
#include <xgboost/logging.h>                    // for Error
#include <xgboost/string_view.h>                // for StringView

#include <cstddef>                              // for size_t
#include <cstdint>                              // for uint32_t
#include <numeric>                              // for iota
#include <utility>                              // for move
#include <vector>                               // for vector

#include "../../../src/common/numeric.h"        // for Iota
#include "../../../src/common/ranking_utils.h"  // for LambdaRankParam, ParseMetricName, MakeMet...
#include "../helpers.h"                         // for EmptyDMatrix

namespace xgboost::ltr {
TEST(RankingUtils, LambdaRankParam) {
  // make sure no memory is shared in dmlc parameter.
  LambdaRankParam p0;
  p0.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "3"}});
  ASSERT_EQ(p0.NumPair(), 3);

  LambdaRankParam p1;
  p1.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "8"}});

  ASSERT_EQ(p0.NumPair(), 3);
  ASSERT_EQ(p1.NumPair(), 8);

  p0.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "17"}});
  ASSERT_EQ(p0.NumPair(), 17);
  ASSERT_EQ(p1.NumPair(), 8);
}

TEST(RankingUtils, ParseMetricName) {
  std::uint32_t topn{32};
  bool minus{false};
  auto name = ParseMetricName("ndcg", "3-", &topn, &minus);
  ASSERT_EQ(name, "ndcg@3-");
  ASSERT_EQ(topn, 3);
  ASSERT_TRUE(minus);

  name = ParseMetricName("ndcg", "6", &topn, &minus);
  ASSERT_EQ(topn, 6);
  ASSERT_TRUE(minus);  // unchanged

  minus = false;
  name = ParseMetricName("ndcg", "-", &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);

  name = ParseMetricName("ndcg", nullptr, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged

  name = ParseMetricName("ndcg", StringView{}, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged
}

TEST(RankingUtils, MakeMetricName) {
  auto name = MakeMetricName("map", LambdaRankParam::NotSet(), true);
  ASSERT_EQ(name, "map-");
  name = MakeMetricName("map", LambdaRankParam::NotSet(), false);
  ASSERT_EQ(name, "map");
  name = MakeMetricName("map", 2, true);
  ASSERT_EQ(name, "map@2-");
  name = MakeMetricName("map", 2, false);
  ASSERT_EQ(name, "map@2");
}

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

  for (std::size_t i = 0; i < rank_idx.size(); ++i) {
    ASSERT_EQ(rank_idx[i], rank_idx.size() - i - 1);
  }
}

TEST(RankingCache, InitFromCPU) {
  Context ctx;
  TestRankingCache(&ctx);
}

void TestNDCGCache(Context const* ctx) {
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  LambdaRankParam param;
  param.UpdateAllowUnknown(Args{});

  {
    // empty
    NDCGCache cache{ctx, info, param};
    ASSERT_EQ(cache.DataGroupPtr(ctx).size(), 2);
  }

  info.num_row_ = 3;
  info.group_ptr_ = {static_cast<bst_group_t>(0), static_cast<bst_group_t>(info.num_row_)};

  {
    auto fail = [&]() { NDCGCache cache{ctx, info, param}; };
    // empty label
    ASSERT_THROW(fail(), dmlc::Error);
    info.labels = linalg::Matrix<float>{{0.0f, 0.1f, 0.2f}, {3}, Context::kCpuId};
    // invalid label
    ASSERT_THROW(fail(), dmlc::Error);
    auto h_labels = info.labels.HostView();
    for (std::size_t i = 0; i < h_labels.Size(); ++i) {
      h_labels(i) *= 10;
    }
    param.UpdateAllowUnknown(Args{{"ndcg_exp_gain", "false"}});
    NDCGCache cache{ctx, info, param};
    Context cpuctx;
    auto inv_idcg = cache.InvIDCG(&cpuctx);
    ASSERT_EQ(inv_idcg.Size(), 1);
    ASSERT_NEAR(1.0 / inv_idcg(0), 2.63093, kRtEps);
  }

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_unbiased", "false"}});

    std::vector<float> h_data(32);

    common::Iota(ctx, h_data.begin(), h_data.end(), 0.0f);
    info.labels.Reshape(h_data.size());
    info.num_row_ = h_data.size();
    info.group_ptr_.back() = info.num_row_;
    info.labels.Data()->HostVector() = std::move(h_data);

    {
      NDCGCache cache{ctx, info, param};
      Context cpuctx;
      auto inv_idcg = cache.InvIDCG(&cpuctx);
      ASSERT_NEAR(inv_idcg(0), 0.00551782, kRtEps);
    }

    param.UpdateAllowUnknown(
        Args{{"lambdarank_num_pair_per_sample", "3"}, {"lambdarank_pair_method", "topk"}});
    {
      NDCGCache cache{ctx, info, param};
      Context cpuctx;
      auto inv_idcg = cache.InvIDCG(&cpuctx);
      ASSERT_NEAR(inv_idcg(0), 0.01552123, kRtEps);
    }
  }
}

TEST(NDCGCache, InitFromCPU) {
  Context ctx;
  TestNDCGCache(&ctx);
}

void TestMAPCache(Context const* ctx) {
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  LambdaRankParam param;
  param.UpdateAllowUnknown(Args{});

  std::vector<float> h_data(32);

  common::Iota(ctx, h_data.begin(), h_data.end(), 0.0f);
  info.labels.Reshape(h_data.size());
  info.num_row_ = h_data.size();
  info.labels.Data()->HostVector() = std::move(h_data);

  auto fail = [&]() { std::make_shared<MAPCache>(ctx, info, param); };
  // binary label
  ASSERT_THROW(fail(), dmlc::Error);

  h_data = std::vector<float>(32, 0.0f);
  h_data[1] = 1.0f;
  info.labels.Data()->HostVector() = h_data;
  auto p_cache = std::make_shared<MAPCache>(ctx, info, param);

  ASSERT_EQ(p_cache->Acc(ctx).size(), info.num_row_);
  ASSERT_EQ(p_cache->NumRelevant(ctx).size(), info.num_row_);
}

TEST(MAPCache, InitFromCPU) {
  Context ctx;
  ctx.Init(Args{});
  TestMAPCache(&ctx);
}
}  // namespace xgboost::ltr
