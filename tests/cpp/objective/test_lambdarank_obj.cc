/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "test_lambdarank_obj.h"

#include <gtest/gtest.h>                        // for Test, Message, TestPartResult, CmpHel...

#include <algorithm>                            // for sort
#include <cstddef>                              // for size_t
#include <initializer_list>                     // for initializer_list
#include <map>                                  // for map
#include <memory>                               // for unique_ptr, shared_ptr, make_shared
#include <numeric>                              // for iota
#include <string>                               // for char_traits, basic_string, string
#include <vector>                               // for vector

#include "../../../src/common/ranking_utils.h"  // for NDCGCache, LambdaRankParam
#include "../helpers.h"                         // for CheckRankingObjFunction, CheckConfigReload
#include "xgboost/base.h"                       // for GradientPair, bst_group_t, Args
#include "xgboost/context.h"                    // for Context
#include "xgboost/data.h"                       // for MetaInfo, DMatrix
#include "xgboost/host_device_vector.h"         // for HostDeviceVector
#include "xgboost/linalg.h"                     // for Tensor, All, TensorView
#include "xgboost/objective.h"                  // for ObjFunction
#include "xgboost/span.h"                       // for Span

namespace xgboost::obj {
TEST(LambdaRank, NDCGJsonIO) {
  Context ctx;
  TestNDCGJsonIO(&ctx);
}

void TestNDCGGPair(Context const* ctx) {
  {
    std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:ndcg", ctx)};
    obj->Configure(Args{{"lambdarank_pair_method", "topk"}});
    CheckConfigReload(obj, "rank:ndcg");

    // No gain in swapping 2 documents.
    CheckRankingObjFunction(obj,
                            {1, 1, 1, 1},
                            {1, 1, 1, 1},
                            {1.0f, 1.0f},
                            {0, 2, 4},
                            {0.0f, -0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 0.0f, 0.0f});
  }
  {
    std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:ndcg", ctx)};
    obj->Configure(Args{{"lambdarank_pair_method", "topk"}});
    // Test with setting sample weight to second query group
    CheckRankingObjFunction(obj,
                            {0, 0.1f, 0, 0.1f},
                            {0,   1, 0, 1},
                            {2.0f, 0.0f},
                            {0, 2, 4},
                            {2.06611f, -2.06611f, 0.0f, 0.0f},
                            {2.169331f, 2.169331f, 0.0f, 0.0f});

    CheckRankingObjFunction(obj,
                            {0, 0.1f, 0, 0.1f},
                            {0,   1, 0, 1},
                            {2.0f, 2.0f},
                            {0, 2, 4},
                            {2.06611f, -2.06611f, 2.06611f, -2.06611f},
                            {2.169331f, 2.169331f, 2.169331f, 2.169331f});
  }

  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:ndcg", ctx)};
  obj->Configure(Args{{"lambdarank_pair_method", "topk"}});

  HostDeviceVector<float> predts{0, 1, 0, 1};
  MetaInfo info;
  info.labels = linalg::Tensor<float, 2>{{0, 1, 0, 1}, {4, 1}, GPUIDX};
  info.group_ptr_ = {0, 2, 4};
  info.num_row_ = 4;
  HostDeviceVector<GradientPair> gpairs;
  obj->GetGradient(predts, info, 0, &gpairs);
  ASSERT_EQ(gpairs.Size(), predts.Size());

  {
    predts = {1, 0, 1, 0};
    HostDeviceVector<GradientPair> gpairs;
    obj->GetGradient(predts, info, 0, &gpairs);
    for (size_t i = 0; i < gpairs.Size(); ++i) {
      ASSERT_GT(gpairs.HostSpan()[i].GetHess(), 0);
    }
    ASSERT_LT(gpairs.HostSpan()[1].GetGrad(), 0);
    ASSERT_LT(gpairs.HostSpan()[3].GetGrad(), 0);

    ASSERT_GT(gpairs.HostSpan()[0].GetGrad(), 0);
    ASSERT_GT(gpairs.HostSpan()[2].GetGrad(), 0);

    info.weights_ = {2, 3};
    HostDeviceVector<GradientPair> weighted_gpairs;
    obj->GetGradient(predts, info, 0, &weighted_gpairs);
    auto const& h_gpairs = gpairs.ConstHostSpan();
    auto const& h_weighted_gpairs = weighted_gpairs.ConstHostSpan();
    for (size_t i : {0ul, 1ul}) {
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetGrad(), h_gpairs[i].GetGrad() * 2.0f);
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetHess(), h_gpairs[i].GetHess() * 2.0f);
    }
    for (size_t i : {2ul, 3ul}) {
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetGrad(), h_gpairs[i].GetGrad() * 3.0f);
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetHess(), h_gpairs[i].GetHess() * 3.0f);
    }
  }

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(LambdaRank, NDCGGPair) {
  Context ctx;
  TestNDCGGPair(&ctx);
}

void TestUnbiasedNDCG(Context const* ctx) {
  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:ndcg", ctx)};
  obj->Configure(Args{{"lambdarank_pair_method", "topk"},
                      {"lambdarank_unbiased", "true"},
                      {"lambdarank_bias_norm", "0"}});
  std::shared_ptr<DMatrix> p_fmat{RandomDataGenerator{10, 1, 0.0f}.GenerateDMatrix(true, false, 2)};
  auto h_label = p_fmat->Info().labels.HostView().Values();
  // Move clicked samples to the beginning.
  std::sort(h_label.begin(), h_label.end(), std::greater<>{});
  HostDeviceVector<float> predt(p_fmat->Info().num_row_, 1.0f);

  HostDeviceVector<GradientPair> out_gpair;
  obj->GetGradient(predt, p_fmat->Info(), 0, &out_gpair);

  Json config{Object{}};
  obj->SaveConfig(&config);
  auto ti_plus = get<F32Array const>(config["ti+"]);
  ASSERT_FLOAT_EQ(ti_plus[0], 1.0);
  // bias is non-increasing when prediction is constant. (constant cost on swapping documents)
  for (std::size_t i = 1; i < ti_plus.size(); ++i) {
    ASSERT_LE(ti_plus[i], ti_plus[i - 1]);
  }
  auto tj_minus = get<F32Array const>(config["tj-"]);
  ASSERT_FLOAT_EQ(tj_minus[0], 1.0);
}

TEST(LambdaRank, UnbiasedNDCG) {
  Context ctx;
  TestUnbiasedNDCG(&ctx);
}

void InitMakePairTest(Context const* ctx, MetaInfo* out_info, HostDeviceVector<float>* out_predt) {
  out_predt->SetDevice(ctx->gpu_id);
  MetaInfo& info = *out_info;
  info.num_row_ = 128;
  info.labels.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    shape[0] = info.num_row_;
    shape[1] = 1;
    auto& h_data = data->HostVector();
    h_data.resize(shape[0]);
    for (std::size_t i = 0; i < h_data.size(); ++i) {
      h_data[i] = i % 2;
    }
  });
  std::vector<float> predt(info.num_row_);
  std::iota(predt.rbegin(), predt.rend(), 0.0f);
  out_predt->HostVector() = predt;
}

TEST(LambdaRank, MakePair) {
  Context ctx;
  MetaInfo info;
  HostDeviceVector<float> predt;

  InitMakePairTest(&ctx, &info, &predt);

  ltr::LambdaRankParam param;
  param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "topk"}});
  ASSERT_TRUE(param.HasTruncation());

  std::shared_ptr<ltr::RankingCache> p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
  auto const& h_predt = predt.ConstHostVector();
  {
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    for (std::size_t i = 0; i < h_predt.size(); ++i) {
      ASSERT_EQ(rank_idx[i], static_cast<std::size_t>(*(h_predt.crbegin() + i)));
    }
    std::int32_t n_pairs{0};
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ASSERT_GT(j, i);
                ASSERT_LT(i, p_cache->Param().NumPair());
                ++n_pairs;
              });
    ASSERT_EQ(n_pairs, 3568);
  }

  auto const h_label = info.labels.HostView();

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "mean"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    ASSERT_FALSE(param.HasTruncation());
    std::int32_t n_pairs = 0;
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ++n_pairs;
                // Not in the same bucket
                ASSERT_NE(h_label(rank_idx[i]), h_label(rank_idx[j]));
              });
    ASSERT_EQ(n_pairs, info.num_row_ * param.NumPair());
  }

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "2"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    std::int32_t n_pairs = 0;
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ++n_pairs;
                // Not in the same bucket
                ASSERT_NE(h_label(rank_idx[i]), h_label(rank_idx[j]));
              });
    ASSERT_EQ(param.NumPair(), 2);
    ASSERT_EQ(n_pairs, info.num_row_ * param.NumPair());
  }
}

void TestMAPStat(Context const* ctx) {
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  ltr::LambdaRankParam param;
  param.UpdateAllowUnknown(Args{});

  {
    std::vector<float> h_data{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    info.labels.Reshape(h_data.size(), 1);
    info.labels.Data()->HostVector() = h_data;
    info.num_row_ = h_data.size();

    HostDeviceVector<float> predt;
    auto& h_predt = predt.HostVector();
    h_predt.resize(h_data.size());
    std::iota(h_predt.rbegin(), h_predt.rend(), 0.0f);

    auto p_cache = std::make_shared<ltr::MAPCache>(ctx, info, param);

    predt.SetDevice(ctx->gpu_id);
    auto rank_idx =
        p_cache->SortedIdx(ctx, ctx->IsCPU() ? predt.ConstHostSpan() : predt.ConstDeviceSpan());

    if (ctx->IsCPU()) {
      obj::cpu_impl::MAPStat(ctx, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
                             p_cache);
    } else {
      obj::cuda_impl::MAPStat(ctx, info, rank_idx, p_cache);
    }

    Context cpu_ctx;
    auto n_rel = p_cache->NumRelevant(&cpu_ctx);
    auto acc = p_cache->Acc(&cpu_ctx);

    ASSERT_EQ(n_rel[0], 1.0);
    ASSERT_EQ(acc[0], 1.0);

    ASSERT_EQ(n_rel.back(), h_data.size() - 1.0);
    ASSERT_NEAR(acc.back(), 1.95 + (1.0 / h_data.size()), kRtEps);
  }
  {
    info.labels.Reshape(16);
    auto& h_label = info.labels.Data()->HostVector();
    info.group_ptr_ = {0, 8, 16};
    info.num_row_ = info.labels.Shape(0);

    std::fill_n(h_label.begin(), 8, 1.0f);
    std::fill_n(h_label.begin() + 8, 8, 0.0f);
    HostDeviceVector<float> predt;
    auto& h_predt = predt.HostVector();
    h_predt.resize(h_label.size());
    std::iota(h_predt.rbegin(), h_predt.rbegin() + 8, 0.0f);
    std::iota(h_predt.rbegin() + 8, h_predt.rend(), 0.0f);

    auto p_cache = std::make_shared<ltr::MAPCache>(ctx, info, param);

    predt.SetDevice(ctx->gpu_id);
    auto rank_idx =
        p_cache->SortedIdx(ctx, ctx->IsCPU() ? predt.ConstHostSpan() : predt.ConstDeviceSpan());

    if (ctx->IsCPU()) {
      obj::cpu_impl::MAPStat(ctx, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
                             p_cache);
    } else {
      obj::cuda_impl::MAPStat(ctx, info, rank_idx, p_cache);
    }

    Context cpu_ctx;
    auto n_rel = p_cache->NumRelevant(&cpu_ctx);
    ASSERT_EQ(n_rel[7], 8);      // first group
    ASSERT_EQ(n_rel.back(), 0);  // second group
  }
}

TEST(LambdaRank, MAPStat) {
  Context ctx;
  TestMAPStat(&ctx);
}

void TestMAPGPair(Context const* ctx) {
  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:map", ctx)};
  Args args;
  obj->Configure(args);

  CheckConfigReload(obj, "rank:map");

  CheckRankingObjFunction(obj,                                                 // obj
                          {0, 0.1f, 0, 0.1f},                                  // score
                          {0, 1, 0, 1},                                        // label
                          {2.0f, 2.0f},                                        // weight
                          {0, 2, 4},                                           // group
                          {1.2054923f, -1.2054923f, 1.2054923f, -1.2054923f},  // out grad
                          {1.2657166f, 1.2657166f, 1.2657166f, 1.2657166f});
  // disable the second query group with 0 weight
  CheckRankingObjFunction(obj,                                  // obj
                          {0, 0.1f, 0, 0.1f},                   // score
                          {0, 1, 0, 1},                         // label
                          {2.0f, 0.0f},                         // weight
                          {0, 2, 4},                            // group
                          {1.2054923f, -1.2054923f, .0f, .0f},  // out grad
                          {1.2657166f, 1.2657166f, .0f, .0f});
}

TEST(LambdaRank, MAPGPair) {
  Context ctx;
  TestMAPGPair(&ctx);
}

void TestPairWiseGPair(Context const* ctx) {
  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:pairwise", ctx)};
  Args args;
  obj->Configure(args);

  args.emplace_back("lambdarank_unbiased", "true");
}

TEST(LambdaRank, Pairwise) {
  Context ctx;
  TestPairWiseGPair(&ctx);
}
}  // namespace xgboost::obj
