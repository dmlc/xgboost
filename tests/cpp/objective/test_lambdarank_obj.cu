/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>                     // for Context

#include <cstdint>                               // for uint32_t
#include <vector>                                // for vector

#include "../../../src/common/cuda_context.cuh"  // for CUDAContext
#include "../../../src/objective/lambdarank_obj.cuh"
#include "test_lambdarank_obj.h"

namespace xgboost::obj {
TEST(LambdaRank, GPUNDCGJsonIO) {
  Context ctx;
  ctx.gpu_id = 0;
  TestNDCGJsonIO(&ctx);
}

TEST(LambdaRank, GPUMAPStat) {
  Context ctx;
  ctx.gpu_id = 0;
  TestMAPStat(&ctx);
}

TEST(LambdaRank, GPUNDCGGPair) {
  Context ctx;
  ctx.gpu_id = 0;
  TestNDCGGPair(&ctx);
}

void TestGPUMakePair() {
  Context ctx;
  ctx.gpu_id = 0;

  MetaInfo info;
  HostDeviceVector<float> predt;
  InitMakePairTest(&ctx, &info, &predt);

  ltr::LambdaRankParam param;

  auto make_args = [&](std::shared_ptr<ltr::RankingCache> p_cache, auto rank_idx,
                       common::Span<std::size_t const> y_sorted_idx) {
    linalg::Vector<double> dummy;
    auto d = dummy.View(ctx.gpu_id);
    linalg::Vector<GradientPair> dgpair;
    auto dg = dgpair.View(ctx.gpu_id);
    cuda_impl::KernelInputs args{d,
                                 d,
                                 d,
                                 d,
                                 p_cache->DataGroupPtr(&ctx),
                                 p_cache->CUDAThreadsGroupPtr(),
                                 rank_idx,
                                 info.labels.View(ctx.gpu_id),
                                 predt.ConstDeviceSpan(),
                                 {},
                                 dg,
                                 nullptr,
                                 y_sorted_idx,
                                 0};
    return args;
  };

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "topk"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    auto rank_idx = p_cache->SortedIdx(&ctx, predt.ConstDeviceSpan());

    ASSERT_EQ(p_cache->CUDAThreads(), 3568);

    auto args = make_args(p_cache, rank_idx, {});
    auto n_pairs = p_cache->Param().NumPair();
    auto make_pair = cuda_impl::MakePairsOp<true>{args};

    dh::LaunchN(p_cache->CUDAThreads(), ctx.CUDACtx()->Stream(),
                [=] XGBOOST_DEVICE(std::size_t idx) {
                  auto [i, j] = make_pair(idx, 0);
                  SPAN_CHECK(j > i);
                  SPAN_CHECK(i < n_pairs);
                });
  }
  {
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "mean"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    auto rank_idx = p_cache->SortedIdx(&ctx, predt.ConstDeviceSpan());
    auto y_sorted_idx = cuda_impl::SortY(&ctx, info, rank_idx, p_cache);

    ASSERT_FALSE(param.HasTruncation());
    ASSERT_EQ(p_cache->CUDAThreads(), info.num_row_ * param.NumPair());

    auto args = make_args(p_cache, rank_idx, y_sorted_idx);
    auto make_pair = cuda_impl::MakePairsOp<false>{args};
    auto n_pairs = p_cache->Param().NumPair();
    ASSERT_EQ(n_pairs, 1);

    dh::LaunchN(
        p_cache->CUDAThreads(), ctx.CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t idx) {
          idx = 97;
          auto [i, j] = make_pair(idx, 0);
          // Not in the same bucket
          SPAN_CHECK(make_pair.args.labels(rank_idx[i]) != make_pair.args.labels(rank_idx[j]));
        });
  }
  {
    param.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "2"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    auto rank_idx = p_cache->SortedIdx(&ctx, predt.ConstDeviceSpan());
    auto y_sorted_idx = cuda_impl::SortY(&ctx, info, rank_idx, p_cache);

    auto args = make_args(p_cache, rank_idx, y_sorted_idx);
    auto make_pair = cuda_impl::MakePairsOp<false>{args};

    dh::LaunchN(
        p_cache->CUDAThreads(), ctx.CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t idx) {
          auto [i, j] = make_pair(idx, 0);
          // Not in the same bucket
          SPAN_CHECK(make_pair.args.labels(rank_idx[i]) != make_pair.args.labels(rank_idx[j]));
        });
    ASSERT_EQ(param.NumPair(), 2);
    ASSERT_EQ(p_cache->CUDAThreads(), info.num_row_ * param.NumPair());
  }
}

TEST(LambdaRank, GPUMakePair) { TestGPUMakePair(); }

TEST(LambdaRank, GPUUnbiasedNDCG) {
  Context ctx;
  ctx.gpu_id = 0;
  TestUnbiasedNDCG(&ctx);
}

template <typename CountFunctor>
void RankItemCountImpl(std::vector<std::uint32_t> const &sorted_items, CountFunctor f,
                       std::uint32_t find_val, std::uint32_t exp_val) {
  EXPECT_NE(std::find(sorted_items.begin(), sorted_items.end(), find_val), sorted_items.end());
  EXPECT_EQ(f(&sorted_items[0], sorted_items.size(), find_val), exp_val);
}

TEST(LambdaRank, RankItemCountOnLeft) {
  // Items sorted descendingly
  std::vector<std::uint32_t> sorted_items{10, 10, 6, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0};
  auto wrapper = [](auto const &...args) { return cuda_impl::CountNumItemsToTheLeftOf(args...); };
  RankItemCountImpl(sorted_items, wrapper, 10, static_cast<uint32_t>(0));
  RankItemCountImpl(sorted_items, wrapper, 6, static_cast<uint32_t>(2));
  RankItemCountImpl(sorted_items, wrapper, 4, static_cast<uint32_t>(3));
  RankItemCountImpl(sorted_items, wrapper, 1, static_cast<uint32_t>(7));
  RankItemCountImpl(sorted_items, wrapper, 0, static_cast<uint32_t>(12));
}

TEST(LambdaRank, RankItemCountOnRight) {
  // Items sorted descendingly
  std::vector<std::uint32_t> sorted_items{10, 10, 6, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0};
  auto wrapper = [](auto const &...args) { return cuda_impl::CountNumItemsToTheRightOf(args...); };
  RankItemCountImpl(sorted_items, wrapper, 10, static_cast<uint32_t>(11));
  RankItemCountImpl(sorted_items, wrapper, 6, static_cast<uint32_t>(10));
  RankItemCountImpl(sorted_items, wrapper, 4, static_cast<uint32_t>(6));
  RankItemCountImpl(sorted_items, wrapper, 1, static_cast<uint32_t>(1));
  RankItemCountImpl(sorted_items, wrapper, 0, static_cast<uint32_t>(0));
}

TEST(LambdaRank, GPUMAPGPair) {
  Context ctx;
  ctx.gpu_id = 0;
  TestMAPGPair(&ctx);
}
}  // namespace xgboost::obj
