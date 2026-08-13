/**
 * Copyright 2015-2026, XGBoost contributors
 *
 * \brief CUDA implementation of lambdarank.
 */
#include <dmlc/registry.h>                      // for DMLC_REGISTRY_FILE_TAG
#include <thrust/fill.h>                        // for fill_n
#include <thrust/for_each.h>                    // for for_each_n
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/iterator/zip_iterator.h>       // for make_zip_iterator
#include <thrust/tuple.h>                       // for make_tuple (zip_iterator)

#include <algorithm>       // for min
#include <cassert>         // for assert
#include <cmath>           // for abs, log2, isinf
#include <cstddef>         // for size_t
#include <cstdint>         // for int32_t
#include <cuda/std/tuple>  // for make_tuple, tuple, get
#include <memory>          // for shared_ptr
#include <utility>

#include "../common/algorithm.cuh"       // for SegmentedArgSort
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/deterministic.cuh"   // for CreateRoundingFactor, TruncateWithRounding
#include "../common/device_helpers.cuh"  // for SegmentId, TemporaryArray, AtomicAddGpair
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "../common/ranking_utils.h"     // for NDCGCache, LambdaRankParam, rel_degree_t
#include "lambdarank_obj.cuh"
#include "lambdarank_obj.h"
#include "xgboost/base.h"                // for bst_group_t, XGBOOST_DEVICE, GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for VectorView, Range, Vector
#include "xgboost/logging.h"
#include "xgboost/span.h"  // for Span

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(lambdarank_obj_cu);

namespace cuda_impl {
namespace {
/**
 * \brief Type for gradient statistic. (Gradient, normalization factor)
 */
using GradNorm = cuda::std::tuple<GradientPair, double>;

/**
 * \brief Obtain and update the gradient for one pair.
 */
template <bool has_truncation, bool norm_by_diff, typename Delta>
struct GetGradOp {
  MakePairsOp<has_truncation> make_pair;
  Delta delta;

  bool const need_update;

  auto __device__ operator()(std::size_t idx) -> GradNorm {
    auto const& args = make_pair.args;
    auto g = dh::SegmentId(args.d_threads_group_ptr, idx);

    auto data_group_begin = static_cast<std::size_t>(args.d_group_ptr[g]);
    std::size_t n_data = args.d_group_ptr[g + 1] - data_group_begin;
    // obtain group segment data.
    auto g_label = args.labels.Slice(linalg::Range(data_group_begin, data_group_begin + n_data), 0);
    auto g_predt = args.predts.subspan(data_group_begin, n_data);
    auto g_gpair = args.gpairs.Slice(linalg::Range(data_group_begin, data_group_begin + n_data));
    auto g_rank = args.d_sorted_idx.subspan(data_group_begin, n_data);

    auto [i, j] = make_pair(idx, g);

    std::size_t rank_high = i, rank_low = j;
    if (g_label(g_rank[i]) == g_label(g_rank[j])) {
      return cuda::std::make_tuple(GradientPair{}, 0.0);
    }
    if (g_label(g_rank[i]) < g_label(g_rank[j])) {
      thrust::swap(rank_high, rank_low);
    }

    auto delta_op = [&](auto const&... args) {
      return delta(args..., g);
    };
    GradientPair pg =
        LambdaGrad<norm_by_diff>(g_label, g_predt, g_rank, rank_high, rank_low, delta_op);

    std::size_t idx_high = g_rank[rank_high];
    std::size_t idx_low = g_rank[rank_low];

    if (need_update) {
      // second run, update the gradient
      auto ng = Repulse(pg);

      auto gr = args.d_roundings(g);
      // positive gradient truncated
      auto pgt = GradientPair{common::TruncateWithRounding(gr.GetGrad(), pg.GetGrad()),
                              common::TruncateWithRounding(gr.GetHess(), pg.GetHess())};
      // negative gradient truncated
      auto ngt = GradientPair{common::TruncateWithRounding(gr.GetGrad(), ng.GetGrad()),
                              common::TruncateWithRounding(gr.GetHess(), ng.GetHess())};

      dh::AtomicAddGpair(&g_gpair(idx_high), pgt);
      dh::AtomicAddGpair(&g_gpair(idx_low), ngt);
    }

    return cuda::std::make_tuple(GradientPair{std::abs(pg.GetGrad()), std::abs(pg.GetHess())},
                                 -2.0 * static_cast<double>(pg.GetGrad()));
  }
};

template <bool has_truncation, bool norm_by_diff, typename Delta>
struct MakeGetGrad {
  MakePairsOp<has_truncation> make_pair;
  Delta delta;

  [[nodiscard]] KernelInputs const& Args() const { return make_pair.args; }

  MakeGetGrad(KernelInputs args, Delta d) : make_pair{args}, delta{std::move(d)} {}

  auto operator()(bool need_update) {
    return GetGradOp<has_truncation, norm_by_diff, Delta>{make_pair, delta, need_update};
  }
};

/**
 * \brief Calculate gradient for all pairs using update op created by make_get_grad.
 *
 * We need to run gradient calculation twice, the first time gathers information like the
 * maximum gradient and normalization term using reduction. The second time performs the
 * actual update.
 *
 * Without normalization, we only need to run it once since we can manually calculate
 * the bounds of gradient (NDCG \in [0, 1], delta_NDCG \in [0, 1]). However, if normalization
 * is used, the delta score is unbounded and we need to obtain the sum gradient. As a tradeoff,
 * we simply run the kernel twice, once as reduction, second one as for_each.
 *
 * Alternatively, we can bound the delta score by limiting the output of the model using
 * sigmoid for binary output and some normalization for multi-level. But effect to the
 * accuracy is not known yet, and it's only used by GPU.
 *
 * For performance, the segmented sort for sorted scores is the bottleneck and takes up
 * about half of the time, while the reduction and for_each takes up the second half.
 */
template <bool has_truncation, bool norm_by_diff, typename Delta>
void CalcGrad(Context const* ctx, MetaInfo const& info, std::shared_ptr<ltr::RankingCache> p_cache,
              MakeGetGrad<has_truncation, norm_by_diff, Delta> make_get_grad) {
  auto n_groups = p_cache->Groups();
  auto d_threads_group_ptr = p_cache->CUDAThreadsGroupPtr();
  auto d_gptr = p_cache->DataGroupPtr(ctx);
  auto d_gpair = make_get_grad.Args().gpairs;

  /**
   * First pass, gather info for normalization and rounding factor.
   */
  auto val_it = dh::MakeTransformIterator<GradNorm>(thrust::make_counting_iterator(0ul),
                                                    make_get_grad(false));
  auto reduction_op = [] XGBOOST_DEVICE(GradNorm const& l, GradNorm const& r) -> GradNorm {
    // Get maximum gradient for each group along with the normalization term.
    auto const& lg = cuda::std::get<0>(l);
    auto const& rg = cuda::std::get<0>(r);
    auto grad = std::max(lg.GetGrad(), rg.GetGrad());
    auto hess = std::max(lg.GetHess(), rg.GetHess());
    double sum_lambda = cuda::std::get<1>(l) + cuda::std::get<1>(r);
    return cuda::std::make_tuple(GradientPair{grad, hess}, sum_lambda);
  };
  auto init = cuda::std::make_tuple(GradientPair{0.0f, 0.0f}, 0.0);
  common::Span<GradNorm> d_max_lambdas = p_cache->MaxLambdas<GradNorm>(ctx, n_groups);
  CHECK_EQ(n_groups * sizeof(GradNorm), d_max_lambdas.size_bytes());
  // Reduce by group.
  std::size_t bytes;
  dh::safe_cuda(cub::DeviceSegmentedReduce::Reduce(
      nullptr, bytes, val_it, d_max_lambdas.data(), n_groups, d_threads_group_ptr.data(),
      d_threads_group_ptr.data() + 1, reduction_op, init, ctx->CUDACtx()->Stream()));
  dh::TemporaryArray<char> temp(bytes);
  dh::safe_cuda(cub::DeviceSegmentedReduce::Reduce(
      temp.data().get(), bytes, val_it, d_max_lambdas.data(), n_groups, d_threads_group_ptr.data(),
      d_threads_group_ptr.data() + 1, reduction_op, init, ctx->CUDACtx()->Stream()));

  auto d_rounding = p_cache->CUDARounding(ctx);
  dh::LaunchN(n_groups, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t g) mutable {
    auto group_size = d_gptr[g + 1] - d_gptr[g];
    auto const& max_grad = cuda::std::get<0>(d_max_lambdas[g]);
    // float group size
    auto fgs = static_cast<float>(group_size);
    auto grad = common::CreateRoundingFactor(fgs * max_grad.GetGrad(), group_size);
    auto hess = common::CreateRoundingFactor(fgs * max_grad.GetHess(), group_size);
    d_rounding(g) = GradientPair{grad, hess};
  });

  /**
   * Second pass, actual update to gradients.
   */
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul),
                     p_cache->CUDAThreads(), make_get_grad(true));

  /**
   * Lastly, normalization and weight.
   */
  auto d_weights = common::MakeOptionalWeights(ctx->Device(), info.weights_);
  auto w_norm = p_cache->WeightNorm();
  auto need_norm = p_cache->Param().lambdarank_normalization;
  auto n_pairs = p_cache->Param().NumPair();
  bool is_mean = p_cache->Param().IsMean();
  CHECK_EQ(is_mean, !has_truncation);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), d_gpair.Size(),
                     [=] XGBOOST_DEVICE(std::size_t i) mutable {
                       auto g = dh::SegmentId(d_gptr, i);
                       if (need_norm) {
                         double norm = 1.0;
                         if (has_truncation) {
                           // Normalize using gradient for top-k.
                           auto sum_lambda = cuda::std::get<1>(d_max_lambdas[g]);
                           if (sum_lambda > 0.0) {
                             norm = std::log2(1.0 + sum_lambda) / sum_lambda;
                           }
                         } else {
                           // Normalize using the number of pairs for mean.
                           double scale = 1.0 / static_cast<double>(n_pairs);
                           norm = scale;
                         }
                         d_gpair(i, 0) *= norm;
                       }

                       d_gpair(i, 0) *= (d_weights[g] * w_norm);
                     });
}

/**
 * @brief Handles boilerplate code like getting device spans.
 */
template <bool norm_by_diff, typename Delta>
void Launch(Context const* ctx, std::uint32_t seed, HostDeviceVector<float> const& preds,
            const MetaInfo& info, std::shared_ptr<ltr::RankingCache> p_cache, Delta delta,
            linalg::Matrix<GradientPair>* out_gpair) {
  // boilerplate
  auto device = ctx->Device();
  dh::safe_cuda(cudaSetDevice(device.ordinal));

  info.labels.SetDevice(device);
  preds.SetDevice(device);
  out_gpair->SetDevice(ctx->Device());
  out_gpair->Reshape(preds.Size(), 1);

  CHECK(p_cache);
  auto d_rounding = p_cache->CUDARounding(ctx);
  CHECK_NE(d_rounding.Size(), 0);

  auto label = info.labels.View(ctx->Device());
  auto predts = preds.ConstDeviceSpan();
  auto gpairs = out_gpair->View(ctx->Device());
  thrust::fill_n(ctx->CUDACtx()->CTP(), gpairs.Values().data(), gpairs.Size(),
                 GradientPair{0.0f, 0.0f});

  auto const d_threads_group_ptr = p_cache->CUDAThreadsGroupPtr();
  auto const d_gptr = p_cache->DataGroupPtr(ctx);
  auto const rank_idx = p_cache->SortedIdx(ctx, predts);

  common::Span<std::size_t const> d_y_sorted_idx;
  if (!p_cache->Param().HasTruncation()) {
    d_y_sorted_idx = SortY(ctx, info, rank_idx, p_cache);
  }

  KernelInputs args{d_gptr, d_threads_group_ptr, rank_idx,       label, predts,
                    gpairs, d_rounding,          d_y_sorted_idx, seed};

  // Dispatch based on truncation.
  if (p_cache->Param().HasTruncation()) {
    CalcGrad(ctx, info, p_cache, MakeGetGrad<true, norm_by_diff, Delta>{args, delta});
  } else {
    CalcGrad(ctx, info, p_cache, MakeGetGrad<false, norm_by_diff, Delta>{args, delta});
  }
}
}  // anonymous namespace

common::Span<std::size_t const> SortY(Context const* ctx, MetaInfo const& info,
                                      common::Span<std::size_t const> d_rank,
                                      std::shared_ptr<ltr::RankingCache> p_cache) {
  auto const d_group_ptr = p_cache->DataGroupPtr(ctx);
  auto label = info.labels.View(ctx->Device());
  // The buffer for ranked y is necessary as cub segmented sort accepts only pointer.
  auto d_y_ranked = p_cache->RankedY(ctx, info.num_row_);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), d_y_ranked.size(),
                     [=] XGBOOST_DEVICE(std::size_t i) {
                       auto g = dh::SegmentId(d_group_ptr, i);
                       auto g_label =
                           label.Slice(linalg::Range(d_group_ptr[g], d_group_ptr[g + 1]), 0);
                       auto g_rank_idx = d_rank.subspan(d_group_ptr[g], g_label.Size());
                       i -= d_group_ptr[g];
                       auto g_y_ranked = d_y_ranked.subspan(d_group_ptr[g], g_label.Size());
                       g_y_ranked[i] = g_label(g_rank_idx[i]);
                     });
  auto d_y_sorted_idx = p_cache->SortedIdxY(ctx, info.num_row_);
  common::SegmentedArgSort<false, true>(ctx, d_y_ranked, d_group_ptr, d_y_sorted_idx);
  return d_y_sorted_idx;
}

void LambdaRankGetGradientNDCG(Context const* ctx, std::uint32_t seed,
                               const HostDeviceVector<float>& preds, const MetaInfo& info,
                               std::shared_ptr<ltr::NDCGCache> p_cache,
                               linalg::Matrix<GradientPair>* out_gpair) {
  // boilerplate
  auto device = ctx->Device();
  dh::safe_cuda(cudaSetDevice(device.ordinal));
  auto const d_inv_IDCG = p_cache->InvIDCG(ctx);
  auto const discount = p_cache->Discount(ctx);

  info.labels.SetDevice(device);
  preds.SetDevice(device);

  auto const exp_gain = p_cache->Param().ndcg_exp_gain;
  auto delta_ndcg = [=] XGBOOST_DEVICE(float y_high, float y_low, std::size_t rank_high,
                                       std::size_t rank_low, bst_group_t g) {
    return exp_gain ? DeltaNDCG<true>(y_high, y_low, rank_high, rank_low, d_inv_IDCG(g), discount)
                    : DeltaNDCG<false>(y_high, y_low, rank_high, rank_low, d_inv_IDCG(g), discount);
  };
  if (p_cache->Param().lambdarank_score_normalization) {
    Launch<true>(ctx, seed, preds, info, p_cache, delta_ndcg, out_gpair);
  } else {
    Launch<false>(ctx, seed, preds, info, p_cache, delta_ndcg, out_gpair);
  }
}

void MAPStat(Context const* ctx, MetaInfo const& info, common::Span<std::size_t const> d_rank_idx,
             std::shared_ptr<ltr::MAPCache> p_cache) {
  common::Span<double> out_n_rel = p_cache->NumRelevant(ctx);
  common::Span<double> out_acc = p_cache->Acc(ctx);

  CHECK_EQ(out_n_rel.size(), info.num_row_);
  CHECK_EQ(out_acc.size(), info.num_row_);

  auto group_ptr = p_cache->DataGroupPtr(ctx);
  auto key_it = dh::MakeTransformIterator<std::size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> std::size_t { return dh::SegmentId(group_ptr, i); });
  auto label = info.labels.View(ctx->Device()).Slice(linalg::All(), 0);
  auto const* cuctx = ctx->CUDACtx();

  {
    // calculate number of relevant documents
    auto val_it = dh::MakeTransformIterator<double>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) -> double {
          auto g = dh::SegmentId(group_ptr, i);
          auto g_label = label.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]));
          auto idx_in_group = i - group_ptr[g];
          auto g_sorted_idx = d_rank_idx.subspan(group_ptr[g], group_ptr[g + 1] - group_ptr[g]);
          return static_cast<double>(g_label(g_sorted_idx[idx_in_group]));
        });
    thrust::inclusive_scan_by_key(cuctx->CTP(), key_it, key_it + info.num_row_, val_it,
                                  out_n_rel.data());
  }
  {
    // \sum l_k/k
    auto val_it = dh::MakeTransformIterator<double>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) -> double {
          auto g = dh::SegmentId(group_ptr, i);
          auto g_label = label.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]));
          auto g_sorted_idx = d_rank_idx.subspan(group_ptr[g], group_ptr[g + 1] - group_ptr[g]);
          auto idx_in_group = i - group_ptr[g];
          double rank_in_group = idx_in_group + 1.0;
          return static_cast<double>(g_label(g_sorted_idx[idx_in_group])) / rank_in_group;
        });
    thrust::inclusive_scan_by_key(cuctx->CTP(), key_it, key_it + info.num_row_, val_it,
                                  out_acc.data());
  }
}

void LambdaRankGetGradientMAP(Context const* ctx, std::uint32_t seed,
                              HostDeviceVector<float> const& predt, const MetaInfo& info,
                              std::shared_ptr<ltr::MAPCache> p_cache,
                              linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  dh::safe_cuda(cudaSetDevice(device.ordinal));

  info.labels.SetDevice(device);
  predt.SetDevice(device);

  CHECK(p_cache);

  auto d_predt = predt.ConstDeviceSpan();
  auto const d_sorted_idx = p_cache->SortedIdx(ctx, d_predt);

  MAPStat(ctx, info, d_sorted_idx, p_cache);
  auto d_n_rel = p_cache->NumRelevant(ctx);
  auto d_acc = p_cache->Acc(ctx);
  auto d_gptr = p_cache->DataGroupPtr(ctx).data();

  auto delta_map = [=] XGBOOST_DEVICE(float y_high, float y_low, std::size_t rank_high,
                                      std::size_t rank_low, bst_group_t g) {
    if (rank_high > rank_low) {
      thrust::swap(rank_high, rank_low);
      thrust::swap(y_high, y_low);
    }
    auto cnt = d_gptr[g + 1] - d_gptr[g];
    auto g_n_rel = d_n_rel.subspan(d_gptr[g], cnt);
    auto g_acc = d_acc.subspan(d_gptr[g], cnt);
    auto d = DeltaMAP(y_high, y_low, rank_high, rank_low, g_n_rel, g_acc);
    return d;
  };
  if (p_cache->Param().lambdarank_score_normalization) {
    Launch<true>(ctx, seed, predt, info, p_cache, delta_map, out_gpair);
  } else {
    Launch<false>(ctx, seed, predt, info, p_cache, delta_map, out_gpair);
  }
}

void LambdaRankGetGradientPairwise(Context const* ctx, std::uint32_t seed,
                                   HostDeviceVector<float> const& predt, const MetaInfo& info,
                                   std::shared_ptr<ltr::RankingCache> p_cache,
                                   linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  dh::safe_cuda(cudaSetDevice(device.ordinal));

  info.labels.SetDevice(device);
  predt.SetDevice(device);

  auto delta = [] XGBOOST_DEVICE(float, float, std::size_t, std::size_t, bst_group_t) {
    return 1.0;
  };

  if (p_cache->Param().lambdarank_score_normalization) {
    Launch<true>(ctx, seed, predt, info, p_cache, delta, out_gpair);
  } else {
    Launch<false>(ctx, seed, predt, info, p_cache, delta, out_gpair);
  }
}

}  // namespace cuda_impl
}  // namespace xgboost::obj
