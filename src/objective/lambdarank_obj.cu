/**
 * Copyright 2015-2023 by XGBoost contributors
 *
 * \brief CUDA implementation of lambdarank.
 */
#include <thrust/fill.h>                        // for fill_n
#include <thrust/for_each.h>                    // for for_each_n
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/iterator/zip_iterator.h>       // for make_zip_iterator
#include <thrust/tuple.h>                       // for make_tuple, tuple, tie, get

#include <algorithm>                            // for min
#include <cassert>                              // for assert
#include <cmath>                                // for abs, log2, isinf
#include <cstddef>                              // for size_t
#include <cstdint>                              // for int32_t
#include <memory>                               // for shared_ptr
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
#include "xgboost/span.h"                // for Span

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(lambdarank_obj_cu);

namespace cuda_impl {
namespace {
/**
 * \brief Calculate minimum value of bias for floating point truncation.
 */
void MinBias(Context const* ctx, std::shared_ptr<ltr::RankingCache> p_cache,
             linalg::VectorView<double const> t_plus, linalg::VectorView<double const> tj_minus,
             common::Span<double> d_min) {
  CHECK_EQ(d_min.size(), 2);
  auto cuctx = ctx->CUDACtx();

  auto k = t_plus.Size();
  auto const& p = p_cache->Param();
  CHECK_GT(k, 0);
  CHECK_EQ(k, p_cache->MaxPositionSize());

  auto key_it = dh::MakeTransformIterator<std::size_t>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) { return i * k; });
  auto val_it = dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul),
                                                  [=] XGBOOST_DEVICE(std::size_t i) {
                                                    if (i >= k) {
                                                      return std::abs(tj_minus(i - k));
                                                    }
                                                    return std::abs(t_plus(i));
                                                  });
  std::size_t bytes;
  cub::DeviceSegmentedReduce::Min(nullptr, bytes, val_it, d_min.data(), 2, key_it, key_it + 1,
                                  cuctx->Stream());
  dh::TemporaryArray<char> temp(bytes);
  cub::DeviceSegmentedReduce::Min(temp.data().get(), bytes, val_it, d_min.data(), 2, key_it,
                                  key_it + 1, cuctx->Stream());
}

/**
 * \brief Type for gradient statistic. (Gradient, cost for unbiased LTR, normalization factor)
 */
using GradCostNorm = thrust::tuple<GradientPair, double, double>;

/**
 * \brief Obtain and update the gradient for one pair.
 */
template <bool unbiased, bool has_truncation, typename Delta>
struct GetGradOp {
  MakePairsOp<has_truncation> make_pair;
  Delta delta;

  bool need_update;

  auto __device__ operator()(std::size_t idx) -> GradCostNorm {
    auto const& args = make_pair.args;
    auto g = dh::SegmentId(args.d_threads_group_ptr, idx);

    auto data_group_begin = static_cast<std::size_t>(args.d_group_ptr[g]);
    std::size_t n_data = args.d_group_ptr[g + 1] - data_group_begin;
    // obtain group segment data.
    auto g_label = args.labels.Slice(linalg::Range(data_group_begin, data_group_begin + n_data), 0);
    auto g_predt = args.predts.subspan(data_group_begin, n_data);
    auto g_gpair = args.gpairs.subspan(data_group_begin, n_data).data();
    auto g_rank = args.d_sorted_idx.subspan(data_group_begin, n_data);

    auto [i, j] = make_pair(idx, g);

    std::size_t rank_high = i, rank_low = j;
    if (g_label(g_rank[i]) == g_label(g_rank[j])) {
      return thrust::make_tuple(GradientPair{}, 0.0, 0.0);
    }
    if (g_label(g_rank[i]) < g_label(g_rank[j])) {
      thrust::swap(rank_high, rank_low);
    }

    double cost{0};

    auto delta_op = [&](auto const&... args) { return delta(args..., g); };
    GradientPair pg = LambdaGrad<unbiased>(g_label, g_predt, g_rank, rank_high, rank_low, delta_op,
                                           args.ti_plus, args.tj_minus, &cost);

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

      dh::AtomicAddGpair(g_gpair + idx_high, pgt);
      dh::AtomicAddGpair(g_gpair + idx_low, ngt);
    }

    if (unbiased && need_update) {
      // second run, update the cost
      assert(args.tj_minus.Size() == args.ti_plus.Size() && "Invalid size of position bias");

      auto g_li = args.li.Slice(linalg::Range(data_group_begin, data_group_begin + n_data));
      auto g_lj = args.lj.Slice(linalg::Range(data_group_begin, data_group_begin + n_data));

      if (idx_high < args.ti_plus.Size() && idx_low < args.ti_plus.Size()) {
        if (args.tj_minus(idx_low) >= Eps64()) {
          // eq.30
          atomicAdd(&g_li(idx_high), common::TruncateWithRounding(args.d_cost_rounding[0],
                                                                  cost / args.tj_minus(idx_low)));
        }
        if (args.ti_plus(idx_high) >= Eps64()) {
          // eq.31
          atomicAdd(&g_lj(idx_low), common::TruncateWithRounding(args.d_cost_rounding[0],
                                                                 cost / args.ti_plus(idx_high)));
        }
      }
    }
    return thrust::make_tuple(GradientPair{std::abs(pg.GetGrad()), std::abs(pg.GetHess())},
                              std::abs(cost), -2.0 * static_cast<double>(pg.GetGrad()));
  }
};

template <bool unbiased, bool has_truncation, typename Delta>
struct MakeGetGrad {
  MakePairsOp<has_truncation> make_pair;
  Delta delta;

  [[nodiscard]] KernelInputs const& Args() const { return make_pair.args; }

  MakeGetGrad(KernelInputs args, Delta d) : make_pair{args}, delta{std::move(d)} {}

  GetGradOp<unbiased, has_truncation, Delta> operator()(bool need_update) {
    return GetGradOp<unbiased, has_truncation, Delta>{make_pair, delta, need_update};
  }
};

/**
 * \brief Calculate gradient for all pairs using update op created by make_get_grad.
 *
 * We need to run gradient calculation twice, the first time gathers infomation like
 * maximum gradient, maximum cost, and the normalization term using reduction. The second
 * time performs the actual update.
 *
 * Without normalization, we only need to run it once since we can manually calculate
 * the bounds of gradient (NDCG \in [0, 1], delta_NDCG \in [0, 1], ti+/tj- are from the
 * previous iteration so the bound can be calculated for current iteration). However, if
 * normalization is used, the delta score is un-bounded and we need to obtain the sum
 * gradient. As a tradeoff, we simply run the kernel twice, once as reduction, second
 * one as for_each.
 *
 * Alternatively, we can bound the delta score by limiting the output of the model using
 * sigmoid for binary output and some normalization for multi-level. But effect to the
 * accuracy is not known yet, and it's only used by GPU.
 *
 * For performance, the segmented sort for sorted scores is the bottleneck and takes up
 * about half of the time, while the reduction and for_each takes up the second half.
 */
template <bool unbiased, bool has_truncation, typename Delta>
void CalcGrad(Context const* ctx, MetaInfo const& info, std::shared_ptr<ltr::RankingCache> p_cache,
              MakeGetGrad<unbiased, has_truncation, Delta> make_get_grad) {
  auto n_groups = p_cache->Groups();
  auto d_threads_group_ptr = p_cache->CUDAThreadsGroupPtr();
  auto d_gptr = p_cache->DataGroupPtr(ctx);
  auto d_gpair = make_get_grad.Args().gpairs;

  /**
   * First pass, gather info for normalization and rounding factor.
   */
  auto val_it = dh::MakeTransformIterator<GradCostNorm>(thrust::make_counting_iterator(0ul),
                                                        make_get_grad(false));
  auto reduction_op = [] XGBOOST_DEVICE(GradCostNorm const& l,
                                        GradCostNorm const& r) -> GradCostNorm {
    // get maximum gradient for each group, along with cost and the normalization term
    auto const& lg = thrust::get<0>(l);
    auto const& rg = thrust::get<0>(r);
    auto grad = std::max(lg.GetGrad(), rg.GetGrad());
    auto hess = std::max(lg.GetHess(), rg.GetHess());
    auto cost = std::max(thrust::get<1>(l), thrust::get<1>(r));
    double sum_lambda = thrust::get<2>(l) + thrust::get<2>(r);
    return thrust::make_tuple(GradientPair{std::abs(grad), std::abs(hess)}, cost, sum_lambda);
  };
  auto init = thrust::make_tuple(GradientPair{0.0f, 0.0f}, 0.0, 0.0);
  common::Span<GradCostNorm> d_max_lambdas = p_cache->MaxLambdas<GradCostNorm>(ctx, n_groups);
  CHECK_EQ(n_groups * sizeof(GradCostNorm), d_max_lambdas.size_bytes());

  std::size_t bytes;
  cub::DeviceSegmentedReduce::Reduce(nullptr, bytes, val_it, d_max_lambdas.data(), n_groups,
                                     d_threads_group_ptr.data(), d_threads_group_ptr.data() + 1,
                                     reduction_op, init, ctx->CUDACtx()->Stream());
  dh::TemporaryArray<char> temp(bytes);
  cub::DeviceSegmentedReduce::Reduce(
      temp.data().get(), bytes, val_it, d_max_lambdas.data(), n_groups, d_threads_group_ptr.data(),
      d_threads_group_ptr.data() + 1, reduction_op, init, ctx->CUDACtx()->Stream());

  dh::TemporaryArray<double> min_bias(2);
  auto d_min_bias = dh::ToSpan(min_bias);
  if (unbiased) {
    MinBias(ctx, p_cache, make_get_grad.Args().ti_plus, make_get_grad.Args().tj_minus, d_min_bias);
  }
  /**
   * Create rounding factors
   */
  auto d_cost_rounding = p_cache->CUDACostRounding(ctx);
  auto d_rounding = p_cache->CUDARounding(ctx);
  dh::LaunchN(n_groups, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t g) mutable {
    auto group_size = d_gptr[g + 1] - d_gptr[g];
    auto const& max_grad = thrust::get<0>(d_max_lambdas[g]);
    // float group size
    auto fgs = static_cast<float>(group_size);
    auto grad = common::CreateRoundingFactor(fgs * max_grad.GetGrad(), group_size);
    auto hess = common::CreateRoundingFactor(fgs * max_grad.GetHess(), group_size);
    d_rounding(g) = GradientPair{grad, hess};

    auto cost = thrust::get<1>(d_max_lambdas[g]);
    if (unbiased) {
      cost /= std::min(d_min_bias[0], d_min_bias[1]);
      d_cost_rounding[0] = common::CreateRoundingFactor(fgs * cost, group_size);
    }
  });

  /**
   * Second pass, actual update to gradient and bias.
   */
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul),
                     p_cache->CUDAThreads(), make_get_grad(true));

  /**
   * Lastly, normalization and weight.
   */
  auto d_weights = common::MakeOptionalWeights(ctx, info.weights_);
  auto w_norm = p_cache->WeightNorm();
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), d_gpair.size(),
                     [=] XGBOOST_DEVICE(std::size_t i) {
                       auto g = dh::SegmentId(d_gptr, i);
                       auto sum_lambda = thrust::get<2>(d_max_lambdas[g]);
                       // Normalization
                       if (sum_lambda > 0.0) {
                         double norm = std::log2(1.0 + sum_lambda) / sum_lambda;
                         d_gpair[i] *= norm;
                       }
                       d_gpair[i] *= (d_weights[g] * w_norm);
                     });
}

/**
 * \brief Handles boilerplate code like getting device span.
 */
template <typename Delta>
void Launch(Context const* ctx, std::int32_t iter, HostDeviceVector<float> const& preds,
            const MetaInfo& info, std::shared_ptr<ltr::RankingCache> p_cache, Delta delta,
            linalg::VectorView<double const> ti_plus,   // input bias ratio
            linalg::VectorView<double const> tj_minus,  // input bias ratio
            linalg::VectorView<double> li, linalg::VectorView<double> lj,
            HostDeviceVector<GradientPair>* out_gpair) {
  // boilerplate
  std::int32_t device_id = ctx->gpu_id;
  dh::safe_cuda(cudaSetDevice(device_id));
  auto n_groups = p_cache->Groups();

  info.labels.SetDevice(device_id);
  preds.SetDevice(device_id);
  out_gpair->SetDevice(device_id);
  out_gpair->Resize(preds.Size());

  CHECK(p_cache);

  auto d_rounding = p_cache->CUDARounding(ctx);
  auto d_cost_rounding = p_cache->CUDACostRounding(ctx);

  CHECK_NE(d_rounding.Size(), 0);

  auto label = info.labels.View(ctx->gpu_id);
  auto predts = preds.ConstDeviceSpan();
  auto gpairs = out_gpair->DeviceSpan();
  thrust::fill_n(ctx->CUDACtx()->CTP(), gpairs.data(), gpairs.size(), GradientPair{0.0f, 0.0f});

  auto const d_threads_group_ptr = p_cache->CUDAThreadsGroupPtr();
  auto const d_gptr = p_cache->DataGroupPtr(ctx);
  auto const rank_idx = p_cache->SortedIdx(ctx, predts);

  auto const unbiased = p_cache->Param().lambdarank_unbiased;

  common::Span<std::size_t const> d_y_sorted_idx;
  if (!p_cache->Param().HasTruncation()) {
    d_y_sorted_idx = SortY(ctx, info, rank_idx, p_cache);
  }

  KernelInputs args{ti_plus,        tj_minus, li,     lj,     d_gptr,     d_threads_group_ptr,
                    rank_idx,       label,    predts, gpairs, d_rounding, d_cost_rounding.data(),
                    d_y_sorted_idx, iter};

  // dispatch based on unbiased and truncation
  if (p_cache->Param().HasTruncation()) {
    if (unbiased) {
      CalcGrad(ctx, info, p_cache, MakeGetGrad<true, true, Delta>{args, delta});
    } else {
      CalcGrad(ctx, info, p_cache, MakeGetGrad<false, true, Delta>{args, delta});
    }
  } else {
    if (unbiased) {
      CalcGrad(ctx, info, p_cache, MakeGetGrad<true, false, Delta>{args, delta});
    } else {
      CalcGrad(ctx, info, p_cache, MakeGetGrad<false, false, Delta>{args, delta});
    }
  }
}
}  // anonymous namespace

common::Span<std::size_t const> SortY(Context const* ctx, MetaInfo const& info,
                                      common::Span<std::size_t const> d_rank,
                                      std::shared_ptr<ltr::RankingCache> p_cache) {
  auto const d_group_ptr = p_cache->DataGroupPtr(ctx);
  auto label = info.labels.View(ctx->gpu_id);
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

void LambdaRankGetGradientNDCG(Context const* ctx, std::int32_t iter,
                               const HostDeviceVector<float>& preds, const MetaInfo& info,
                               std::shared_ptr<ltr::NDCGCache> p_cache,
                               linalg::VectorView<double const> ti_plus,   // input bias ratio
                               linalg::VectorView<double const> tj_minus,  // input bias ratio
                               linalg::VectorView<double> li, linalg::VectorView<double> lj,
                               HostDeviceVector<GradientPair>* out_gpair) {
  // boilerplate
  std::int32_t device_id = ctx->gpu_id;
  dh::safe_cuda(cudaSetDevice(device_id));
  auto const d_inv_IDCG = p_cache->InvIDCG(ctx);
  auto const discount = p_cache->Discount(ctx);

  info.labels.SetDevice(device_id);
  preds.SetDevice(device_id);

  auto const exp_gain = p_cache->Param().ndcg_exp_gain;
  auto delta_ndcg = [=] XGBOOST_DEVICE(float y_high, float y_low, std::size_t rank_high,
                                       std::size_t rank_low, bst_group_t g) {
    return exp_gain ? DeltaNDCG<true>(y_high, y_low, rank_high, rank_low, d_inv_IDCG(g), discount)
                    : DeltaNDCG<false>(y_high, y_low, rank_high, rank_low, d_inv_IDCG(g), discount);
  };
  Launch(ctx, iter, preds, info, p_cache, delta_ndcg, ti_plus, tj_minus, li, lj, out_gpair);
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
  auto label = info.labels.View(ctx->gpu_id).Slice(linalg::All(), 0);
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

void LambdaRankGetGradientMAP(Context const* ctx, std::int32_t iter,
                              HostDeviceVector<float> const& predt, const MetaInfo& info,
                              std::shared_ptr<ltr::MAPCache> p_cache,
                              linalg::VectorView<double const> ti_plus,   // input bias ratio
                              linalg::VectorView<double const> tj_minus,  // input bias ratio
                              linalg::VectorView<double> li, linalg::VectorView<double> lj,
                              HostDeviceVector<GradientPair>* out_gpair) {
  std::int32_t device_id = ctx->gpu_id;
  dh::safe_cuda(cudaSetDevice(device_id));

  info.labels.SetDevice(device_id);
  predt.SetDevice(device_id);

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

  Launch(ctx, iter, predt, info, p_cache, delta_map, ti_plus, tj_minus, li, lj, out_gpair);
}

void LambdaRankGetGradientPairwise(Context const* ctx, std::int32_t iter,
                                   HostDeviceVector<float> const& predt, const MetaInfo& info,
                                   std::shared_ptr<ltr::RankingCache> p_cache,
                                   linalg::VectorView<double const> ti_plus,   // input bias ratio
                                   linalg::VectorView<double const> tj_minus,  // input bias ratio
                                   linalg::VectorView<double> li, linalg::VectorView<double> lj,
                                   HostDeviceVector<GradientPair>* out_gpair) {
  std::int32_t device_id = ctx->gpu_id;
  dh::safe_cuda(cudaSetDevice(device_id));

  info.labels.SetDevice(device_id);
  predt.SetDevice(device_id);

  auto d_predt = predt.ConstDeviceSpan();
  auto const d_sorted_idx = p_cache->SortedIdx(ctx, d_predt);

  auto delta = [] XGBOOST_DEVICE(float, float, std::size_t, std::size_t, bst_group_t) {
    return 1.0;
  };

  Launch(ctx, iter, predt, info, p_cache, delta, ti_plus, tj_minus, li, lj, out_gpair);
}

namespace {
struct ReduceOp {
  template <typename Tup>
  Tup XGBOOST_DEVICE operator()(Tup const& l, Tup const& r) {
    return thrust::make_tuple(thrust::get<0>(l) + thrust::get<0>(r),
                              thrust::get<1>(l) + thrust::get<1>(r));
  }
};
}  // namespace

void LambdaRankUpdatePositionBias(Context const* ctx, linalg::VectorView<double const> li_full,
                                  linalg::VectorView<double const> lj_full,
                                  linalg::Vector<double>* p_ti_plus,
                                  linalg::Vector<double>* p_tj_minus,
                                  linalg::Vector<double>* p_li,  // loss
                                  linalg::Vector<double>* p_lj,
                                  std::shared_ptr<ltr::RankingCache> p_cache) {
  auto const d_group_ptr = p_cache->DataGroupPtr(ctx);
  auto n_groups = d_group_ptr.size() - 1;

  auto ti_plus = p_ti_plus->View(ctx->gpu_id);
  auto tj_minus = p_tj_minus->View(ctx->gpu_id);

  auto li = p_li->View(ctx->gpu_id);
  auto lj = p_lj->View(ctx->gpu_id);
  CHECK_EQ(li.Size(), ti_plus.Size());

  auto const& param = p_cache->Param();
  auto regularizer = param.Regularizer();
  std::size_t k = p_cache->MaxPositionSize();

  CHECK_EQ(li.Size(), k);
  CHECK_EQ(lj.Size(), k);
  // reduce li_full to li for each group.
  auto make_iter = [&](linalg::VectorView<double const> l_full) {
    auto l_it = [=] XGBOOST_DEVICE(std::size_t i) {
      // group index
      auto g = i % n_groups;
      // rank is the position within a group, also the segment index
      auto r = i / n_groups;

      auto begin = d_group_ptr[g];
      std::size_t group_size = d_group_ptr[g + 1] - begin;
      auto n = std::min(group_size, k);
      // r can be greater than n since we allocate threads based on truncation level
      // instead of actual group size.
      if (r >= n) {
        return 0.0;
      }
      return l_full(r + begin);
    };
    return l_it;
  };
  auto li_it =
      dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul), make_iter(li_full));
  auto lj_it =
      dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul), make_iter(lj_full));
  // k segments, each segment has size n_groups.
  auto key_it = dh::MakeTransformIterator<std::size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return i * n_groups; });
  auto val_it = thrust::make_zip_iterator(thrust::make_tuple(li_it, lj_it));
  auto out_it =
      thrust::make_zip_iterator(thrust::make_tuple(li.Values().data(), lj.Values().data()));

  auto init = thrust::make_tuple(0.0, 0.0);
  std::size_t bytes;
  cub::DeviceSegmentedReduce::Reduce(nullptr, bytes, val_it, out_it, k, key_it, key_it + 1,
                                     ReduceOp{}, init, ctx->CUDACtx()->Stream());
  dh::TemporaryArray<char> temp(bytes);
  cub::DeviceSegmentedReduce::Reduce(temp.data().get(), bytes, val_it, out_it, k, key_it,
                                     key_it + 1, ReduceOp{}, init, ctx->CUDACtx()->Stream());

  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), li.Size(),
                     [=] XGBOOST_DEVICE(std::size_t i) mutable {
                       if (li(0) >= Eps64()) {
                         ti_plus(i) = std::pow(li(i) / li(0), regularizer);
                       }
                       if (lj(0) >= Eps64()) {
                         tj_minus(i) = std::pow(lj(i) / lj(0), regularizer);
                       }
                       assert(!std::isinf(ti_plus(i)));
                       assert(!std::isinf(tj_minus(i)));
                     });
}
}  // namespace cuda_impl
}  // namespace xgboost::obj
