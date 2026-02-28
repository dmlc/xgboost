/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <thrust/copy.h>                         // for copy_n
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>
#include <cstdint>          // uint32_t, int32_t
#include <cuda/functional>  // for proclaim_copyable_arguments
#include <vector>           // for vector

#include "../../collective/aggregator.h"
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../common/linalg_op.cuh"  // for tbegin, tcbegin
#include "quantiser.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
struct Pair {
  GradientPairPrecise first;
  GradientPairPrecise second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

struct Clip {
  static XGBOOST_DEV_INLINE float Pclip(float v) { return v > 0 ? v : 0; }
  static XGBOOST_DEV_INLINE float Nclip(float v) { return v < 0 ? abs(v) : 0; }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return {GradientPairPrecise{pg, ph}, GradientPairPrecise{ng, nh}};
  }
};

/**
 * In algorithm 5 (see common::CreateRoundingFactor) the bound is calculated as
 * $max(|v_i|) * n$.  Here we use the bound:
 *
 * @begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * @end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
Pair MakeQuantiserForTarget(Context const* ctx, linalg::VectorView<GradientPair const> gpair) {
  using T = typename GradientPairPrecise::ValueT;

  auto beg = thrust::make_transform_iterator(linalg::tcbegin(gpair), Clip{});
  Pair p =
      dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.Size(), Pair{}, cuda::std::plus<Pair>{});
  return p;
}

GradientQuantiser BuildQuantiserFromPair(Pair const& p, std::size_t total_rows) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;

  GradientSumT positive_sum{p.first}, negative_sum{p.second};

  auto histogram_rounding =
      GradientSumT{common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), total_rows),
                   common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetHess(), negative_sum.GetHess()), total_rows)};

  using IntT = typename GradientPairInt64::ValueT;

  auto to_floating_point =
      histogram_rounding /
      static_cast<T>(static_cast<IntT>(1)
                     << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit
  auto to_fixed_point = GradientSumT{static_cast<T>(1) / to_floating_point.GetGrad(),
                                     static_cast<T>(1) / to_floating_point.GetHess()};
  return GradientQuantiser{to_fixed_point, to_floating_point};
}
}  // anonymous namespace

GradientQuantiserGroup::GradientQuantiserGroup(Context const* ctx,
                                               linalg::MatrixView<GradientPair const> gpair,
                                               MetaInfo const& info) {
  auto n_targets = gpair.Shape(1);
  CHECK_GE(n_targets, 1);

  // Local reduction per target — these are fast device-local operations.
  using ReduceT = typename GradientPairPrecise::ValueT;
  std::vector<Pair> h_pairs(n_targets);
  std::size_t n_samples = gpair.Shape(0);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    h_pairs[t] = MakeQuantiserForTarget(ctx, gpair.Slice(linalg::All(), t));
  }

  auto rc = collective::Success() << [&]() {
    static_assert(sizeof(Pair) == sizeof(ReduceT) * 4);
    auto casted = linalg::MakeVec(reinterpret_cast<ReduceT*>(h_pairs.data()), 4 * n_targets);
    return collective::GlobalSum(ctx, info, casted);
  } << [&] {
    // Single GlobalSum for total_rows (shared across targets).
    return collective::GlobalSum(ctx, info, linalg::MakeVec(&n_samples, 1));
  };
  collective::SafeColl(rc);

  // Build quantisers on host from the reduced pairs.
  h_quantizers_.resize(n_targets);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    h_quantizers_[t] = BuildQuantiserFromPair(h_pairs[t], n_samples);
  }

  // Copy to device.
  d_quantizers_.resize(n_targets);
  dh::safe_cuda(cudaMemcpyAsync(d_quantizers_.data(), h_quantizers_.data(),
                                n_targets * sizeof(GradientQuantiser), cudaMemcpyHostToDevice,
                                ctx->CUDACtx()->Stream()));
}

GradientQuantiserGroup::GradientQuantiserGroup(Context const* ctx,
                                               linalg::VectorView<GradientPair const> gpair,
                                               MetaInfo const& info)
    : GradientQuantiserGroup(
          ctx, linalg::MakeTensorView(ctx, gpair.Values(), gpair.Size(), bst_target_t{1}), info) {}

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out) {
  auto shape = gpairs.Shape();
  if (p_out->Empty()) {
    *p_out = linalg::Matrix<GradientPairInt64>{shape, ctx->Device(), linalg::kF};
  } else {
    p_out->Reshape(shape);
  }

  auto out_gpair = p_out->View(ctx->Device());
  CHECK(out_gpair.FContiguous());
  auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto [ridx, target_idx] = linalg::UnravelIndex(i, gpairs.Shape());
    auto g = gpairs(ridx, target_idx);
    return roundings[target_idx].ToFixedPoint(g);
  });
  thrust::copy_n(ctx->CUDACtx()->CTP(), it, gpairs.Size(), linalg::tbegin(out_gpair));
}
}  // namespace xgboost::tree
