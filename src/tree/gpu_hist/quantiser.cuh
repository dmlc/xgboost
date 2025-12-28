/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#pragma once
#include <thrust/copy.h>  // for copy_n

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for ToSpan, MakeIndexTransformIter
#include "../../common/device_vector.cuh"   // for device_vector
#include "../../common/linalg_op.cuh"       // for tbegin
#include "xgboost/base.h"                   // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/data.h"                   // for MetaInfo
#include "xgboost/linalg.h"                 // for VectorView

namespace xgboost::tree {
class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  // Used for test
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float)
      : to_fixed_point_{to_fixed}, to_floating_point_{to_float} {}
  GradientQuantiser(Context const* ctx, linalg::VectorView<GradientPair const> gpair,
                    MetaInfo const& info);
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64 ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPoint(GradientPairPrecise const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise
  ToFloatingPoint(const GradientPairInt64& gpair) const {
    auto g = gpair.GetQuantisedGrad() * to_floating_point_.GetGrad();
    auto h = gpair.GetQuantisedHess() * to_floating_point_.GetHess();
    return {g, h};
  }
};

// For vector leaf
class MultiGradientQuantiser {
 private:
  dh::device_vector<GradientQuantiser> quantizers_;

 public:
  MultiGradientQuantiser(Context const* ctx, linalg::MatrixView<GradientPair const> gpair,
                         MetaInfo const& info);

  [[nodiscard]] auto Quantizers() const { return dh::ToSpan(this->quantizers_); }
};

inline void CalcQuantizedGpairs(Context const* ctx, linalg::Matrix<GradientPair>* const gpairs,
                                common::Span<GradientQuantiser const> roundings,
                                linalg::Matrix<GradientPairInt64>* p_out) {
  auto shape = gpairs->Shape();
  if (p_out->Empty()) {
    *p_out = linalg::Matrix<GradientPairInt64>{shape, ctx->Device(), linalg::kF};
  } else {
    p_out->Reshape(shape);
  }

  auto in_gpair = gpairs->View(ctx->Device());
  auto out_gpair = p_out->View(ctx->Device());
  CHECK(out_gpair.FContiguous());
  auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto [ridx, target_idx] = linalg::UnravelIndex(i, in_gpair.Shape());
    auto g = in_gpair(ridx, target_idx);
    return roundings[target_idx].ToFixedPoint(g);
  });
  thrust::copy_n(ctx->CUDACtx()->CTP(), it, in_gpair.Size(), linalg::tbegin(out_gpair));
}
}  // namespace xgboost::tree
