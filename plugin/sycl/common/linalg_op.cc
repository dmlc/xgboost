/**
 * Copyright 2021-2025, XGBoost Contributors
 * \file linalg_op.h
 */

#include "../data.h"
#include "../device_manager.h"

#include "../../../src/common/optional_weight.h"  // for OptionalWeights
#include "xgboost/context.h"  // for Context

#include <sycl/sycl.hpp>

namespace xgboost::sycl::linalg {

void SmallHistogram(Context const* ctx, xgboost::linalg::MatrixView<float const> indices,
                    xgboost::common::OptionalWeights const& weights,
                    xgboost::linalg::VectorView<float> bins) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(ctx->Device());

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(indices.Size()),
                       [=](::sycl::id<1> pid) {
      const size_t i = pid[0];
      auto y = indices(i);
      auto w = weights[i];
      AtomicRef<float> bin_val(const_cast<float&>(bins(static_cast<std::size_t>(y))));
      bin_val += w;
    });
  }).wait();
}

void VecScaMul(Context const* ctx, xgboost::linalg::VectorView<float> x, double mul) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(ctx->Device());

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(x.Size()),
                       [=](::sycl::id<1> pid) {
      const size_t i = pid[0];
      const_cast<float&>(x(i)) *= mul;
    });
  }).wait();
}
}  // namespace xgboost::sycl::linalg

namespace xgboost::linalg::sycl_impl {
void VecScaMul(Context const* ctx, xgboost::linalg::VectorView<float> x, double mul) {
  xgboost::sycl::linalg::VecScaMul(ctx, x, mul);
}
}  // namespace xgboost::linalg::sycl_impl
