/**
 * Copyright 2021-2025, XGBoost Contributors
 * \file linalg_op.h
 */

#include <sycl/sycl.hpp>

#include "../../../src/common/optional_weight.h"  // for OptionalWeights
#include "../data.h"
#include "../device_manager.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::sycl::linalg {
void SmallHistogram(Context const* ctx, xgboost::linalg::MatrixView<float const> indices,
                    xgboost::common::OptionalWeights const& weights,
                    xgboost::linalg::VectorView<float> bins) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(ctx->Device());

  qu->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(indices.Size()), [=](::sycl::id<1> pid) {
        const size_t i = pid[0];
        auto y = indices(i);
        auto w = weights[i];
        AtomicRef<float> bin_val(const_cast<float&>(bins(static_cast<std::size_t>(y))));
        bin_val += w;
      });
    }).wait();
}
}  // namespace xgboost::sycl::linalg
