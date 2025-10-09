/*!
 * Copyright 2017-2025 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#include "../../src/common/linalg_op.h"
#include "../../../src/common/optional_weight.h"  // for MakeOptionalWeights
#include "sycl_helpers.h"

namespace xgboost::sycl::linalg {
TEST(SyclLinalg, SmallHistogram) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  std::size_t cnt = 32, n_bins = 4;
  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  HostDeviceVector<float> values(cnt * n_bins);
  values.SetDevice(ctx.Device());
  float* values_host_ptr = values.HostPointer();
  for (std::size_t i = 0; i < n_bins; ++i) {
    std::fill(values_host_ptr + i * cnt, values_host_ptr + (i  + 1) * cnt, i);
  }

  std::mt19937 rng;
  rng.seed(2025);
  std::shuffle(values_host_ptr, values_host_ptr + cnt * n_bins, rng);

  float* values_device_ptr = values.DevicePointer();
  xgboost::linalg::MatrixView<float> indices =
      xgboost::linalg::MakeTensorView(&ctx, xgboost::common::Span(values_device_ptr, cnt * n_bins),
                                      cnt * n_bins, 1);
  HostDeviceVector<float> bins(n_bins, 0);
  bins.SetDevice(ctx.Device());

  HostDeviceVector<float> weights;
  xgboost::linalg::SmallHistogram(&ctx, indices, xgboost::common::MakeOptionalWeights(ctx.Device(), weights),
                 xgboost::linalg::MakeTensorView(&ctx, xgboost::common::Span(bins.DevicePointer(), n_bins), n_bins));

  for (std::size_t i = 0; i < n_bins; ++i) {
    ASSERT_EQ(bins.HostVector()[i], cnt);
  }
}
}  // namespace xgboost::linalg