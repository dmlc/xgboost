/**
 * Copyright 2026, XGBoost Contributors
 */

#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "xgboost/context.h"             // for Context
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::c_api::cuda_impl {
void SplitShapValues(Context const *ctx, HostDeviceVector<float> const &contribs, std::size_t rows,
                     std::size_t cols, std::size_t groups, HostDeviceVector<float> *out_values,
                     HostDeviceVector<float> *out_bias) {
  auto cuctx = ctx->CUDACtx();
  auto contribs_d = contribs.ConstDeviceSpan();
  auto values_d = out_values->DeviceSpan();
  auto bias_d = out_bias->DeviceSpan();

  dh::LaunchN(values_d.size(), cuctx->Stream(), [=] __device__(std::size_t idx) {
    auto group = idx % groups;
    auto col = (idx / groups) % cols;
    auto row = idx / (cols * groups);
    auto contrib_offset = row * groups * (cols + 1) + group * (cols + 1);
    values_d[idx] = contribs_d[contrib_offset + col];
  });
  dh::LaunchN(bias_d.size(), cuctx->Stream(), [=] __device__(std::size_t idx) {
    auto group = idx % groups;
    auto row = idx / groups;
    auto contrib_offset = row * groups * (cols + 1) + group * (cols + 1);
    bias_d[idx] = contribs_d[contrib_offset + cols];
  });
}
}  // namespace xgboost::c_api::cuda_impl
