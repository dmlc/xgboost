/*!
 * Copyright 2021 by Contributors
 */
#include "../common/device_helpers.cuh"
#include "xgboost/context.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost {
namespace gbm {

void GPUCopyGradient(HostDeviceVector<GradientPair> const *in_gpair,
                     bst_group_t n_groups, bst_group_t group_id,
                     HostDeviceVector<GradientPair> *out_gpair) {
  auto mat = linalg::TensorView<GradientPair const, 2>(
      in_gpair->ConstDeviceSpan(),
      {in_gpair->Size() / n_groups, static_cast<size_t>(n_groups)},
      in_gpair->DeviceIdx());
  auto v_in = mat.Slice(linalg::All(), group_id);
  out_gpair->Resize(v_in.Size());
  auto d_out = out_gpair->DeviceSpan();
  dh::LaunchN(v_in.Size(), [=] __device__(size_t i) { d_out[i] = v_in(i); });
}

void GPUDartPredictInc(common::Span<float> out_predts,
                       common::Span<float> predts, float tree_w, size_t n_rows,
                       bst_group_t n_groups, bst_group_t group) {
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] * tree_w);
  });
}

void GPUDartInplacePredictInc(common::Span<float> out_predts, common::Span<float> predts,
                              float tree_w, size_t n_rows,
                              linalg::TensorView<float const, 1> base_score, bst_group_t n_groups,
                              bst_group_t group) {
  CHECK_EQ(base_score.Size(), 1);
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] - base_score(0)) * tree_w;
  });
}
}  // namespace gbm
}  // namespace xgboost
