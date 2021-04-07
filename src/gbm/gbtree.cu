/*!
 * Copyright 2021 by Contributors
 */
#include "xgboost/span.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace gbm {
void GPUDartPredictInc(common::Span<float> out_predts,
                       common::Span<float> predts, float tree_w, size_t n_rows,
                       bst_group_t n_groups, bst_group_t group) {
  dh::LaunchN(dh::CurrentDevice(), n_rows, [=]XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] * tree_w);
  });
}

void GPUDartInplacePredictInc(common::Span<float> out_predts,
                              common::Span<float> predts, float tree_w,
                              size_t n_rows, float base_score,
                              bst_group_t n_groups, bst_group_t group) {
  dh::LaunchN(dh::CurrentDevice(), n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] - base_score) * tree_w;
  });
}
}  // namespace gbm
}  // namespace xgboost
