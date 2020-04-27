/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_
#include <xgboost/span.h>
#include "../../data/ellpack_page.cuh"
#include "../constraints.cuh"
#include "../updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

template <typename GradientSumT>
struct EvaluateSplitInputs {
  int nidx;
  GradientSumT parent_sum;
  GPUTrainingParam param;
  common::Span<const bst_feature_t> feature_set;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  common::Span<const float> min_fvalue;
  common::Span<const GradientSumT> gradient_histogram;
  ValueConstraint value_constraint;
  common::Span<const int> monotonic_constraints;
};
template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right);
template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         EvaluateSplitInputs<GradientSumT> input);
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
