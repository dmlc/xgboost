/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_
#include <xgboost/span.h>
#include "../../data/ellpack_page.cuh"
#include "../../common/device_helpers.cuh"
#include "../split_evaluator.h"
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
  common::Span<FeatureType const> feature_types;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  common::Span<const float> min_fvalue;
  common::Span<const GradientSumT> gradient_histogram;
};

struct EvaluateSplitsHistEntry {
  uint32_t node_idx;
  uint32_t hist_idx;
  bool forward;
};

template <typename GradientSumT>
struct ScanElem {
  uint32_t node_idx;
  uint32_t hist_idx;
  int32_t findex{-1};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  bool forward{true};
  GradientSumT gpair{0.0, 0.0};
  GradientSumT partial_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};
};

template <typename GradientSumT>
struct ScanValueOp {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  __device__ ScanElemT MapEvaluateSplitsHistEntryToScanElem(
      EvaluateSplitsHistEntry entry,
      EvaluateSplitInputs<GradientSumT> split_input);
  __device__ ScanElemT
  operator() (EvaluateSplitsHistEntry entry);
};

template <typename GradientSumT>
struct ScanOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  __device__ ScanElemT operator() (ScanElemT lhs, ScanElemT rhs);
};

template <typename GradientSumT>
struct ReduceElem {
  GradientSumT partial_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};
  float loss_chg{std::numeric_limits<float>::lowest()};
  int32_t findex{-1};
  uint32_t node_idx{0};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  DefaultDirection direction{DefaultDirection::kLeftDir};
};

template <typename GradientSumT>
struct ReduceValueOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;
  using ReduceElemT = ReduceElem<GradientSumT>;

  __device__ ReduceElemT operator() (ScanElemT e);
};

template <typename GradientSumT>
dh::device_vector<ReduceElem<GradientSumT>>
EvaluateSplitsGenerateSplitCandidatesViaScan(
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right);

template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right);

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input);
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
