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

enum class ChildNodeIndicator : int8_t {
  kLeftChild, kRightChild
};

struct EvaluateSplitsHistEntry {
  ChildNodeIndicator indicator;
  uint64_t hist_idx;
};

template <typename GradientSumT>
struct ScanComputedElem {
  GradientSumT left_sum{0.0, 0.0};
  GradientSumT right_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};
  float best_loss_chg{std::numeric_limits<float>::lowest()};
  int32_t best_findex{-1};
  bool is_cat{false};
  float best_fvalue{std::numeric_limits<float>::quiet_NaN()};
  DefaultDirection best_direction{DefaultDirection::kLeftDir};

  __noinline__ __device__ bool Update(
      GradientSumT left_sum_in,
      GradientSumT right_sum_in,
      GradientSumT parent_sum_in,
      float loss_chg_in,
      int32_t findex_in,
      bool is_cat_in,
      float fvalue_in,
      DefaultDirection dir_in,
      const GPUTrainingParam& param);
};

template <typename GradientSumT>
struct ScanElem {
  ChildNodeIndicator indicator{ChildNodeIndicator::kLeftChild};
  uint64_t hist_idx;
  GradientSumT gpair{0.0, 0.0};
  int32_t findex{-1};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  ScanComputedElem<GradientSumT> computed_result{};
};

template <typename GradientSumT>
struct ScanValueOp {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  template <bool forward>
  __noinline__ __device__ ScanElemT MapEvaluateSplitsHistEntryToScanElem(
      EvaluateSplitsHistEntry entry,
      EvaluateSplitInputs<GradientSumT> split_input);
  __noinline__ __device__ thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup);
};

template <typename GradientSumT>
struct ScanOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  template<bool forward>
  __noinline__ __device__ ScanElem<GradientSumT>
  DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs);
  __noinline__ __device__ thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<ScanElemT, ScanElemT> lhs, thrust::tuple<ScanElemT, ScanElemT> rhs);
};

template <typename GradientSumT>
struct WriteScan {
  EvaluateSplitInputs<GradientSumT> left, right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;
  common::Span<ScanComputedElem<GradientSumT>> d_out_scan;

  using ScanElemT = ScanElem<GradientSumT>;

  template <bool forward>
  __noinline__ __device__ void DoIt(ScanElemT e);

  __noinline__ __device__ thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<ScanElemT, ScanElemT> e);
};

template <typename GradientSumT>
dh::device_vector<ScanComputedElem<GradientSumT>>
EvaluateSplitsFindOptimalSplitsViaScan(
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
