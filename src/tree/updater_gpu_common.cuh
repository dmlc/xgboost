/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#pragma once
#include <thrust/random.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>
#include <vector>
#include "../common/device_helpers.cuh"
#include "../common/random.h"
#include "param.h"

namespace xgboost {
namespace tree {

// Atomic add function for gradients
template <typename OutputGradientT, typename InputGradientT>
DEV_INLINE void AtomicAddGpair(OutputGradientT* dest,
                               const InputGradientT& gpair) {
  auto dst_ptr = reinterpret_cast<typename OutputGradientT::ValueT*>(dest);

  atomicAdd(dst_ptr,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetGrad()));
  atomicAdd(dst_ptr + 1,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetHess()));
}

/**
 * \fn  void CheckGradientMax(const dh::dvec<bst_gpair>& gpair)
 *
 * \brief Check maximum gradient value is below max allowed. This is to prevent
 * overflow when using integer gradient summation.
 */

inline void CheckGradientMax(HostDeviceVector<GradientPair> *gpair_ptr) {
  auto* ptr = reinterpret_cast<const float*>(gpair_ptr->HostVector().data());
  float abs_max =
    std::accumulate(ptr, ptr + (gpair_ptr->Size() * 2) , 0.f,
                      [=](float a, float b) { return max(abs(a), abs(b)); });

  float max_allowed = 1E-4f*std::pow(2.0f, 63.0f)/(1+ gpair_ptr->Size());
  CHECK_LT(abs_max, max_allowed)
      << "Labels are too large for this algorithm. Rescale to much less than " << max_allowed << ".";

// This is a weaker issue.  And (say) if have 0000011111 in labels, may not have taken gradient of that 1 case due to sampling, but that's ok.
//  CHECK_GT(abs_max, 1e-4f)
//      << "Labels are too small for this algorithm. Rescale to much more than 1E-4.";
}

struct GPUTrainingParam {
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;
  float learning_rate;

  GPUTrainingParam() = default;

  XGBOOST_DEVICE explicit GPUTrainingParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step),
        learning_rate{param.learning_rate} {}
};

using NodeIdT = int32_t;

/** used to assign default id to a Node */
static const bst_node_t kUnusedNode = -1;

/**
 * @enum DefaultDirection node.cuh
 * @brief Default direction to be followed in case of missing values
 */
enum DefaultDirection {
  /** move to left child */
  kLeftDir = 0,
  /** move to right child */
  kRightDir
};

struct DeviceSplitCandidate {
  float loss_chg {-FLT_MAX};
  DefaultDirection dir {kLeftDir};
  int findex {-1};
  float fvalue {0};
  bool is_cat { false };

  GradientPair left_sum;
  GradientPair right_sum;

  XGBOOST_DEVICE DeviceSplitCandidate() {}  // NOLINT

  template <typename ParamT>
  XGBOOST_DEVICE void Update(const DeviceSplitCandidate& other,
                             const ParamT& param) {
    if (other.loss_chg > loss_chg &&
        other.left_sum.GetHess() >= param.min_child_weight &&
        other.right_sum.GetHess() >= param.min_child_weight) {
      *this = other;
    }
  }

  XGBOOST_DEVICE void Update(float loss_chg_in, DefaultDirection dir_in,
                             float fvalue_in, int findex_in,
                             GradientPair left_sum_in,
                             GradientPair right_sum_in,
                             bool cat,
                             const GPUTrainingParam& param) {
    if (loss_chg_in > loss_chg &&
        left_sum_in.GetHess() >= param.min_child_weight &&
        right_sum_in.GetHess() >= param.min_child_weight) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = fvalue_in;
      is_cat = cat;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
    }
  }
  XGBOOST_DEVICE bool IsValid() const { return loss_chg > 0.0f; }

  friend std::ostream& operator<<(std::ostream& os, DeviceSplitCandidate const& c) {
    os << "loss_chg:" << c.loss_chg << ", "
       << "dir: " << c.dir << ", "
       << "findex: " << c.findex << ", "
       << "fvalue: " << c.fvalue << ", "
       << "is_cat: " << c.is_cat << ", "
       << "left sum: " << c.left_sum << ", "
       << "right sum: " << c.right_sum << std::endl;
    return os;
  }
};

struct DeviceSplitCandidateReduceOp {
  GPUTrainingParam param;
  explicit DeviceSplitCandidateReduceOp(GPUTrainingParam param) : param(std::move(param)) {}
  XGBOOST_DEVICE DeviceSplitCandidate operator()(
      const DeviceSplitCandidate& a, const DeviceSplitCandidate& b) const {
    DeviceSplitCandidate best;
    best.Update(a, param);
    best.Update(b, param);
    return best;
  }
};

template <typename T>
struct SumCallbackOp {
  // Running prefix
  T running_total;
  // Constructor
  XGBOOST_DEVICE SumCallbackOp() : running_total(T()) {}
  XGBOOST_DEVICE T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};
}  // namespace tree
}  // namespace xgboost
