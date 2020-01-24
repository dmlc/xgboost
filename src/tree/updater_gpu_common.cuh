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

  GPUTrainingParam() = default;

  XGBOOST_DEVICE explicit GPUTrainingParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step) {}
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
  float loss_chg;
  DefaultDirection dir;
  int findex;
  float fvalue;

  GradientPair left_sum;
  GradientPair right_sum;

  XGBOOST_DEVICE DeviceSplitCandidate()
      : loss_chg(-FLT_MAX), dir(kLeftDir), fvalue(0), findex(-1) {}

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
                             const GPUTrainingParam& param) {
    if (loss_chg_in > loss_chg &&
        left_sum_in.GetHess() >= param.min_child_weight &&
        right_sum_in.GetHess() >= param.min_child_weight) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = fvalue_in;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
    }
  }
  XGBOOST_DEVICE bool IsValid() const { return loss_chg > 0.0f; }
};

struct DeviceSplitCandidateReduceOp {
  GPUTrainingParam param;
  DeviceSplitCandidateReduceOp(GPUTrainingParam param) : param(param) {}
  XGBOOST_DEVICE DeviceSplitCandidate operator()(
      const DeviceSplitCandidate& a, const DeviceSplitCandidate& b) const {
    DeviceSplitCandidate best;
    best.Update(a, param);
    best.Update(b, param);
    return best;
  }
};

struct DeviceNodeStats {
  GradientPair sum_gradients;
  float root_gain;
  float weight;

  /** default direction for missing values */
  DefaultDirection dir;
  /** threshold value for comparison */
  float fvalue;
  GradientPair left_sum;
  GradientPair right_sum;
  /** \brief The feature index. */
  int fidx;
  /** node id (used as key for reduce/scan) */
  NodeIdT idx;

  HOST_DEV_INLINE DeviceNodeStats()
      : sum_gradients(),
        root_gain(-FLT_MAX),
        weight(-FLT_MAX),
        dir(kLeftDir),
        fvalue(0.f),
        left_sum(),
        right_sum(),
        fidx(kUnusedNode),
        idx(kUnusedNode) {}

  template <typename ParamT>
  HOST_DEV_INLINE DeviceNodeStats(GradientPair sum_gradients, NodeIdT nidx,
                                  const ParamT& param)
      : sum_gradients(sum_gradients),
        dir(kLeftDir),
        fvalue(0.f),
        fidx(kUnusedNode),
        idx(nidx) {
    this->root_gain =
        CalcGain(param, sum_gradients.GetGrad(), sum_gradients.GetHess());
    this->weight =
        CalcWeight(param, sum_gradients.GetGrad(), sum_gradients.GetHess());
  }

  HOST_DEV_INLINE void SetSplit(float fvalue, int fidx, DefaultDirection dir,
                                GradientPair left_sum, GradientPair right_sum) {
    this->fvalue = fvalue;
    this->fidx = fidx;
    this->dir = dir;
    this->left_sum = left_sum;
    this->right_sum = right_sum;
  }

  HOST_DEV_INLINE void SetSplit(const DeviceSplitCandidate& split) {
    this->SetSplit(split.fvalue, split.findex, split.dir, split.left_sum,
                   split.right_sum);
  }

  /** Tells whether this node is part of the decision tree */
  HOST_DEV_INLINE bool IsUnused() const { return (idx == kUnusedNode); }

  /** Tells whether this node is a leaf of the decision tree */
  HOST_DEV_INLINE bool IsLeaf() const {
    return (!IsUnused() && (fidx == kUnusedNode));
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

// Total number of nodes in tree, given depth
XGBOOST_DEVICE inline int MaxNodesDepth(int depth) {
  return (1 << (depth + 1)) - 1;
}

/*
 * Random
 */
struct BernoulliRng {
  float p;
  uint32_t seed;

  XGBOOST_DEVICE BernoulliRng(float p, size_t seed_) : p(p) {
    seed = static_cast<uint32_t>(seed_);
  }

  XGBOOST_DEVICE bool operator()(const int i) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng) <= p;
  }
};

// Set gradient pair to 0 with p = 1 - subsample
inline void SubsampleGradientPair(int device_idx,
                                  common::Span<GradientPair> d_gpair,
                                  float subsample, int offset = 0) {
  if (subsample == 1.0) {
    return;
  }

  BernoulliRng rng(subsample, common::GlobalRandom()());

  dh::LaunchN(device_idx, d_gpair.size(), [=] XGBOOST_DEVICE(int i) {
    if (!rng(i + offset)) {
      d_gpair[i] = GradientPair();
    }
  });
}

}  // namespace tree
}  // namespace xgboost
