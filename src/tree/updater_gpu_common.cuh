/*!
 * Copyright 2017 XGBoost contributors
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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
XGBOOST_DEVICE __forceinline__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull =
      (unsigned long long int*)address;                   // NOLINT
  unsigned long long int old = *address_as_ull, assumed;  // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

namespace xgboost {
namespace tree {

// Atomic add function for double precision gradients
__device__ __forceinline__ void AtomicAddGpair(GradientPairPrecise* dest,
                                               const GradientPair& gpair) {
  auto dst_ptr = reinterpret_cast<double*>(dest);

  atomicAdd(dst_ptr, static_cast<double>(gpair.GetGrad()));
  atomicAdd(dst_ptr + 1, static_cast<double>(gpair.GetHess()));
}
// used by shared-memory atomics code
__device__ __forceinline__ void AtomicAddGpair(GradientPairPrecise* dest,
                                               const GradientPairPrecise& gpair) {
  auto dst_ptr = reinterpret_cast<double*>(dest);

  atomicAdd(dst_ptr, gpair.GetGrad());
  atomicAdd(dst_ptr + 1, gpair.GetHess());
}

// For integer gradients
__device__ __forceinline__ void AtomicAddGpair(GradientPairInteger* dest,
                                               const GradientPair& gpair) {
  auto dst_ptr = reinterpret_cast<unsigned long long int*>(dest);  // NOLINT
  GradientPairInteger tmp(gpair.GetGrad(), gpair.GetHess());
  auto src_ptr = reinterpret_cast<GradientPairInteger::ValueT*>(&tmp);

  atomicAdd(dst_ptr,
            static_cast<unsigned long long int>(*src_ptr));  // NOLINT
  atomicAdd(dst_ptr + 1,
            static_cast<unsigned long long int>(*(src_ptr + 1)));  // NOLINT
}

/**
 * \brief Check maximum gradient value is below 2^16. This is to prevent
 * overflow when using integer gradient summation.
 */

inline void CheckGradientMax(const std::vector<GradientPair>& gpair) {
  auto* ptr = reinterpret_cast<const float*>(gpair.data());
  float abs_max =
      std::accumulate(ptr, ptr + (gpair.size() * 2), 0.f,
                      [=](float a, float b) { return max(abs(a), abs(b)); });

  CHECK_LT(abs_max, std::pow(2.0f, 16.0f))
      << "Labels are too large for this algorithm. Rescale to less than 2^16.";
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

using NodeIdT = int;

/** used to assign default id to a Node */
static const int kUnusedNode = -1;

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
  float fvalue;
  int findex;
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

template <typename GradientPairT>
XGBOOST_DEVICE inline float DeviceCalcLossChange(const GPUTrainingParam& param,
                                             const GradientPairT& left,
                                             const GradientPairT& parent_sum,
                                             const float& parent_gain) {
  GradientPairT right = parent_sum - left;
  float left_gain = CalcGain(param, left.GetGrad(), left.GetHess());
  float right_gain = CalcGain(param, right.GetGrad(), right.GetHess());
  return left_gain + right_gain - parent_gain;
}

// Without constraints
template <typename GradientPairT>
XGBOOST_DEVICE float inline LossChangeMissing(const GradientPairT& scan,
                                         const GradientPairT& missing,
                                         const GradientPairT& parent_sum,
                                         const float& parent_gain,
                                         const GPUTrainingParam& param,
                                         bool& missing_left_out) {  // NOLINT
  float missing_left_loss =
      DeviceCalcLossChange(param, scan + missing, parent_sum, parent_gain);
  float missing_right_loss =
      DeviceCalcLossChange(param, scan, parent_sum, parent_gain);

  if (missing_left_loss >= missing_right_loss) {
    missing_left_out = true;
    return missing_left_loss;
  } else {
    missing_left_out = false;
    return missing_right_loss;
  }
}

// With constraints
template <typename GradientPairT>
XGBOOST_DEVICE float inline LossChangeMissing(
    const GradientPairT& scan, const GradientPairT& missing, const GradientPairT& parent_sum,
    const float& parent_gain, const GPUTrainingParam& param, int constraint,
    const ValueConstraint& value_constraint,
    bool& missing_left_out) {  // NOLINT
  float missing_left_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan + missing),
      GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain >= missing_right_gain) {
    missing_left_out = true;
    return missing_left_gain - parent_gain;
  } else {
    missing_left_out = false;
    return missing_right_gain - parent_gain;
  }
}

// Total number of nodes in tree, given depth
XGBOOST_DEVICE inline int MaxNodesDepth(int depth) {
  return (1 << (depth + 1)) - 1;
}

// Number of nodes at this level of the tree
XGBOOST_DEVICE inline int MaxNodesLevel(int depth) { return 1 << depth; }

// Whether a node is currently being processed at current depth
XGBOOST_DEVICE inline bool IsNodeActive(int nidx, int depth) {
  return nidx >= MaxNodesDepth(depth - 1);
}

XGBOOST_DEVICE inline int ParentNodeIdx(int nidx) { return (nidx - 1) / 2; }

XGBOOST_DEVICE inline int LeftChildNodeIdx(int nidx) {
  return nidx * 2 + 1;
}

XGBOOST_DEVICE inline int RightChildNodeIdx(int nidx) {
  return nidx * 2 + 2;
}

XGBOOST_DEVICE inline bool IsLeftChild(int nidx) {
  return nidx % 2 == 1;
}

// Copy gpu dense representation of tree to xgboost sparse representation
inline void Dense2SparseTree(RegTree* p_tree,
                              const dh::DVec<DeviceNodeStats>& nodes,
                              const TrainParam& param) {
  RegTree& tree = *p_tree;
  std::vector<DeviceNodeStats> h_nodes = nodes.AsVector();

  int nid = 0;
  for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
    const DeviceNodeStats& n = h_nodes[gpu_nid];
    if (!n.IsUnused() && !n.IsLeaf()) {
      tree.AddChilds(nid);
      tree[nid].SetSplit(n.fidx, n.fvalue, n.dir == kLeftDir);
      tree.Stat(nid).loss_chg = n.root_gain;
      tree.Stat(nid).base_weight = n.weight;
      tree.Stat(nid).sum_hess = n.sum_gradients.GetHess();
      tree[tree[nid].LeftChild()].SetLeaf(0);
      tree[tree[nid].RightChild()].SetLeaf(0);
      nid++;
    } else if (n.IsLeaf()) {
      tree[nid].SetLeaf(n.weight * param.learning_rate);
      tree.Stat(nid).sum_hess = n.sum_gradients.GetHess();
      nid++;
    }
  }
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
inline void SubsampleGradientPair(dh::DVec<GradientPair>* p_gpair, float subsample,
                            int offset = 0) {
  if (subsample == 1.0) {
    return;
  }

  dh::DVec<GradientPair>& gpair = *p_gpair;

  auto d_gpair = gpair.Data();
  BernoulliRng rng(subsample, common::GlobalRandom()());

  dh::LaunchN(gpair.DeviceIdx(), gpair.Size(), [=] XGBOOST_DEVICE(int i) {
    if (!rng(i + offset)) {
      d_gpair[i] = GradientPair();
    }
  });
}

inline std::vector<int> ColSample(std::vector<int> features, float colsample) {
  CHECK_GT(features.size(), 0);
  int n = std::max(1, static_cast<int>(colsample * features.size()));

  std::shuffle(features.begin(), features.end(), common::GlobalRandom());
  features.resize(n);
  std::sort(features.begin(), features.end());

  return features;
}

/**
 * \class ColumnSampler
 *
 * \brief Handles selection of columns due to colsample_bytree and
 * colsample_bylevel parameters. Should be initialised the before tree
 * construction and to reset When tree construction is completed.
 */

class ColumnSampler {
  std::vector<int> feature_set_tree_;
  std::map<int, std::vector<int>> feature_set_level_;
  TrainParam param_;

 public:
  /**
   * \fn  void Init(int64_t num_col, const TrainParam& param)
   *
   * \brief Initialise this object before use.
   *
   * \param num_col Number of cols.
   * \param param   The parameter.
   */

  void Init(int64_t num_col, const TrainParam& param) {
    this->Reset();
    this->param_ = param;
    feature_set_tree_.resize(num_col);
    std::iota(feature_set_tree_.begin(), feature_set_tree_.end(), 0);
    feature_set_tree_ = ColSample(feature_set_tree_, param.colsample_bytree);
  }

  /**
   * \fn  void Reset()
   *
   * \brief Resets this object.
   */

  void Reset() {
    feature_set_tree_.clear();
    feature_set_level_.clear();
  }

  /**
   * \fn  bool ColumnUsed(int column, int depth)
   *
   * \brief Whether the current column should be considered as a split.
   *
   * \param column  The column index.
   * \param depth   The current tree depth.
   *
   * \return  True if it should be used, false if it should not be used.
   */

  bool ColumnUsed(int column, int depth) {
    if (feature_set_level_.count(depth) == 0) {
      feature_set_level_[depth] =
          ColSample(feature_set_tree_, param_.colsample_bylevel);
    }

    return std::binary_search(feature_set_level_[depth].begin(),
                              feature_set_level_[depth].end(), column);
  }
};

}  // namespace tree
}  // namespace xgboost
