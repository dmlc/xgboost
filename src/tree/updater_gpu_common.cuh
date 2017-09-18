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

namespace xgboost {
namespace tree {

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

  __host__ __device__ GPUTrainingParam() {}

  __host__ __device__ GPUTrainingParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step) {}
};

typedef int node_id_t;

/** used to assign default id to a Node */
static const int UNUSED_NODE = -1;

/**
 * @enum DefaultDirection node.cuh
 * @brief Default direction to be followed in case of missing values
 */
enum DefaultDirection {
  /** move to left child */
  LeftDir = 0,
  /** move to right child */
  RightDir
};

struct DeviceSplitCandidate {
  float loss_chg;
  DefaultDirection dir;
  float fvalue;
  int findex;
  bst_gpair_integer left_sum;
  bst_gpair_integer right_sum;

  __host__ __device__ DeviceSplitCandidate()
      : loss_chg(-FLT_MAX), dir(LeftDir), fvalue(0), findex(-1) {}

  template <typename param_t>
  __host__ __device__ void Update(const DeviceSplitCandidate &other,
                         const param_t& param) {
    if (other.loss_chg > loss_chg &&
        other.left_sum.GetHess() >= param.min_child_weight &&
        other.right_sum.GetHess() >= param.min_child_weight) {
      *this = other;
    }
  }

  __device__ void Update(float loss_chg_in, DefaultDirection dir_in,
                         float fvalue_in, int findex_in,
                         bst_gpair_integer left_sum_in,
                         bst_gpair_integer right_sum_in,
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
  __device__ bool IsValid() const { return loss_chg > 0.0f; }
};

struct DeviceNodeStats {
  bst_gpair sum_gradients;
  float root_gain;
  float weight;

  /** default direction for missing values */
  DefaultDirection dir;
  /** threshold value for comparison */
  float fvalue;
  bst_gpair left_sum;
  bst_gpair right_sum;
  /** \brief The feature index. */
  int fidx;
  /** node id (used as key for reduce/scan) */
  node_id_t idx;

  HOST_DEV_INLINE DeviceNodeStats()
      : sum_gradients(),
        root_gain(-FLT_MAX),
        weight(-FLT_MAX),
        dir(LeftDir),
        fvalue(0.f),
        left_sum(),
        right_sum(),
        fidx(UNUSED_NODE),
        idx(UNUSED_NODE) {}

  template <typename param_t>
  HOST_DEV_INLINE DeviceNodeStats(bst_gpair sum_gradients, node_id_t nidx,
                                  const param_t& param)
      : sum_gradients(sum_gradients),
        dir(LeftDir),
        fvalue(0.f),
        fidx(UNUSED_NODE),
        idx(nidx) {
    this->root_gain =
        CalcGain(param, sum_gradients.GetGrad(), sum_gradients.GetHess());
    this->weight =
        CalcWeight(param, sum_gradients.GetGrad(), sum_gradients.GetHess());
  }

  HOST_DEV_INLINE void SetSplit(float fvalue, int fidx, DefaultDirection dir,
                                bst_gpair left_sum, bst_gpair right_sum) {
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
  HOST_DEV_INLINE bool IsUnused() const { return (idx == UNUSED_NODE); }

  /** Tells whether this node is a leaf of the decision tree */
  HOST_DEV_INLINE bool IsLeaf() const {
    return (!IsUnused() && (fidx == UNUSED_NODE));
  }
};

template <typename T>
struct SumCallbackOp {
  // Running prefix
  T running_total;
  // Constructor
  __device__ SumCallbackOp() : running_total(T()) {}
  __device__ T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename gpair_t>
__device__ inline float device_calc_loss_chg(
    const GPUTrainingParam& param, const gpair_t&  left, const gpair_t& parent_sum, const float& parent_gain) {
  gpair_t right = parent_sum - left;
  float left_gain = CalcGain(param, left.GetGrad(), left.GetHess());
  float right_gain = CalcGain(param, right.GetGrad(), right.GetHess());
  return left_gain + right_gain - parent_gain;
}

template <typename gpair_t>
__device__ float inline loss_chg_missing(const gpair_t& scan,
                                         const gpair_t& missing,
                                         const gpair_t& parent_sum,
                                         const float& parent_gain,
                                         const GPUTrainingParam& param,
                                         bool& missing_left_out) {  // NOLINT
  float missing_left_loss =
      device_calc_loss_chg(param, scan + missing, parent_sum, parent_gain);
  float missing_right_loss = device_calc_loss_chg(
      param, scan, parent_sum, parent_gain);

  if (missing_left_loss >= missing_right_loss) {
    missing_left_out = true;
    return missing_left_loss;
  } else {
    missing_left_out = false;
    return missing_right_loss;
  }
}

// Total number of nodes in tree, given depth
__host__ __device__ inline int n_nodes(int depth) {
  return (1 << (depth + 1)) - 1;
}

// Number of nodes at this level of the tree
__host__ __device__ inline int n_nodes_level(int depth) { return 1 << depth; }

// Whether a node is currently being processed at current depth
__host__ __device__ inline bool is_active(int nidx, int depth) {
  return nidx >= n_nodes(depth - 1);
}

__host__ __device__ inline int parent_nidx(int nidx) { return (nidx - 1) / 2; }

__host__ __device__ inline int left_child_nidx(int nidx) {
  return nidx * 2 + 1;
}

__host__ __device__ inline int right_child_nidx(int nidx) {
  return nidx * 2 + 2;
}

__host__ __device__ inline bool is_left_child(int nidx) {
  return nidx % 2 == 1;
}

// Copy gpu dense representation of tree to xgboost sparse representation
inline void dense2sparse_tree(RegTree* p_tree,
                              const dh::dvec<DeviceNodeStats>& nodes,
                              const TrainParam& param) {
  RegTree& tree = *p_tree;
  std::vector<DeviceNodeStats> h_nodes = nodes.as_vector();

  int nid = 0;
  for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
    const DeviceNodeStats& n = h_nodes[gpu_nid];
    if (!n.IsUnused() && !n.IsLeaf()) {
      tree.AddChilds(nid);
      tree[nid].set_split(n.fidx, n.fvalue, n.dir == LeftDir);
      tree.stat(nid).loss_chg = n.root_gain;
      tree.stat(nid).base_weight = n.weight;
      tree.stat(nid).sum_hess = n.sum_gradients.GetHess();
      tree[tree[nid].cleft()].set_leaf(0);
      tree[tree[nid].cright()].set_leaf(0);
      nid++;
    } else if (n.IsLeaf()) {
      tree[nid].set_leaf(n.weight * param.learning_rate);
      tree.stat(nid).sum_hess = n.sum_gradients.GetHess();
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

  __host__ __device__ BernoulliRng(float p, size_t seed_) : p(p) {
    seed = static_cast<uint32_t>(seed_);
  }

  __host__ __device__ bool operator()(const int i) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng) <= p;
  }
};

// Set gradient pair to 0 with p = 1 - subsample
inline void subsample_gpair(dh::dvec<bst_gpair>* p_gpair, float subsample,
                            int offset = 0) {
  if (subsample == 1.0) {
    return;
  }

  dh::dvec<bst_gpair>& gpair = *p_gpair;

  auto d_gpair = gpair.data();
  BernoulliRng rng(subsample, common::GlobalRandom()());

  dh::launch_n(gpair.device_idx(), gpair.size(), [=] __device__(int i) {
    if (!rng(i + offset)) {
      d_gpair[i] = bst_gpair();
    }
  });
}

inline std::vector<int> col_sample(std::vector<int> features, float colsample) {
  CHECK_GT(features.size(), 0);
  int n = std::max(1, static_cast<int>(colsample * features.size()));

  std::shuffle(features.begin(), features.end(), common::GlobalRandom());
  features.resize(n);

  return features;
}
}  // namespace tree
}  // namespace xgboost
