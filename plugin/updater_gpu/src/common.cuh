/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <vector>
#include "../../../src/common/random.h"
#include "../../../src/tree/param.h"
#include "device_helpers.cuh"
#include "types.cuh"

namespace xgboost {
namespace tree {
// When we split on a value which has no left neighbour, define its left
// neighbour as having left_fvalue = current_fvalue - FVALUE_EPS
// This produces a split value slightly lower than the current instance
#define FVALUE_EPS 0.0001

__device__ inline float device_calc_loss_chg(const GPUTrainingParam& param,
                                             const gpu_gpair& scan,
                                             const gpu_gpair& missing,
                                             const gpu_gpair& parent_sum,
                                             const float& parent_gain,
                                             bool missing_left) {
  gpu_gpair left = scan;

  if (missing_left) {
    left += missing;
  }

  gpu_gpair right = parent_sum - left;

  float left_gain = CalcGain(param, left.grad(), left.hess());
  float right_gain = CalcGain(param, right.grad(), right.hess());
  return left_gain + right_gain - parent_gain;
}

__device__ float inline loss_chg_missing(const gpu_gpair& scan,
                                         const gpu_gpair& missing,
                                         const gpu_gpair& parent_sum,
                                         const float& parent_gain,
                                         const GPUTrainingParam& param,
                                         bool& missing_left_out) {  // NOLINT
  float missing_left_loss =
      device_calc_loss_chg(param, scan, missing, parent_sum, parent_gain, true);
  float missing_right_loss = device_calc_loss_chg(
      param, scan, missing, parent_sum, parent_gain, false);

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

enum NodeType {
  NODE = 0,
  LEAF = 1,
  UNUSED = 2,
};

// Recursively label node types
inline void flag_nodes(const thrust::host_vector<Node>& nodes,
                       std::vector<NodeType>* node_flags, int nid,
                       NodeType type) {
  if (nid >= nodes.size() || type == UNUSED) {
    return;
  }

  const Node& n = nodes[nid];

  // Current node and all children are valid
  if (n.split.loss_chg > rt_eps) {
    (*node_flags)[nid] = NODE;
    flag_nodes(nodes, node_flags, nid * 2 + 1, NODE);
    flag_nodes(nodes, node_flags, nid * 2 + 2, NODE);
  } else {
    // Current node is leaf, therefore is valid but all children are invalid
    (*node_flags)[nid] = LEAF;
    flag_nodes(nodes, node_flags, nid * 2 + 1, UNUSED);
    flag_nodes(nodes, node_flags, nid * 2 + 2, UNUSED);
  }
}

// Copy gpu dense representation of tree to xgboost sparse representation
inline void dense2sparse_tree(RegTree* p_tree,
                              thrust::device_ptr<Node> nodes_begin,
                              thrust::device_ptr<Node> nodes_end,
                              const TrainParam& param) {
  RegTree& tree = *p_tree;
  thrust::host_vector<Node> h_nodes(nodes_begin, nodes_end);
  std::vector<NodeType> node_flags(h_nodes.size(), UNUSED);
  flag_nodes(h_nodes, &node_flags, 0, NODE);

  int nid = 0;
  for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
    NodeType flag = node_flags[gpu_nid];
    const Node& n = h_nodes[gpu_nid];
    if (flag == NODE) {
      tree.AddChilds(nid);
      tree[nid].set_split(n.split.findex, n.split.fvalue, n.split.missing_left);
      tree.stat(nid).loss_chg = n.split.loss_chg;
      tree.stat(nid).base_weight = n.weight;
      tree.stat(nid).sum_hess = n.sum_gradients.hess();
      tree[tree[nid].cleft()].set_leaf(0);
      tree[tree[nid].cright()].set_leaf(0);
      nid++;
    } else if (flag == LEAF) {
      tree[nid].set_leaf(n.weight * param.learning_rate);
      tree.stat(nid).sum_hess = n.sum_gradients.hess();
      nid++;
    }
  }
}

// Set gradient pair to 0 with p = 1 - subsample
inline void subsample_gpair(dh::dvec<gpu_gpair>* p_gpair, float subsample) {
  if (subsample == 1.0) {
    return;
  }

  dh::dvec<gpu_gpair>& gpair = *p_gpair;

  auto d_gpair = gpair.data();
  dh::BernoulliRng rng(subsample, common::GlobalRandom()());

  dh::launch_n(gpair.size(), [=] __device__(int i) {
    if (!rng(i)) {
      d_gpair[i] = gpu_gpair();
    }
  });
}

inline std::vector<int> col_sample(std::vector<int> features, float colsample) {
  int n = colsample * features.size();
  CHECK_GT(n, 0);

  std::shuffle(features.begin(), features.end(), common::GlobalRandom());
  features.resize(n);

  return features;
}
}  // namespace tree
}  // namespace xgboost
