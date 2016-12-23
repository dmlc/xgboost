/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <cub/cub.cuh>
#include <xgboost/base.h>
#include <vector>
#include "device_helpers.cuh"
#include "find_split_multiscan.cuh"
#include "find_split_sorting.cuh"
#include "gpu_data.cuh"
#include "types_functions.cuh"

namespace xgboost {
namespace tree {

__global__ void
reduce_split_candidates_kernel(Split *d_split_candidates, Node *d_current_nodes,
                               Node *d_new_nodes, int n_current_nodes,
                               int n_features, const GPUTrainingParam param) {
  int nid = blockIdx.x * blockDim.x + threadIdx.x;

  if (nid >= n_current_nodes) {
    return;
  }

  // Find best split for each node
  Split best;

  for (int i = 0; i < n_features; i++) {
    best.Update(d_split_candidates[n_current_nodes * i + nid]);
  }

  // Update current node
  d_current_nodes[nid].split = best;

  // Generate new nodes
  d_new_nodes[nid * 2] =
      Node(best.left_sum,
           CalcGain(param, best.left_sum.grad(), best.left_sum.hess()),
           CalcWeight(param, best.left_sum.grad(), best.left_sum.hess()));
  d_new_nodes[nid * 2 + 1] =
      Node(best.right_sum,
           CalcGain(param, best.right_sum.grad(), best.right_sum.hess()),
           CalcWeight(param, best.right_sum.grad(), best.right_sum.hess()));
}

void reduce_split_candidates(Split *d_split_candidates, Node *d_nodes,
                             int level, int n_features,
                             const GPUTrainingParam param) {
  // Current level nodes (before split)
  Node *d_current_nodes = d_nodes + (1 << (level)) - 1;
  // Next level nodes (after split)
  Node *d_new_nodes = d_nodes + (1 << (level + 1)) - 1;
  // Number of existing nodes on this level
  int n_current_nodes = 1 << level;

  const int BLOCK_THREADS = 512;
  const int GRID_SIZE = dh::div_round_up(n_current_nodes, BLOCK_THREADS);

  reduce_split_candidates_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(
      d_split_candidates, d_current_nodes, d_new_nodes, n_current_nodes,
      n_features, param);
  dh::safe_cuda(cudaDeviceSynchronize());
}

void colsample_level(GPUData *data, const TrainParam xgboost_param,
                     const std::vector<int> &feature_set_tree,
                     std::vector<int> *feature_set_level) {
  unsigned n_bytree =
      static_cast<unsigned>(xgboost_param.colsample_bytree * data->n_features);
  unsigned n =
      static_cast<unsigned>(n_bytree * xgboost_param.colsample_bylevel);
  CHECK_GT(n, 0);

  *feature_set_level = feature_set_tree;

  std::shuffle((*feature_set_level).begin(),
               (*feature_set_level).begin() + n_bytree, common::GlobalRandom());

  data->feature_set = *feature_set_level;

  data->feature_flags.fill(0);
  auto d_feature_set = data->feature_set.data();
  auto d_feature_flags = data->feature_flags.data();

  dh::launch_n(
      n, [=] __device__(int i) { d_feature_flags[d_feature_set[i]] = 1; });
}

void find_split(GPUData *data, const TrainParam xgboost_param, const int level,
                bool multiscan_algorithm,
                const std::vector<int> &feature_set_tree,
                std::vector<int> *feature_set_level) {
  colsample_level(data, xgboost_param, feature_set_tree, feature_set_level);
  // Reset split candidates
  data->split_candidates.fill(Split());

  if (multiscan_algorithm) {
    find_split_candidates_multiscan(data, level);
  } else {
    find_split_candidates_sorted(data, level);
  }

  // Find the best split for each node
  reduce_split_candidates(data->split_candidates.data(), data->nodes.data(),
                          level, data->n_features, data->param);
}
}  // namespace tree
}  // namespace xgboost
