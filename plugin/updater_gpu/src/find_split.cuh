/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <cub/cub.cuh>
#include <xgboost/base.h>
#include "device_helpers.cuh"
#include "find_split_multiscan.cuh"
#include "find_split_sorting.cuh"
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

void find_split(const ItemIter items_iter, Split *d_split_candidates,
                Node *d_nodes, bst_uint num_items, int num_features,
                const int *d_feature_offsets, gpu_gpair *d_node_sums,
                int *d_node_offsets, const GPUTrainingParam param,
                const int level, bool multiscan_algorithm) {
  if (multiscan_algorithm) {
    find_split_candidates_multiscan(items_iter, d_split_candidates, d_nodes,
                                    num_items, num_features, d_feature_offsets,
                                    param, level);
  } else {
    find_split_candidates_sorted(items_iter, d_split_candidates, d_nodes,
                                 num_items, num_features, d_feature_offsets,
                                 d_node_sums, d_node_offsets, param, level);
  }

  // Find the best split for each node
  reduce_split_candidates(d_split_candidates, d_nodes, level, num_features,
                          param);
}
}  // namespace tree
}  // namespace xgboost
