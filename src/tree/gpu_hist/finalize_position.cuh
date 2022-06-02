/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#pragma once
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/span.h"
#include "../../data/ellpack_page.cuh"

namespace xgboost {
namespace tree {

template <int kBlockSize, bool kUseShared>
__device__ const RegTree::Node *LoadTree(common::Span<const RegTree::Node> d_nodes, int *smem) {
  if (!kUseShared) {
    return d_nodes.data();
  }

  auto nodes = reinterpret_cast<RegTree::Node*>(smem);
  for (int i = threadIdx.x; i < d_nodes.size(); i += kBlockSize) {
      nodes[i]=d_nodes[i];
  }
  __syncthreads();
  return nodes;
}

template <int kBlockSize, bool kUseShared>
__global__ __launch_bounds__(kBlockSize) void FinalizePositionKernel(
    common::Span<const RegTree::Node> d_nodes, common::Span<const FeatureType> feature_types,
    common::Span<const uint32_t> categories,
    common::Span<const RegTree::Segment> categories_segments,
    common::Span<const GradientPair> gradients, const EllpackDeviceAccessor dmatrix,
    common::Span<bst_float> predictions, common::Span<bst_node_t> position) {
  extern __shared__ int s[];
  auto nodes = LoadTree<kBlockSize,kUseShared>(d_nodes, s);
  auto new_position_op = [&] __device__(size_t row_id) {
    // What happens if user prune the tree?
    if (!dmatrix.IsInRange(row_id)) {
      return -1;
    }
    int row_position = RegTree::kRoot;
    auto node = nodes[row_position];

    while (!node.IsLeaf()) {
      bst_float element = dmatrix.GetFvalue(row_id, node.SplitIndex());
      // Missing value
      if (isnan(element)) {
        row_position = node.DefaultChild();
      } else {
        bool go_left = true;
        if (common::IsCat(feature_types, row_position)) {
          auto node_cats = categories.subspan(categories_segments[row_position].beg,
                                              categories_segments[row_position].size);
          go_left = common::Decision<false>(node_cats, element, node.DefaultLeft());
        } else {
          go_left = element <= node.SplitCond();
        }
        if (go_left) {
          row_position = node.LeftChild();
        } else {
          row_position = node.RightChild();
        }
      }
      node = nodes[row_position];
    }

    return row_position;
  };  // NOLINT

  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < position.size();
       idx += blockDim.x * gridDim.x) {
      bst_node_t row_position = new_position_op(idx);
      predictions[idx] = nodes[row_position].LeafValue();
      // FIXME(jiamingy): Doesn't work when sampling is used with external memory as
      // the sampler compacts the gradient vector.
      bool is_sampled = gradients[idx].GetHess() - .0f == 0.f;
      position[idx] = is_sampled ? ~row_position : row_position;
  }
}

inline void CallFinalizePosition(common::Span<const RegTree::Node> nodes,
                              common::Span<const FeatureType> feature_types,
                              common::Span<const uint32_t> categories,
                              common::Span<const RegTree::Segment> categories_segments,
                              common::Span<const GradientPair> gradients,
                              const EllpackDeviceAccessor dmatrix,
                              common::Span<bst_float> predictions,
                              common::Span<bst_node_t> position){

  // Use shared memory?
  int device = 0;
  dh::safe_cuda(cudaGetDevice(&device));
  int max_shared_memory = dh::MaxSharedMemoryOptin(device);
  size_t smem_size = sizeof( RegTree::Node) *
                     nodes.size();
  bool shared = smem_size <= max_shared_memory;
  smem_size = shared ? smem_size : 0;
  constexpr int kBlockSize = 256;
  const int grid_size =
      std::min(256, static_cast<int>(xgboost::common::DivRoundUp(position.size(), kBlockSize)));

  if (shared) {
    FinalizePositionKernel<kBlockSize, true>
        <<<grid_size, kBlockSize, smem_size>>>(nodes, feature_types, categories, categories_segments,
                                    gradients, dmatrix, predictions, position);
  } else {
    FinalizePositionKernel<kBlockSize, false>
        <<<grid_size, kBlockSize, smem_size>>>(nodes, feature_types, categories, categories_segments,
                                    gradients, dmatrix, predictions, position);
  }
}
};  // namespace tree
};  // namespace xgboost