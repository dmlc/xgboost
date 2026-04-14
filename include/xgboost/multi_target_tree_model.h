/**
 * Copyright 2023-2026, XGBoost contributors
 *
 * @brief Core data structure for multi-target trees.
 */
#ifndef XGBOOST_MULTI_TARGET_TREE_MODEL_H_
#define XGBOOST_MULTI_TARGET_TREE_MODEL_H_

#include <xgboost/base.h>                // for bst_node_t, bst_target_t, bst_feature_t
#include <xgboost/context.h>             // for Context
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/linalg.h>              // for VectorView, MatrixView
#include <xgboost/model.h>               // for Model
#include <xgboost/span.h>                // for Span

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t
#include <vector>   // for vector

namespace xgboost {
namespace tree {
struct MultiTargetTreeView;
}
struct TreeParam;

/**
 * @brief Tree structure for multi-target model.
 *
 * In order to support reduced gradient, the internal storage distinguishes weights
 * between base weights and leaf weights. The former is the weight calculated from split
 * gradient, and the later is the weight calculated from value gradient and used as
 * outputs. Every node has a base weight, but only leaves have leaf weights.
 *
 * To access the leaf weights, we re-use the right child to store leaf indices. For split
 * nodes, the `right_` member stores their right child node indices, for leaf nodes, the
 * `right_` member stores the corresponding leaf weight indices.
 */
class MultiTargetTree : public Model {
 public:
  static bst_node_t constexpr InvalidNodeId() { return -1; }
  friend struct tree::MultiTargetTreeView;

 private:
  TreeParam const* param_;
  // Mapping from node index to its left child. -1 for a leaf node.
  HostDeviceVector<bst_node_t> left_;
  // Mapping from node index to its right child. Maps to leaf weight for a leaf node.
  HostDeviceVector<bst_node_t> right_;
  // Mapping from node index to its parent.
  HostDeviceVector<bst_node_t> parent_;
  // Feature index for node split.
  HostDeviceVector<bst_feature_t> split_index_;
  // Whether the left child is the default node when split feature is missing.
  HostDeviceVector<std::uint8_t> default_left_;
  // Threshold for splitting a node.
  HostDeviceVector<float> split_conds_;
  // Internal base weights.
  HostDeviceVector<float> weights_;
  // Output weights.
  HostDeviceVector<float> leaf_weights_;
  // Loss change for each node.
  HostDeviceVector<float> loss_chg_;
  // Sum of hessians for each node (coverage).
  HostDeviceVector<float> sum_hess_;

  [[nodiscard]] linalg::VectorView<float const> NodeWeight(bst_node_t nidx) const {
    auto beg = nidx * this->NumSplitTargets();
    auto v = this->weights_.ConstHostSpan().subspan(beg, this->NumSplitTargets());
    return linalg::MakeTensorView(DeviceOrd::CPU(), v, v.size());
  }
  // Unlike the const version, `NumSplitTargets` is not reliable if the tree can change.
  [[nodiscard]] linalg::VectorView<float> NodeWeight(bst_node_t nidx,
                                                     bst_target_t n_split_targets) {
    auto beg = nidx * n_split_targets;
    auto v = this->weights_.HostSpan().subspan(beg, n_split_targets);
    return linalg::MakeTensorView(DeviceOrd::CPU(), v, v.size());
  }
  [[nodiscard]] bst_node_t LeafIdx(bst_node_t nidx) const { return this->RightChild(nidx); }

 public:
  explicit MultiTargetTree(TreeParam const* param);
  MultiTargetTree(MultiTargetTree const& that);
  MultiTargetTree& operator=(MultiTargetTree const& that) = delete;
  MultiTargetTree(MultiTargetTree&& that) = delete;
  MultiTargetTree& operator=(MultiTargetTree&& that) = delete;

  /**
   * @brief Set the weight and statistics for the root.
   *
   * @param weight   The weight vector for the root node.
   * @param sum_hess The sum of hessians for the root node (coverage).
   */
  void SetRoot(linalg::VectorView<float const> weight, float sum_hess);
  /**
   * @brief Expand a leaf into split node.
   */
  void Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond, bool default_left,
              linalg::VectorView<float const> base_weight,
              linalg::VectorView<float const> left_weight,
              linalg::VectorView<float const> right_weight, float loss_chg, float sum_hess,
              float left_sum, float right_sum);
  /** @see RegTree::SetLeaves */
  void SetLeaves(std::vector<bst_node_t> leaves, common::Span<float const> weights);
  /** @brief Copy base weight into leaf weight for a non-reduced multi-target tree. */
  void SetLeaves();

  [[nodiscard]] bool IsLeaf(bst_node_t nidx) const {
    return left_.ConstHostVector()[nidx] == InvalidNodeId();
  }
  [[nodiscard]] bst_node_t LeftChild(bst_node_t nidx) const {
    return left_.ConstHostVector().at(nidx);
  }
  [[nodiscard]] bst_node_t RightChild(bst_node_t nidx) const {
    return right_.ConstHostVector().at(nidx);
  }
  /**
   * @brief Number of targets (size of a leaf).
   */
  [[nodiscard]] bst_target_t NumTargets() const;
  /**
   * @brief Number of reduced targets.
   */
  [[nodiscard]] bst_target_t NumSplitTargets() const;
  [[nodiscard]] auto NumLeaves() const { return this->leaf_weights_.Size() / this->NumTargets(); }

  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] MultiTargetTree* Copy(TreeParam const* param) const;

  common::Span<float const> LeafWeights(DeviceOrd device) const {
    if (device.IsCPU()) {
      return this->leaf_weights_.ConstHostSpan();
    }
    this->leaf_weights_.SetDevice(device);
    return this->leaf_weights_.ConstDeviceSpan();
  }

  [[nodiscard]] linalg::VectorView<float const> LeafValue(bst_node_t nidx) const {
    CHECK(IsLeaf(nidx));
    auto n_targets = this->NumTargets();
    auto h_leaf_mapping = this->right_.ConstHostSpan();
    auto h_leaf_weights = this->leaf_weights_.ConstHostSpan();
    auto lidx = h_leaf_mapping[nidx];
    CHECK_NE(lidx, InvalidNodeId());
    auto weight = h_leaf_weights.subspan(lidx * n_targets, n_targets);
    return linalg::MakeVec(DeviceOrd::CPU(), weight);
  }

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;

  [[nodiscard]] std::size_t MemCostBytes() const;
};
}  // namespace xgboost
#endif  // XGBOOST_MULTI_TARGET_TREE_MODEL_H_
