/**
 * Copyright 2023-2025, XGBoost contributors
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
#include <mutex>    // for mutex
#include <vector>   // for vector

namespace xgboost {
struct TreeParam;
/**
 * @brief A view to the @MultiTargetTree suitable for both host and device.
 */
struct MultiTargetTreeView {
  static bst_node_t constexpr InvalidNodeId() { return -1; }

  bst_node_t const* left;
  bst_node_t const* right;
  bst_node_t const* parent;

  bst_feature_t const* split_index;
  std::uint8_t const* default_left;
  float const* split_conds;

  // The number of nodes
  std::size_t n{0};

  linalg::MatrixView<float const> weights;

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf(bst_node_t nidx) const {
    return left[nidx] == InvalidNodeId();
  }

  [[nodiscard]] XGBOOST_DEVICE bst_node_t LeftChild(bst_node_t nidx) const { return left[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t RightChild(bst_node_t nidx) const { return right[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_feature_t SplitIndex(bst_node_t nidx) const {
    return split_index[nidx];
  }
  [[nodiscard]] XGBOOST_DEVICE float SplitCond(bst_node_t nidx) const { return split_conds[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bool DefaultLeft(bst_node_t nidx) const {
    return default_left[nidx];
  }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t DefaultChild(bst_node_t nidx) const {
    return this->DefaultLeft(nidx) ? this->LeftChild(nidx) : this->RightChild(nidx);
  }
  [[nodiscard]] XGBOOST_DEVICE linalg::VectorView<float const> LeafValue(bst_node_t nidx) const {
    return this->weights.Slice(nidx, linalg::All());
  }

  [[nodiscard]] bst_target_t NumTargets() const { return this->weights.Shape(1); }
  [[nodiscard]] bst_node_t Size() const { return this->n; }
};

/**
 * @brief Tree structure for multi-target model.
 */
class MultiTargetTree : public Model {
 public:
  static bst_node_t constexpr InvalidNodeId() { return MultiTargetTreeView::InvalidNodeId(); }

 private:
  TreeParam const* param_;
  HostDeviceVector<bst_node_t> left_;
  HostDeviceVector<bst_node_t> right_;
  HostDeviceVector<bst_node_t> parent_;
  HostDeviceVector<bst_feature_t> split_index_;
  HostDeviceVector<std::uint8_t> default_left_;
  HostDeviceVector<float> split_conds_;
  HostDeviceVector<float> weights_;

  mutable std::mutex tree_view_lock_;

  [[nodiscard]] linalg::VectorView<float const> NodeWeight(bst_node_t nidx) const {
    auto beg = nidx * this->NumTargets();
    auto v = this->weights_.ConstHostSpan().subspan(beg, this->NumTargets());
    return linalg::MakeTensorView(DeviceOrd::CPU(), v, v.size());
  }
  [[nodiscard]] linalg::VectorView<float> NodeWeight(bst_node_t nidx) {
    auto beg = nidx * this->NumTargets();
    auto v = this->weights_.HostSpan().subspan(beg, this->NumTargets());
    return linalg::MakeTensorView(DeviceOrd::CPU(), v, v.size());
  }

 public:
  explicit MultiTargetTree(TreeParam const* param);
  MultiTargetTree(MultiTargetTree const& that);
  MultiTargetTree& operator=(MultiTargetTree const& that) = delete;
  MultiTargetTree(MultiTargetTree&& that) = delete;
  MultiTargetTree& operator=(MultiTargetTree&& that) = delete;

  /**
   * @brief Set the weight for a leaf.
   */
  void SetLeaf(bst_node_t nidx, linalg::VectorView<float const> weight);
  /**
   * @brief Expand a leaf into split node.
   */
  void Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond, bool default_left,
              linalg::VectorView<float const> base_weight,
              linalg::VectorView<float const> left_weight,
              linalg::VectorView<float const> right_weight);

  [[nodiscard]] bool IsLeaf(bst_node_t nidx) const {
    return left_.ConstHostVector()[nidx] == InvalidNodeId();
  }
  [[nodiscard]] bst_node_t Parent(bst_node_t nidx) const {
    return parent_.ConstHostVector().at(nidx);
  }
  [[nodiscard]] bst_node_t LeftChild(bst_node_t nidx) const {
    return left_.ConstHostVector().at(nidx);
  }
  [[nodiscard]] bst_node_t RightChild(bst_node_t nidx) const {
    return right_.ConstHostVector().at(nidx);
  }

  [[nodiscard]] bst_feature_t SplitIndex(bst_node_t nidx) const {
    return split_index_.ConstHostVector()[nidx];
  }
  [[nodiscard]] float SplitCond(bst_node_t nidx) const {
    return split_conds_.ConstHostVector()[nidx];
  }
  [[nodiscard]] bool DefaultLeft(bst_node_t nidx) const {
    return default_left_.ConstHostVector()[nidx];
  }
  [[nodiscard]] bst_node_t DefaultChild(bst_node_t nidx) const {
    return this->DefaultLeft(nidx) ? this->LeftChild(nidx) : this->RightChild(nidx);
  }

  [[nodiscard]] bst_target_t NumTargets() const;

  [[nodiscard]] std::size_t Size() const;

  [[nodiscard]] bst_node_t Depth(bst_node_t nidx) const {
    bst_node_t depth{0};
    while (Parent(nidx) != InvalidNodeId()) {
      ++depth;
      nidx = Parent(nidx);
    }
    return depth;
  }

  [[nodiscard]] linalg::VectorView<float const> LeafValue(bst_node_t nidx) const {
    CHECK(IsLeaf(nidx));
    return this->NodeWeight(nidx);
  }
  /**
   * @brief Get a view to the tree.
   *
   *   This method is NOT thread-safe.
   */
  [[nodiscard]] MultiTargetTreeView View(Context const* ctx) const;

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;

  [[nodiscard]] std::size_t MemCostBytes() const;
};
}  // namespace xgboost
#endif  // XGBOOST_MULTI_TARGET_TREE_MODEL_H_
