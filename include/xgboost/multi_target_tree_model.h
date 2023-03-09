/**
 * Copyright 2023 by XGBoost contributors
 *
 * \brief Core data structure for multi-target trees.
 */
#ifndef XGBOOST_MULTI_TARGET_TREE_MODEL_H_
#define XGBOOST_MULTI_TARGET_TREE_MODEL_H_
#include <xgboost/base.h>     // for bst_node_t, bst_target_t, bst_feature_t
#include <xgboost/context.h>  // for Context
#include <xgboost/linalg.h>   // for VectorView
#include <xgboost/model.h>    // for Model
#include <xgboost/span.h>     // for Span

#include <cinttypes>          // for uint8_t
#include <cstddef>            // for size_t
#include <vector>             // for vector

namespace xgboost {
struct TreeParam;
/**
 * \brief Tree structure for multi-target model.
 */
class MultiTargetTree : public Model {
 public:
  static bst_node_t constexpr InvalidNodeId() { return -1; }

 private:
  TreeParam const* param_;
  std::vector<bst_node_t> left_;
  std::vector<bst_node_t> right_;
  std::vector<bst_node_t> parent_;
  std::vector<bst_feature_t> split_index_;
  std::vector<std::uint8_t> default_left_;
  std::vector<float> split_conds_;
  std::vector<float> weights_;

  [[nodiscard]] linalg::VectorView<float const> NodeWeight(bst_node_t nidx) const {
    auto beg = nidx * this->NumTarget();
    auto v = common::Span<float const>{weights_}.subspan(beg, this->NumTarget());
    return linalg::MakeTensorView(Context::kCpuId, v, v.size());
  }
  [[nodiscard]] linalg::VectorView<float> NodeWeight(bst_node_t nidx) {
    auto beg = nidx * this->NumTarget();
    auto v = common::Span<float>{weights_}.subspan(beg, this->NumTarget());
    return linalg::MakeTensorView(Context::kCpuId, v, v.size());
  }

 public:
  explicit MultiTargetTree(TreeParam const* param);
  /**
   * \brief Set the weight for a leaf.
   */
  void SetLeaf(bst_node_t nidx, linalg::VectorView<float const> weight);
  /**
   * \brief Expand a leaf into split node.
   */
  void Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond, bool default_left,
              linalg::VectorView<float const> base_weight,
              linalg::VectorView<float const> left_weight,
              linalg::VectorView<float const> right_weight);

  [[nodiscard]] bool IsLeaf(bst_node_t nidx) const { return left_[nidx] == InvalidNodeId(); }
  [[nodiscard]] bst_node_t Parent(bst_node_t nidx) const { return parent_.at(nidx); }
  [[nodiscard]] bst_node_t LeftChild(bst_node_t nidx) const { return left_.at(nidx); }
  [[nodiscard]] bst_node_t RightChild(bst_node_t nidx) const { return right_.at(nidx); }

  [[nodiscard]] bst_feature_t SplitIndex(bst_node_t nidx) const { return split_index_[nidx]; }
  [[nodiscard]] float SplitCond(bst_node_t nidx) const { return split_conds_[nidx]; }
  [[nodiscard]] bool DefaultLeft(bst_node_t nidx) const { return default_left_[nidx]; }
  [[nodiscard]] bst_node_t DefaultChild(bst_node_t nidx) const {
    return this->DefaultLeft(nidx) ? this->LeftChild(nidx) : this->RightChild(nidx);
  }

  [[nodiscard]] bst_target_t NumTarget() const;

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

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;
};
}  // namespace xgboost
#endif  // XGBOOST_MULTI_TARGET_TREE_MODEL_H_
