/**
 * Copyright 2021-2026, XGBoost Contributors
 * \file array_tree_layout.h
 * \brief Implementation of array tree layout -- a powerfull inference optimization method.
 */
#ifndef XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
#define XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_

#include <algorithm>  // for min
#include <array>      // for array
#include <cstddef>    // for size_t
#include <cstdint>    // for uint8_t, uint32_t
#include <limits>     // for numeric_limits
#include <vector>     // for vector

#include "../common/categorical.h"  // for IsCat, Decision
#include "xgboost/base.h"           // for bst_node_t, bst_feature_t
#include "xgboost/span.h"           // for Span
#include "xgboost/tree_model.h"     // for RegTree

namespace xgboost::predictor {
/**
 * @brief Array-based representation of the top levels of a single tree.
 *
 * The nodes at tree levels 0, 1, ..., n_levels - 1 are unrolled into a complete binary
 * tree stored in arrays: if a node at the current level has index nidx (relative to the
 * first node of its level), then its left child at the next level has index 2*nidx and
 * its right child 2*nidx+1.  This gives a compact, cache friendly structure for
 * traversing a block of samples level by level.
 *
 * The layout is built once per tree for a prediction call and is then shared read-only
 * by all blocks of samples (and all threads).  Trees deeper than @ref kMaxNumDeepLevels
 * are traversed with the array layout for the first @ref kMaxNumDeepLevels levels and
 * with the regular node walk for the remaining levels, starting from the node index
 * stored in @ref NidxInTree.
 */
class ArrayTreeLayout {
 public:
  /* Ad-hoc value.
   * Increasing doesn't lead to perf gain, since bottleneck is now at gather instructions.
   */
  constexpr static int kMaxNumDeepLevels = 6;
  /* Number of nodes in the array based representation of the top levels of the tree */
  constexpr static std::size_t kNodesCount = (1u << kMaxNumDeepLevels) - 1;

 private:
  std::array<bst_feature_t, kNodesCount> split_index_{};
  std::array<float, kNodesCount> split_cond_{};
  std::array<std::uint8_t, kNodesCount> default_left_{};
  /* If the tree has more levels than n_levels_, this array stores the node indices of the
   * sub-trees at level n_levels_ (one entry for each of the 2^n_levels_ positions), so
   * that the traversal can continue in the original tree.  For a leaf that is reached
   * before that level, the same leaf is stored.
   */
  std::array<bst_node_t, kNodesCount + 1> nidx_in_tree_{};
  // Categorical splits are rare and keep their data outside of the hot arrays.
  std::vector<std::uint8_t> is_cat_;
  std::vector<common::Span<std::uint32_t const>> cat_segment_;
  // Depth of the original tree.
  bst_node_t tree_depth_{0};
  // Number of tree levels unrolled into the arrays.
  int n_levels_{0};

  /**
   * @brief Traverse the top levels of original tree and fill internal arrays
   *
   * @param tree the original tree
   * @param cats matrix of categorical splits
   * @param depth the tree level being processed
   * @param nidx_array node idx in the array layout
   * @param nidx node idx in the original tree
   */
  template <typename TreeView>
  void Populate(TreeView const& tree, RegTree::CategoricalSplitMatrix const& cats, int depth,
                bst_node_t nidx_array, bst_node_t nidx) {
    if (depth == n_levels_) {
      /* We store the node index in the original tree to ensure continued processing
       * for nodes that are not eligible for array layout optimization.
       */
      nidx_in_tree_[nidx_array - ((1u << n_levels_) - 1)] = nidx;
      return;
    }
    bool const has_categorical = !is_cat_.empty();
    if (tree.IsLeaf(nidx)) {
      split_index_[nidx_array] = 0;
      /*
       * If the tree is not fully populated, we can reduce transfer costs.
       * The values for the unpopulated parts of the tree are set to ensure
       * that any move will always proceed in the "right" direction.
       * This is achieved by exploiting the fact that comparisons with NaN always result in false.
       */
      default_left_[nidx_array] = 0;
      if (has_categorical) {
        is_cat_[nidx_array] = 0;
      }
      split_cond_[nidx_array] = std::numeric_limits<float>::quiet_NaN();

      this->Populate(tree, cats, depth + 1, 2 * nidx_array + 2, nidx);
    } else {
      default_left_[nidx_array] = tree.DefaultLeft(nidx);
      if (has_categorical) {
        is_cat_[nidx_array] = common::IsCat(cats.split_type, nidx);
        if (is_cat_[nidx_array]) {
          cat_segment_[nidx_array] =
              cats.categories.subspan(cats.node_ptr[nidx].beg, cats.node_ptr[nidx].size);
        }
      }

      split_index_[nidx_array] = tree.SplitIndex(nidx);
      split_cond_[nidx_array] = tree.SplitCond(nidx);

      /*
       * LeftChild is used to determine if a node is a leaf, so it is always a valid value.
       * However, RightChild can be invalid in some exotic cases.
       * A tree with an invalid RightChild can still be correctly processed using classical methods
       * if the split conditions are correct.
       * However, in an array layout, an invalid RightChild, even if unreachable, can lead to memory corruption.
       * A check should be added to prevent this.
       */
      this->Populate(tree, cats, depth + 1, 2 * nidx_array + 1, tree.LeftChild(nidx));
      bst_node_t right_child = tree.RightChild(nidx);
      if (right_child != RegTree::kInvalidNodeId) {
        this->Populate(tree, cats, depth + 1, 2 * nidx_array + 2, right_child);
      }
    }
  }

  template <bool has_categorical>
  [[nodiscard]] bool GetDecision(float fvalue, std::size_t nidx) const {
    if constexpr (has_categorical) {
      if (is_cat_[nidx]) {
        return common::Decision(cat_segment_[nidx], fvalue);
      }
    }
    return fvalue < split_cond_[nidx];
  }

 public:
  ArrayTreeLayout() = default;

  /**
   * @brief Build the layout for a tree.
   *
   * @param tree       The tree view.
   * @param max_levels Upper bound for the number of unrolled levels.
   */
  template <typename TreeView>
  explicit ArrayTreeLayout(TreeView const& tree, int max_levels = kMaxNumDeepLevels) {
    this->Build(tree, max_levels);
  }

  template <typename TreeView>
  void Build(TreeView const& tree, int max_levels = kMaxNumDeepLevels) {
    tree_depth_ = tree.MaxDepth();
    n_levels_ = std::min(static_cast<int>(tree_depth_), std::min(max_levels, kMaxNumDeepLevels));
    if (tree.HasCategoricalSplit()) {
      is_cat_.assign(kNodesCount, 0);
      cat_segment_.assign(kNodesCount, {});
    } else {
      is_cat_.clear();
      cat_segment_.clear();
    }
    this->Populate(tree, tree.GetCategoriesMatrix(), 0, 0, RegTree::kRoot);
  }

  /** @brief Number of tree levels unrolled into the arrays. */
  [[nodiscard]] int NumLevels() const { return n_levels_; }
  /** @brief Depth of the original tree. */
  [[nodiscard]] bst_node_t TreeDepth() const { return tree_depth_; }
  /**
   * @brief Whether the whole tree is covered by the layout, in which case @ref Process
   *        outputs leaf indices.
   */
  [[nodiscard]] bool IsComplete() const { return tree_depth_ <= n_levels_; }

  [[nodiscard]] auto const& SplitIndex() const { return split_index_; }
  [[nodiscard]] auto const& SplitCond() const { return split_cond_; }
  [[nodiscard]] auto const& DefaultLeft() const { return default_left_; }
  [[nodiscard]] auto const& NidxInTree() const { return nidx_in_tree_; }

  /**
   * @brief Traverse the top levels of the tree for the entire block_size.
   *
   * @tparam has_categorical Whether the tree has categorical splits.
   * @tparam any_missing     Whether the block may contain missing values.
   *
   * @param fvec_tloc buffer holding the feature values
   * @param block_size size of the current block (1 < block_size <= 64)
   * @param p_nidx Pointer to the vector of node indexes in the original tree with size
   *               equals to the block size. (One node per sample). The value corresponds
   *               to the level next after n_levels_
   */
  template <bool has_categorical, bool any_missing>
  void Process(common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
               bst_node_t* p_nidx) const {
    // Raw pointers: no per-element span bounds check and no reload of the span in the
    // innermost loop.
    RegTree::FVec const* feats = fvec_tloc.data();
    bst_feature_t const* split_index = split_index_.data();
    for (int depth = 0; depth < n_levels_; ++depth) {
      std::size_t const first_node = (1u << depth) - 1;

      for (std::size_t i = 0; i < block_size; ++i) {
        bst_node_t const idx = p_nidx[i];
        std::size_t const node = first_node + idx;

        bst_feature_t const split = split_index[node];
        auto const fvalue = feats[i].GetFvalue(split);
        if constexpr (any_missing) {
          bool go_left = feats[i].IsMissing(split)
                             ? default_left_[node]
                             : this->GetDecision<has_categorical>(fvalue, node);
          p_nidx[i] = 2 * idx + !go_left;
        } else {
          p_nidx[i] = 2 * idx + !this->GetDecision<has_categorical>(fvalue, node);
        }
      }
    }
    // Remap to the original index.
    bst_node_t const* nidx_in_tree = nidx_in_tree_.data();
    for (std::size_t i = 0; i < block_size; ++i) {
      p_nidx[i] = nidx_in_tree[p_nidx[i]];
    }
  }
};
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
