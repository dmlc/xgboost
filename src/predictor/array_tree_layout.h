/**
 * Copyright 2021-2025, XGBoost Contributors
 * \file array_tree_layout.cc
 * \brief Implementation of array tree layout -- a powerfull inference optimization method.
 */
#ifndef XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
#define XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_

#include <array>
#include <limits>
#include <type_traits>  // for conditional_t

#include "../common/categorical.h"  // for IsCat
#include "xgboost/tree_model.h"     // for RegTree

namespace xgboost::predictor {

/**
 * @brief The class holds the array-based representation of the top levels of a single tree.
 *
 * @tparam has_categorical if the tree has categorical features
 *
 * @tparam any_missing if the class is able to process missing values
 *
 * @tparam kNumDeepLevels number of tree leveles being unrolled into array-based structure
 */
template <bool has_categorical, bool any_missing, int kNumDeepLevels>
class ArrayTreeLayout {
 private:
  /* Number of nodes in the array based representation of the top levels of the tree
   */
  constexpr static size_t kNodesCount = (1u << kNumDeepLevels) - 1;

  struct Empty {};
  using DefaultLeftType =
      typename std::conditional_t<any_missing, std::array<uint8_t, kNodesCount>, Empty>;
  using IsCatType =
      typename std::conditional_t<has_categorical, std::array<uint8_t, kNodesCount>, Empty>;
  using CatSegmentType =
      typename std::conditional_t<has_categorical,
                                  std::array<common::Span<uint32_t const>, kNodesCount>, Empty>;

  DefaultLeftType default_left_;
  IsCatType is_cat_;
  CatSegmentType cat_segment_;

  std::array<bst_feature_t, kNodesCount> split_index_;
  std::array<float, kNodesCount> split_cond_;
  /* The nodes at tree levels 0, 1, ..., kNumDeepLevels - 1 are unrolled into an array-based structure.
   *  If the tree has additional levels, this array stores the node indices of the sub-trees at level kNumDeepLevels.
   *  This is necessary to continue processing nodes that are not eligible for array-based unrolling.
   *  The number of sub-trees packed into this array is equal to the number of nodes at tree level kNumDeepLevels,
   *  which is calculated as (1u << kNumDeepLevels) == kNodesCount + 1.
   */
  // Mapping from array node index to the RegTree node index.
  std::array<bst_node_t, kNodesCount + 1> nidx_in_tree_;

 /**
 * @brief Traverse the top levels of original tree and fill internal arrays
 *
 * @tparam depth the tree level being processing
 *
 * @param tree the original tree
 * @param cats matrix of categorical splits
 * @param nidx_array node idx in the array layout
 * @param nidx node idx in the original tree
 */
  template <int depth = 0>
  void Populate(const RegTree& tree, RegTree::CategoricalSplitMatrix const& cats,
                bst_node_t nidx_array = 0, bst_node_t nidx = 0) {
    if constexpr (depth == kNumDeepLevels + 1) {
      return;
    } else if constexpr (depth == kNumDeepLevels) {
        /* We store the node index in the original tree to ensure continued processing
         * for nodes that are not eligible for array layout optimization.
         */
        nidx_in_tree_[nidx_array - kNodesCount] = nidx;
    } else {
      if (tree.IsLeaf(nidx)) {
        split_index_[nidx_array]  = 0;

        /*
         * If the tree is not fully populated, we can reduce transfer costs.
         * The values for the unpopulated parts of the tree are set to ensure
         * that any move will always proceed in the "right" direction.
         * This is achieved by exploiting the fact that comparisons with NaN always result in false.
         */
        if constexpr (any_missing) default_left_[nidx_array] = 0;
        if constexpr (has_categorical) is_cat_[nidx_array] = 0;
        split_cond_[nidx_array]   = std::numeric_limits<float>::quiet_NaN();

        Populate<depth + 1>(tree, cats, 2 * nidx_array + 2, nidx);
      } else {
        if constexpr (any_missing) default_left_[nidx_array] = tree.DefaultLeft(nidx);
        if constexpr (has_categorical) {
          is_cat_[nidx_array] = common::IsCat(cats.split_type, nidx);
          if (is_cat_[nidx_array]) {
            cat_segment_[nidx_array] = cats.categories.subspan(cats.node_ptr[nidx].beg,
                                                               cats.node_ptr[nidx].size);
          }
        }

        split_index_[nidx_array]  = tree.SplitIndex(nidx);
        split_cond_[nidx_array]   = tree.SplitCond(nidx);

        /*
         * LeftChild is used to determine if a node is a leaf, so it is always a valid value.
         * However, RightChild can be invalid in some exotic cases.
         * A tree with an invalid RightChild can still be correctly processed using classical methods
         * if the split conditions are correct.
         * However, in an array layout, an invalid RightChild, even if unreachable, can lead to memory corruption.
         * A check should be added to prevent this.
         */
        Populate<depth + 1>(tree, cats, 2 * nidx_array + 1, tree.LeftChild(nidx));
        bst_node_t right_child = tree.RightChild(nidx);
        if (right_child != RegTree::kInvalidNodeId) {
          Populate<depth + 1>(tree, cats, 2 * nidx_array + 2, right_child);
        }
      }
    }
  }

  bool GetDecision(float fvalue, bst_node_t nidx) const {
    if constexpr (has_categorical) {
      if (is_cat_[nidx]) {
       return common::Decision(cat_segment_[nidx], fvalue);
      } else {
        return fvalue < split_cond_[nidx];
      }
    } else {
      return fvalue < split_cond_[nidx];
    }
  }

 public:
  /* Ad-hoc value.
   * Increasing doesn't lead to perf gain, since bottleneck is now at gather instructions.
   */
  constexpr static int kMaxNumDeepLevels = 6;
  static_assert(kNumDeepLevels <= kMaxNumDeepLevels);

  ArrayTreeLayout(const RegTree& tree, RegTree::CategoricalSplitMatrix const &cats) {
    Populate(tree, cats);
  }

  const auto& SplitIndex() const {
    return split_index_;
  }

  const auto& SplitCond() const {
    return split_cond_;
  }

  const auto& DefaultLeft() const {
    return default_left_;
  }

  const auto& NidxInTree() const {
    return nidx_in_tree_;
  }

  /**
   * @brief Traverse the top levels of the tree for the entire block_size.
   *
   * In the array layout, it is organized to guarantee that if a node at the current level
   * has index nidx, then the node index for the left child at the next level is always
   * 2*nidx, and the node index for the right child at the next level is always 2*nidx+1.
   * This greatly improves data locality.
   *
   * @param fvec_tloc buffer holding the feature values
   * @param block_size size of the current block (1 < block_size <= 64)
   * @param p_nidx Pointer to the vector of node indexes in the original tree with size
   *               equals to the block size. (One node per sample). The value corresponds
   *               to the level next after kNumDeepLevels
   */
  void Process(common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
               bst_node_t* p_nidx) {
    for (int depth = 0; depth < kNumDeepLevels; ++depth) {
      std::size_t first_node = (1u << depth) - 1;

      for (std::size_t i = 0; i < block_size; ++i) {
        bst_node_t idx = p_nidx[i];

        const auto& feat = fvec_tloc[i];
        bst_feature_t split = split_index_[first_node + idx];
        auto fvalue = feat.GetFvalue(split);
        if constexpr (any_missing) {
          bool go_left = feat.IsMissing(split) ? default_left_[first_node + idx]
                                               : GetDecision(fvalue, first_node + idx);
          p_nidx[i] = 2 * idx + !go_left;
        } else {
          p_nidx[i] = 2 * idx + !GetDecision(fvalue, first_node + idx);
        }
      }
    }
    // Remap to the original index.
    for (std::size_t i = 0; i < block_size; ++i) {
      p_nidx[i] = nidx_in_tree_[p_nidx[i]];
    }
  }
};

template <bool has_categorical, bool any_missing, int num_deep_levels = 1>
void ProcessArrayTree(const RegTree& tree, RegTree::CategoricalSplitMatrix const& cats,
                      common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                      bst_node_t* p_nidx, int tree_depth) {
  constexpr int kMaxNumDeepLevels =
      ArrayTreeLayout<has_categorical, any_missing, 0>::kMaxNumDeepLevels;

  // Fill the array tree, then output predicted node idx.
  if constexpr (num_deep_levels == kMaxNumDeepLevels) {
    ArrayTreeLayout<has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
    buffer.Process(fvec_tloc, block_size, p_nidx);
  } else {
    if (tree_depth <= num_deep_levels) {
      ArrayTreeLayout<has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
      buffer.Process(fvec_tloc, block_size, p_nidx);
    } else {
      ProcessArrayTree<has_categorical, any_missing, num_deep_levels + 1>
        (tree, cats, fvec_tloc, block_size, p_nidx, tree_depth);
    }
  }
}

}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
