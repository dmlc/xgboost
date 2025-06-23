/**
 * Copyright 2021-2025, XGBoost Contributors
 * \file array_tree_layout.cc
 * \brief Implementation of array tree layout -- a powerfull inference optimization method.
 */
#ifndef XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
#define XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_

#include <limits>
#include <vector>

namespace xgboost::predictor {

/**
 * @brief The class holds the array-based representation of the top levels of a single tree.
 * 
 * \tparam TreeType The type of the origianl tree (RegTree or MultiTargetTree)
 *
 * \tparam has_categorical if the tree has categorical features
 *
 * \tparam any_missing if the class is able to process missing values
 * 
 * \tparam kNumDeepLevels number of tree leveles being unrolled into array-based structure
 */
template <class TreeType, bool has_categorical, bool any_missing, int kNumDeepLevels>
class ArrayTreeLayout {
 private:
  constexpr static size_t kNodesCount = (1u << kNumDeepLevels) - 1;

  struct Empty {};
  using DefaultLeftType =
        typename std::conditional_t<any_missing,
                                   std::array<uint8_t, kNodesCount>,
                                   struct Empty>;
  using IsCatType =
        typename std::conditional_t<has_categorical,
                                   std::array<uint8_t, kNodesCount>,
                                   struct Empty>;
  using CatSegmentType =
        typename std::conditional_t<has_categorical,
                                   std::array<common::Span<uint32_t const>, kNodesCount>,
                                   struct Empty>;

  DefaultLeftType default_left_;
  IsCatType is_cat_;
  CatSegmentType cat_segment_;

  std::array<bst_feature_t, kNodesCount> split_index_;
  std::array<float, kNodesCount> split_cond_;
  std::array<bst_node_t, kNodesCount + 1> nidx_in_tree_;

  inline static bool IsLeaf(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].IsLeaf();
  }

  inline static bool IsLeaf(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.IsLeaf(nidx);
  }

  inline static uint8_t DefaultLeft(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].DefaultLeft();
  }

  inline static uint8_t DefaultLeft(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.DefaultLeft(nidx);
  }

  inline static bst_feature_t SplitIndex(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].SplitIndex();
  }

  inline static bst_feature_t SplitIndex(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.SplitIndex(nidx);
  }

  inline static float SplitCond(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].SplitCond();
  }

  inline static float SplitCond(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.SplitCond(nidx);
  }

  inline static bst_feature_t LeftChild(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].LeftChild();
  }

  inline static bst_feature_t LeftChild(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.LeftChild(nidx);
  }

  inline static bst_feature_t RightChild(const RegTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<RegTree, TreeType>);
    return tree[nidx].LeftChild() + 1;
  }

  inline static bst_feature_t RightChild(const MultiTargetTree& tree, bst_feature_t nidx) {
    static_assert(std::is_same_v<MultiTargetTree, TreeType>);
    return tree.RightChild(nidx);
  }

 /**
 * @brief Traverse the top levels of original tree and fill internal arrays
 * 
 * \tparam TreeType The type of the origianl tree (RegTree or MultiTargetTree)
 *
 * \tparam depth the tree level being processing
 *
 * \param tree the original tree
 * 
 * \param cats matrix of categorical splits
 * 
 * \param nidx_array node idx in the array layout
 * 
 * \param nidx node idx in the original tree
 * 
 */
  template <int depth = 0>
  void inline Populate(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats,
                       bst_feature_t nidx_array = 0, bst_feature_t nidx = 0) {
    if constexpr (depth == kNumDeepLevels + 1) {
      return;
    } else if constexpr (depth == kNumDeepLevels) {
        /* We save the node index in the origianl tree to able to continue processing
         * for nodes not egligable for array layout optimisation. 
         */
        nidx_in_tree_[nidx_array - kNodesCount] = nidx;
    } else {
      if (IsLeaf(tree, nidx)) {
        split_index_[nidx_array]  = 0;

        /* 
         * if the tree is not fully populated, we can reduce transfering costs.
         * the values for unpopulated part of the tree are set in a way to guarantie
         * that a moove will always done in "right" direction.
         * here we exploiting that comparison with nan always results to false.
         */
        if constexpr (any_missing) default_left_[nidx_array] = 0;
        if constexpr (has_categorical) is_cat_[nidx_array] = 0;
        split_cond_[nidx_array]   = std::numeric_limits<float>::quiet_NaN();

        Populate<depth + 1>(tree, cats, 2 * nidx_array + 2, nidx);
      } else {
        if constexpr (any_missing) default_left_[nidx_array] = DefaultLeft(tree, nidx);
        if constexpr (has_categorical) {
          is_cat_[nidx_array] = common::IsCat(cats.split_type, nidx);
          if (is_cat_[nidx_array]) {
            cat_segment_[nidx_array] = cats.categories.subspan(cats.node_ptr[nidx].beg,
                                                               cats.node_ptr[nidx].size);
          }
        }

        split_index_[nidx_array]  = SplitIndex(tree, nidx);
        split_cond_[nidx_array]   = SplitCond(tree, nidx);

        Populate<depth + 1>(tree, cats, 2 * nidx_array + 1, LeftChild(tree, nidx));
        Populate<depth + 1>(tree, cats, 2 * nidx_array + 2, RightChild(tree, nidx));
      }
    }
  }

  bool inline GetDecision(float fvalue, bst_node_t nidx) const {
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

  ArrayTreeLayout(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats) {
    Populate(tree, cats);
  }

  /**
   * @brief
   * Traverse top levels of the tree for an entire block_size.
   * In array layout is orginised to garantie that
   * if the node at the current level has index nidx, than
   * the node index for left child at the next level is always 2*nidx
   * the node index for right child at the next level is always 2*nidx+1
   * This greatly improve data locality
   * 
   * \param thread_temp buffer holding the feature values
   * 
   * \param offset offset of the current data block
   * 
   * \param block_size size of the current block (1 < block_size <= 64)
   * 
   * \param p_nidx pointer to the vector of node indexes in the original tree,
   *               corresponding to the level next after kNumDeepLevels
   */
  void inline Process(std::vector<RegTree::FVec> const &thread_temp, std::size_t const offset,
                      std::size_t const block_size, bst_node_t* p_nidx) {
    for (int depth = 0; depth < kNumDeepLevels; ++depth) {
      std::size_t first_node = (1u << depth) - 1;

      for (std::size_t i = 0; i < block_size; ++i) {
        bst_node_t idx = p_nidx[i];

        const auto& feat = thread_temp[offset + i];
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
    for (std::size_t i = 0; i < block_size; ++i) {
      p_nidx[i] = nidx_in_tree_[p_nidx[i]];
    }
  }
};

template <class TreeType, bool has_categorical, bool any_missing, int num_deep_levels = 1>
void inline ProcessArrayTree(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats,
                             std::vector<RegTree::FVec> const &thread_temp,
                             std::size_t const offset, std::size_t const block_size,
                             bst_node_t* p_nidx, int tree_depth) {
  constexpr int kMaxNumDeepLevels =
    ArrayTreeLayout<TreeType, has_categorical, any_missing, 0>::kMaxNumDeepLevels;

  if constexpr (num_deep_levels == kMaxNumDeepLevels) {
    ArrayTreeLayout<TreeType, has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
    buffer.Process(thread_temp, offset, block_size, p_nidx);
  } else {
    if (tree_depth <= num_deep_levels) {
      ArrayTreeLayout<TreeType, has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
      buffer.Process(thread_temp, offset, block_size, p_nidx);
    } else {
      ProcessArrayTree<TreeType, has_categorical, any_missing, num_deep_levels + 1>
        (tree, cats, thread_temp, offset, block_size, p_nidx, tree_depth);
    }
  }
}

}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_ARRAY_TREE_LAYOUT_H_
