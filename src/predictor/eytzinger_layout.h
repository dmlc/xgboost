/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_EYTZINGER_LAYOUT_H_
#define XGBOOST_PREDICTOR_EYTZINGER_LAYOUT_H_

#include <limits>
#include <vector>

namespace xgboost::predictor {

template <class TreeType, bool has_categorical, bool any_missing, int kNumDeepLevels>
// Eytzinger Layout
class EytzingerLayout {
 private:
  constexpr static size_t kNodesCount = (1u << kNumDeepLevels) - 1;

  struct Empty {};
  using DefaultLeftType =
        typename std::conditional<any_missing,
                                  std::array<uint8_t, kNodesCount>,
                                  struct Empty>::type;
  using IsCatType =
        typename std::conditional<has_categorical,
                                  std::array<uint8_t, kNodesCount>,
                                  struct Empty>::type;
  using CatSegmentType =
        typename std::conditional<has_categorical,
                                  std::array<common::Span<uint32_t const>, kNodesCount>,
                                  struct Empty>::type;

  DefaultLeftType default_left;
  IsCatType is_cat;
  CatSegmentType cat_segment;

  std::array<bst_feature_t, kNodesCount> split_index;
  std::array<float, kNodesCount> split_cond;
  std::array<bst_node_t, kNodesCount + 1> global_nidx;

  inline static bool IsLeaf(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].IsLeaf();
  }

  inline static bool IsLeaf(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.IsLeaf(nidx);
  }

  inline static uint8_t DefaultLeft(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].DefaultLeft();
  }

  inline static uint8_t DefaultLeft(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.DefaultLeft(nidx);
  }

  inline static bst_feature_t SplitIndex(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].SplitIndex();
  }

  inline static bst_feature_t SplitIndex(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.SplitIndex(nidx);
  }

  inline static float SplitCond(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].SplitCond();
  }

  inline static float SplitCond(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.SplitCond(nidx);
  }

  inline static bst_feature_t LeftChild(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].LeftChild();
  }

  inline static bst_feature_t LeftChild(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.LeftChild(nidx);
  }

  inline static bst_feature_t RightChild(const RegTree& tree, bst_feature_t nidx) {
    return tree[nidx].LeftChild() + 1;
  }

  inline static bst_feature_t RightChild(const MultiTargetTree& tree, bst_feature_t nidx) {
    return tree.RightChild(nidx);
  }

  template <int depth = 0>
  void inline Populate(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats,
                       bst_feature_t new_nidx = 0, bst_feature_t nidx = 0) {
    if constexpr (depth == kNumDeepLevels + 1) {
      return;
    } else if constexpr (depth == kNumDeepLevels) {
        global_nidx[new_nidx - kNodesCount] = nidx;
    } else {
      if (IsLeaf(tree, nidx)) {
        split_index[new_nidx]  = 0;

        // always go right
        if constexpr (any_missing) default_left[new_nidx] = 0;
        if constexpr (has_categorical) is_cat[new_nidx] = 0;
        split_cond[new_nidx]   = std::numeric_limits<float>::quiet_NaN();

        Populate<depth + 1>(tree, cats, 2 * new_nidx + 2, nidx);
      } else {
        if constexpr (any_missing) default_left[new_nidx] = DefaultLeft(tree, nidx);
        if constexpr (has_categorical) {
          is_cat[new_nidx] = common::IsCat(cats.split_type, nidx);
          cat_segment[new_nidx] = cats.categories.subspan(cats.node_ptr[nidx].beg,
                                                          cats.node_ptr[nidx].size);
        }

        split_index[new_nidx]  = SplitIndex(tree, nidx);
        split_cond[new_nidx]   = SplitCond(tree, nidx);

        Populate<depth + 1>(tree, cats, 2 * new_nidx + 1, LeftChild(tree, nidx));
        Populate<depth + 1>(tree, cats, 2 * new_nidx + 2, RightChild(tree, nidx));
      }
    }
  }

  bool inline GoLeft(float fvalue, bst_node_t nidx) const {
    if constexpr (has_categorical) {
      if (is_cat[nidx]) {
       return common::Decision(cat_segment[nidx], fvalue);
      } else {
        return fvalue < split_cond[nidx];
      }
    } else {
      return fvalue < split_cond[nidx];
    }
  }

 public:
  constexpr static int kMaxNumDeepLevels = 6;
  static_assert(kNumDeepLevels <= kMaxNumDeepLevels);

  EytzingerLayout(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats) {
    Populate(tree, cats);
  }

  void inline Process(std::vector<RegTree::FVec> const &thread_temp, std::size_t const offset,
                      std::size_t const block_size, bst_node_t* p_nidx) {
    for (int depth = 0; depth < kNumDeepLevels; ++depth) {
      std::size_t first_node = (1u << depth) - 1;

      for (std::size_t i = 0; i < block_size; ++i) {
        bst_node_t idx = p_nidx[i];

        const auto& feat = thread_temp[offset + i];
        bst_feature_t split = split_index[first_node + idx];
        auto fvalue = feat.GetFvalue(split);
        if constexpr (any_missing) {
          bool go_left = feat.IsMissing(split) ? default_left[first_node + idx]
                                               : GoLeft(fvalue, first_node + idx);
          p_nidx[i] = 2 * idx + !go_left;
        } else {
          p_nidx[i] = 2 * idx + !GoLeft(fvalue, first_node + idx);
        }
      }
    }
    for (std::size_t i = 0; i < block_size; ++i) {
      p_nidx[i] = global_nidx[p_nidx[i]];
    }
  }
};

template <class TreeType, bool has_categorical, bool any_missing, int num_deep_levels = 1>
void inline ProcessEytzinger(const TreeType& tree, RegTree::CategoricalSplitMatrix const &cats,
                             std::vector<RegTree::FVec> const &thread_temp,
                             std::size_t const offset, std::size_t const block_size,
                             bst_node_t* p_nidx, int tree_depth) {
  if constexpr (num_deep_levels == EytzingerLayout<TreeType, 0, 0, 0>::kMaxNumDeepLevels) {
    EytzingerLayout<TreeType, has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
    buffer.Process(thread_temp, offset, block_size, p_nidx);
  } else {
    if (tree_depth <= num_deep_levels) {
      EytzingerLayout<TreeType, has_categorical, any_missing, num_deep_levels> buffer(tree, cats);
      buffer.Process(thread_temp, offset, block_size, p_nidx);
    } else {
      ProcessEytzinger<TreeType, has_categorical, any_missing, num_deep_levels + 1>
        (tree, cats, thread_temp, offset, block_size, p_nidx, tree_depth);
    }
  }
}

}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_EYTZINGER_LAYOUT_H_
