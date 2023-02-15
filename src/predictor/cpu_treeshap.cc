/**
 * Copyright by XGBoost Contributors 2017-2022
 */
#include "cpu_treeshap.h"

#include <algorithm>             // copy
#include <cinttypes>             // std::uint32_t

#include "predict_fn.h"          // GetNextNode
#include "xgboost/base.h"        // bst_node_t
#include "xgboost/logging.h"
#include "xgboost/tree_model.h"  // RegTree

namespace xgboost {
// Used by TreeShap
// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permutation weight of paths with i-1 ones in them
struct PathElement {
  int feature_index;
  float zero_fraction;
  float one_fraction;
  float pweight;
  PathElement() = default;
  PathElement(int i, float z, float o, float w)
      : feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};

// extend our decision path with a fraction of one and zero extensions
void ExtendPath(PathElement* unique_path, std::uint32_t unique_depth, float zero_fraction,
                float one_fraction, int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
  for (int i = unique_depth - 1; i >= 0; i--) {
    unique_path[i + 1].pweight +=
        one_fraction * unique_path[i].pweight * (i + 1) / static_cast<float>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i) /
                             static_cast<float>(unique_depth + 1);
  }
}

// undo a previous extension of the decision path
void UnwindPath(PathElement* unique_path, std::uint32_t unique_depth, std::uint32_t path_index) {
  const float one_fraction = unique_path[path_index].one_fraction;
  const float zero_fraction = unique_path[path_index].zero_fraction;
  float next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const float tmp = unique_path[i].pweight;
      unique_path[i].pweight =
          next_one_portion * (unique_depth + 1) / static_cast<float>((i + 1) * one_fraction);
      next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i) /
                                   static_cast<float>(unique_depth + 1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1)) /
                               static_cast<float>(zero_fraction * (unique_depth - i));
    }
  }

  for (auto i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i + 1].feature_index;
    unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
  }
}

// determine what the total permutation weight would be if
// we unwound a previous extension in the decision path
float UnwoundPathSum(const PathElement* unique_path, std::uint32_t unique_depth,
                     std::uint32_t path_index) {
  const float one_fraction = unique_path[path_index].one_fraction;
  const float zero_fraction = unique_path[path_index].zero_fraction;
  float next_one_portion = unique_path[unique_depth].pweight;
  float total = 0;
  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const float tmp =
          next_one_portion * (unique_depth + 1) / static_cast<float>((i + 1) * one_fraction);
      total += tmp;
      next_one_portion =
          unique_path[i].pweight -
          tmp * zero_fraction * ((unique_depth - i) / static_cast<float>(unique_depth + 1));
    } else if (zero_fraction != 0) {
      total += (unique_path[i].pweight / zero_fraction) /
               ((unique_depth - i) / static_cast<float>(unique_depth + 1));
    } else {
      CHECK_EQ(unique_path[i].pweight, 0) << "Unique path " << i << " must have zero weight";
    }
  }
  return total;
}

/**
 * \brief Recursive function that computes the feature attributions for a single tree.
 * \param feat dense feature vector, if the feature is missing the field is set to NaN
 * \param phi dense output vector of feature attributions
 * \param node_index the index of the current node in the tree
 * \param unique_depth how many unique features are above the current node in the tree
 * \param parent_unique_path a vector of statistics about our current path through the tree
 * \param parent_zero_fraction what fraction of the parent path weight is coming as 0 (integrated)
 * \param parent_one_fraction what fraction of the parent path weight is coming as 1 (fixed)
 * \param parent_feature_index what feature the parent node used to split
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param condition_feature the index of the feature to fix
 * \param condition_fraction what fraction of the current weight matches our conditioning feature
 */
void TreeShap(RegTree const& tree, const RegTree::FVec& feat, float* phi, bst_node_t node_index,
              std::uint32_t unique_depth, PathElement* parent_unique_path,
              float parent_zero_fraction, float parent_one_fraction, int parent_feature_index,
              int condition, std::uint32_t condition_feature, float condition_fraction) {
  const auto node = tree[node_index];

  // stop if we have no weight coming down to us
  if (condition_fraction == 0) return;

  // extend the unique path
  PathElement* unique_path = parent_unique_path + unique_depth + 1;
  std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

  if (condition == 0 || condition_feature != static_cast<std::uint32_t>(parent_feature_index)) {
    ExtendPath(unique_path, unique_depth, parent_zero_fraction, parent_one_fraction,
               parent_feature_index);
  }
  const std::uint32_t split_index = node.SplitIndex();

  // leaf node
  if (node.IsLeaf()) {
    for (std::uint32_t i = 1; i <= unique_depth; ++i) {
      const float w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement& el = unique_path[i];
      phi[el.feature_index] +=
          w * (el.one_fraction - el.zero_fraction) * node.LeafValue() * condition_fraction;
    }

    // internal node
  } else {
    // find which branch is "hot" (meaning x would follow it)
    auto const& cats = tree.GetCategoriesMatrix();
    bst_node_t hot_index = predictor::GetNextNode<true, true>(
        node, node_index, feat.GetFvalue(split_index), feat.IsMissing(split_index), cats);

    const auto cold_index = (hot_index == node.LeftChild() ? node.RightChild() : node.LeftChild());
    const float w = tree.Stat(node_index).sum_hess;
    const float hot_zero_fraction = tree.Stat(hot_index).sum_hess / w;
    const float cold_zero_fraction = tree.Stat(cold_index).sum_hess / w;
    float incoming_zero_fraction = 1;
    float incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    std::uint32_t path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (static_cast<std::uint32_t>(unique_path[path_index].feature_index) == split_index) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    // divide up the condition_fraction among the recursive calls
    float hot_condition_fraction = condition_fraction;
    float cold_condition_fraction = condition_fraction;
    if (condition > 0 && split_index == condition_feature) {
      cold_condition_fraction = 0;
      unique_depth -= 1;
    } else if (condition < 0 && split_index == condition_feature) {
      hot_condition_fraction *= hot_zero_fraction;
      cold_condition_fraction *= cold_zero_fraction;
      unique_depth -= 1;
    }

    TreeShap(tree, feat, phi, hot_index, unique_depth + 1, unique_path,
             hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction, split_index,
             condition, condition_feature, hot_condition_fraction);

    TreeShap(tree, feat, phi, cold_index, unique_depth + 1, unique_path,
             cold_zero_fraction * incoming_zero_fraction, 0, split_index, condition,
             condition_feature, cold_condition_fraction);
  }
}

void CalculateContributions(RegTree const& tree, const RegTree::FVec& feat,
                            std::vector<float>* mean_values, float* out_contribs, int condition,
                            std::uint32_t condition_feature) {
  // find the expected value of the tree's predictions
  if (condition == 0) {
    float node_value = (*mean_values)[0];
    out_contribs[feat.Size()] += node_value;
  }

  // Preallocate space for the unique path data
  const int maxd = tree.MaxDepth(0) + 2;
  std::vector<PathElement> unique_path_data((maxd * (maxd + 1)) / 2);

  TreeShap(tree, feat, out_contribs, 0, 0, unique_path_data.data(), 1, 1, -1, condition,
           condition_feature, 1);
}
}  // namespace xgboost
