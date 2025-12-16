/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#include "treeshap.h"

#include <algorithm>  // copy
#include <cmath>      // std::tgamma
#include <cstdint>    // std::uint32_t
#include <iostream>   // std::cout
#include <map>        // std::map
#include <set>        // std::set
#include <vector>     // std::vector

#include "../tree/tree_view.h"  // for ScalarTreeView
#include "predict_fn.h"         // GetNextNode
#include "xgboost/base.h"       // bst_node_t
#include "xgboost/logging.h"
#include "xgboost/tree_model.h"  // RegTree

namespace xgboost {
void CalculateContributionsApprox(tree::ScalarTreeView const& tree, const RegTree::FVec& feat,
                                  std::vector<float>* mean_values, float* out_contribs) {
  CHECK_GT(mean_values->size(), 0U);
  bst_feature_t split_index = 0;
  // update bias value
  float node_value = (*mean_values)[0];
  out_contribs[feat.Size()] += node_value;
  if (tree.IsLeaf(RegTree::kRoot)) {
    // nothing to do anymore
    return;
  }

  bst_node_t nidx = 0;
  auto const& cats = tree.GetCategoriesMatrix();

  while (!tree.IsLeaf(nidx)) {
    split_index = tree.SplitIndex(nidx);
    nidx = predictor::GetNextNode<true, true>(tree, nidx, feat.GetFvalue(split_index),
                                              feat.IsMissing(split_index), cats);
    bst_float new_value = (*mean_values)[nidx];
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  float leaf_value = tree.LeafValue(nidx);
  // update leaf feature weight
  out_contribs[split_index] += leaf_value - node_value;
}

std::uint64_t ExtractBinaryPath(tree::ScalarTreeView const& tree, const RegTree::FVec& feat, 
                                                       std::uint64_t leaf_path) {
  std::uint64_t consumer_path = 0;
  bst_node_t nidx = 0;
  int depth =0;
  while (!tree.IsLeaf(nidx)) {
    auto split_index = tree.SplitIndex(nidx);
    auto child_index = predictor::GetNextNode<true, true>(
        tree, nidx, feat.GetFvalue(split_index), feat.IsMissing(split_index), tree.GetCategoriesMatrix());
    auto path_index = leaf_path & 1 ? tree.RightChild(nidx) : tree.LeftChild(nidx);
    leaf_path >>= 1;
    consumer_path |= (child_index == path_index) << depth;
    nidx = path_index;
    depth++;
  }


  return consumer_path;
}

// Sparse storage for leaf information
struct LeafData {
  bst_node_t node_id;
  double leaf_weight;
  double probability;  // Probability of reaching this leaf
  std::vector<int> features;
  std::uint64_t path;
};

void GetLeafDataRecursive(tree::ScalarTreeView const& tree,
                   std::vector<LeafData>* leaf_data, bst_node_t nidx,  
                   std::vector<int> const& features, std::uint64_t path, int depth, double probability) {

  if (tree.IsLeaf(nidx)) {
    // Store sparse leaf data
    leaf_data->push_back({nidx, tree.LeafValue(nidx), probability, features, path});
  } else {
    const std::uint32_t split_index = tree.SplitIndex(nidx);
    // Create updated feature list and path for children
    std::vector<int> child_features = features;
    child_features.push_back(split_index);

    auto left_probability = tree.SumHess(tree.LeftChild(nidx)) / tree.SumHess(nidx);
    auto right_probability = 1.0 - left_probability;
    
    GetLeafDataRecursive(tree, leaf_data, tree.LeftChild(nidx), child_features, path, depth + 1, probability * left_probability);
    GetLeafDataRecursive(tree, leaf_data, tree.RightChild(nidx), child_features, path | (1ULL << depth), depth + 1, probability * right_probability);
  }
}

// Type alias for the pattern-to-cube mapping structure
// Maps (consumer_pattern, background_pattern) -> (S+, S-)
using PatternCubeMap = std::map<std::uint64_t, std::map<std::uint64_t, std::pair<std::set<int>, std::set<int>>>>;

PatternCubeMap MapPatternsToCube(const std::vector<int>& features_in_path) {
  PatternCubeMap updated_wdnf_table;
  
  // Initialize with the root: empty consumer and background patterns
  updated_wdnf_table[0][0] = {std::set<int>(), std::set<int>()};
  
  for (auto feature : features_in_path) {
    PatternCubeMap current_wdnf_table = updated_wdnf_table;
    updated_wdnf_table.clear();
    
    for (const auto& [consumer_pattern, background_map] : current_wdnf_table) {
      for (const auto& [background_pattern, cube] : background_map) {
        const auto& [s_plus, s_minus] = cube;
        
        // Rule 1: Consumer takes feature=1, Background takes feature=0
        std::uint64_t new_consumer_1 = (consumer_pattern << 1) | 1;
        std::uint64_t new_background_0 = (background_pattern << 1) | 0;
        auto& cube_1_0 = updated_wdnf_table[new_consumer_1][new_background_0];
        cube_1_0.first = s_plus;
        cube_1_0.first.insert(feature);
        cube_1_0.second = s_minus;
        
        // Rule 2: Consumer takes feature=0, Background takes feature=1
        std::uint64_t new_consumer_0 = (consumer_pattern << 1) | 0;
        std::uint64_t new_background_1 = (background_pattern << 1) | 1;
        auto& cube_2_1 = updated_wdnf_table[new_consumer_0][new_background_1];
        cube_2_1.first = s_plus;
        cube_2_1.second = s_minus;
        cube_2_1.second.insert(feature);
        
        // Rule 3: Both take feature=1 (feature is not in coalition)
        std::uint64_t new_both_1 = (consumer_pattern << 1) | 1;
        std::uint64_t new_background_both_1 = (background_pattern << 1) | 1;
        auto& cube_1_1 = updated_wdnf_table[new_both_1][new_background_both_1];
        cube_1_1.first = s_plus;
        cube_1_1.second = s_minus;
      }
    }
  }
  
  return updated_wdnf_table;
}

std::vector<std::pair<int, double>> v(const std::pair<std::set<int>, std::set<int>>& cube) {
  auto &[s_plus, s_minus] = cube;
  std::vector<std::pair<int, double>> res;
  
  auto n_choose_k = [](int n, int k) {
    return tgamma(n + 1) / (tgamma(k + 1) * tgamma(n - k + 1));
  };
  if (!s_plus.empty() && !s_minus.empty()) {
    std::set<int> intersection;
    std::set_intersection(s_plus.begin(), s_plus.end(),
                          s_minus.begin(), s_minus.end(),
                          std::inserter(intersection, intersection.begin()));
    if (!intersection.empty()) {
      return res; // s_plus and s_minus must be disjoint
    }
  }
  std::set<int> s;
  std::set_union(s_plus.begin(), s_plus.end(),
                 s_minus.begin(), s_minus.end(),
                 std::inserter(s, s.begin()));
  if (!s_plus.empty()) {
    double contribution = 1.0 / (static_cast<double>(s_plus.size()) *
                                static_cast<double>(n_choose_k(s.size(), s_plus.size())));
    for (auto must_exist_feature : s_plus) {
      res.emplace_back(must_exist_feature, contribution);
    }
  }
  if (!s_minus.empty()) {
    double contribution = -1.0 / (static_cast<double>(s_minus.size()) *
                                 static_cast<double>(n_choose_k(s.size(), s_minus.size())));
    for (auto must_be_missing_feature : s_minus) {
      res.emplace_back(must_be_missing_feature, contribution);
    }
  }
  return res;
}

std::map<int, std::vector<double>> path_dependant_frequencies(tree::ScalarTreeView const& tree){
  std::map<int, std::vector<double>> leaves_freq;
  std::map<int, std::vector<double>> inner_freq;
  inner_freq[0] = {1.0};
  // Walk tree breadth first
  tree.WalkTree([&](auto const& node) {
    auto& current_freq = inner_freq[node];
    if(tree.IsLeaf(node)) {
      leaves_freq[node] = inner_freq[node];
    } else {
      auto left_prob = tree.SumHess(tree.LeftChild(node)) / tree.SumHess(node);
      auto right_prob = tree.SumHess(tree.RightChild(node)) / tree.SumHess(node);
      std::vector<double> left;
      left.reserve(current_freq.size()*2);
      for(auto freq: current_freq){
        left.emplace_back(freq * right_prob);
        left.emplace_back(freq * left_prob);
      }
      inner_freq[tree.LeftChild(node)] = left;

      std::vector<double> right;
      right.reserve(current_freq.size()*2);
      for(auto freq: current_freq){
        right.emplace_back(freq * left_prob);
        right.emplace_back(freq * right_prob);
      }
      inner_freq[tree.RightChild(node)] = right;
    }
    return true;
  });
  return leaves_freq;
}

std::vector<PreprocessedLeaf> PreprocessTree(int tree_idx, tree::ScalarTreeView const& tree) {

    std::vector<LeafData> leaf_data;
    leaf_data.reserve(tree.Size());
    GetLeafDataRecursive(tree, &leaf_data, 0, {}, 0, 0, 1.0);
    auto f = path_dependant_frequencies(tree);
    
    std::vector<PreprocessedLeaf> preprocessed_leaves(leaf_data.size());
    for(std::size_t i = 0; i < leaf_data.size(); ++i) {
      const auto& leaf = leaf_data[i];

      // Get the pattern-to-cube mapping for this leaf's path
      auto pc_pb_to_cube = MapPatternsToCube(leaf.features);
      const std::size_t n_paths = 1 << leaf.features.size();

      // Create dense matrices for each feature: row=consumer_pattern, col=background_pattern
      std::map<int, std::vector<double>> M;
      for (auto feature : leaf.features) {
        M[feature] = std::vector<double>(n_paths * n_paths, 0.0);
      }

      // Fill M matrices with contributions from v(cube)
      for (const auto& [consumer_pattern, background_map] : pc_pb_to_cube) {
        for (const auto& [background_pattern, cube] : background_map) {
          const auto values = v(cube);
          for (const auto& [feature, contribution] : values) {
            auto& mat = M[feature];
            const std::size_t idx = static_cast<std::size_t>(consumer_pattern) * n_paths +
                                    static_cast<std::size_t>(background_pattern);
            if (idx < mat.size()) {
              mat[idx] = contribution;
            }
          }
        }
      }
      
      // Compute S: full matrix-vector multiplication S = w * M * f_l
      auto&preprocessed_leaf = preprocessed_leaves[i];
      preprocessed_leaf.tree_idx = tree_idx;
      preprocessed_leaf.leaf_path = leaf.path;
      preprocessed_leaf.null_coalition_weight = leaf.probability * leaf.leaf_weight;
      auto &S = preprocessed_leaf.S;
      auto &f_l = f[leaf.node_id];
      for (auto feature : leaf.features) {
        const auto& mat = M[feature];
        S[feature] = std::vector<double>(n_paths, 0.0);
        auto& S_vec = S[feature];
        for (std::size_t row = 0; row < n_paths; ++row) {
          double dot = 0.0;
          for (std::size_t col = 0; col < n_paths && col < f_l.size(); ++col) {
            dot += mat[row * n_paths + col] * f_l[col];
          }
          S_vec[row] = dot * leaf.leaf_weight;
        }
      }
    }
  return preprocessed_leaves;
}

}  // namespace xgboost
