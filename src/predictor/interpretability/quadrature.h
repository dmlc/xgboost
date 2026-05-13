/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_
#define XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/logging.h"
#include "xgboost/tree_model.h"

namespace xgboost::interpretability::detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::size_t kQuadratureTreeShapPoints = 8;
constexpr float kQuadratureTreeShapUnseen = -999.0f;
constexpr float kQuadratureTreeShapMinChildWeight = 1e-12f;

XGBOOST_DEVICE inline float BranchWeight(float cover, float parent_cover) {
  // In a well-formed tree, split-node cover is positive and each child cover is a valid
  // fraction of the parent cover. Zero cover can still appear after model refresh or when
  // loading externally produced models, so fall back to a neutral branch probability instead
  // of allowing NaN/Inf weights to propagate through SHAP.
  if (parent_cover <= 0.0f) {
    return 0.5f;
  }
  auto weight = cover / parent_cover;
  // A zero-cover child is not expected for a normally trained split, but can occur in
  // refreshed trees. Keep the path reachable with a tiny probability so quadrature remains
  // numerically well-defined while preserving ordinary nonzero cover ratios unchanged.
  if (weight < kQuadratureTreeShapMinChildWeight) {
    return kQuadratureTreeShapMinChildWeight;
  }
  return weight;
}

struct QuadratureRule {
  float nodes[kQuadratureTreeShapPoints];
  float weights[kQuadratureTreeShapPoints];
};

inline double LegendrePolynomial(std::size_t n, double x) {
  double p0 = 1.0;
  if (n == 0) {
    return p0;
  }
  double p1 = x;
  if (n == 1) {
    return p1;
  }
  for (std::size_t k = 2; k <= n; ++k) {
    double pk =
        ((2.0 * static_cast<double>(k) - 1.0) * x * p1 - (static_cast<double>(k) - 1.0) * p0) /
        static_cast<double>(k);
    p0 = p1;
    p1 = pk;
  }
  return p1;
}

inline double LegendreDerivative(std::size_t n, double x, double pn) {
  auto n_d = static_cast<double>(n);
  return n_d * (x * pn - LegendrePolynomial(n - 1, x)) / (x * x - 1.0);
}

inline QuadratureRule MakeEndpointQuadrature() {
  constexpr std::size_t kN = kQuadratureTreeShapPoints;
  constexpr double kConvergenceEps = 1e-15;
  QuadratureRule rule;

  for (std::size_t i = 0; i < kN; ++i) {
    double theta = kPi * (static_cast<double>(i) + 0.75) / (static_cast<double>(kN) + 0.5);
    double x = std::cos(theta);
    for (std::size_t iter = 0; iter < 64; ++iter) {
      auto pn = LegendrePolynomial(kN, x);
      auto dpn = LegendreDerivative(kN, x, pn);
      auto dx = pn / dpn;
      x -= dx;
      if (std::abs(dx) < kConvergenceEps) {
        break;
      }
    }

    auto pn = LegendrePolynomial(kN, x);
    auto dpn = LegendreDerivative(kN, x, pn);
    auto w = 2.0 / ((1.0 - x * x) * dpn * dpn);
    double s = 0.5 * (x + 1.0);
    double ws = 0.5 * w;
    auto out_idx = kN - 1 - i;
    rule.nodes[out_idx] = static_cast<float>(s * s);
    rule.weights[out_idx] = static_cast<float>(2.0 * s * ws);
  }
  return rule;
}

inline QuadratureRule const& GetQuadratureRule() {
  static QuadratureRule const kRule = MakeEndpointQuadrature();
  return kRule;
}

template <typename Tree, typename LeafValueFn>
double FillRootMeanValue(Tree const& tree, bst_node_t nidx, LeafValueFn const& leaf_value) {
  if (tree.IsLeaf(nidx)) {
    return leaf_value(nidx);
  }
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  CHECK_GE(tree.SumHess(nidx), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with negative cover at split nodes.";
  CHECK_GE(tree.SumHess(left), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with negative child cover.";
  CHECK_GE(tree.SumHess(right), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with negative child cover.";
  auto const parent_cover = tree.SumHess(nidx);
  auto const left_mean = FillRootMeanValue(tree, left, leaf_value);
  auto const right_mean = FillRootMeanValue(tree, right, leaf_value);
  if (parent_cover == 0.0f) {
    return 0.5 * (left_mean + right_mean);
  }
  return (left_mean * tree.SumHess(left) + right_mean * tree.SumHess(right)) / parent_cover;
}

template <typename Tree>
double FillRootMeanValue(Tree const& tree, bst_node_t nidx) {
  return FillRootMeanValue(tree, nidx, [&](bst_node_t leaf) { return tree.LeafValue(leaf); });
}

template <typename TreeGroups, typename GetTree>
std::vector<float> MakeGroupRootMeanSums(TreeGroups const& tree_groups, bst_target_t n_groups,
                                         bst_tree_t tree_end,
                                         std::vector<float> const* tree_weights,
                                         GetTree&& get_tree) {
  std::vector<double> group_root_mean_sums(n_groups, 0.0);
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    group_root_mean_sums[tree_groups[tree_idx]] +=
        FillRootMeanValue(get_tree(tree_idx), RegTree::kRoot) * weight;
  }

  std::vector<float> out(group_root_mean_sums.size());
  std::transform(group_root_mean_sums.cbegin(), group_root_mean_sums.cend(), out.begin(),
                 [](double v) { return static_cast<float>(v); });
  return out;
}

}  // namespace xgboost::interpretability::detail

#endif  // XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_
