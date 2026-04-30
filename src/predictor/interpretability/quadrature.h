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

XGBOOST_DEVICE inline float GuardChildWeight(float cover, float parent_cover) {
  auto weight = cover / parent_cover;
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
  constexpr std::size_t n = kQuadratureTreeShapPoints;
  constexpr double convergence_eps = 1e-15;
  QuadratureRule rule;

  for (std::size_t i = 0; i < n; ++i) {
    double theta = kPi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5);
    double x = std::cos(theta);
    for (std::size_t iter = 0; iter < 64; ++iter) {
      auto pn = LegendrePolynomial(n, x);
      auto dpn = LegendreDerivative(n, x, pn);
      auto dx = pn / dpn;
      x -= dx;
      if (std::abs(dx) < convergence_eps) {
        break;
      }
    }

    auto pn = LegendrePolynomial(n, x);
    auto dpn = LegendreDerivative(n, x, pn);
    auto w = 2.0 / ((1.0 - x * x) * dpn * dpn);
    double s = 0.5 * (x + 1.0);
    double ws = 0.5 * w;
    auto out_idx = n - 1 - i;
    rule.nodes[out_idx] = static_cast<float>(s * s);
    rule.weights[out_idx] = static_cast<float>(2.0 * s * ws);
  }
  return rule;
}

inline QuadratureRule const& GetQuadratureRule() {
  static QuadratureRule const kRule = MakeEndpointQuadrature();
  return kRule;
}

template <typename Tree>
double FillRootMeanValue(Tree const& tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return tree.LeafValue(nidx);
  }
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  CHECK_GT(tree.SumHess(nidx), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with non-positive cover at split nodes.";
  CHECK_GE(tree.SumHess(left), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with negative child cover.";
  CHECK_GE(tree.SumHess(right), 0.0f)
      << "QuadratureTreeSHAP is undefined for trees with negative child cover.";
  double result = FillRootMeanValue(tree, left) * tree.SumHess(left);
  result += FillRootMeanValue(tree, right) * tree.SumHess(right);
  result /= tree.SumHess(nidx);
  return result;
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
