/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_
#define XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/logging.h"

namespace xgboost::interpretability::detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::size_t kQuadratureTreeShapPoints = 8;
constexpr double kQuadratureTreeShapBuildQeps = 1e-15;
constexpr float kQuadratureTreeShapUnseen = -999.0f;
constexpr float kQuadratureTreeShapMinChildWeight = 1e-12f;

XGBOOST_DEVICE inline float GuardChildWeight(float cover, float parent_cover) {
  auto weight = cover / parent_cover;
  if (weight < kQuadratureTreeShapMinChildWeight) {
    return kQuadratureTreeShapMinChildWeight;
  }
  return weight;
}

template <std::size_t Points>
struct FloatQuadratureRule {
  float nodes[Points];
  float weights[Points];
};

using QuadratureTreeShapRule = FloatQuadratureRule<kQuadratureTreeShapPoints>;

template <std::size_t MaxPoints>
struct EndpointQuadratureRule {
  std::size_t points{0};
  std::array<double, MaxPoints> nodes{};
  std::array<double, MaxPoints> weights{};
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

template <std::size_t MaxPoints>
inline EndpointQuadratureRule<MaxPoints> MakeEndpointQuadrature(std::size_t n,
                                                                double convergence_eps) {
  CHECK_GE(n, 2);
  CHECK_LE(n, MaxPoints);

  EndpointQuadratureRule<MaxPoints> rule;
  rule.points = n;
  std::vector<std::pair<double, double>> nodes_weights;
  nodes_weights.reserve(n);

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
    nodes_weights.emplace_back(s * s, 2.0 * s * ws);
  }

  std::sort(nodes_weights.begin(), nodes_weights.end(),
            [](auto const& l, auto const& r) { return l.first < r.first; });
  for (std::size_t i = 0; i < n; ++i) {
    rule.nodes[i] = nodes_weights[i].first;
    rule.weights[i] = nodes_weights[i].second;
  }
  return rule;
}

template <std::size_t Points>
inline EndpointQuadratureRule<Points> MakeEndpointQuadrature(double convergence_eps) {
  return MakeEndpointQuadrature<Points>(Points, convergence_eps);
}

template <std::size_t Points>
inline FloatQuadratureRule<Points> MakeFloatQuadratureRule(double convergence_eps) {
  auto const rule_d = MakeEndpointQuadrature<Points>(convergence_eps);
  FloatQuadratureRule<Points> out;
  for (std::size_t i = 0; i < Points; ++i) {
    out.nodes[i] = static_cast<float>(rule_d.nodes[i]);
    out.weights[i] = static_cast<float>(rule_d.weights[i]);
  }
  return out;
}

template <typename Tree>
double FillRootMeanValue(Tree const& tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return tree.LeafValue(nidx);
  }
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  double result = FillRootMeanValue(tree, left) * tree.SumHess(left);
  result += FillRootMeanValue(tree, right) * tree.SumHess(right);
  result /= tree.SumHess(nidx);
  return result;
}

template <typename Tree>
void ValidateQuadratureTreeShapCovers(Tree const& tree, bst_node_t nidx, char const* device_name) {
  if (tree.IsLeaf(nidx)) {
    return;
  }

  CHECK_GT(tree.SumHess(nidx), 0.0f)
      << device_name
      << " QuadratureTreeSHAP is undefined for trees with non-positive cover at split nodes.";

  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  CHECK_GE(tree.SumHess(left), 0.0f)
      << device_name << " QuadratureTreeSHAP is undefined for trees with negative child cover.";
  CHECK_GE(tree.SumHess(right), 0.0f)
      << device_name << " QuadratureTreeSHAP is undefined for trees with negative child cover.";

  ValidateQuadratureTreeShapCovers(tree, left, device_name);
  ValidateQuadratureTreeShapCovers(tree, right, device_name);
}

template <typename Lhs, typename Rhs>
XGBOOST_DEVICE void AddQuadratureInPlace(Lhs&& lhs, Rhs&& rhs) {
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    lhs[i] += rhs[i];
  }
}

template <typename Rule, typename HVals>
XGBOOST_DEVICE float ExtractQuadratureDelta(Rule const& rule, HVals&& h_vals, float p_enter,
                                            float p_exit) {
  float acc = 0.0f;
  if (p_enter != 1.0f) {
    auto const alpha_enter = p_enter - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc += alpha_enter * h_vals[i] / (1.0f + alpha_enter * rule.nodes[i]);
    }
  }
  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc -= alpha_exit * h_vals[i] / (1.0f + alpha_exit * rule.nodes[i]);
    }
  }
  return acc;
}

template <typename Rule, typename HVals>
XGBOOST_DEVICE float ExtractQuadratureInteractionDelta(Rule const& rule, HVals&& h_vals,
                                                       float p_enter, float p_exit,
                                                       float q_partner) {
  if (q_partner == 1.0f) {
    return 0.0f;
  }

  auto const alpha_partner = q_partner - 1.0f;
  auto const alpha_enter = p_enter - 1.0f;

  float acc = 0.0f;
  if (p_exit == 1.0f) {
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  } else {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]) -
                              alpha_exit / (1.0f + alpha_exit * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  }
  return acc;
}

template <typename Rule, typename CVals, typename Out>
XGBOOST_DEVICE void WriteQuadratureLeafReturn(Rule const& rule, CVals&& c_vals, float leaf_scale,
                                              Out&& out_h) {
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    out_h[i] = c_vals[i] * leaf_scale * rule.weights[i];
  }
}

}  // namespace xgboost::interpretability::detail

#endif  // XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_
