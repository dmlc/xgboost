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

#include "xgboost/logging.h"

namespace xgboost::interpretability::detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

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
            [](auto const &l, auto const &r) { return l.first < r.first; });
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

}  // namespace xgboost::interpretability::detail

#endif  // XGBOOST_PREDICTOR_INTERPRETABILITY_QUADRATURE_H_
