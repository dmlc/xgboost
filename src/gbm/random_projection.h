/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t

#include "xgboost/base.h"     // for GradientPair
#include "xgboost/context.h"  // for Context
#include "xgboost/linalg.h"   // for Matrix, MatrixView
#include "xgboost/logging.h"  // for CHECK_EQ

namespace xgboost::gbm {
/**
 * @brief Draw an objective-independent orthonormal Gaussian basis.
 */
linalg::Matrix<double> MakeRandomProjection(Context const* ctx, std::size_t n_targets,
                                            std::size_t n_components);

/**
 * @brief Project gradients with Q and diagonal Hessians with elementwise Q squared.
 */
struct ProjectGradientKernel {
  using Signature = void(Context const*, linalg::MatrixView<GradientPair const>,
                         linalg::MatrixView<double const>, linalg::MatrixView<GradientPair>);

  linalg::MatrixView<GradientPair const> in;
  linalg::MatrixView<double const> basis;
  linalg::MatrixView<GradientPair> out;

  ProjectGradientKernel(linalg::MatrixView<GradientPair const> in,
                        linalg::MatrixView<double const> basis,
                        linalg::MatrixView<GradientPair> out)
      : in{in}, basis{basis}, out{out} {
    CHECK_EQ(in.Shape(0), out.Shape(0));
    CHECK_EQ(in.Shape(1), basis.Shape(0));
    CHECK_EQ(out.Shape(1), basis.Shape(1));
  }

  XGBOOST_DEVICE void operator()(std::size_t i, std::size_t j) {
    double grad{0.0}, hess{0.0};
    for (std::size_t t = 0; t < in.Shape(1); ++t) {
      auto q = basis(t, j);
      grad += q * in(i, t).GetGrad();
      // diag(Q^T diag(h_i) Q); off-diagonal curvature is omitted for split search.
      hess += q * q * in(i, t).GetHess();
    }
    out(i, j) = GradientPair{static_cast<float>(grad), static_cast<float>(hess)};
  }
};
}  // namespace xgboost::gbm
