/**
 * Copyright 2026, XGBoost Contributors
 */
#include "random_projection.h"

#include <cmath>    // for sqrt
#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t
#include <random>   // for normal_distribution

#include "../collective/broadcast.h"    // for Broadcast
#include "../common/kernel.h"           // for KernelRegistration
#include "../common/linalg_op.h"        // for ElementWiseKernel
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/logging.h"            // for CHECK_GT, CHECK_LE

namespace xgboost::gbm {
linalg::Matrix<double> MakeRandomProjection(Context const* ctx, std::size_t n_targets,
                                            std::size_t n_components) {
  CHECK_GT(n_components, 0);
  CHECK_LE(n_components, n_targets);

  std::uint32_t seed = ctx->Rng()();
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(collective::Broadcast(&cpu_ctx, linalg::MakeVec(&seed, 1), 0));
  RandomEngine rng{seed};
  std::normal_distribution<double> normal;
  auto basis = linalg::Empty<double>(&cpu_ctx, n_targets, n_components);
  // An orthonormal basis preserves the L2 regularization within the retained subspace. In
  // addition, constant Hessian objectives can keep the scale of the Hessian, which
  // preserves `min_child_weight`.
  auto q = basis.HostView();
  for (std::size_t j = 0; j < n_components; ++j) {
    for (std::size_t t = 0; t < n_targets; ++t) {
      q(t, j) = normal(rng);
    }
    // Two-pass modified Gram-Schmidt (MGS2).
    for (auto pass = 0; pass < 2; ++pass) {
      for (std::size_t prev = 0; prev < j; ++prev) {
        double dot{0.0};
        for (std::size_t t = 0; t < n_targets; ++t) {
          dot += q(t, prev) * q(t, j);
        }
        for (std::size_t t = 0; t < n_targets; ++t) {
          q(t, j) -= dot * q(t, prev);
        }
      }
    }
    // Reorthogonalize
    double norm{0.0};
    for (std::size_t t = 0; t < n_targets; ++t) {
      norm += q(t, j) * q(t, j);
    }
    norm = std::sqrt(norm);
    CHECK_GT(norm, 0.0);
    for (std::size_t t = 0; t < n_targets; ++t) {
      q(t, j) /= norm;
    }
  }
  return basis;
}

namespace {
void ProjectGradient(Context const* ctx, linalg::MatrixView<GradientPair const> in,
                     linalg::MatrixView<double const> basis, linalg::MatrixView<GradientPair> out) {
  linalg::cpu_impl::ElementWiseKernel(out, ctx->Threads(), ProjectGradientKernel{in, basis, out});
}

common::KernelRegistration<ProjectGradientKernel> const kProjectGradientCPU{DeviceOrd::kCPU,
                                                                            &ProjectGradient};
}  // namespace
}  // namespace xgboost::gbm
