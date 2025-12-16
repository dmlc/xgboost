/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>    // for GradientPair
#include <xgboost/linalg.h>  // for Matrix
#include <xgboost/linalg.h>  // for UnravelIndex, MatrixView
#include <xgboost/logging.h>
#include <xgboost/string_view.h>  // for StringView
#include <xgboost/data.h>

#include <cstddef>  // for size_t
#include <utility>  // for move

namespace xgboost {
namespace data {
class ArrayPageSource;
struct ArrayPage;

class HostGpairsCache;
}

/**
 * @brief Container for gradient produced by objective.
 */
struct GradientContainer {
 private:
  std::shared_ptr<data::HostGpairsCache> gpair_cache_;

 public:
  /** @brief Gradient used for multi-target tree split and linear model, required. */
  linalg::Matrix<GradientPair> gpair;
  /** @brief Gradient used for tree leaf value, optional. */
  linalg::Matrix<GradientPair> value_gpair;

  std::shared_ptr<data::ArrayPageSource> gpair_iter;

  [[nodiscard]] bool HasValueGrad() const noexcept { return !value_gpair.Empty(); }

  [[nodiscard]] std::size_t NumSplitTargets() const noexcept { return gpair.Shape(1); }
  [[nodiscard]] std::size_t NumTargets() const noexcept {
    return HasValueGrad() ? value_gpair.Shape(1) : this->gpair.Shape(1);
  }

  linalg::MatrixView<GradientPair const> ValueGrad(Context const* ctx) const {
    if (HasValueGrad()) {
      return this->value_gpair.View(ctx->Device());
    }
    return this->gpair.View(ctx->Device());
  }

  [[nodiscard]] linalg::Matrix<GradientPair> const* Grad() const { return &this->gpair; }
  [[nodiscard]] linalg::Matrix<GradientPair>* Grad() { return &this->gpair; }

  [[nodiscard]] linalg::Matrix<GradientPair> const* FullGradOnly() const {
    if (this->HasValueGrad()) {
      LOG(FATAL) << "Reduced gradient is not yet supported.";
    }
    return this->Grad();
  }
  [[nodiscard]] linalg::Matrix<GradientPair>* FullGradOnly() {
    if (this->HasValueGrad()) {
      LOG(FATAL) << "Reduced gradient is not yet supported.";
    }
    return this->Grad();
  }

  // Push a batch of split gradient
  void PushGrad(Context const* ctx, StringView grad, StringView hess);
  // Push a batch of value gradient
  void PushValueGrad(Context const* ctx, StringView grad, StringView hess);

  BatchSet<data::ArrayPage> GetGrad();
};

template <typename G, typename H>
struct CustomGradHessOp {
  linalg::MatrixView<G> t_grad;
  linalg::MatrixView<H> t_hess;
  linalg::MatrixView<GradientPair> d_gpair;

  CustomGradHessOp(linalg::MatrixView<G> t_grad, linalg::MatrixView<H> t_hess,
                   linalg::MatrixView<GradientPair> d_gpair)
      : t_grad{std::move(t_grad)}, t_hess{std::move(t_hess)}, d_gpair{std::move(d_gpair)} {}

  XGBOOST_DEVICE void operator()(std::size_t i) {
    auto [m, n] = linalg::UnravelIndex(i, t_grad.Shape(0), t_grad.Shape(1));
    auto g = t_grad(m, n);
    auto h = t_hess(m, n);
    // from struct of arrays to array of structs.
    d_gpair(m, n) = GradientPair{static_cast<float>(g), static_cast<float>(h)};
  }
};
}  // namespace xgboost
