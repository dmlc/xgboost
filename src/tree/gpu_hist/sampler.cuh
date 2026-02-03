/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t

#include "../../common/device_vector.cuh"  // for device_vector, caching_device_vector
#include "../../common/timer.h"            // for Monitor
#include "quantiser.cuh"                   // for GradientQuantiser
#include "xgboost/base.h"                  // for GradientPair
#include "xgboost/data.h"                  // for BatchParam
#include "xgboost/linalg.h"                // for MatrixView

namespace xgboost::tree::cuda_impl {
class SamplingStrategy {
 public:
  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  virtual void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                      common::Span<GradientQuantiser const> roundings) = 0;
  virtual ~SamplingStrategy() = default;
};

/** @brief No-op. */
class NoSampling : public SamplingStrategy {
 public:
  void Sample(Context const*, linalg::MatrixView<GradientPairInt64>,
              common::Span<GradientQuantiser const>) override {}
};

/** @brief Uniform sampling */
class UniformSampling : public SamplingStrategy {
 public:
  explicit UniformSampling(float subsample) : subsample_{subsample} {}
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings) override;

 private:
  float subsample_;
};

/** @brief Gradient-based sampling. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(std::size_t n_samples, float subsample);
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings) override;

 private:
  float subsample_;
  // abs gradient
  dh::device_vector<float> reg_abs_grad_;
  // sorted abs gradient
  dh::device_vector<float> thresholds_;
  // csum of sorted abs gradient
  dh::device_vector<float> grad_csum_;
};

/**
 * @brief Draw sample rows by setting non-selected gradient to 0.
 *
 * @see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * @see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * @see Ohlsson, E. (1998). Sequential Poisson sampling. Journal of official Statistics, 14(2), 149.
 * @see Rong Ou. (2020). Out-of-Core GPU Gradient Boosting.
 */
class GradientBasedSampler {
 public:
  GradientBasedSampler(bst_idx_t n_samples, float subsample, int sampling_method);

  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings);

 private:
  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;
};

/**
 * @brief Apply sampling mask from sampled split gradient to value gradient.
 */
void ApplySampling(Context const* ctx, linalg::Matrix<GradientPairInt64> const& sampled_split_gpair,
                   linalg::Matrix<GradientPair>* value_gpair);

std::size_t CalculateThresholdIndex(Context const* ctx, common::Span<float> sorted_rag,
                                    common::Span<float> grad_csum, bst_idx_t n_samples,
                                    bst_idx_t sample_rows);
}  // namespace xgboost::tree::cuda_impl
