/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t

#include "../../common/device_vector.cuh"  // for device_vector, caching_device_vector
#include "../../common/timer.h"            // for Monitor
#include "xgboost/base.h"                  // for GradientPair
#include "xgboost/data.h"                  // for BatchParam
#include "xgboost/span.h"                  // for Span

namespace xgboost::tree {
struct GradientBasedSample {
  /** @brief Sampled rows in ELLPACK format. */
  DMatrix* p_fmat;
  /** @brief Gradient pairs for the sampled rows. */
  common::Span<GradientPair const> gpair;
};

class SamplingStrategy {
 public:
  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  virtual GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                                     DMatrix* dmat) = 0;
  virtual ~SamplingStrategy() = default;
};

/**
 * @brief No-op.
 */
class NoSampling : public SamplingStrategy {
 public:
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;
};

/**
 * @brief Uniform sampling in in-memory mode.
 */
class UniformSampling : public SamplingStrategy {
 public:
  explicit UniformSampling(float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  float subsample_;
};

/** @brief Gradient-based sampling. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(std::size_t n_rows, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  float subsample_;
  dh::caching_device_vector<float> threshold_;
  dh::caching_device_vector<float> grad_sum_;
};

/**
 * @brief Draw a sample of rows from a DMatrix.
 *
 * @see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * @see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * @see Ohlsson, E. (1998). Sequential Poisson sampling. Journal of official Statistics, 14(2), 149.
 * @see Rong Ou. (2020. Out-of-Core GPU Gradient Boosting
 */
class GradientBasedSampler {
 public:
  GradientBasedSampler(Context const* ctx, size_t n_rows, float subsample, int sampling_method);

  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair, DMatrix* dmat);

  /*! \brief Calculate the threshold used to normalize sampling probabilities. */
  static size_t CalculateThresholdIndex(Context const* ctx, common::Span<GradientPair> gpair,
                                        common::Span<float> threshold, common::Span<float> grad_sum,
                                        size_t sample_rows);

 private:
  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;
};
};  // namespace xgboost::tree
