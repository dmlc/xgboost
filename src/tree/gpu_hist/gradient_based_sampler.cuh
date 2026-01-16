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

namespace xgboost::tree {
class SamplingStrategy {
 public:
  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  virtual void Sample(Context const* ctx, linalg::VectorView<GradientPairInt64> gpair,
                      GradientQuantiser const& rounding, DMatrix* dmat) = 0;
  virtual ~SamplingStrategy() = default;
};

/** @brief No-op. */
class NoSampling : public SamplingStrategy {
 public:
  void Sample(Context const*, linalg::VectorView<GradientPairInt64>, GradientQuantiser const&,
              DMatrix*) override {}
};

/**
 * @brief Uniform sampling in in-memory mode.
 */
class UniformSampling : public SamplingStrategy {
 public:
  UniformSampling(BatchParam batch_param, float subsample);
  void Sample(Context const* ctx, linalg::VectorView<GradientPairInt64> gpair,
              GradientQuantiser const& rounding, DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
};

/** @brief Gradient-based sampling. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(std::size_t n_rows, BatchParam batch_param, float subsample);
  void Sample(Context const* ctx, linalg::VectorView<GradientPairInt64> gpair,
              GradientQuantiser const& rounding, DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  dh::device_vector<float> threshold_;
  dh::device_vector<float> grad_sum_;
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
  GradientBasedSampler(Context const* ctx, size_t n_rows, const BatchParam& batch_param,
                       float subsample, int sampling_method);

  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  void Sample(Context const* ctx, linalg::VectorView<GradientPairInt64> gpair,
              GradientQuantiser const& rounding, DMatrix* dmat);

 private:
  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;
};
};  // namespace xgboost::tree
