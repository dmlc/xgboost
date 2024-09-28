/**
 * Copyright 2019-2024, XGBoost Contributors
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
  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  virtual GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                                     DMatrix* dmat) = 0;
  virtual ~SamplingStrategy() = default;
  /**
   * @brief Whether pages are concatenated after sampling.
   */
  [[nodiscard]] virtual bool ConcatPages() const { return false; }
};

class ExtMemSamplingStrategy : public SamplingStrategy {
 public:
  [[nodiscard]] bool ConcatPages() const final { return true; }
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
  UniformSampling(BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
};

/*! \brief No sampling in external memory mode. */
class ExternalMemoryUniformSampling : public ExtMemSamplingStrategy {
 public:
  ExternalMemoryUniformSampling(size_t n_rows, BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  std::unique_ptr<DMatrix> p_fmat_new_{nullptr};
  dh::device_vector<GradientPair> gpair_{};
  dh::caching_device_vector<bst_idx_t> sample_row_index_;
  dh::device_vector<bst_idx_t> compact_row_index_;
};

/*! \brief Gradient-based sampling in in-memory mode.. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(std::size_t n_rows, BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  dh::caching_device_vector<float> threshold_;
  dh::caching_device_vector<float> grad_sum_;
};

/*! \brief Gradient-based sampling in external memory mode.. */
class ExternalMemoryGradientBasedSampling : public ExtMemSamplingStrategy {
 public:
  ExternalMemoryGradientBasedSampling(size_t n_rows, BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  dh::device_vector<float> threshold_;
  dh::device_vector<float> grad_sum_;
  std::unique_ptr<DMatrix> p_fmat_new_{nullptr};
  dh::device_vector<GradientPair> gpair_;
  dh::device_vector<bst_idx_t> sample_row_index_;
  dh::device_vector<bst_idx_t> compact_row_index_;
};

/*! \brief Draw a sample of rows from a DMatrix.
 *
 * \see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * \see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * \see Ohlsson, E. (1998). Sequential Poisson sampling. Journal of official Statistics, 14(2), 149.
 */
class GradientBasedSampler {
 public:
  GradientBasedSampler(Context const* ctx, size_t n_rows, const BatchParam& batch_param,
                       float subsample, int sampling_method, bool concat_pages);

  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair, DMatrix* dmat);

  /*! \brief Calculate the threshold used to normalize sampling probabilities. */
  static size_t CalculateThresholdIndex(Context const* ctx, common::Span<GradientPair> gpair,
                                        common::Span<float> threshold, common::Span<float> grad_sum,
                                        size_t sample_rows);

  [[nodiscard]] bool ConcatPages() const { return this->strategy_->ConcatPages(); }

 private:
  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;
};
};  // namespace xgboost::tree
