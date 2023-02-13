/*!
 * Copyright 2019 by XGBoost Contributors
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/span.h>

#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"

namespace xgboost {
namespace tree {

struct GradientBasedSample {
  /*!\brief Number of sampled rows. */
  size_t sample_rows;
  /*!\brief Sampled rows in ELLPACK format. */
  EllpackPageImpl const* page;
  /*!\brief Gradient pairs for the sampled rows. */
  common::Span<GradientPair> gpair;
};

class SamplingStrategy {
 public:
  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  virtual GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                                     DMatrix* dmat) = 0;
  virtual ~SamplingStrategy() = default;
};

/*! \brief No sampling in in-memory mode. */
class NoSampling : public SamplingStrategy {
 public:
  explicit NoSampling(EllpackPageImpl const* page);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  EllpackPageImpl const* page_;
};

/*! \brief No sampling in external memory mode. */
class ExternalMemoryNoSampling : public SamplingStrategy {
 public:
  ExternalMemoryNoSampling(Context const* ctx, EllpackPageImpl const* page, size_t n_rows,
                           BatchParam batch_param);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  std::unique_ptr<EllpackPageImpl> page_;
  bool page_concatenated_{false};
};

/*! \brief Uniform sampling in in-memory mode. */
class UniformSampling : public SamplingStrategy {
 public:
  UniformSampling(EllpackPageImpl const* page, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  EllpackPageImpl const* page_;
  float subsample_;
};

/*! \brief No sampling in external memory mode. */
class ExternalMemoryUniformSampling : public SamplingStrategy {
 public:
  ExternalMemoryUniformSampling(size_t n_rows, BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  std::unique_ptr<EllpackPageImpl> page_;
  dh::device_vector<GradientPair> gpair_{};
  dh::caching_device_vector<size_t> sample_row_index_;
};

/*! \brief Gradient-based sampling in in-memory mode.. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(EllpackPageImpl const* page, size_t n_rows, const BatchParam& batch_param,
                        float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  EllpackPageImpl const* page_;
  float subsample_;
  dh::caching_device_vector<float> threshold_;
  dh::caching_device_vector<float> grad_sum_;
};

/*! \brief Gradient-based sampling in external memory mode.. */
class ExternalMemoryGradientBasedSampling : public SamplingStrategy {
 public:
  ExternalMemoryGradientBasedSampling(size_t n_rows, BatchParam batch_param, float subsample);
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair,
                             DMatrix* dmat) override;

 private:
  BatchParam batch_param_;
  float subsample_;
  dh::caching_device_vector<float> threshold_;
  dh::caching_device_vector<float> grad_sum_;
  std::unique_ptr<EllpackPageImpl> page_;
  dh::device_vector<GradientPair> gpair_;
  dh::caching_device_vector<size_t> sample_row_index_;
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
  GradientBasedSampler(Context const* ctx, EllpackPageImpl const* page, size_t n_rows,
                       const BatchParam& batch_param, float subsample, int sampling_method);

  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  GradientBasedSample Sample(Context const* ctx, common::Span<GradientPair> gpair, DMatrix* dmat);

  /*! \brief Calculate the threshold used to normalize sampling probabilities. */
  static size_t CalculateThresholdIndex(common::Span<GradientPair> gpair,
                                        common::Span<float> threshold,
                                        common::Span<float> grad_sum,
                                        size_t sample_rows);

 private:
  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;
};
};  // namespace tree
};  // namespace xgboost
