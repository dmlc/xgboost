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
  EllpackPageImpl* page;
  /*!\brief Gradient pairs for the sampled rows. */
  common::Span<GradientPair> gpair;
};

class SamplingStrategy {
 public:
  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  virtual GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) = 0;
};

/*! \brief No sampling in in-memory mode. */
class NoSampling : public SamplingStrategy {
 public:
  explicit NoSampling(EllpackPageImpl* page);

  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;

 private:
  EllpackPageImpl* page_;
};

/*! \brief No sampling in external memory mode. */
class ExternalMemoryNoSampling : public SamplingStrategy {
 public:
  ExternalMemoryNoSampling(EllpackPageImpl* page,
                           size_t n_rows,
                           const BatchParam& batch_param);

  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;

 private:
  /*! \brief Concatenate all the rows from a DMatrix into a single ELLPACK page. */
  void ConcatenatePages(DMatrix* dmat);

  BatchParam batch_param_;
  std::unique_ptr<EllpackPageImpl> page_;
  bool page_concatenated_{false};
};

/*! \brief Uniform sampling in in-memory mode. */
class UniformSampling : public SamplingStrategy {
 public:
  UniformSampling(EllpackPageImpl* page, float subsample);

  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;

 private:
  EllpackPageImpl* page_;
  float subsample_;
};

/*! \brief No sampling in external memory mode. */
class ExternalMemoryUniformSampling : public SamplingStrategy {
 public:
  ExternalMemoryUniformSampling(float subsample);

  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;

 private:
  dh::BulkAllocator ba_;
  EllpackPageImpl* original_page_;
  float subsample_;
  BatchParam batch_param_;
  std::unique_ptr<EllpackPageImpl> page_;
  common::Span<GradientPair> gpair_;
  common::Span<size_t> sample_row_index_;
};

class GradientBasedSampling : public SamplingStrategy {
 public:
  /*! \brief Gradient-based sampling in in-memory mode.. */
  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;
};

class ExternalMemoryGradientBasedSampling : public SamplingStrategy {
 public:
  /*! \brief Gradient-based sampling in external memory mode.. */
  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat) override;
};

/*! \brief Draw a sample of rows from a DMatrix.
 *
 * \see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * \see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * \see Ohlsson, E. (1998). Sequential poisson sampling. Journal of official Statistics, 14(2), 149.
 */
class GradientBasedSampler {
 public:
  GradientBasedSampler(EllpackPageImpl* page,
                       size_t n_rows,
                       const BatchParam& batch_param,
                       float subsample,
                       int sampling_method);

  /*! \brief Sample from a DMatrix based on the given gradient pairs. */
  GradientBasedSample Sample(common::Span<GradientPair> gpair, DMatrix* dmat);

 private:
  /*! \brief Calculate the threshold used to normalize sampling probabilities. */
  size_t CalculateThresholdIndex(common::Span<GradientPair> gpair);

  /*! \brief Fixed-size Poisson sampling after the row weights are calculated. */
  GradientBasedSample SequentialPoissonSampling(common::Span<GradientPair> gpair, DMatrix* dmat);

  common::Monitor monitor_;
  std::unique_ptr<SamplingStrategy> strategy_;


  dh::BulkAllocator ba_;
  EllpackPageImpl* original_page_;
  float subsample_;
  bool is_external_memory_;
  bool is_sampling_;
  BatchParam batch_param_;
  int sampling_method_;
  size_t sample_rows_;
  std::unique_ptr<EllpackPageImpl> page_;
  common::Span<GradientPair> gpair_;
  common::Span<float> row_weight_;
  common::Span<float> threshold_;
  common::Span<size_t> row_index_;
  common::Span<size_t> sample_row_index_;
  bool page_concatenated_{false};
};
};  // namespace tree
};  // namespace xgboost
