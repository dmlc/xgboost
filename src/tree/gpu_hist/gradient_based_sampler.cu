/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <limits>

#include "../../common/compressed_iterator.h"
#include "../../common/random.h"
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

/*! \brief A functor that returns random weights. */
class RandomWeight : public thrust::unary_function<size_t, float> {
 public:
  explicit RandomWeight(size_t seed) : seed_(seed) {}

  XGBOOST_DEVICE float operator()(size_t i) const {
    thrust::default_random_engine rng(seed_);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng);
  }

 private:
  uint32_t seed_;
};

/*! \brief A functor that performs a Bernoulli trial to discard a gradient pair. */
class BernoulliTrial : public thrust::unary_function<size_t, bool> {
 public:
  BernoulliTrial(size_t seed, float p) : rnd_(seed), p_(p) {}

  XGBOOST_DEVICE bool operator()(size_t i) const {
    return rnd_(i) > p_;
  }

 private:
  RandomWeight rnd_;
  float p_;
};

/*! \brief A functor that returns true if the gradient pair is non-zero. */
struct IsNonZero : public thrust::unary_function<GradientPair, bool> {
  XGBOOST_DEVICE bool operator()(const GradientPair& gpair) const {
    return gpair.GetGrad() != 0 || gpair.GetHess() != 0;
  }
};

/*! \brief A functor that clears the row indexes with empty gradient. */
struct ClearEmptyRows : public thrust::binary_function<GradientPair, size_t, size_t> {
  XGBOOST_DEVICE size_t operator()(const GradientPair& gpair, size_t row_index) const {
    if (gpair.GetGrad() != 0 || gpair.GetHess() != 0) {
      return row_index;
    } else {
      return std::numeric_limits<std::size_t>::max();
    }
  }
};

/*! \brief A functor that combines the gradient pair into a single float.
 *
 * The approach here is based on Minimal Variance Sampling (MVS), with lambda set to 0.1.
 *
 * \see Ibragimov, B., & Gusev, G. (2019). Minimal Variance Sampling in Stochastic Gradient
 * Boosting. In Advances in Neural Information Processing Systems (pp. 15061-15071).
 */
class CombineGradientPair : public thrust::unary_function<GradientPair, float> {
 public:
  XGBOOST_DEVICE float operator()(const GradientPair& gpair) const {
    return sqrtf(powf(gpair.GetGrad(), 2) + kLambda * powf(gpair.GetHess(), 2));
  }

 private:
  static constexpr float kLambda = 0.1f;
};

/*! \brief A functor that calculates the difference between the sample rate and the desired sample
 * rows, given a cumulative gradient sum.
 */
class SampleRateDelta : public thrust::binary_function<float, size_t, float> {
 public:
  SampleRateDelta(common::Span<float> threshold, size_t n_rows, size_t sample_rows)
      : threshold_(threshold), n_rows_(n_rows), sample_rows_(sample_rows) {}

  XGBOOST_DEVICE float operator()(float gradient_sum, size_t row_index) const {
    float lower = threshold_[row_index];
    float upper = threshold_[row_index + 1];
    float u = gradient_sum / static_cast<float>(sample_rows_ - n_rows_ + row_index + 1);
    if (u > lower && u <= upper) {
      threshold_[row_index + 1] = u;
      return 0.0f;
    } else {
      return std::numeric_limits<float>::max();
    }
  }

 private:
  common::Span<float> threshold_;
  size_t n_rows_;
  size_t sample_rows_;
};

/*! \brief A functor that performs Poisson sampling, and scales gradient pairs by 1/p_i. */
class PoissonSampling : public thrust::binary_function<GradientPair, size_t, GradientPair> {
 public:
  PoissonSampling(common::Span<float> threshold, size_t threshold_index, RandomWeight rnd)
      : threshold_(threshold), threshold_index_(threshold_index), rnd_(rnd) {}

  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t i) {
    // If the gradient and hessian are both empty, we should never select this row.
    if (gpair.GetGrad() == 0 && gpair.GetHess() == 0) {
      return gpair;
    }
    float combined_gradient = combine_(gpair);
    float u = threshold_[threshold_index_];
    float p = combined_gradient / u;
    if (p >= 1) {
      // Always select this row.
      return gpair;
    } else {
      // Select this row randomly with probability proportional to the combined gradient.
      // Scale gpair by 1/p.
      if (rnd_(i) <= p) {
        return gpair / p;
      } else {
        return GradientPair();
      }
    }
  }

 private:
  common::Span<float> threshold_;
  size_t threshold_index_;
  RandomWeight rnd_;
  CombineGradientPair combine_;
};

NoSampling::NoSampling(EllpackPageImpl* page) : page_(page) {}

GradientBasedSample NoSampling::Sample(common::Span<GradientPair> gpair, DMatrix* dmat) {
  return {dmat->Info().num_row_, page_, gpair};
}

ExternalMemoryNoSampling::ExternalMemoryNoSampling(EllpackPageImpl* page,
                                                   size_t n_rows,
                                                   const BatchParam& batch_param)
    : batch_param_(batch_param),
      page_(new EllpackPageImpl(batch_param.gpu_id, page->Cuts(), page->is_dense,
                                page->row_stride, n_rows)) {}

GradientBasedSample ExternalMemoryNoSampling::Sample(common::Span<GradientPair> gpair,
                                                     DMatrix* dmat) {
  if (!page_concatenated_) {
    // Concatenate all the external memory ELLPACK pages into a single in-memory page.
    size_t offset = 0;
    for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
      auto page = batch.Impl();
      size_t num_elements = page_->Copy(batch_param_.gpu_id, page, offset);
      offset += num_elements;
    }
    page_concatenated_ = true;
  }
  return {dmat->Info().num_row_, page_.get(), gpair};
}

UniformSampling::UniformSampling(EllpackPageImpl* page, float subsample)
    : page_(page), subsample_(subsample) {}

GradientBasedSample UniformSampling::Sample(common::Span<GradientPair> gpair, DMatrix* dmat) {
  // Set gradient pair to 0 with p = 1 - subsample
  thrust::replace_if(dh::tbegin(gpair), dh::tend(gpair),
                     thrust::counting_iterator<size_t>(0),
                     BernoulliTrial(common::GlobalRandom()(), subsample_),
                     GradientPair());
  return {dmat->Info().num_row_, page_, gpair};
}

ExternalMemoryUniformSampling::ExternalMemoryUniformSampling(EllpackPageImpl* page,
                                                             size_t n_rows,
                                                             const BatchParam& batch_param,
                                                             float subsample)
    : original_page_(page),
      batch_param_(batch_param),
      subsample_(subsample),
      sample_row_index_(n_rows) {}

GradientBasedSample ExternalMemoryUniformSampling::Sample(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  // Set gradient pair to 0 with p = 1 - subsample
  thrust::replace_if(dh::tbegin(gpair), dh::tend(gpair),
                     thrust::counting_iterator<size_t>(0),
                     BernoulliTrial(common::GlobalRandom()(), subsample_),
                     GradientPair());

  // Count the sampled rows.
  size_t sample_rows = thrust::count_if(dh::tbegin(gpair), dh::tend(gpair), IsNonZero());

  // Compact gradient pairs.
  gpair_.resize(sample_rows);
  thrust::copy_if(dh::tbegin(gpair), dh::tend(gpair), gpair_.begin(), IsNonZero());

  // Index the sample rows.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair), sample_row_index_.begin(), IsNonZero());
  thrust::exclusive_scan(sample_row_index_.begin(), sample_row_index_.end(),
                         sample_row_index_.begin());
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    sample_row_index_.begin(),
                    sample_row_index_.begin(),
                    ClearEmptyRows());

  // Create a new ELLPACK page with empty rows.
  page_.reset();  // Release the device memory first before reallocating
  page_.reset(new EllpackPageImpl(
      batch_param_.gpu_id, original_page_->Cuts(), original_page_->is_dense,
                                  original_page_->row_stride, sample_rows));

  // Compact the ELLPACK pages into the single sample page.
  thrust::fill(dh::tbegin(page_->gidx_buffer), dh::tend(page_->gidx_buffer), 0);
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), dh::ToSpan(sample_row_index_));
  }

  return {sample_rows, page_.get(), dh::ToSpan(gpair_)};
}

GradientBasedSampling::GradientBasedSampling(EllpackPageImpl* page,
                                             size_t n_rows,
                                             const BatchParam&,
                                             float subsample)
    : page_(page),
      subsample_(subsample),
      threshold_(n_rows + 1, 0.0f),
      grad_sum_(n_rows, 0.0f) {}

GradientBasedSample GradientBasedSampling::Sample(common::Span<GradientPair> gpair,
                                                  DMatrix* dmat) {
  size_t n_rows = dmat->Info().num_row_;
  size_t threshold_index = GradientBasedSampler::CalculateThresholdIndex(
      gpair, dh::ToSpan(threshold_), dh::ToSpan(grad_sum_), n_rows * subsample_);

  // Perform Poisson sampling in place.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(gpair),
                    PoissonSampling(dh::ToSpan(threshold_),
                                    threshold_index,
                                    RandomWeight(common::GlobalRandom()())));
  return {n_rows, page_, gpair};
}

ExternalMemoryGradientBasedSampling::ExternalMemoryGradientBasedSampling(
    EllpackPageImpl* page,
    size_t n_rows,
    const BatchParam& batch_param,
    float subsample)
    : original_page_(page),
      batch_param_(batch_param),
      subsample_(subsample),
      threshold_(n_rows + 1, 0.0f),
      grad_sum_(n_rows, 0.0f),
      sample_row_index_(n_rows) {}

GradientBasedSample ExternalMemoryGradientBasedSampling::Sample(common::Span<GradientPair> gpair,
                                                                DMatrix* dmat) {
  size_t n_rows = dmat->Info().num_row_;
  size_t threshold_index = GradientBasedSampler::CalculateThresholdIndex(
      gpair, dh::ToSpan(threshold_), dh::ToSpan(grad_sum_), n_rows * subsample_);

  // Perform Poisson sampling in place.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(gpair),
                    PoissonSampling(dh::ToSpan(threshold_),
                                    threshold_index,
                                    RandomWeight(common::GlobalRandom()())));

  // Count the sampled rows.
  size_t sample_rows = thrust::count_if(dh::tbegin(gpair), dh::tend(gpair), IsNonZero());

  // Compact gradient pairs.
  gpair_.resize(sample_rows);
  thrust::copy_if(dh::tbegin(gpair), dh::tend(gpair), gpair_.begin(), IsNonZero());

  // Index the sample rows.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair), sample_row_index_.begin(), IsNonZero());
  thrust::exclusive_scan(sample_row_index_.begin(), sample_row_index_.end(),
    sample_row_index_.begin());
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    sample_row_index_.begin(),
                    sample_row_index_.begin(),
                    ClearEmptyRows());

  // Create a new ELLPACK page with empty rows.
  page_.reset();  // Release the device memory first before reallocating
  page_.reset(new EllpackPageImpl(batch_param_.gpu_id, original_page_->Cuts(),
                                  original_page_->is_dense,
                                  original_page_->row_stride, sample_rows));

  // Compact the ELLPACK pages into the single sample page.
  thrust::fill(dh::tbegin(page_->gidx_buffer), dh::tend(page_->gidx_buffer), 0);
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), dh::ToSpan(sample_row_index_));
  }

  return {sample_rows, page_.get(), dh::ToSpan(gpair_)};
}

GradientBasedSampler::GradientBasedSampler(EllpackPageImpl* page,
                                           size_t n_rows,
                                           const BatchParam& batch_param,
                                           float subsample,
                                           int sampling_method) {
  monitor_.Init("gradient_based_sampler");

  bool is_sampling = subsample < 1.0;
  bool is_external_memory = page->n_rows != n_rows;

  if (is_sampling) {
    switch (sampling_method) {
      case TrainParam::kUniform:
        if (is_external_memory) {
          strategy_.reset(new ExternalMemoryUniformSampling(page, n_rows, batch_param, subsample));
        } else {
          strategy_.reset(new UniformSampling(page, subsample));
        }
        break;
      case TrainParam::kGradientBased:
        if (is_external_memory) {
          strategy_.reset(
              new ExternalMemoryGradientBasedSampling(page, n_rows, batch_param, subsample));
        } else {
          strategy_.reset(new GradientBasedSampling(page, n_rows, batch_param, subsample));
        }
        break;
      default:LOG(FATAL) << "unknown sampling method";
    }
  } else {
    if (is_external_memory) {
      strategy_.reset(new ExternalMemoryNoSampling(page, n_rows, batch_param));
    } else {
      strategy_.reset(new NoSampling(page));
    }
  }
}

// Sample a DMatrix based on the given gradient pairs.
GradientBasedSample GradientBasedSampler::Sample(common::Span<GradientPair> gpair,
                                                 DMatrix* dmat) {
  monitor_.Start("Sample");
  GradientBasedSample sample = strategy_->Sample(gpair, dmat);
  monitor_.Stop("Sample");
  return sample;
}

size_t GradientBasedSampler::CalculateThresholdIndex(
    common::Span<GradientPair> gpair, common::Span<float> threshold,
    common::Span<float> grad_sum, size_t sample_rows) {
  thrust::fill(dh::tend(threshold) - 1, dh::tend(threshold),
               std::numeric_limits<float>::max());
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair), dh::tbegin(threshold),
                    CombineGradientPair());
  thrust::sort(dh::tbegin(threshold), dh::tend(threshold) - 1);
  thrust::inclusive_scan(dh::tbegin(threshold), dh::tend(threshold) - 1,
                         dh::tbegin(grad_sum));
  thrust::transform(dh::tbegin(grad_sum), dh::tend(grad_sum),
                    thrust::counting_iterator<size_t>(0), dh::tbegin(grad_sum),
                    SampleRateDelta(threshold, gpair.size(), sample_rows));
  thrust::device_ptr<float> min =
      thrust::min_element(dh::tbegin(grad_sum), dh::tend(grad_sum));
  return thrust::distance(dh::tbegin(grad_sum), min) + 1;
}

};  // namespace tree
};  // namespace xgboost
