/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <limits>

#include "../../common/compressed_iterator.h"
#include "../../common/random.h"
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

GradientBasedSampler::GradientBasedSampler(EllpackPageImpl* page,
                                           size_t n_rows,
                                           BatchParam batch_param,
                                           float subsample,
                                           int sampling_method)
    : original_page_(page),
      batch_param_(batch_param),
      is_external_memory_(page->matrix.n_rows != n_rows),
      subsample_(subsample),
      is_sampling_(subsample < 1.0),
      sampling_method_(sampling_method),
      sample_rows_(n_rows * subsample) {
  monitor_.Init("gradient_based_sampler");

  if (is_sampling_ || is_external_memory_) {
    // Create a new ELLPACK page with empty rows.
    page_.reset(new EllpackPageImpl(batch_param.gpu_id,
                                    original_page_->matrix.info,
                                    sample_rows_));
  }
  // Allocate GPU memory for sampling.
  if (is_sampling_) {
    ba_.Allocate(batch_param_.gpu_id,
                 &gpair_, sample_rows_,
                 &row_weight_, n_rows,
                 &threshold_, n_rows + 1,
                 &row_index_, n_rows,
                 &sample_row_index_, n_rows);
    thrust::copy(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator<size_t>(n_rows),
                 dh::tbegin(row_index_));
  }
}

// Sample a DMatrix based on the given gradient pairs.
GradientBasedSample GradientBasedSampler::Sample(common::Span<GradientPair> gpair,
                                                 DMatrix* dmat) {
  monitor_.StartCuda("Sample");
  GradientBasedSample sample;
  if (is_sampling_) {
    switch (sampling_method_) {
      case TrainParam::kUniform:
        sample = UniformSampling(gpair, dmat);
        break;
      case TrainParam::kGradientBased:
        sample = GradientBasedSampling(gpair, dmat);
        break;
      default:
        LOG(FATAL) << "unknown sampling method";
        sample = {0, nullptr, gpair};
    }
  } else {
    sample = NoSampling(gpair, dmat);
  }
  monitor_.StopCuda("Sample");
  return sample;
}

GradientBasedSample GradientBasedSampler::NoSampling(common::Span<GradientPair> gpair,
                                                     DMatrix* dmat) {
  if (is_external_memory_) {
    ConcatenatePages(dmat);
    return {dmat->Info().num_row_, page_.get(), gpair};
  } else {
    return {dmat->Info().num_row_, original_page_, gpair};
  }
}

// When not sampling, concatenate all the external memory ELLPACK pages into a single in-memory
// page.
void GradientBasedSampler::ConcatenatePages(DMatrix* dmat) {
  if (page_concatenated_) {
    return;
  }

  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    auto page = batch.Impl();
    size_t num_elements = page_->Copy(batch_param_.gpu_id, page, offset);
    offset += num_elements;
  }
  page_concatenated_ = true;
}

/*! \brief A functor that returns random weights. */
struct RandomWeight : public thrust::unary_function<size_t, float> {
  uint32_t seed;

  XGBOOST_DEVICE explicit RandomWeight(size_t _seed) : seed(_seed) {}

  XGBOOST_DEVICE float operator()(size_t i) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng);
  }
};

/*! \brief A functor that scales gradient pairs by 1/p. */
struct FixedScaling : public thrust::unary_function<GradientPair, GradientPair> {
  float p;

  XGBOOST_DEVICE explicit FixedScaling(float _p) : p(_p) {}

  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair) const {
    return gpair / p;
  }
};

GradientBasedSample GradientBasedSampler::UniformSampling(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  // Generate random weights.
  thrust::transform(thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(gpair.size()),
                    dh::tbegin(row_weight_),
                    RandomWeight(common::GlobalRandom()()));
  // Scale gradient pairs by 1/subsample.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(gpair),
                    FixedScaling(subsample_));
  return SequentialPoissonSampling(gpair, dmat);
}

/*! \brief A functor that combines the gradient pair into a single float.
 *
 * The approach here is based on Minimal Variance Sampling (MVS), with lambda set to 0.1.
 *
 * \see Ibragimov, B., & Gusev, G. (2019). Minimal Variance Sampling in Stochastic Gradient
 * Boosting. In Advances in Neural Information Processing Systems (pp. 15061-15071).
 */
struct CombineGradientPair : public thrust::unary_function<GradientPair, float> {
  static constexpr float kLambda = 0.1f;

  XGBOOST_DEVICE float operator()(const GradientPair& gpair) const {
    return sqrtf(powf(gpair.GetGrad(), 2) + kLambda * powf(gpair.GetHess(), 2));
  }
};

/*! \brief A functor that calculates the difference between the sample rate and the desired sample
 * rows, given a cumulative gradient sum.
 */
struct SampleRateDelta : public thrust::binary_function<float, size_t, float> {
  common::Span<float> threshold;
  size_t n_rows;
  size_t sample_rows;

  XGBOOST_DEVICE SampleRateDelta(common::Span<float> _threshold,
                                 size_t _n_rows,
                                 size_t _sample_rows)
      : threshold(_threshold), n_rows(_n_rows), sample_rows(_sample_rows) {}

  XGBOOST_DEVICE float operator()(float gradient_sum, size_t row_index) const {
    // For the last row, if gradient_sum/sample_rows > gradient, that means no row will be sampled
    // with probability equal to 1, so we use the mean as the threshold.
    if (row_index == n_rows - 1) {
      float last = threshold[n_rows - 1];
      float mean = gradient_sum / sample_rows;
      if (mean > last) {
        threshold[n_rows] = mean;
        return 0.0f;
      } else {
        return std::numeric_limits<float>::max();
      }
    }

    // For a given u = threshold[row_index], the summed sample rate for the rows above the current
    // row is `gradient_sum / u`.
    //
    // Rows below (including the current row) are sampled with probability equal to 1, thus adding
    // up to `n_rows - row_index`.
    //
    // The total sample rate is therefore `gradient_sum / u + n_rows - row_index`.
    //
    // We want to choose the threshold that makes this value as close to `sample_rows` as possible.
    float u = threshold[row_index + 1];
    if (u == 0.0f) {
      return std::numeric_limits<float>::max();
    } else {
      return fabsf(gradient_sum / u + n_rows - row_index - sample_rows);
    }
  }
};

size_t GradientBasedSampler::CalculateThresholdIndex(common::Span<GradientPair> gpair) {
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(threshold_),
                    CombineGradientPair());
  thrust::sort(dh::tbegin(threshold_), dh::tend(threshold_) - 1);
  thrust::inclusive_scan(dh::tbegin(threshold_), dh::tend(threshold_) - 1, dh::tbegin(row_weight_));
  thrust::transform(dh::tbegin(row_weight_), dh::tend(row_weight_),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(row_weight_),
                    SampleRateDelta(threshold_, gpair.size(), sample_rows_));
  thrust::device_ptr<float> min = thrust::min_element(dh::tbegin(row_weight_),
                                                      dh::tend(row_weight_));
  return thrust::distance(dh::tbegin(row_weight_), min) + 1;
}

/*! \brief A functor that calculates the weight of each row, and scales gradient pairs by 1/p_i. */
struct CalculateWeight
    : public thrust::binary_function<GradientPair, size_t, thrust::tuple<float, GradientPair>> {
  common::Span<float> threshold;
  size_t threshold_index;
  RandomWeight rnd;
  CombineGradientPair combine;

  XGBOOST_DEVICE CalculateWeight(common::Span<float> _threshold,
                                 size_t _threshold_index,
                                 RandomWeight _rnd)
    : threshold(_threshold), threshold_index(_threshold_index), rnd(_rnd) {}

  XGBOOST_DEVICE thrust::tuple<float, GradientPair> operator()(const GradientPair& gpair,
                                                               size_t i) {
    // If the gradient and hessian are both empty, we should never select this row.
    if (gpair.GetGrad() == 0 && gpair.GetHess() == 0) {
      return thrust::make_tuple(std::numeric_limits<float>::max(), gpair);
    }
    float combined_gradient = combine(gpair);
    float u = threshold[threshold_index];
    float p = combined_gradient / u;
    if (p >= 1) {
      // Always select this row.
      return thrust::make_tuple(0.0f, gpair);
    } else {
      // Select this row randomly with probability proportional to the combined gradient.
      // Scale gpair by 1/p.
      return thrust::make_tuple(rnd(i) / combined_gradient, gpair / p);
    }
  }
};

GradientBasedSample GradientBasedSampler::GradientBasedSampling(
    common::Span<GradientPair> gpair, DMatrix* dmat) {
  size_t threshold_index = CalculateThresholdIndex(gpair);
  printf("threshold_index=%lu\n", threshold_index);
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        dh::tbegin(row_weight_), dh::tbegin(gpair))),
                    CalculateWeight(threshold_, threshold_index,
                        RandomWeight(common::GlobalRandom()())));
  return SequentialPoissonSampling(gpair, dmat);
}

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

// Perform sampling after the weights are calculated.
GradientBasedSample GradientBasedSampler::SequentialPoissonSampling(
      common::Span<GradientPair> gpair, DMatrix* dmat) {
  // Sort the gradient pairs and row indexes by weight.
  thrust::sort_by_key(dh::tbegin(row_weight_), dh::tend(row_weight_),
                      thrust::make_zip_iterator(thrust::make_tuple(dh::tbegin(gpair),
                                                                   dh::tbegin(row_index_))));

  // Clear the gradient pairs not included in the sample.
  thrust::fill(dh::tbegin(gpair) + sample_rows_, dh::tend(gpair), GradientPair());

  // Mask the sample rows.
  thrust::fill(dh::tbegin(sample_row_index_), dh::tbegin(sample_row_index_) + sample_rows_, 1);
  thrust::fill(dh::tbegin(sample_row_index_) + sample_rows_, dh::tend(sample_row_index_), 0);

  // Sort the gradient pairs and sample row indexes by the original row index.
  thrust::sort_by_key(dh::tbegin(row_index_), dh::tend(row_index_),
                      thrust::make_zip_iterator(thrust::make_tuple(dh::tbegin(gpair),
                                                                   dh::tbegin(sample_row_index_))));

  // Compact the non-zero gradient pairs.
  thrust::fill(dh::tbegin(gpair_), dh::tend(gpair_), GradientPair());
  thrust::copy_if(dh::tbegin(gpair), dh::tend(gpair), dh::tbegin(gpair_), IsNonZero());

  // Index the sample rows.
  thrust::exclusive_scan(dh::tbegin(sample_row_index_), dh::tend(sample_row_index_),
                         dh::tbegin(sample_row_index_));
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(sample_row_index_),
                    dh::tbegin(sample_row_index_),
                    ClearEmptyRows());

  // Compact the ELLPACK pages into the single sample page.
  thrust::fill(dh::tbegin(page_->gidx_buffer), dh::tend(page_->gidx_buffer), 0);
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), sample_row_index_);
  }

  return {sample_rows_, page_.get(), gpair_};
}

};  // namespace tree
};  // namespace xgboost
