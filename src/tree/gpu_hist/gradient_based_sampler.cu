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

  if (is_external_memory_) {
    // Create a new ELLPACK page with empty rows.
    page_.reset(new EllpackPageImpl(batch_param.gpu_id,
                                    original_page_->matrix.info,
                                    sample_rows_));
    // Allocate GPU memory for sampling.
    if (is_sampling_) {
      ba_.Allocate(batch_param_.gpu_id,
                   &gpair_, sample_rows_,
                   &row_weight_, n_rows,
                   &row_index_, n_rows,
                   &sample_row_index_, n_rows);
      thrust::copy(thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(n_rows),
                   dh::tbegin(row_index_));
    }
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
    CollectPages(dmat);
    return {dmat->Info().num_row_, page_.get(), gpair};
  } else {
    return {dmat->Info().num_row_, original_page_, gpair};
  }
}

// When not sampling, collect all the external memory ELLPACK pages into a single in-memory page.
void GradientBasedSampler::CollectPages(DMatrix* dmat) {
  if (page_collected_) {
    return;
  }

  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    auto page = batch.Impl();
    size_t num_elements = page_->Copy(batch_param_.gpu_id, page, offset);
    offset += num_elements;
  }
  page_collected_ = true;
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

/*! \brief A functor that performs Bernoulli sampling on gradient pairs. */
struct BernoulliSampling : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  float p;
  RandomWeight rnd;

  XGBOOST_DEVICE BernoulliSampling(float _p, RandomWeight _rnd) : p(_p), rnd(_rnd) {}

  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t i) const {
    if (rnd(i) <= p) {
      return gpair;
    } else {
      return GradientPair();
    }
  }
};

GradientBasedSample GradientBasedSampler::UniformSampling(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  RandomWeight rnd(common::GlobalRandom()());
  if (is_external_memory_) {
    // Generate random weights.
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(gpair.size()),
                      dh::tbegin(row_weight_),
                      rnd);
    return SequentialPoissonSampling(gpair, dmat);
  } else {
    // Set gradient pair to 0 with p = 1 - subsample
    thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                      thrust::counting_iterator<size_t>(0),
                      dh::tbegin(gpair),
                      BernoulliSampling(subsample_, rnd));
    return {dmat->Info().num_row_, original_page_, gpair};
  }
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

/*! \brief A functor that calculates the weight of each row.
 */
struct CalculateWeight : public thrust::binary_function<GradientPair, size_t, float> {
  RandomWeight rnd;
  CombineGradientPair combine{};

  XGBOOST_DEVICE explicit CalculateWeight(RandomWeight _rnd) : rnd(_rnd) {}

  XGBOOST_DEVICE float operator()(const GradientPair& gpair, size_t i) {
    if (gpair.GetGrad() == 0 && gpair.GetHess() == 0) {
      return FLT_MAX;
    }
    return rnd(i) / combine(gpair);
  }
};

/*! \brief A functor that performs Poisson sampling with probability proportional to the combined
 * gradient pair.
 */
struct PoissonSampling : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  size_t sample_rows;
  float normalization;
  const RandomWeight rnd;
  const CombineGradientPair combine{};

  XGBOOST_DEVICE PoissonSampling(size_t _sample_rows, float _normalization, RandomWeight _rnd)
      : sample_rows(_sample_rows), normalization(_normalization), rnd(_rnd) {}

  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t i) const {
    if (rnd(i) <= sample_rows * combine(gpair) / normalization) {
      return gpair;
    } else {
      return GradientPair();
    }
  }
};

GradientBasedSample GradientBasedSampler::GradientBasedSampling(
    common::Span<GradientPair> gpair, DMatrix* dmat) {
  RandomWeight rnd(common::GlobalRandom()());
  if (is_external_memory_) {
    thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                      thrust::counting_iterator<size_t>(0),
                      dh::tbegin(row_weight_),
                      CalculateWeight(rnd));
    return SequentialPoissonSampling(gpair, dmat);
  } else {
    float normalization = thrust::transform_reduce(dh::tbegin(gpair), dh::tend(gpair),
                                                   CombineGradientPair(),
                                                   0.0f,
                                                   thrust::plus<float>());
    // Set gradient pair to 0 with p = 1 - p_i
    thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                      thrust::counting_iterator<size_t>(0),
                      dh::tbegin(gpair),
                      PoissonSampling(sample_rows_, normalization, rnd));
    return {dmat->Info().num_row_, original_page_, gpair};
  }
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
      return SIZE_MAX;
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
