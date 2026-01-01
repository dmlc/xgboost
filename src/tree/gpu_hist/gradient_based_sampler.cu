/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>  // for sort
#include <thrust/transform.h>

#include <cstddef>            // for size_t
#include <cuda/std/iterator>  // for distance
#include <limits>
#include <utility>

#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/random.h"
#include "../param.h"
#include "gradient_based_sampler.cuh"
#include "xgboost/logging.h"

namespace xgboost::tree {
/*! \brief A functor that returns random weights. */
class RandomWeight {
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
class BernoulliTrial {
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
struct IsNonZero {
  XGBOOST_DEVICE bool operator()(const GradientPair& gpair) const {
    return gpair.GetGrad() != 0 || gpair.GetHess() != 0;
  }
};

/*! \brief A functor that clears the row indexes with empty gradient. */
struct ClearEmptyRows {
  static constexpr bst_idx_t InvalidRow() { return std::numeric_limits<std::size_t>::max(); }

  XGBOOST_DEVICE size_t operator()(const GradientPair& gpair, size_t row_index) const {
    if (gpair.GetGrad() != 0 || gpair.GetHess() != 0) {
      return row_index;
    } else {
      return InvalidRow();
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
class CombineGradientPair {
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
class SampleRateDelta {
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
class PoissonSampling {
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
        return {};
      }
    }
  }

 private:
  common::Span<float> threshold_;
  size_t threshold_index_;
  RandomWeight rnd_;
  CombineGradientPair combine_;
};

GradientBasedSample NoSampling::Sample(Context const*, common::Span<GradientPair> gpair,
                                       DMatrix* p_fmat) {
  return {p_fmat, gpair};
}

UniformSampling::UniformSampling(BatchParam batch_param, float subsample)
    : batch_param_{std::move(batch_param)}, subsample_{subsample} {}

GradientBasedSample UniformSampling::Sample(Context const* ctx,
                                            common::Span<GradientPair> gpair,
                                            DMatrix* p_fmat) {
  // Set gradient pair to 0 with p = 1 - subsample
  auto cuctx = ctx->CUDACtx();
  thrust::replace_if(cuctx->CTP(), dh::tbegin(gpair), dh::tend(gpair),
                     thrust::counting_iterator<std::size_t>(0),
                     BernoulliTrial(common::GlobalRandom()(), subsample_), GradientPair());
  return {p_fmat, gpair};
}

GradientBasedSampling::GradientBasedSampling(std::size_t n_rows, BatchParam batch_param,
                                             float subsample)
    : subsample_(subsample),
      batch_param_{std::move(batch_param)},
      threshold_(n_rows + 1, 0.0f),
      grad_sum_(n_rows, 0.0f) {}

/** @brief Calculate the threshold used to normalize sampling probabilities. */
std::size_t CalculateThresholdIndex(Context const* ctx, common::Span<GradientPair const> gpair,
                                    common::Span<float> threshold, common::Span<float> grad_sum,
                                    size_t sample_rows) {
  auto cuctx = ctx->CUDACtx();
  thrust::fill(cuctx->CTP(), dh::tend(threshold) - 1, dh::tend(threshold),
               std::numeric_limits<float>::max());
  thrust::transform(cuctx->CTP(), dh::tbegin(gpair), dh::tend(gpair), dh::tbegin(threshold),
                    CombineGradientPair{});
  thrust::sort(cuctx->TP(), dh::tbegin(threshold), dh::tend(threshold) - 1);
  thrust::inclusive_scan(cuctx->CTP(), dh::tbegin(threshold), dh::tend(threshold) - 1,
                         dh::tbegin(grad_sum));
  thrust::transform(cuctx->CTP(), dh::tbegin(grad_sum), dh::tend(grad_sum),
                    thrust::counting_iterator<size_t>(0), dh::tbegin(grad_sum),
                    SampleRateDelta(threshold, gpair.size(), sample_rows));
  thrust::device_ptr<float> min =
      thrust::min_element(cuctx->CTP(), dh::tbegin(grad_sum), dh::tend(grad_sum));
  return cuda::std::distance(dh::tbegin(grad_sum), min) + 1;
}

GradientBasedSample GradientBasedSampling::Sample(Context const* ctx,
                                                  common::Span<GradientPair> gpair,
                                                  DMatrix* p_fmat) {
  auto cuctx = ctx->CUDACtx();
  size_t n_rows = p_fmat->Info().num_row_;
  size_t threshold_index = CalculateThresholdIndex(ctx, gpair, dh::ToSpan(threshold_),
                                                   dh::ToSpan(grad_sum_), n_rows * subsample_);

  // Perform Poisson sampling in place.
  thrust::transform(cuctx->CTP(), dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0), dh::tbegin(gpair),
                    PoissonSampling(dh::ToSpan(threshold_), threshold_index,
                                    RandomWeight(common::GlobalRandom()())));
  return {p_fmat, gpair};
}

GradientBasedSampler::GradientBasedSampler(Context const* /*ctx*/, size_t n_rows,
                                           const BatchParam& batch_param, float subsample,
                                           int sampling_method) {
  // The ctx is kept here for future development of stream-based operations.
  monitor_.Init(__func__);

  bool is_sampling = subsample < 1.0;

  if (!is_sampling) {
    strategy_.reset(new NoSampling{});
    return;
  }

  switch (sampling_method) {
    case TrainParam::kUniform: {
      strategy_.reset(new UniformSampling(batch_param, subsample));
      break;
    }
    case TrainParam::kGradientBased: {
      strategy_.reset(new GradientBasedSampling(n_rows, batch_param, subsample));
      break;
    }
    default:
      LOG(FATAL) << "unknown sampling method";
  }
}

// Sample a DMatrix based on the given gradient pairs.
GradientBasedSample GradientBasedSampler::Sample(Context const* ctx,
                                                 common::Span<GradientPair> gpair, DMatrix* dmat) {
  monitor_.Start(__func__);
  GradientBasedSample sample = strategy_->Sample(ctx, gpair, dmat);
  monitor_.Stop(__func__);
  return sample;
}
};  // namespace xgboost::tree
