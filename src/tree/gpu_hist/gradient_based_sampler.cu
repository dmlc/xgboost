/**
 * Copyright 2019-2026, XGBoost Contributors
 */

#include <thrust/copy.h>  // for copy_n
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>  // for sort
#include <thrust/transform.h>
#include <thrust/version.h>

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
#include <cub/device/device_segmented_reduce.cuh>  // for DeviceSegmentedReduce
#else
#include <thrust/reduce.h>  // for reduce_by_key
#endif

#include <cstddef>            // for size_t
#include <cuda/std/iterator>  // for distance
#include <limits>
#include <utility>

#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/math.h"            // for Sqr
#include "../../common/random.h"
#include "../param.h"
#include "gradient_based_sampler.cuh"
#include "quantiser.cuh"  // for GradientQuantiser

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

  XGBOOST_DEVICE bool operator()(size_t i) const { return rnd_(i) > p_; }

 private:
  RandomWeight rnd_;
  float p_;
};

/**
 * @brief A functor that combines the gradient pair into a single float.
 *
 * The approach here is based on Minimal Variance Sampling (MVS), with lambda set to 0.1.
 *
 * @see Ibragimov, B., & Gusev, G. (2019). Minimal Variance Sampling in Stochastic Gradient
 * Boosting. In Advances in Neural Information Processing Systems (pp. 15061-15071).
 */
class CombineGradientPair {
 public:
  XGBOOST_DEVICE float operator()(const GradientPairPrecise& gpair) const {
    auto [g, h] = std::make_pair(gpair.GetGrad(), gpair.GetHess());
    return cuda::std::sqrt((g * g) + kLambda * (h * h));
  }

 private:
  static constexpr float kLambda = 0.1f;
};

/**
 * @brief A functor that calculates the difference between the sample rate and the desired
 *        sample rows, given a cumulative gradient sum.
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

/** @brief A functor that performs Poisson sampling, and scales gradient pairs by 1/p_i. */
class PoissonSampling {
 public:
  PoissonSampling(common::Span<GradientQuantiser const> roundings,
                  common::Span<float const> threshold, common::Span<float const> rag,
                  std::size_t threshold_index, RandomWeight rnd)
      : roundings_{roundings},
        threshold_{threshold},
        regularized_abs_grad_{rag},
        threshold_index_(threshold_index),
        rnd_(rnd) {}

  XGBOOST_DEVICE GradientPairInt64 operator()(GradientPairInt64 const& gpair, std::size_t i) {
    // If the gradient and hessian are both empty, we should never select this row.
    if (gpair.GetQuantisedGrad() == 0 && gpair.GetQuantisedHess() == 0) {
      return gpair;
    }
    auto n_samples = threshold_.size() - 1;
    auto [ridx, tidx] = linalg::UnravelIndex(i, n_samples, roundings_.size());
    auto q = roundings_[tidx];

    float u = threshold_[threshold_index_];
    float combined_gradient = regularized_abs_grad_[ridx];
    float p = combined_gradient / u;
    if (p >= 1) {
      // Always select this row.
      return gpair;
    } else {
      // Select this row randomly with probability proportional to the combined gradient.
      // Scale gpair by 1/p.
      if (rnd_(ridx) <= p) {
        return q.ToFixedPoint(q.ToFloatingPoint(gpair) / p);
      } else {
        return {};
      }
    }
  }

 private:
  common::Span<GradientQuantiser const> roundings_;
  common::Span<float const> threshold_;
  common::Span<float const> regularized_abs_grad_;
  std::size_t threshold_index_;
  RandomWeight rnd_;
  CombineGradientPair combine_;
};

void UniformSampling::Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                             common::Span<GradientQuantiser const>) {
  // Set gradient pair to 0 with p = 1 - subsample
  auto cuctx = ctx->CUDACtx();
  auto n_targets = gpair.Shape(1);
  BernoulliTrial trial{common::GlobalRandom()(), subsample_};
  thrust::replace_if(
      cuctx->CTP(), linalg::tbegin(gpair), linalg::tend(gpair), thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto ridx = i / n_targets;
        return trial(ridx);
      },
      GradientPairInt64{});
}

GradientBasedSampling::GradientBasedSampling(std::size_t n_rows, float subsample)
    : subsample_(subsample),
      reg_abs_grad_(n_rows, 0.0f),
      threshold_(n_rows + 1, 0.0f),
      grad_sum_(n_rows, 0.0f) {}

void ReduceGrad(Context const* ctx, linalg::MatrixView<GradientPairInt64 const> gpairs,
                common::Span<GradientQuantiser const> roundings, common::Span<float> reg_abs_grad) {
  constexpr float kLambda = 0.1f;
  auto n_segments = gpairs.Shape(0);
  CHECK_EQ(n_segments, reg_abs_grad.size());
  auto n_targets = gpairs.Shape(1);
  auto op = [=] XGBOOST_DEVICE(cuda::std::tuple<std::size_t, GradientPairInt64> tup) -> float {
    auto [i, gpairs_i64] = tup;
    auto cidx = i % n_targets;
    auto gpair = roundings[cidx].ToFloatingPoint(gpairs_i64);
    auto [g, h] = cuda::std::make_pair(gpair.GetGrad(), gpair.GetHess());
    return common::Sqr(g) + kLambda * common::Sqr(h);
  };
  auto in_it = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_counting_iterator(0ul), linalg::tcbegin(gpairs)), op);

  if (gpairs.Shape(1) <= 1) {
    CHECK_EQ(gpairs.Size(), reg_abs_grad.size());
    thrust::copy_n(ctx->CUDACtx()->CTP(), in_it, gpairs.Size(), dh::tbegin(reg_abs_grad));
    return;
  }

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
  // Fixed size segment support:
  // https://github.com/NVIDIA/cccl/commit/ae0bbef407fa8fea2b654f35f886a6f3420f5897
  auto s = ctx->CUDACtx()->Stream();
  std::size_t n_bytes = 0;
  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(nullptr, n_bytes, in_it, dh::tbegin(reg_abs_grad),
                                                /*num_segments*/ gpairs.Shape(0),
                                                /*segment_size=*/gpairs.Shape(1), s));
  dh::TemporaryArray<char> alloc(n_bytes);
  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(alloc.data().get(), n_bytes, /*d_in=*/in_it,
                                                /*d_out=*/dh::tbegin(reg_abs_grad),
                                                /*num_segments=*/gpairs.Shape(0),
                                                /*segment_size=*/gpairs.Shape(1), s));
#else
  auto key_it =
      dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return i / n_targets; });
  thrust::reduce_by_key(ctx->CUDACtx()->CTP(), key_it, key_it + gpairs.Size(), in_it,
                        thrust::make_discard_iterator(), dh::tbegin(reg_abs_grad));
#endif
}

/** @brief Calculate the threshold used to normalize sampling probabilities. */
std::size_t CalculateThresholdIndex(Context const* ctx,
                                    linalg::MatrixView<GradientPairInt64 const> gpairs,
                                    common::Span<GradientQuantiser const> roundings,
                                    common::Span<float> reg_abs_grad, common::Span<float> threshold,
                                    common::Span<float> grad_csum, std::size_t sample_rows) {
  auto cuctx = ctx->CUDACtx();

  thrust::fill(cuctx->CTP(), dh::tend(threshold) - 1, dh::tend(threshold),
               std::numeric_limits<float>::max());
  // Create the regularized absolute gradient
  ReduceGrad(ctx, gpairs, roundings, dh::ToSpan(reg_abs_grad));
  thrust::transform(cuctx->CTP(), dh::tcbegin(reg_abs_grad), dh::tcend(reg_abs_grad) - 1,
                    dh::tbegin(reg_abs_grad),
                    [] XGBOOST_DEVICE(float gpair) { return cuda::std::sqrt(gpair); });

  // Sort and calculate csum
  thrust::copy(dh::tbegin(reg_abs_grad), dh::tend(reg_abs_grad), dh::tbegin(threshold));
  thrust::sort(cuctx->TP(), dh::tbegin(threshold), dh::tend(threshold) - 1);
  thrust::inclusive_scan(cuctx->CTP(), dh::tbegin(threshold), dh::tend(threshold) - 1,
                         dh::tbegin(grad_csum));

  // sets the u for each row.
  thrust::transform(cuctx->CTP(), dh::tbegin(grad_csum), dh::tend(grad_csum),
                    thrust::make_counting_iterator(0ul), dh::tbegin(grad_csum),
                    SampleRateDelta{threshold, gpairs.Shape(0), sample_rows});
  // Find the first 0 element in grad_sum, which is within the threshold bound
  thrust::device_ptr<float> min =
      thrust::min_element(cuctx->CTP(), dh::tbegin(grad_csum), dh::tend(grad_csum));
  return cuda::std::distance(dh::tbegin(grad_csum), min) + 1;
}

void GradientBasedSampling::Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                                   common::Span<GradientQuantiser const> roundings) {
  auto cuctx = ctx->CUDACtx();
  std::size_t n_samples = gpair.Shape(0);
  CHECK_EQ(n_samples, this->reg_abs_grad_.size());
  CHECK_EQ(n_samples, this->grad_sum_.size());
  CHECK_EQ(n_samples + 1, this->threshold_.size());
  std::size_t threshold_index = CalculateThresholdIndex(
      ctx, gpair, roundings, dh::ToSpan(reg_abs_grad_), dh::ToSpan(this->threshold_),
      dh::ToSpan(grad_sum_), n_samples * subsample_);
  auto seed = common::GlobalRandom()();
  // Perform sequential Poisson sampling in place.
  // Only the threshold_[threshold_index] is used. (that is the \mu in the paper)
  thrust::transform(cuctx->CTP(), linalg::tcbegin(gpair), linalg::tcend(gpair),
                    thrust::make_counting_iterator(0ul), linalg::tbegin(gpair),
                    PoissonSampling{roundings, dh::ToSpan(threshold_), dh::ToSpan(reg_abs_grad_),
                                    threshold_index, RandomWeight{seed}});
}

GradientBasedSampler::GradientBasedSampler(bst_idx_t n_samples, float subsample,
                                           int sampling_method) {
  monitor_.Init(__func__);

  bool is_sampling = subsample < 1.0;

  if (!is_sampling) {
    strategy_.reset(new NoSampling{});
    return;
  }

  switch (sampling_method) {
    case TrainParam::kUniform: {
      strategy_.reset(new UniformSampling{subsample});
      break;
    }
    case TrainParam::kGradientBased: {
      strategy_.reset(new GradientBasedSampling{n_samples, subsample});
      break;
    }
    default:
      LOG(FATAL) << "unknown sampling method";
  }
}

// Sample a DMatrix based on the given gradient pairs.
void GradientBasedSampler::Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                                  common::Span<GradientQuantiser const> roundings) {
  monitor_.Start(__func__);
  strategy_->Sample(ctx, gpair, roundings);
  monitor_.Stop(__func__);
}
namespace cuda_impl {
void ApplySamplingMask(Context const* ctx,
                       linalg::Matrix<GradientPairInt64> const& sampled_split_gpair,
                       linalg::Matrix<GradientPair>* value_gpair) {
  CHECK_EQ(sampled_split_gpair.Shape(0), value_gpair->Shape(0));
  auto d_split = sampled_split_gpair.View(ctx->Device());
  auto d_value = value_gpair->View(ctx->Device());
  auto n_targets = value_gpair->Shape(1);
  thrust::replace_if(
      ctx->CUDACtx()->CTP(), linalg::tbegin(d_value), linalg::tend(d_value),
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto ridx = i / n_targets;
        // Check if this row was not sampled (hessian is zero in split gradient)
        return d_split(ridx, 0).GetQuantisedHess() == 0;
      },
      GradientPair{});
}
}  // namespace cuda_impl
}  // namespace xgboost::tree
