/**
 * Copyright 2019-2026, XGBoost Contributors
 */

#include <thrust/copy.h>  // for copy_n
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>         // for make_transform_iterator
#include <thrust/iterator/transform_output_iterator.h>  // for make_transform_output_iterator
#include <thrust/random.h>
#include <thrust/sort.h>  // for sort
#include <thrust/transform.h>
#include <thrust/version.h>

#include "../../common/nvtx_utils.h"

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
#include <cub/device/device_segmented_reduce.cuh>  // for DeviceSegmentedReduce
#else
#include <thrust/reduce.h>  // for reduce_by_key
#endif

#include <cstddef>            // for size_t
#include <cuda/std/iterator>  // for distance
#include <limits>

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "../../common/random.h"
#include "../hist/sampler.h"  // for kDefaultMvsLambda
#include "../param.h"
#include "quantiser.cuh"  // for GradientQuantiser
#include "sampler.cuh"

namespace xgboost::tree::cuda_impl {
/** @brief A functor that returns random weights. */
class RandomWeight {
 public:
  explicit RandomWeight(std::size_t seed) : seed_(seed) {}

  XGBOOST_DEVICE float operator()(std::size_t i) const {
    thrust::default_random_engine rng(seed_);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng);
  }

 private:
  std::uint32_t seed_;
};

/** @brief A functor that performs a Bernoulli trial to discard a gradient pair. */
class BernoulliTrial {
 public:
  BernoulliTrial(std::size_t seed, float p) : rnd_(seed), p_(p) {}

  XGBOOST_DEVICE bool operator()(std::size_t i) const { return rnd_(i) > p_; }

 private:
  RandomWeight rnd_;
  float p_;
};

/**
 * @brief A functor that calculates the difference between the sample rate and the desired
 *        sample rows, given a cumulative gradient sum.
 */
class SampleRateDelta {
 public:
  SampleRateDelta(common::Span<float> threshold, bst_idx_t n_samples, bst_idx_t sample_rows)
      : threshold_(threshold), n_samples_(n_samples), sample_rows_(sample_rows) {}

  XGBOOST_DEVICE float operator()(float gradient_sum, bst_idx_t i) const {
    float lower = threshold_[i];
    float upper = threshold_[i + 1];

    bst_idx_t n_above = n_samples_ - i - 1;
    float denom = static_cast<float>(sample_rows_) - static_cast<float>(n_above);
    // i is too small, sampling too many rows
    if (denom <= 0) {
      return std::numeric_limits<float>::max();
    }

    float u = gradient_sum / denom;
    if (u > lower && u <= upper) {
      // Found it, set the value for future use.
      threshold_[i + 1] = u;
      return 0.0f;
    } else {
      return std::numeric_limits<float>::max();
    }
  }

 private:
  common::Span<float> threshold_;
  bst_idx_t n_samples_;
  bst_idx_t sample_rows_;
};

namespace {
[[nodiscard]] std::size_t CalcThresholdIndex(Context const* ctx, common::Span<float> reg_abs_grad,
                                             common::Span<float> thresholds,
                                             common::Span<float> grad_csum, bst_idx_t sample_rows) {
  auto cuctx = ctx->CUDACtx();
  // Set a sentinel for upper bound.
  thrust::fill(cuctx->CTP(), dh::tend(thresholds) - 1, dh::tend(thresholds),
               std::numeric_limits<float>::max());
  // Sort thresholds
  thrust::copy(cuctx->CTP(), dh::tcbegin(reg_abs_grad), dh::tcend(reg_abs_grad),
               dh::tbegin(thresholds));
  thrust::sort(cuctx->TP(), dh::tbegin(thresholds), dh::tend(thresholds) - 1);
  auto n_samples = reg_abs_grad.size();
  return CalculateThresholdIndex(ctx, thresholds, grad_csum, n_samples, sample_rows);
}
}  // anonymous namespace

/** @brief A functor that performs Poisson sampling, and scales gradient pairs by 1/p_i. */
class PoissonSampling {
 public:
  PoissonSampling(common::Span<GradientQuantiser const> roundings,
                  common::Span<float const> threshold, common::Span<float const> rag,
                  std::size_t threshold_index, RandomWeight rnd)
      : roundings_{roundings},
        threshold_{threshold},
        regularized_abs_grad_{rag},
        threshold_index_{threshold_index},
        rnd_{rnd} {}

  XGBOOST_DEVICE GradientPairInt64 operator()(GradientPairInt64 const& gpair, std::size_t i) {
    // If the gradient and hessian are both empty, we should never select this row.
    if (gpair.GetQuantisedGrad() == 0 && gpair.GetQuantisedHess() == 0) {
      return gpair;
    }
    auto n_samples = threshold_.size() - 1;
    auto [ridx, tidx] = linalg::UnravelIndex(i, n_samples, roundings_.size());
    auto q = roundings_[tidx];

    float p = SamplingProbability(threshold_[threshold_index_], regularized_abs_grad_[ridx]);
    if (p >= 1.0f) {
      // Always select this row.
      return gpair;
    } else {
      // Select this row randomly with probability proportional to the combined gradient.
      // Scale gpair by 1/p.
      if (rnd_(ridx) <= p) {
        return q.ToFixedPoint(RescaleGrad(p, q.ToFloatingPoint(gpair)));
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
};

void UniformSampling::Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                             common::Span<GradientQuantiser const>) {
  // Set gradient pair to 0 with p = 1 - subsample
  auto cuctx = ctx->CUDACtx();
  auto n_targets = gpair.Shape(1);
  BernoulliTrial trial{ctx->Rng()(), subsample_};
  thrust::replace_if(
      cuctx->CTP(), linalg::tbegin(gpair), linalg::tend(gpair), thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto ridx = i / n_targets;
        return trial(ridx);
      },
      GradientPairInt64{});
}

void UniformSampling::ApplySampling(Context const* ctx,
                                    linalg::MatrixView<GradientPairInt64 const> sampled_split_gpair,
                                    linalg::Matrix<GradientPair>* value_gpair) {
  CHECK_EQ(sampled_split_gpair.Shape(0), value_gpair->Shape(0));
  auto d_split_gpair = sampled_split_gpair;
  auto d_value = value_gpair->View(ctx->Device());
  auto n_targets = value_gpair->Shape(1);
  thrust::replace_if(
      ctx->CUDACtx()->CTP(), linalg::tbegin(d_value), linalg::tend(d_value),
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto ridx = i / n_targets;
        // Check if this row was not sampled (hessian is zero in split gradient)
        return d_split_gpair(ridx, 0).GetQuantisedHess() == 0;
      },
      GradientPair{});
}

GradientBasedSampling::GradientBasedSampling(std::size_t n_samples, float subsample)
    : subsample_{subsample},
      reg_abs_grad_(n_samples, 0.0f),
      thresholds_(n_samples + 1, 0.0f),
      grad_csum_(n_samples, 0.0f) {}

template <typename GPair, typename ToFloat>
void ReduceGradImpl(Context const* ctx, linalg::MatrixView<GPair const> gpairs, ToFloat&& to_float,
                    common::Span<float> reg_abs_grad) {
  float mvs_lambda = kDefaultMvsLambda;
  auto n_segments = gpairs.Shape(0);
  CHECK_EQ(n_segments, reg_abs_grad.size());
  auto n_targets = gpairs.Shape(1);
  auto grad_op = MvsGradOp{mvs_lambda};

  auto op = [=] XGBOOST_DEVICE(cuda::std::tuple<std::size_t, GPair> tup) -> float {
    auto [i, gpair] = tup;
    return grad_op(to_float(i, gpair));
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
                                                /*num_segments=*/n_segments,
                                                /*segment_size=*/n_targets, s));
  dh::TemporaryArray<char> alloc(n_bytes);
  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(alloc.data().get(), n_bytes, /*d_in=*/in_it,
                                                /*d_out=*/dh::tbegin(reg_abs_grad),
                                                /*num_segments=*/n_segments,
                                                /*segment_size=*/n_targets, s));
#else
  auto key_it =
      dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return i / n_targets; });
  thrust::reduce_by_key(ctx->CUDACtx()->CTP(), key_it, key_it + gpairs.Size(), in_it,
                        thrust::make_discard_iterator(), dh::tbegin(reg_abs_grad));
#endif
}

void ReduceGrad(Context const* ctx, linalg::MatrixView<GradientPairInt64 const> gpairs,
                common::Span<GradientQuantiser const> roundings, common::Span<float> reg_abs_grad) {
  auto n_targets = gpairs.Shape(1);
  auto to_float = [=] XGBOOST_DEVICE(std::size_t i, GradientPairInt64 gpair) {
    auto cidx = i % n_targets;
    return roundings[cidx].ToFloatingPoint(gpair);
  };
  ReduceGradImpl(ctx, gpairs, to_float, reg_abs_grad);
}

void ReduceGradValue(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                     common::Span<float> reg_abs_grad) {
  auto to_float = [=] XGBOOST_DEVICE(std::size_t, GradientPair gpair) {
    return gpair;
  };
  ReduceGradImpl(ctx, gpairs, to_float, reg_abs_grad);
}

std::size_t CalculateThresholdIndex(Context const* ctx, common::Span<float> sorted_rag,
                                    common::Span<float> grad_csum, bst_idx_t n_samples,
                                    bst_idx_t sample_rows) {
  auto cuctx = ctx->CUDACtx();

  // scan is not yet made deterministic
  double h_total_sum = thrust::reduce(cuctx->CTP(), dh::tbegin(sorted_rag),
                                      dh::tend(sorted_rag) - 1, 0.0, cuda::std::plus{});
  FloatQuantiser quantiser{h_total_sum, n_samples};
  auto in_it =
      dh::MakeTransformIterator<std::int64_t>(dh::tbegin(sorted_rag), ToFixedPointOp{quantiser});
  auto out_it =
      thrust::make_transform_output_iterator(dh::tbegin(grad_csum), ToFloatingPointOp{quantiser});
  thrust::inclusive_scan(cuctx->CTP(), in_it, in_it + n_samples, out_it);

  // Find the threshold u for each row.
  thrust::transform(cuctx->CTP(), dh::tbegin(grad_csum), dh::tend(grad_csum),
                    thrust::make_counting_iterator(0ul), dh::tbegin(grad_csum),
                    SampleRateDelta{sorted_rag, n_samples, sample_rows});
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
  CHECK_EQ(n_samples, this->grad_csum_.size());
  CHECK_EQ(n_samples + 1, this->thresholds_.size());

  // Create the regularized absolute gradient.
  ReduceGrad(ctx, gpair, roundings, dh::ToSpan(reg_abs_grad_));
  thrust::transform(cuctx->CTP(), reg_abs_grad_.cbegin(), reg_abs_grad_.cend(),
                    reg_abs_grad_.begin(),
                    [] XGBOOST_DEVICE(float gpair) { return cuda::std::sqrt(gpair); });

  bst_idx_t sample_rows = n_samples * subsample_;
  auto threshold_index = CalcThresholdIndex(ctx, dh::ToSpan(reg_abs_grad_), dh::ToSpan(thresholds_),
                                            dh::ToSpan(grad_csum_), sample_rows);

  auto seed = ctx->Rng()();
  // Perform sequential Poisson sampling in place.
  // Only the threshold_[threshold_index] is used. (that is the \mu in the paper)
  thrust::transform(cuctx->CTP(), linalg::tcbegin(gpair), linalg::tcend(gpair),
                    thrust::make_counting_iterator(0ul), linalg::tbegin(gpair),
                    PoissonSampling{roundings, dh::ToSpan(thresholds_), dh::ToSpan(reg_abs_grad_),
                                    threshold_index, RandomWeight{seed}});
}

void GradientBasedSampling::ApplySampling(
    Context const* ctx, linalg::MatrixView<GradientPairInt64 const> sampled_split_gpair,
    linalg::Matrix<GradientPair>* value_gpair) {
  CHECK_EQ(sampled_split_gpair.Shape(0), value_gpair->Shape(0));
  auto d_split_gpair = sampled_split_gpair;
  auto d_value = value_gpair->View(ctx->Device());
  auto n_targets = value_gpair->Shape(1);
  auto n_samples = value_gpair->Shape(0);
  CHECK_EQ(n_samples, this->reg_abs_grad_.size());
  CHECK_EQ(n_samples, this->grad_csum_.size());
  CHECK_EQ(n_samples + 1, this->thresholds_.size());

  // Create the regularized absolute gradient from value gradient.
  ReduceGradValue(ctx, d_value, dh::ToSpan(reg_abs_grad_));
  thrust::transform(ctx->CUDACtx()->CTP(), reg_abs_grad_.cbegin(), reg_abs_grad_.cend(),
                    reg_abs_grad_.begin(),
                    [] XGBOOST_DEVICE(float gpair) { return cuda::std::sqrt(gpair); });
  bst_idx_t sample_rows = n_samples * subsample_;
  auto threshold_index = CalcThresholdIndex(ctx, dh::ToSpan(reg_abs_grad_), dh::ToSpan(thresholds_),
                                            dh::ToSpan(grad_csum_), sample_rows);

  auto threshold = dh::ToSpan(thresholds_);
  auto rag = dh::ToSpan(reg_abs_grad_);
  thrust::transform(ctx->CUDACtx()->CTP(), linalg::tcbegin(d_value), linalg::tcend(d_value),
                    thrust::make_counting_iterator(0ul), linalg::tbegin(d_value),
                    [=] XGBOOST_DEVICE(GradientPair gpair, std::size_t i) {
                      auto ridx = i / n_targets;
                      // Check if this row was not sampled (hessian is zero in split gradient)
                      if (d_split_gpair(ridx, 0).GetQuantisedHess() == 0) {
                        return GradientPair{};
                      }
                      float p = SamplingProbability(threshold[threshold_index], rag[ridx]);
                      return RescaleGrad(p, gpair);
                    });
}

Sampler::Sampler(bst_idx_t n_samples, float subsample, int sampling_method) {
  bool is_sampling = subsample < 1.0;

  if (!is_sampling) {
    strategy_ = std::make_unique<SamplingStrategy>();
    return;
  }

  switch (sampling_method) {
    case TrainParam::kUniform: {
      strategy_ = std::make_unique<UniformSampling>(subsample);
      break;
    }
    case TrainParam::kGradientBased: {
      strategy_ = std::make_unique<GradientBasedSampling>(n_samples, subsample);
      break;
    }
    default:
      LOG(FATAL) << "Unknown sampling method.";
  }
}

void Sampler::Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
                     common::Span<GradientQuantiser const> roundings) {
  xgboost_NVTX_FN_RANGE();
  strategy_->Sample(ctx, gpair, roundings);
}

void Sampler::ApplySampling(Context const* ctx,
                            linalg::Matrix<GradientPairInt64> const& sampled_split_gpair,
                            linalg::Matrix<GradientPair>* value_gpair) {
  xgboost_NVTX_FN_RANGE();
  strategy_->ApplySampling(ctx, sampled_split_gpair.View(ctx->Device()), value_gpair);
}
}  // namespace xgboost::tree::cuda_impl
