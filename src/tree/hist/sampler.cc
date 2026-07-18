/**
 * Copyright 2026, XGBoost Contributors
 */
#include "sampler.h"  // for kDefaultMvsLambda

#include <algorithm>  // for fill
#include <cmath>      // for sqrt
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <numeric>    // for partial_sum
#include <random>     // for default_random_engine, uniform_real_distribution
#include <vector>     // for vector

#include "../../common/algorithm.h"  // for Sort
#include "xgboost/base.h"            // for GradientPair, GradientPairPrecise
#include "xgboost/linalg.h"          // for MatrixView
#include "xgboost/span.h"            // for Span

namespace xgboost::tree::cpu_impl {
template <typename Fn>
void ParallelSampling(Context const* ctx, bst_idx_t n_samples, std::uint64_t initial_seed,
                      Fn&& fn) {
  auto n_threads = ctx->Threads();
  std::size_t const discard_size = n_samples / n_threads;
  common::ParallelFor(n_threads, n_threads, [&](auto tid) {
    std::size_t ibegin = tid * discard_size;
    std::size_t iend = (tid == (n_threads - 1)) ? n_samples : ibegin + discard_size;

    // Setup the eng
    const uint64_t displaced_seed =
        RandomReplace::SimpleSkip(ibegin, initial_seed, RandomReplace::kBase, RandomReplace::kMod);
    RandomReplace::EngineT eng(displaced_seed);

    fn(ibegin, iend, eng);
  });
}

namespace {
[[nodiscard]] float CalcSamplingInfo(Context const* ctx,
                                     linalg::MatrixView<GradientPair const> gpairs, float subsample,
                                     std::vector<float>* p_reg_abs_grad) {
  std::size_t n_samples = gpairs.Shape(0);
  std::size_t sample_rows = static_cast<std::size_t>(n_samples * subsample);

  std::vector<float> thresholds;
  *p_reg_abs_grad = CalcRegAbsGrad(ctx, gpairs, &thresholds);

  std::vector<float> grad_csum(n_samples);
  std::partial_sum(thresholds.begin(), thresholds.end() - 1, grad_csum.begin());
  float threshold =
      CalculateThreshold(common::Span{thresholds}, common::Span{grad_csum}, n_samples, sample_rows);
  return threshold;
}

[[nodiscard]] bool IsSampled(float p, float rnd) { return p >= 1.0f || (p > 0.0f && rnd <= p); }

void ZeroGradientPairs(linalg::MatrixView<GradientPair> gpairs) {
  std::fill(linalg::begin(gpairs), linalg::end(gpairs), GradientPair{});
}

template <typename Fn>
void ForEachGradientSample(Context const* ctx, bst_idx_t n_samples,
                           common::Span<float const> reg_abs_grad, float threshold,
                           std::uint64_t initial_seed, Fn&& fn) {
  ParallelSampling(ctx, n_samples, initial_seed,
                   [&](std::size_t ibegin, std::size_t iend, auto& eng) {
                     std::uniform_real_distribution<float> dist{0.0f, 1.0f};
                     for (std::size_t i = ibegin; i < iend; ++i) {
                       auto p = SamplingProbability(threshold, reg_abs_grad[i]);
                       // Consume exactly one random value per row for thread-count
                       // independence and replay.
                       auto rnd = dist(eng);
                       fn(i, p, IsSampled(p, rnd));
                     }
                   });
}

void GradientBasedSampling(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                           common::Span<float const> reg_abs_grad, float threshold,
                           std::uint64_t initial_seed) {
  auto n_samples = gpairs.Shape(0);
  auto n_targets = gpairs.Shape(1);
  auto apply = [&](std::size_t i, float p, bool is_sampled) {
    for (std::size_t t = 0; t < n_targets; ++t) {
      gpairs(i, t) = is_sampled ? RescaleGrad(p, gpairs(i, t)) : GradientPair{};
    }
  };
  ForEachGradientSample(ctx, n_samples, reg_abs_grad, threshold, initial_seed, apply);
}

void ApplyMvsWeights(Context const* ctx, linalg::Matrix<GradientPair>* value_gpair,
                     common::Span<float const> reg_abs_grad, float threshold,
                     std::uint64_t initial_seed) {
  auto h_value = value_gpair->HostView();
  auto n_samples = h_value.Shape(0);
  auto n_targets = h_value.Shape(1);
  auto apply = [&](std::size_t i, float p, bool is_sampled) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      h_value(i, t) = is_sampled ? RescaleGrad(p, h_value(i, t)) : GradientPair{};
    }
  };
  ForEachGradientSample(ctx, n_samples, reg_abs_grad, threshold, initial_seed, apply);
}

void UniformSample(Context const* ctx, linalg::MatrixView<GradientPair> out, float subsample,
                   std::uint64_t initial_seed) {
  auto n_samples = out.Shape(0);
  auto n_targets = out.Shape(1);
  CHECK_GE(n_targets, 1);

  ParallelSampling(ctx, n_samples, initial_seed,
                   [&](std::size_t ibegin, std::size_t iend, auto& eng) {
                     std::bernoulli_distribution coin_flip{subsample};
                     for (std::size_t i = ibegin; i < iend; ++i) {
                       if (!coin_flip(eng)) {
                         for (std::size_t t = 0; t < n_targets; ++t) {
                           out(i, t) = GradientPair{};
                         }
                       }
                     }
                   });
}
}  // namespace

std::vector<float> CalcRegAbsGrad(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                                  std::vector<float>* p_thresholds) {
  float mvs_lambda = kDefaultMvsLambda;
  std::size_t n_samples = gpairs.Shape(0);
  std::size_t n_targets = gpairs.Shape(1);
  std::vector<float> reg_abs_grad(n_samples);
  auto grad_op = MvsGradOp{mvs_lambda};
  common::ParallelFor(n_samples, ctx->Threads(), [&](auto i) {
    float sum_sq = 0.0f;
    for (std::size_t t = 0; t < n_targets; ++t) {
      sum_sq += grad_op(gpairs(i, t));
    }
    reg_abs_grad[i] = std::sqrt(sum_sq);
  });

  auto& thresholds = *p_thresholds;
  thresholds = reg_abs_grad;                                // Copy for sorting
  thresholds.push_back(std::numeric_limits<float>::max());  // sentinel
  common::Sort(ctx, thresholds.begin(), thresholds.end() - 1, std::less{});

  return reg_abs_grad;
}

float CalculateThreshold(common::Span<float const> sorted_rag, common::Span<float const> grad_csum,
                         bst_idx_t n_samples, bst_idx_t sample_rows) {
  CHECK_GE(n_samples, 1);
  // Use binary search to find the threshold index
  std::int64_t low_idx = 0;
  std::int64_t high_idx = n_samples - 1;
  while (low_idx <= high_idx) {
    std::int64_t i = low_idx + (high_idx - low_idx) / 2;

    float lower = sorted_rag[i];
    // Upper bound is next element or max for last element
    float upper = sorted_rag[i + 1];

    bst_idx_t n_above = n_samples - i - 1;
    float denom = static_cast<float>(sample_rows) - static_cast<float>(n_above);

    if (denom <= 0) {
      // i is too small, need to go right to increase denom
      low_idx = i + 1;
      continue;
    }

    float u = grad_csum[i] / denom;

    if (u > lower && u <= upper) {
      return u;
    }

    if (u <= lower) {
      high_idx = i - 1;
    } else {
      low_idx = i + 1;
    }
  }

  // p will be extremely small, no row can be sampled.
  if (sample_rows == 0) {
    return std::numeric_limits<float>::max();
  }
  // Degenerate case: all gradients are the same, so u cannot be greater than the lower
  // bound. Fall back to using the total sum divided by sample_rows.
  return grad_csum.back() / sample_rows;
}

void Sampler::Sample(Context const* ctx, linalg::MatrixView<GradientPair> out) {
  CHECK(out.Contiguous());
  std::size_t n_samples = out.Shape(0);
  std::size_t sample_rows = static_cast<std::size_t>(n_samples * subsample_);
  if (sample_rows >= n_samples || n_samples == 0) {
    is_sampling_ = false;
    return;
  }
  is_sampling_ = true;
  initial_seed_ = ctx->Rng()();

  switch (sampling_method_) {
    case TrainParam::kUniform:
      UniformSample(ctx, out, subsample_, initial_seed_);
      break;
    case TrainParam::kGradientBased: {
      if (sample_rows == 0) {
        ZeroGradientPairs(out);
        break;
      }
      std::vector<float> reg_abs_grad;
      auto threshold = CalcSamplingInfo(ctx, out, subsample_, &reg_abs_grad);
      GradientBasedSampling(ctx, out, common::Span{reg_abs_grad}, threshold, initial_seed_);
      break;
    }
    default:
      LOG(FATAL) << "Unknown sampling method: " << sampling_method_;
  }
}

void Sampler::ApplySampling(Context const* ctx, linalg::MatrixView<GradientPair const> split_gpair,
                            linalg::Matrix<GradientPair>* value_gpair) const {
  if (!is_sampling_) {
    return;
  }
  CHECK_EQ(split_gpair.Shape(0), value_gpair->Shape(0));
  switch (sampling_method_) {
    case TrainParam::kUniform: {
      UniformSample(ctx, value_gpair->HostView(), subsample_, initial_seed_);
      break;
    }
    case TrainParam::kGradientBased: {
      auto sample_rows = static_cast<std::size_t>(split_gpair.Shape(0) * subsample_);
      if (sample_rows == 0) {
        ZeroGradientPairs(value_gpair->HostView());
        break;
      }
      std::vector<float> reg_abs_grad;
      auto threshold = CalcSamplingInfo(ctx, split_gpair, subsample_, &reg_abs_grad);
      ApplyMvsWeights(ctx, value_gpair, reg_abs_grad, threshold, initial_seed_);
      break;
    }
    default:
      LOG(FATAL) << "Unknown sampling method: " << sampling_method_;
  }
}
}  // namespace xgboost::tree::cpu_impl
