/**
 * Copyright 2026, XGBoost Contributors
 */
#include "sampler.h"  // for kDefaultMvsLambda

#include <cmath>    // for sqrt
#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <numeric>  // for partial_sum
#include <random>   // for default_random_engine, uniform_real_distribution
#include <vector>   // for vector

#include "../../common/algorithm.h"  // for Sort
#include "../../common/random.h"     // for GlobalRandom
#include "xgboost/base.h"            // for GradientPair, GradientPairPrecise
#include "xgboost/linalg.h"          // for MatrixView
#include "xgboost/span.h"            // for Span

namespace xgboost::tree::cpu_impl {
template <typename Fn>
void ParallelSampling(bst_idx_t n_samples, std::int32_t n_threads, Fn&& fn) {
  auto& rnd = common::GlobalRandom();
  std::uint64_t initial_seed = rnd();
  std::size_t const discard_size = n_samples / n_threads;

  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      // setup the block
      const std::int32_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (n_threads - 1)) ? n_samples : ibegin + discard_size;

      // Setup the eng
      const uint64_t displaced_seed = RandomReplace::SimpleSkip(
          ibegin, initial_seed, RandomReplace::kBase, RandomReplace::kMod);
      RandomReplace::EngineT eng(displaced_seed);

      fn(ibegin, iend, eng);
    });
  }
  exc.Rethrow();
}

namespace {
[[nodiscard]] float CalcSamplingInfo(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                                     float subsample, std::vector<float>* p_reg_abs_grad) {
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

void GradientBasedSampling(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                           common::Span<float const> reg_abs_grad, float threshold) {
  std::uniform_real_distribution<float> dist{0.0f, 1.0f};
  auto n_samples = gpairs.Shape(0);
  auto n_targets = gpairs.Shape(1);
  ParallelSampling(n_samples, ctx->Threads(), [&](std::size_t ibegin, std::size_t iend, auto& eng) {
    for (std::size_t i = ibegin; i < iend; ++i) {
      float p = SamplingProbability(threshold, reg_abs_grad[i]);
      // Skip rows with zero gradient (already zero)
      if (gpairs(i, 0).GetGrad() == 0.0 && gpairs(i, 0).GetHess() == 0.0) {
        continue;
      }

      if (p >= 1.0f) {
        // Always select this row.
        continue;
      }
      float rand_val = dist(eng);
      if (rand_val <= p) {
        for (std::size_t t = 0; t < n_targets; ++t) {
          gpairs(i, t) = RescaleGrad(p, gpairs(i, t));
        }
      } else {
        // Not selected: zero out
        for (std::size_t t = 0; t < n_targets; ++t) {
          gpairs(i, t) = GradientPair{};
        }
      }
    }
  });
}

void ApplyMvsWeights(Context const* ctx, linalg::MatrixView<GradientPair const> sampled_split_gpair,
                     linalg::Matrix<GradientPair>* value_gpair,
                     common::Span<float const> reg_abs_grad, float threshold) {
  CHECK_EQ(sampled_split_gpair.Shape(0), value_gpair->Shape(0));
  auto h_split = sampled_split_gpair;
  auto h_value = value_gpair->HostView();
  auto n_samples = h_value.Shape(0);
  auto n_targets = h_value.Shape(1);

  common::ParallelFor(n_samples, ctx->Threads(), [&](bst_idx_t i) {
    // Check if this row was not sampled (hessian is zero in split gradient)
    if (h_split(i, 0).GetHess() == 0.0f) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        h_value(i, t) = GradientPair{};
      }
      return;
    }
    float p = SamplingProbability(threshold, reg_abs_grad[i]);
    for (bst_target_t t = 0; t < n_targets; ++t) {
      h_value(i, t) = RescaleGrad(p, h_value(i, t));
    }
  });
}

void ApplySamplingMask(Context const* ctx,
                       linalg::MatrixView<GradientPair const> sampled_split_gpair,
                       linalg::Matrix<GradientPair>* value_gpair) {
  CHECK_EQ(sampled_split_gpair.Shape(0), value_gpair->Shape(0));
  auto h_split = sampled_split_gpair;
  auto h_value = value_gpair->HostView();
  auto n_samples = h_value.Shape(0);
  auto n_targets = h_value.Shape(1);

  common::ParallelFor(n_samples, ctx->Threads(), [&](bst_idx_t i) {
    // Check if this row was not sampled (hessian is zero in split gradient)
    if (h_split(i, 0).GetHess() == 0.0f) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        h_value(i, t) = GradientPair{};
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

void UniformSample(Context const* ctx, linalg::MatrixView<GradientPair> out, float subsample) {
  bst_idx_t n_samples = out.Shape(0);
  std::size_t n_targets = out.Shape(1);
  std::bernoulli_distribution coin_flip{subsample};
  CHECK_GE(n_targets, 1);

  ParallelSampling(n_samples, ctx->Threads(), [&](std::size_t ibegin, std::size_t iend, auto& eng) {
    for (std::size_t i = ibegin; i < iend; ++i) {
      if (!coin_flip(eng)) {
        for (std::size_t j = 0; j < n_targets; ++j) {
          out(i, j) = GradientPair{};
        }
      }
    }
  });
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

  switch (sampling_method_) {
    case TrainParam::kUniform:
      UniformSample(ctx, out, subsample_);
      break;
    case TrainParam::kGradientBased: {
      std::vector<float> reg_abs_grad;
      auto threshold = CalcSamplingInfo(ctx, out, subsample_, &reg_abs_grad);
      GradientBasedSampling(ctx, out, common::Span{reg_abs_grad}, threshold);
      break;
    }
    default:
      LOG(FATAL) << "Unknown sampling method: " << sampling_method_;
  }
}

void Sampler::ApplySampling(Context const* ctx,
                            linalg::MatrixView<GradientPair const> sampled_split_gpair,
                            linalg::Matrix<GradientPair>* value_gpair) const {
  if (!is_sampling_) {
    return;
  }
  switch (sampling_method_) {
    case TrainParam::kUniform: {
      ApplySamplingMask(ctx, sampled_split_gpair, value_gpair);
      break;
    }
    case TrainParam::kGradientBased: {
      std::vector<float> reg_abs_grad;
      auto threshold = CalcSamplingInfo(ctx, value_gpair->HostView(), subsample_, &reg_abs_grad);
      ApplyMvsWeights(ctx, sampled_split_gpair, value_gpair, reg_abs_grad, threshold);
      break;
    }
    default:
      LOG(FATAL) << "Unknown sampling method: " << sampling_method_;
  }
}
}  // namespace xgboost::tree::cpu_impl
