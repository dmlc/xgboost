/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_SAMPLER_H_
#define XGBOOST_TREE_HIST_SAMPLER_H_

#include <cstdint>  // for uint64_t
#include <random>   // for bernoulli_distribution, linear_congruential_engine

#include "../../common/math.h"  // for Sqr
#include "../param.h"           // for TrainParam
#include "xgboost/base.h"       // for GradientPair, bst_idx_t
#include "xgboost/context.h"    // for Context
#include "xgboost/data.h"       // for MetaInfo
#include "xgboost/linalg.h"     // for TensorView

namespace xgboost::tree {
struct RandomReplace {
 public:
  // similar value as for minstd_rand
  static constexpr std::uint64_t kBase = 16807;
  static constexpr std::uint64_t kMod = static_cast<std::uint64_t>(1) << 63;

  using EngineT = std::linear_congruential_engine<uint64_t, kBase, 0, kMod>;

  /*
    Right-to-left binary method: https://en.wikipedia.org/wiki/Modular_exponentiation
  */
  static std::uint64_t SimpleSkip(std::uint64_t exponent, std::uint64_t initial_seed,
                                  std::uint64_t base, std::uint64_t mod) {
    CHECK_LE(exponent, mod);
    std::uint64_t result = 1;
    while (exponent > 0) {
      if (exponent % 2 == 1) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exponent = exponent >> 1;
    }
    // with result we can now find the new seed
    return (result * initial_seed) % mod;
  }
};

constexpr float kDefaultMvsLambda = 0.1f;

struct MvsGradOp {
  float lambda;
  template <typename GradientType>
  XGBOOST_DEVICE float operator()(GradientType const& gpair) const {
    auto g = gpair.GetGrad();
    auto h = gpair.GetHess();
    return common::Sqr(g) + lambda * common::Sqr(h);
  }
};

namespace cpu_impl {
void UniformSample(Context const* ctx, linalg::MatrixView<GradientPair> out, float subsample);

void GradientBasedSample(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                         float subsample);

/**
 * @brief Sample gradients based on the configured sampling method.
 *
 * Supports both uniform and gradient-based (MVS) sampling methods.
 */
inline void SampleGradient(Context const* ctx, TrainParam const& param,
                           linalg::MatrixView<GradientPair> out) {
  CHECK(out.Contiguous());

  std::size_t n_samples = out.Shape(0);
  std::size_t sample_rows = static_cast<std::size_t>(n_samples * param.subsample);
  if (sample_rows >= n_samples) {
    return;  // No sampling needed
  }
  if (n_samples == 0) {
    return;
  }

  switch (param.sampling_method) {
    case TrainParam::kUniform:
      UniformSample(ctx, out, param.subsample);
      break;
    case TrainParam::kGradientBased:
      GradientBasedSample(ctx, out, param.subsample);
      break;
    default:
      LOG(FATAL) << "Unknown sampling method: " << param.sampling_method;
  }
}

/**
 * @brief Apply sampling mask from sampled split gradient to value gradient.
 *
 * Zero out rows in value gradient where the corresponding row in split gradient was not
 * sampled (has zero hessian). Value gradient may have more targets than split gradient.
 */
void ApplySamplingMask(Context const* ctx, linalg::Matrix<GradientPair> const& sampled_split_gpair,
                       linalg::Matrix<GradientPair>* value_gpair);
}  // namespace cpu_impl
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_SAMPLER_H_
