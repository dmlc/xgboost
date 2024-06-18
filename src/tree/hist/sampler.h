/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_SAMPLER_H_
#define XGBOOST_TREE_HIST_SAMPLER_H_

#include <cstddef>  // std::size-t
#include <cstdint>  // std::uint64_t
#include <random>   // bernoulli_distribution, linear_congruential_engine

#include "../../common/random.h"  // GlobalRandom
#include "../param.h"             // TrainParam
#include "xgboost/base.h"         // GradientPair
#include "xgboost/context.h"      // Context
#include "xgboost/data.h"         // MetaInfo
#include "xgboost/linalg.h"       // TensorView

namespace xgboost {
namespace tree {
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

// Only uniform sampling, no gradient-based yet.
inline void SampleGradient(Context const* ctx, TrainParam param,
                           linalg::MatrixView<GradientPair> out) {
  CHECK(out.Contiguous());
  CHECK_EQ(param.sampling_method, TrainParam::kUniform)
      << "Only uniform sampling is supported, gradient-based sampling is only support by GPU Hist.";

  if (param.subsample >= 1.0) {
    return;
  }
  bst_row_t n_samples = out.Shape(0);
  auto& rnd = common::GlobalRandom();

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param.subsample);
  CHECK_EQ(out.Shape(1), 1) << "Multi-target with sampling for R is not yet supported.";
  for (size_t i = 0; i < n_samples; ++i) {
    if (!(out(i, 0).GetHess() >= 0.0f && coin_flip(rnd)) || out(i, 0).GetGrad() == 0.0f) {
      out(i, 0) = GradientPair(0);
    }
  }
#else
  std::uint64_t initial_seed = rnd();

  auto n_threads = static_cast<size_t>(ctx->Threads());
  std::size_t const discard_size = n_samples / n_threads;
  std::bernoulli_distribution coin_flip(param.subsample);

  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (n_threads - 1)) ? n_samples : ibegin + discard_size;

      const uint64_t displaced_seed = RandomReplace::SimpleSkip(
          ibegin, initial_seed, RandomReplace::kBase, RandomReplace::kMod);
      RandomReplace::EngineT eng(displaced_seed);
      std::size_t n_targets = out.Shape(1);
      if (n_targets > 1) {
        for (std::size_t i = ibegin; i < iend; ++i) {
          if (!coin_flip(eng)) {
            for (std::size_t j = 0; j < n_targets; ++j) {
              out(i, j) = GradientPair{};
            }
          }
        }
      } else {
        for (std::size_t i = ibegin; i < iend; ++i) {
          if (!coin_flip(eng)) {
            out(i, 0) = GradientPair{};
          }
        }
      }
    });
  }
  exc.Rethrow();
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_HIST_SAMPLER_H_
