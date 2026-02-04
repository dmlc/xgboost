/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_SAMPLER_H_
#define XGBOOST_TREE_HIST_SAMPLER_H_

#include <cstdint>  // for uint64_t
#include <limits>   // for numeric_limits
#include <random>   // for bernoulli_distribution, linear_congruential_engine
#include <vector>   // for vector

#include "../../common/math.h"  // for Sqr
#include "../param.h"           // for TrainParam
#include "xgboost/base.h"       // for GradientPair, bst_idx_t
#include "xgboost/context.h"    // for Context
#include "xgboost/data.h"       // for MetaInfo
#include "xgboost/linalg.h"     // for TensorView
#include "xgboost/span.h"       // for Span

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
// Calculate regularized absolute gradient for each row.
std::vector<float> CalcRegAbsGrad(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                                  std::vector<float>* p_thresholds);

float CalculateThreshold(common::Span<float const> sorted_rag, common::Span<float const> grad_csum,
                         bst_idx_t n_samples, bst_idx_t sample_rows);

class Sampler {
 public:
  explicit Sampler(TrainParam const& param)
      : sampling_method_{param.sampling_method}, subsample_{param.subsample} {}

  void Sample(Context const* ctx, linalg::MatrixView<GradientPair> out);
  void ApplySampling(Context const* ctx, linalg::MatrixView<GradientPair const> sampled_split_gpair,
                     linalg::Matrix<GradientPair>* value_gpair) const;

 private:
  int sampling_method_{TrainParam::kUniform};
  float subsample_{1.0f};
  bool is_sampling_{false};
};
}  // namespace cpu_impl
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_SAMPLER_H_
