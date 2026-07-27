/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t

#include "../../common/device_vector.cuh"  // for device_vector, caching_device_vector
#include "../../common/random.cuh"         // for DefaultRng, UniformRealDistribution
#include "../hist/sampler.h"               // for SamplingProbability
#include "quantiser.cuh"                   // for GradientQuantiser
#include "xgboost/base.h"                  // for GradientPair
#include "xgboost/data.h"                  // for BatchParam
#include "xgboost/linalg.h"                // for MatrixView

namespace xgboost::tree::cuda_impl {
/** @brief A deterministic random weight for a row. */
class RandomWeight {
 public:
  XGBOOST_DEVICE explicit RandomWeight(std::size_t seed)
      : seed_{static_cast<std::uint32_t>(seed)} {}

  XGBOOST_DEVICE float operator()(std::size_t i) const {
    common::cuda_impl::DefaultRng rng{seed_};
    common::cuda_impl::UniformRealDistribution<float> dist;
    rng.discard(i);
    return dist(rng);
  }

 private:
  std::uint32_t seed_;
};

enum class SamplingMethod : std::int32_t {
  kNone,
  kUniform,
  kGradientBased,
};

/**
 * @brief Scalar and device-view state required to replay a row sampling decision.
 */
struct SamplingInfo {
  SamplingMethod method{SamplingMethod::kNone};
  float subsample{1.0f};
  std::uint32_t seed{0};
  common::Span<float const> thresholds;
  common::Span<float const> reg_abs_grad;
  std::size_t threshold_index{0};

  XGBOOST_DEVICE float Probability(std::size_t ridx) const {
    switch (method) {
      case SamplingMethod::kNone:
        return 1.0f;
      case SamplingMethod::kUniform:
        return subsample;
      case SamplingMethod::kGradientBased:
        return SamplingProbability(thresholds[threshold_index], reg_abs_grad[ridx]);
    }
    KERNEL_CHECK(false);
    return 1.0;
  }

  XGBOOST_DEVICE bool IsSampled(std::size_t ridx, float p) const {
    return p >= 1.0f || (p > 0.0f && RandomWeight{seed}(ridx) <= p);
  }

  XGBOOST_DEVICE bool IsSampled(std::size_t ridx) const {
    return this->IsSampled(ridx, this->Probability(ridx));
  }
};

// no-op base class.
class SamplingStrategy {
 public:
  virtual void Sample(Context const*, linalg::MatrixView<GradientPairInt64>,
                      common::Span<GradientQuantiser const>) {}
  virtual void ApplySampling(Context const*, linalg::Matrix<GradientPair>*) {};
  [[nodiscard]] virtual SamplingInfo GetSamplingInfo() const { return {}; }
  virtual ~SamplingStrategy() = default;
};

/** @brief Uniform sampling */
class UniformSampling : public SamplingStrategy {
 public:
  UniformSampling(std::size_t n_samples, float subsample)
      : n_samples_{n_samples}, subsample_{subsample} {}
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings) override;
  void ApplySampling(Context const* ctx, linalg::Matrix<GradientPair>* value_gpair) override;
  [[nodiscard]] SamplingInfo GetSamplingInfo() const override {
    SamplingInfo info;
    info.method = SamplingMethod::kUniform;
    info.subsample = subsample_;
    info.seed = seed_;
    return info;
  }

 private:
  std::size_t const n_samples_;
  float const subsample_;
  std::uint32_t seed_{0};
};

/** @brief Gradient-based sampling. */
class GradientBasedSampling : public SamplingStrategy {
 public:
  GradientBasedSampling(std::size_t n_samples, float subsample);
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings) override;
  void ApplySampling(Context const* ctx, linalg::Matrix<GradientPair>* value_gpair) override;
  [[nodiscard]] SamplingInfo GetSamplingInfo() const override;

 private:
  float const subsample_;
  std::uint32_t seed_{0};
  std::size_t threshold_index_{0};
  // abs gradient
  dh::device_vector<float> reg_abs_grad_;
  // sorted abs gradient
  dh::device_vector<float> thresholds_;
  // csum of sorted abs gradient
  dh::device_vector<float> grad_csum_;
};

/**
 * @brief Draw sample rows by setting non-selected gradient to 0.
 *
 * @see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * @see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * @see Ohlsson, E. (1998). Sequential Poisson sampling. Journal of official Statistics, 14(2), 149.
 * @see Rong Ou. (2020). Out-of-Core GPU Gradient Boosting.
 */
class Sampler {
 public:
  Sampler(bst_idx_t n_samples, float subsample, int sampling_method);

  /** @brief Sample from a DMatrix based on the given gradient pairs. */
  void Sample(Context const* ctx, linalg::MatrixView<GradientPairInt64> gpair,
              common::Span<GradientQuantiser const> roundings);
  /** @brief Apply sampling weights to value gradient. */
  void ApplySampling(Context const* ctx, linalg::Matrix<GradientPair>* value_gpair);
  /** @brief Get the state required to replay sampling for any row. */
  [[nodiscard]] SamplingInfo GetSamplingInfo() const;

 private:
  std::unique_ptr<SamplingStrategy> strategy_;
};

std::size_t CalculateThresholdIndex(Context const* ctx, common::Span<float> sorted_rag,
                                    common::Span<float> grad_csum, bst_idx_t n_samples,
                                    bst_idx_t sample_rows);
}  // namespace xgboost::tree::cuda_impl
