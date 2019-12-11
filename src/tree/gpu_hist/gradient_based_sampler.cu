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

GradientBasedSampler::GradientBasedSampler(BatchParam batch_param,
                                           EllpackInfo info,
                                           size_t n_rows,
                                           float subsample)
    : batch_param_(batch_param), info_(info), sampling_method_(kDefaultSamplingMethod) {
  monitor_.Init("gradient_based_sampler");

  if (subsample == 0.0f || subsample == 1.0f) {
    sample_rows_ = MaxSampleRows();
  } else {
    sample_rows_ = n_rows * subsample;
  }

  if (sample_rows_ >= n_rows) {
    sampling_method_ = kNoSampling;
    sample_rows_ = n_rows;
    LOG(CONSOLE) << "Keeping " << sample_rows_ << " in GPU memory, not sampling";
  } else {
    LOG(CONSOLE) << "Sampling " << sample_rows_ << " rows";
  }

  page_.reset(new EllpackPageImpl(batch_param.gpu_id, info, sample_rows_));
  if (sampling_method_ != kNoSampling) {
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

size_t GradientBasedSampler::MaxSampleRows() {
  size_t available_memory = dh::AvailableMemory(batch_param_.gpu_id);
  size_t usable_memory = available_memory * 0.95;
  size_t extra_bytes = sizeof(GradientPair) + sizeof(float) + 2 * sizeof(size_t);
  size_t max_rows = common::CompressedBufferWriter::CalculateMaxRows(
      usable_memory, info_.NumSymbols(), info_.row_stride, extra_bytes);
  return max_rows;
}

/*! \brief A functor that returns the absolute value of gradient from a gradient pair. */
struct AbsoluteGradient : public thrust::unary_function<GradientPair, float> {
  XGBOOST_DEVICE float operator()(const GradientPair& gpair) const {
    return fabsf(gpair.GetGrad());
  }
};

/*! \brief A functor that returns true if the gradient pair is non-zero. */
struct IsNonZero : public thrust::unary_function<GradientPair, bool> {
  XGBOOST_DEVICE bool operator()(const GradientPair& gpair) const {
    return gpair.GetGrad() != 0 || gpair.GetHess() != 0;
  }
};

GradientBasedSample GradientBasedSampler::Sample(common::Span<GradientPair> gpair,
                                                 DMatrix* dmat,
                                                 SamplingMethod sampling_method) {
  if (sampling_method_ != kNoSampling) {
    sampling_method_ = sampling_method;
  }

  switch (sampling_method_) {
    case kNoSampling:
      return NoSampling(gpair, dmat);
    case kPoissonSampling:
      return PoissonSampling(gpair, dmat);
    case kSequentialPoissonSampling:
      return SequentialPoissonSampling(gpair, dmat);
    case kUniformSampling:
      return UniformSampling(gpair, dmat);
    default:
      LOG(FATAL) << "unknown sampling method";
      return {sample_rows_, page_.get(), gpair};
  }
}

GradientBasedSample GradientBasedSampler::NoSampling(common::Span<GradientPair> gpair,
                                                     DMatrix* dmat) {
  CollectPages(dmat);
  return {sample_rows_, page_.get(), gpair};
}

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

/*! \brief A functor that samples a gradient pair.
 *
 * Sampling probability is proportional to the absolute value of the gradient.
 */
struct PoissonSamplingFunction
    : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  const size_t sample_rows;
  const float sum_abs_gradient;
  const uint32_t seed;

  XGBOOST_DEVICE PoissonSamplingFunction(size_t _sample_rows, float _sum_abs_gradient, size_t _seed)
      : sample_rows(_sample_rows), sum_abs_gradient(_sum_abs_gradient), seed(_seed) {}

  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t i) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    float p = sample_rows * fabsf(gpair.GetGrad()) / sum_abs_gradient;
    if (p > 1.0f) {
      p = 1.0f;
    }
    if (dist(rng) <= p) {
      return gpair;
    } else {
      return GradientPair();
    }
  }
};

/*! \brief A functor that clears the row indexes with empty gradient. */
struct ClearEmptyRows : public thrust::binary_function<GradientPair, size_t, size_t> {
  const size_t max_rows;

  XGBOOST_DEVICE explicit ClearEmptyRows(size_t max_rows) : max_rows(max_rows) {}

  XGBOOST_DEVICE size_t operator()(const GradientPair& gpair, size_t row_index) const {
    if ((gpair.GetGrad() != 0 || gpair.GetHess() != 0) && row_index < max_rows) {
      return row_index;
    } else {
      return SIZE_MAX;
    }
  }
};

/*! \brief A functor that trims extra sampled rows. */
struct TrimExtraRows : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t row_index) const {
    if (row_index == SIZE_MAX) {
      return GradientPair();
    } else {
      return gpair;
    }
  }
};

GradientBasedSample GradientBasedSampler::PoissonSampling(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  // Sum the absolute value of gradients as the denominator to normalize the probability.
  float sum_abs_gradient = thrust::transform_reduce(dh::tbegin(gpair), dh::tend(gpair),
                                                    AbsoluteGradient(),
                                                    0.0f, thrust::plus<float>());

  // Poisson sampling of the gradient pairs based on the absolute value of the gradient.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(gpair),
                    PoissonSamplingFunction(sample_rows_,
                                            sum_abs_gradient,
                                            common::GlobalRandom()()));

  // Map the original row index to the sample row index.
  thrust::fill(dh::tbegin(sample_row_index_), dh::tend(sample_row_index_), 0);
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(sample_row_index_),
                    IsNonZero());
  thrust::exclusive_scan(dh::tbegin(sample_row_index_), dh::tend(sample_row_index_),
                         dh::tbegin(sample_row_index_));
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(sample_row_index_),
                    dh::tbegin(sample_row_index_),
                    ClearEmptyRows(sample_rows_));

  // Zero out the gradient pairs if there are more rows than desired.
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    dh::tbegin(sample_row_index_),
                    dh::tbegin(gpair),
                    TrimExtraRows());

  // Compact the non-zero gradient pairs.
  thrust::copy_if(dh::tbegin(gpair), dh::tend(gpair), dh::tbegin(gpair_), IsNonZero());

  // Compact the ELLPACK pages into the single sample page.
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), sample_row_index_);
  }

  return {sample_rows_, page_.get(), gpair_};
}

/*! \brief A functor that samples gradient pairs using sequential Poisson sampling.
 *
 * Sampling probability is proportional to the absolute value of the gradient.
 */
struct SequentialPoissonSamplingFunction
    : public thrust::binary_function<GradientPair, size_t, float> {
  const uint32_t seed;

  XGBOOST_DEVICE explicit SequentialPoissonSamplingFunction(size_t _seed) : seed(_seed) {}

  XGBOOST_DEVICE float operator()(const GradientPair& gpair, size_t i) {
    if (gpair.GetGrad() == 0) {
      return FLT_MAX;
    }
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng) / fabsf(gpair.GetGrad());
  }
};

GradientBasedSample GradientBasedSampler::SequentialPoissonSampling(
    common::Span<xgboost::GradientPair> gpair, DMatrix* dmat) {
  // Transform the gradient to weight = random(0, 1) / abs(grad).
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(row_weight_),
                    SequentialPoissonSamplingFunction(common::GlobalRandom()()));

  // Sort the gradient pairs and row indexes by weight.
  thrust::sort_by_key(dh::tbegin(row_weight_), dh::tend(row_weight_),
                      thrust::make_zip_iterator(thrust::make_tuple(dh::tbegin(gpair),
                                                                   dh::tbegin(row_index_))));

  // Clear the gradient pairs not in the sample.
  thrust::fill(dh::tbegin(gpair) + sample_rows_, dh::tend(gpair), GradientPair());

  // Mask the sample rows.
  thrust::fill(dh::tbegin(sample_row_index_), dh::tbegin(sample_row_index_) + sample_rows_, 1);
  thrust::fill(dh::tbegin(sample_row_index_) + sample_rows_, dh::tend(sample_row_index_), 0);

  // Sort the gradient pairs and sample row indexed by the original row index.
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
                    ClearEmptyRows(sample_rows_));

  // Compact the ELLPACK pages into the single sample page.
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), sample_row_index_);
  }

  return {sample_rows_, page_.get(), gpair_};
}

GradientBasedSample GradientBasedSampler::UniformSampling(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  // TODO(rongou): implement this.
  return {sample_rows_, page_.get(), gpair_};
}
};  // namespace tree
};  // namespace xgboost
