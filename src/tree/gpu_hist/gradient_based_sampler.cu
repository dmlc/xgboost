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
                                           float subsample,
                                           SamplingMethod sampling_method)
    : batch_param_(batch_param), info_(info), sampling_method_(sampling_method) {
  monitor_.Init("gradient_based_sampler");

  if (subsample == 0.0f || subsample == 1.0f) {
    sample_rows_ = MaxSampleRows(n_rows);
  } else {
    sample_rows_ = n_rows * subsample;
  }

  if (sample_rows_ >= n_rows) {
    sampling_method_ = kNoSampling;
    sample_rows_ = n_rows;
    LOG(CONSOLE) << "Keeping " << sample_rows_ << " rows in GPU memory, not sampling";
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

size_t GradientBasedSampler::MaxSampleRows(size_t n_rows) {
  size_t available_memory = dh::AvailableMemory(batch_param_.gpu_id);
  // Subtract row_weight_, row_index_, and sample_row_index_.
  available_memory -= n_rows * (sizeof(float) + 2 * sizeof(size_t));
  size_t usable_memory = available_memory * 0.7;
  size_t extra_bytes = sizeof(GradientPair);
  size_t max_rows = common::CompressedBufferWriter::CalculateMaxRows(
      usable_memory, info_.NumSymbols(), info_.row_stride, extra_bytes);
  return max_rows;
}

GradientBasedSample GradientBasedSampler::Sample(common::Span<GradientPair> gpair,
                                                 DMatrix* dmat) {
  switch (sampling_method_) {
    case kNoSampling:
      return NoSampling(gpair, dmat);
    case kSequentialPoissonSampling:
      return SequentialPoissonSampling(gpair, dmat);
    case kUniformSampling:
      return UniformSampling(gpair, dmat);
    default:
      LOG(FATAL) << "unknown sampling method";
      return {sample_rows_, page_.get(), gpair};
  }
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

GradientBasedSample GradientBasedSampler::NoSampling(common::Span<GradientPair> gpair,
                                                     DMatrix* dmat) {
  CollectPages(dmat);
  return {sample_rows_, page_.get(), gpair};
}

/*! \brief A functor that calculate the weight of each row as random(0, 1) / abs(grad). */
struct CalculateWeight : public thrust::binary_function<GradientPair, size_t, float> {
  const uint32_t seed;

  XGBOOST_DEVICE explicit CalculateWeight(size_t _seed) : seed(_seed) {}

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

/*! \brief A functor that returns true if the gradient pair is non-zero. */
struct IsNonZero : public thrust::unary_function<GradientPair, bool> {
  XGBOOST_DEVICE bool operator()(const GradientPair& gpair) const {
    return gpair.GetGrad() != 0 || gpair.GetHess() != 0;
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

GradientBasedSample GradientBasedSampler::SequentialPoissonSampling(
    common::Span<xgboost::GradientPair> gpair, DMatrix* dmat) {
  // Transform the gradient to weight = random(0, 1) / abs(grad).
  thrust::transform(dh::tbegin(gpair), dh::tend(gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(row_weight_),
                    CalculateWeight(common::GlobalRandom()()));
  return WeightedSampling(gpair, dmat);
}

GradientBasedSample GradientBasedSampler::WeightedSampling(
      common::Span<xgboost::GradientPair> gpair, DMatrix* dmat) {
  // Sort the gradient pairs and row indexes by weight.
  thrust::sort_by_key(dh::tbegin(row_weight_), dh::tend(row_weight_),
                      thrust::make_zip_iterator(thrust::make_tuple(dh::tbegin(gpair),
                                                                   dh::tbegin(row_index_))));

  // Clear the gradient pairs not in the sample.
  thrust::fill(dh::tbegin(gpair) + sample_rows_, dh::tend(gpair), GradientPair());

  // Mask the sample rows.
  thrust::fill(dh::tbegin(sample_row_index_), dh::tbegin(sample_row_index_) + sample_rows_, 1);
  thrust::fill(dh::tbegin(sample_row_index_) + sample_rows_, dh::tend(sample_row_index_), 0);

  // Sort the gradient pairs and sample row indexes by the original row index.
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
  thrust::fill(dh::tbegin(page_->gidx_buffer), dh::tend(page_->gidx_buffer), 0);
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), sample_row_index_);
  }

  return {sample_rows_, page_.get(), gpair_};
}

/*! \brief A functor that returns random weights. */
struct RandomWeight : public thrust::unary_function<size_t, float> {
  const uint32_t seed;

  XGBOOST_DEVICE explicit RandomWeight(size_t _seed) : seed(_seed) {}

  XGBOOST_DEVICE float operator()(size_t i) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng);
  }
};

GradientBasedSample GradientBasedSampler::UniformSampling(common::Span<GradientPair> gpair,
                                                          DMatrix* dmat) {
  // Generate random weights.
  thrust::transform(thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(0) + gpair.size(),
                    dh::tbegin(row_weight_),
                    RandomWeight(common::GlobalRandom()()));
  return WeightedSampling(gpair, dmat);
}
};  // namespace tree
};  // namespace xgboost
