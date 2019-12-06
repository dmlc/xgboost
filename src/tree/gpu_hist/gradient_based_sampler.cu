/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <xgboost/host_device_vector.h>
#include <xgboost/logging.h>

#include "../../common/compressed_iterator.h"
#include "../../common/random.h"
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

GradientBasedSampler::GradientBasedSampler(BatchParam batch_param,
                                           EllpackInfo info,
                                           size_t n_rows,
                                           size_t sample_rows)
    : batch_param_(batch_param), info_(info), sample_rows_(sample_rows) {
  monitor_.Init("gradient_based_sampler");

  if (sample_rows_ == 0) {
    sample_rows_ = MaxSampleRows();
  }
  if (sample_rows_ >= n_rows) {
    is_sampling_ = false;
    sample_rows_ = n_rows;
    LOG(CONSOLE) << "Keeping " << sample_rows_ << " in GPU memory, not sampling";
  } else {
    is_sampling_ = true;
    LOG(CONSOLE) << "Sampling " << sample_rows_ << " rows";
  }

  page_.reset(new EllpackPageImpl(batch_param.gpu_id, info, sample_rows_));
  if (is_sampling_) {
    gpair_.SetDevice(batch_param_.gpu_id);
    gpair_.Resize(sample_rows_, GradientPair());
    sample_row_index_.SetDevice(batch_param_.gpu_id);
    sample_row_index_.Resize(n_rows, 0);
  }
}

size_t GradientBasedSampler::MaxSampleRows() {
  size_t available_memory = dh::AvailableMemory(batch_param_.gpu_id);
  size_t usable_memory = available_memory * 0.95;
  size_t extra_bytes = sizeof(GradientPair) + sizeof(size_t);
  size_t max_rows = common::CompressedBufferWriter::CalculateMaxRows(
      usable_memory, info_.NumSymbols(), info_.row_stride, extra_bytes);
  return max_rows;
}

/*! \brief A functor that returns the absolute value of gradient from a gradient pair. */
struct abs_grad : public thrust::unary_function<GradientPair, float> {
  XGBOOST_DEVICE float operator()(const GradientPair& gpair) const {
    return fabsf(gpair.GetGrad());
  }
};

/*! \brief A functor that samples a gradient pair.
 *
 * Sampling probability is proportional to the absolute value of the gradient.
 */
struct sample_gradient : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  const size_t sample_rows;
  const float sum_abs_gradient;
  const uint32_t seed;

  XGBOOST_DEVICE sample_gradient(size_t _sample_rows, float _sum_abs_gradient, size_t _seed)
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

/*! \brief A functor that returns true if the gradient pair is non-zero. */
struct is_non_zero : public thrust::unary_function<GradientPair, bool> {
  XGBOOST_DEVICE bool operator()(const GradientPair& gpair) const {
    return gpair.GetGrad() != 0 || gpair.GetHess() != 0;
  }
};

/*! \brief A functor that clears the row indexes with empty gradient. */
struct clear_empty_rows : public thrust::binary_function<GradientPair, size_t, size_t> {
  const size_t max_rows;

  XGBOOST_DEVICE clear_empty_rows(size_t max_rows) : max_rows(max_rows) {}

  XGBOOST_DEVICE size_t operator()(const GradientPair& gpair, size_t row_index) const {
    if ((gpair.GetGrad() != 0 || gpair.GetHess() != 0) && row_index < max_rows) {
      return row_index;
    } else {
      return SIZE_MAX;
    }
  }
};

/*! \brief A functor that trims extra sampled rows. */
struct trim_extra_rows : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  XGBOOST_DEVICE GradientPair operator()(const GradientPair& gpair, size_t row_index) const {
    if (row_index == SIZE_MAX) {
      return GradientPair();
    } else {
      return gpair;
    }
  }
};

GradientBasedSample GradientBasedSampler::Sample(HostDeviceVector<GradientPair>* gpair,
                                                 DMatrix* dmat) {
  // If there is enough space for all rows, just collect them in a single ELLPACK page and return.
  if (!is_sampling_) {
    CollectPages(dmat);
    auto out_gpair = gpair->DeviceSpan();
    return {sample_rows_, page_.get(), out_gpair};
  }

  // Sum the absolute value of gradients as the denominator to normalize the probability.
  float sum_abs_gradient = thrust::transform_reduce(
      dh::tbegin(*gpair), dh::tend(*gpair), abs_grad(), 0.0f, thrust::plus<float>());

  // Poisson sampling of the gradient pairs based on the absolute value of the gradient.
  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(*gpair),
                    sample_gradient(sample_rows_, sum_abs_gradient, common::GlobalRandom()()));

  // Map the original row index to the sample row index.
  sample_row_index_.Fill(0);
  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair),
                    dh::tbegin(sample_row_index_),
                    is_non_zero());
  thrust::exclusive_scan(dh::tbegin(sample_row_index_), dh::tend(sample_row_index_),
                         dh::tbegin(sample_row_index_));
  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair),
                    dh::tbegin(sample_row_index_),
                    dh::tbegin(sample_row_index_),
                    clear_empty_rows(sample_rows_));

  // Zero out the gradient pairs if there are more rows than desired.
  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair),
                    dh::tbegin(sample_row_index_),
                    dh::tbegin(*gpair),
                    trim_extra_rows());

  // Compact the non-zero gradient pairs.
  thrust::copy_if(dh::tbegin(*gpair), dh::tend(*gpair), dh::tbegin(gpair_), is_non_zero());

  // Compact the ELLPACK pages into the single sample page.
  for (auto& batch : dmat->GetBatches<EllpackPage>(batch_param_)) {
    page_->Compact(batch_param_.gpu_id, batch.Impl(), sample_row_index_.DeviceSpan());
  }

  return {sample_rows_, page_.get(), gpair_.DeviceSpan()};
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

};  // namespace tree
};  // namespace xgboost
