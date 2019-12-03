/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <xgboost/host_device_vector.h>

#include "../../common/compressed_iterator.h"
#include "../../common/device_helpers.cuh"
#include "../../common/random.h"
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

size_t GradientBasedSampler::MaxSampleRows(int device, const EllpackInfo& info) {
  size_t available_memory = dh::AvailableMemory(device);
  size_t usable_memory = available_memory * 0.95;
  size_t max_rows = common::CompressedBufferWriter::CalculateMaxRows(
      usable_memory, info.NumSymbols(), info.row_stride, 2 * sizeof(float));
  return max_rows;
}

/*! \brief A functor that returns the absolute value of gradient from a gradient pair. */
struct abs_grad : public thrust::unary_function<GradientPair, float> {
  XGBOOST_DEVICE float operator()(const GradientPair& gpair) const {
    return fabsf(gpair.GetGrad());
  }
};

/*! \brief A functor that samples and scales a gradient pair.
 *
 * Sampling probability is proportional to the absolute value of the gradient. If selected, the
 * gradient pair is re-scaled proportional to (1 / probability).
 */
struct sample_and_scale : public thrust::binary_function<GradientPair, size_t, GradientPair> {
  const size_t sample_rows;
  const float sum_abs_gradient;
  const uint32_t seed;

  XGBOOST_DEVICE sample_and_scale(size_t _sample_rows, float _sum_abs_gradient, size_t _seed)
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
      return gpair / p;
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

GradientBasedSample GradientBasedSampler::Sample(HostDeviceVector<GradientPair>* gpair,
                                                 DMatrix* dmat,
                                                 BatchParam batch_param,
                                                 size_t sample_rows) {
  float sum_abs_gradient = thrust::transform_reduce(
      dh::tbegin(*gpair), dh::tend(*gpair), abs_grad(), 0.0f, thrust::plus<float>());

  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair),
                    thrust::counting_iterator<size_t>(0),
                    dh::tbegin(*gpair),
                    sample_and_scale(sample_rows, sum_abs_gradient, common::GlobalRandom()()));

  size_t out_size = thrust::count_if(dh::tbegin(*gpair), dh::tend(*gpair), is_non_zero());

  HostDeviceVector<GradientPair> out_gpair(out_size, GradientPair(), batch_param.gpu_id);
  thrust::copy_if(dh::tbegin(*gpair), dh::tend(*gpair), dh::tbegin(out_gpair), is_non_zero());

  auto page = (*dmat->GetBatches<EllpackPage>(batch_param).begin()).Impl();
  return {page, out_gpair};
}

};  // namespace tree
};  // namespace xgboost
