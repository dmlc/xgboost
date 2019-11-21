/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <curand_kernel.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <xgboost/host_device_vector.h>

#include "../../common/device_helpers.cuh"
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

/*! \brief A functor that returns the absolute value of gradient from a gradient pair. */
struct abs_grad : public thrust::unary_function<GradientPair, float> {
  __device__
  float operator()(const GradientPair& gpair) const {
    return fabsf(gpair.GetGrad());
  }
};

struct sample_and_scale : public thrust::unary_function<GradientPair, GradientPair> {
  const size_t expected_sample_rows;
  const float sum_abs_gradient;

  sample_and_scale(size_t _expected_sample_rows, float _sum_abs_gradient)
      : expected_sample_rows(_expected_sample_rows), sum_abs_gradient(_sum_abs_gradient) {}

  __device__
  GradientPair operator()(const GradientPair& gpair) {
    return GradientPair();
  }
};

GradientBasedSample GradientBasedSampler::Sample(HostDeviceVector<GradientPair>* gpair,
                                                 DMatrix* dmat,
                                                 BatchParam batch_param,
                                                 size_t sample_rows) {
  float sum_abs_gradient = thrust::transform_reduce(
      dh::tbegin(*gpair), dh::tend(*gpair), abs_grad(), 0.0f, thrust::plus<float>());

  HostDeviceVector<GradientPair> scaled_gpair(gpair->Size());
  scaled_gpair.SetDevice(batch_param.gpu_id);
  thrust::transform(dh::tbegin(*gpair), dh::tend(*gpair), dh::tbegin(scaled_gpair),
      sample_and_scale(sample_rows, sum_abs_gradient));

  auto page = (*dmat->GetBatches<EllpackPage>(batch_param).begin()).Impl();
  return {page, gpair};
}

};  // namespace tree
};  // namespace xgboost
