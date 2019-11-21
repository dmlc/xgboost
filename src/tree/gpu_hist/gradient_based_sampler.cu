/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include "gradient_based_sampler.cuh"

namespace xgboost {
namespace tree {

GradientBasedSample GradientBasedSampler::Sample(HostDeviceVector<GradientPair>* gpair,
                                                 DMatrix* dmat,
                                                 BatchParam batch_param) {
  auto page = (*dmat->GetBatches<EllpackPage>(batch_param).begin()).Impl();
  return {page, gpair};
}

void GradientBasedSampler::Sample(HostDeviceVector<GradientPair>* gpair,
                                  DMatrix* dmat,
                                  size_t sample_rows) {}

};  // namespace tree
};  // namespace xgboost
