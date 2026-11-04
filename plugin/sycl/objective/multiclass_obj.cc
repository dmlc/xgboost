/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_obj.cc
 * \brief SYCL implementation of the multiclass objective kernels.
 */
#include "../../../src/objective/multiclass_obj.h"

#include <dmlc/registry.h>

#include <cmath>    // for expf, fmaxf, fabsf
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, int64_t
#include <limits>   // for numeric_limits

#include "../../../src/collective/aggregator.h"        // for GlobalSum
#include "../../../src/collective/communicator-inl.h"  // for IsDistributed
#include "../../../src/common/kernel.h"                // for KernelRegistration
#include "../../../src/common/linalg_op.h"             // for SmallHistogram, vector operations
#include "../../../src/common/math.h"                  // for FindMaxIndex, Softmax
#include "../../../src/common/optional_weight.h"       // for MakeOptionalWeights
#include "../../../src/common/stats.h"                 // for Mean
#include "../common/linalg_op.h"                       // for ElementWiseKernel
#include "../device_manager.h"                         // for DeviceManager
#include "elementwise_objective.h"

namespace xgboost::sycl::obj {
DMLC_REGISTRY_FILE_TAG(multiclass_kernel_sycl);

namespace {
void MulticlassGradient(Context const* ctx, HostDeviceVector<float> const& preds,
                        MetaInfo const& info, std::int64_t n_classes,
                        xgboost::linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  preds.SetDevice(device);
  auto predt = xgboost::linalg::MakeTensorView(ctx, &preds, info.num_row_, n_classes);
  auto labels = info.labels.View(device);
  auto weights = xgboost::common::MakeOptionalWeights(device, info.weights_);
  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_classes);
  auto gpair = out_gpair->View(device);

  auto y1d = labels.Slice(xgboost::linalg::All(), 0);
  sycl::linalg::ElementWiseKernel(y1d, [=](std::size_t row) mutable {
    auto point = predt.Slice(row, xgboost::linalg::All());
    float wmax = std::numeric_limits<float>::lowest();
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wmax = fmaxf(point(k), wmax);
    }
    double wsum{0.0};
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wsum += expf(point(k) - wmax);
    }
    auto label = y1d(row);
    auto weight = weights[row];
    for (std::int64_t k{0}; k < n_classes; ++k) {
      auto probability = expf(point(k) - wmax) / static_cast<float>(wsum);
      auto grad = label == k ? probability - 1.0f : probability;
      // Absolute-residual pseudo-Hessian |p - y|; see the CPU kernel for the rationale.
      auto hess = fmaxf(fabsf(grad) * weight, 1e-16f);
      gpair(row, k) = {grad * weight, hess};
    }
  });
}

void MulticlassTransform(Context const* ctx, HostDeviceVector<float>* predictions,
                         std::int32_t n_classes, bool probability) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  auto n_samples = predictions->Size() / n_classes;
  predictions->SetDevice(device);
  auto values = predictions->DeviceSpan();

  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(device);
  if (probability) {
    qu->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<>(::sycl::range<1>(n_samples), [=](::sycl::id<1> pid) {
          auto point = values.begin() + pid[0] * n_classes;
          xgboost::common::Softmax(point, point + n_classes);
        });
      }).wait_and_throw();
  } else {
    HostDeviceVector<float> output(n_samples, 0.0f, device);
    auto out = output.DeviceSpan();
    qu->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<>(::sycl::range<1>(n_samples), [=](::sycl::id<1> pid) {
          auto point = values.cbegin() + pid[0] * n_classes;
          out[pid[0]] = xgboost::common::FindMaxIndex(point, point + n_classes) - point;
        });
      }).wait_and_throw();
    predictions->Resize(output.Size());
    predictions->Copy(output);
  }
}

void MulticlassInitEstimation(Context const* ctx, MetaInfo const& info, std::int64_t n_classes,
                              xgboost::linalg::Vector<float>* base_score) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  *base_score = xgboost::linalg::Zeros<float>(ctx, n_classes);
  auto labels = info.labels.View(device);
  auto weights = xgboost::common::MakeOptionalWeights(device, info.weights_);
  auto intercept = base_score->View(device);
  xgboost::linalg::SmallHistogram(ctx, labels, weights, intercept);
  auto sum_weight = xgboost::common::SumOptionalWeights(ctx, weights, info.labels.Size());
  if (xgboost::collective::IsDistributed()) {
    // The CPU allreduce can't access device memory; re-fetch the view to sync the result back.
    auto cpu_ctx = ctx->MakeCPU();
    xgboost::collective::SafeColl(
        xgboost::collective::GlobalSum(&cpu_ctx, base_score->HostView(), &sum_weight));
    intercept = base_score->View(device);
  }
  CHECK_GE(sum_weight, kRtEps);
  xgboost::linalg::VecScaDiv(ctx, intercept, sum_weight);
  xgboost::linalg::LogE(ctx, intercept, kRtEps);
  xgboost::linalg::Vector<float> mean;
  xgboost::common::Mean(ctx, intercept, &mean);
  auto mean_d = mean.View(device);
  sycl::linalg::ElementWiseKernel(intercept,
                                  [=](std::size_t i) mutable { intercept(i) -= mean_d(0); });
}

auto const kRegisterMulticlassGradient =
    xgboost::common::KernelRegistration<xgboost::obj::MulticlassGradientKernel>{
        {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
        &xgboost::sycl::obj::MulticlassGradient};
auto const kRegisterMulticlassInitEstimation =
    xgboost::common::KernelRegistration<xgboost::obj::MulticlassInitEstimationKernel>{
        {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
        &xgboost::sycl::obj::MulticlassInitEstimation};
auto const kRegisterMulticlassTransform =
    xgboost::common::KernelRegistration<xgboost::obj::MulticlassTransformKernel>{
        {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
        &xgboost::sycl::obj::MulticlassTransform};
auto const kRegisterMulticlassValidation =
    xgboost::sycl::obj::elementwise::RegisterValidation<xgboost::obj::MulticlassLabelCheck>();
auto const kRegisterMulticlassCenter =
    xgboost::sycl::obj::elementwise::RegisterTransform<xgboost::obj::MulticlassCenter>();
}  // namespace
}  // namespace xgboost::sycl::obj
