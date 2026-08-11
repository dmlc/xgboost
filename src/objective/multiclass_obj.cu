/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_obj.cu
 * \brief CUDA implementations of multiclass objective kernels.
 */
#include <dmlc/registry.h>

#include <cassert>  // for assert
#include <cmath>    // for expf, fmaxf
#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "../collective/aggregator.h"    // for GlobalSum
#include "../common/device_helpers.cuh"  // for LaunchN
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.cuh"       // for SmallHistogram, vector operations
#include "../common/math.h"              // for FindMaxIndex, Softmax
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "../common/stats.h"             // for Mean
#include "../common/transform.h"         // for Transform
#include "elementwise_objective.cuh"     // for CUDA elementwise kernels
#include "multiclass_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(multiclass_kernel_cuda);
namespace {
void MulticlassGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                            MetaInfo const& info, std::int64_t n_classes,
                            linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds.SetDevice(device);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, n_classes);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_classes);
  auto gpair = out_gpair->View(device);

  dh::LaunchN(info.num_row_, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t row) mutable {
    auto point = predt.Slice(row, linalg::All());
    assert(point.Size() == static_cast<std::size_t>(n_classes));
    float wmax = std::numeric_limits<float>::min();
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wmax = fmaxf(point(k), wmax);
    }
    double wsum{0.0};
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wsum += expf(point(k) - wmax);
    }
    auto label = labels(row, 0);
    auto weight = weights[row];
    for (std::int64_t k{0}; k < n_classes; ++k) {
      auto probability = expf(point(k) - wmax) / static_cast<float>(wsum);
      auto hess = fmaxf(2.0f * probability * (1.0f - probability) * weight, 1e-16f);
      auto grad = label == k ? probability - 1.0f : probability;
      gpair(row, k) = {grad * weight, hess};
    }
  });
}

void MulticlassTransformCuda(Context const* ctx, HostDeviceVector<float>* predictions,
                             std::int32_t n_classes, bool probability) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  auto n_samples = predictions->Size() / n_classes;
  if (probability) {
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(std::size_t row, common::Span<float> values) {
          auto point = values.subspan(row * n_classes, n_classes);
          common::Softmax(point.begin(), point.end());
        },
        common::Range{0, static_cast<std::int64_t>(n_samples)}, ctx->Threads(), device)
        .Eval(predictions);
  } else {
    HostDeviceVector<float> output(n_samples, 0.0f, device);
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(std::size_t row, common::Span<float const> values,
                           common::Span<float> output) {
          auto point = values.subspan(row * n_classes, n_classes);
          output[row] = common::FindMaxIndex(point.cbegin(), point.cend()) - point.cbegin();
        },
        common::Range{0, static_cast<std::int64_t>(n_samples)}, ctx->Threads(), device)
        .Eval(predictions, &output);
    predictions->Resize(output.Size());
    predictions->Copy(output);
  }
}

void MulticlassInitEstimationCuda(Context const* ctx, MetaInfo const& info, std::int64_t n_classes,
                                  linalg::Vector<float>* base_score) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  *base_score = linalg::Zeros<float>(ctx, n_classes);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  auto intercept = base_score->View(device);
  linalg::SmallHistogram(ctx, labels, weights, intercept);
  auto sum_weight = common::SumOptionalWeights(ctx, weights, info.labels.Size());
  collective::SafeColl(collective::GlobalSum(ctx, intercept, &sum_weight));
  CHECK_GE(sum_weight, kRtEps);
  linalg::VecScaDiv(ctx, intercept, sum_weight);
  linalg::LogE(ctx, intercept, kRtEps);
  linalg::Vector<float> mean;
  common::Mean(ctx, intercept, &mean);
  auto mean_d = mean.View(device);
  dh::LaunchN(intercept.Size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) mutable { intercept(i) -= mean_d(0); });
}

auto const kRegisterMulticlassGradientCuda =
    common::KernelRegistration<MulticlassGradientKernel>{DeviceOrd::kCUDA, &MulticlassGradientCuda};
auto const kRegisterMulticlassInitEstimationCuda =
    common::KernelRegistration<MulticlassInitEstimationKernel>{DeviceOrd::kCUDA,
                                                               &MulticlassInitEstimationCuda};
auto const kRegisterMulticlassTransformCuda = common::KernelRegistration<MulticlassTransformKernel>{
    DeviceOrd::kCUDA, &MulticlassTransformCuda};
auto const kRegisterMulticlassValidationCuda =
    elementwise::RegisterValidationCuda<MulticlassLabelCheck>();
auto const kRegisterMulticlassCenterCuda = elementwise::RegisterTransformCuda<MulticlassCenter>();
}  // namespace
}  // namespace xgboost::obj
