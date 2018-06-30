/*!
 * Copyright 2017 XGBoost contributors
 */
// GPU implementation of objective function.
// Necessary to avoid extra copying of data to CPU.
#include <dmlc/omp.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <memory>
#include <vector>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"
#include "./regression_loss.h"


namespace xgboost {
namespace obj {

using dh::DVec;

DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);

struct GPURegLossParam : public dmlc::Parameter<GPURegLossParam> {
  float scale_pos_weight;
  int n_gpus;
  int gpu_id;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPURegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(-1)
      .describe("Number of GPUs to use for multi-gpu algorithms (NOT IMPLEMENTED)");
    DMLC_DECLARE_FIELD(gpu_id)
      .set_lower_bound(0)
      .set_default(0)
      .describe("gpu to use for objective function evaluation");
  }
};

// GPU kernel for gradient computation
template<typename Loss>
__global__ void get_gradient_k
(GradientPair *__restrict__ out_gpair,  unsigned int *__restrict__ label_correct,
 const float * __restrict__ preds, const float * __restrict__ labels,
 const float * __restrict__ weights, int n, float scale_pos_weight) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;
  float p = Loss::PredTransform(preds[i]);
  float w = weights == nullptr ? 1.0f : weights[i];
  float label = labels[i];
  if (label == 1.0f)
    w *= scale_pos_weight;
  if (!Loss::CheckLabel(label))
    atomicAnd(label_correct, 0);
  out_gpair[i] = GradientPair
    (Loss::FirstOrderGradient(p, label) * w, Loss::SecondOrderGradient(p, label) * w);
}

// GPU kernel for predicate transformation
template<typename Loss>
__global__ void pred_transform_k(float * __restrict__ preds, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;
  preds[i] = Loss::PredTransform(preds[i]);
}

// regression loss function for evaluation on GPU (eventually)
template<typename Loss>
class GPURegLossObj : public ObjFunction {
 protected:
  bool copied_;
  HostDeviceVector<bst_float> labels_, weights_;
  HostDeviceVector<unsigned int> label_correct_;

  // allocate device data for n elements, do nothing if memory is allocated already
  void LazyResize(size_t n, size_t n_weights) {
    if (labels_.Size() == n && weights_.Size() == n_weights)
      return;
    copied_ = false;

    labels_.Reshard(devices_);
    weights_.Reshard(devices_);
    label_correct_.Reshard(devices_);

    if (labels_.Size() != n) {
      labels_.Resize(n);
      label_correct_.Resize(devices_.Size());
    }
    if (weights_.Size() != n_weights)
      weights_.Resize(n_weights);
  }

 public:
  GPURegLossObj() : copied_(false) {}

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    devices_ = GPUSet::Range(param_.gpu_id, dh::NDevicesAll(param_.n_gpus));
  }

  void GetGradient(HostDeviceVector<float>* preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size())
      << "labels are not correctly provided"
      << "preds.size=" << preds->Size() << ", label.size=" << info.labels_.size();
    size_t ndata = preds->Size();
    preds->Reshard(devices_);
    out_gpair->Reshard(devices_);
    out_gpair->Resize(ndata);
    LazyResize(ndata, info.weights_.size());
    GetGradientDevice(preds, info, iter, out_gpair);
  }

 private:
  void GetGradientDevice(HostDeviceVector<float>* preds,
                         const MetaInfo &info,
                         int iter,
                         HostDeviceVector<GradientPair>* out_gpair) {
    label_correct_.Fill(1);
    // only copy the labels and weights once, similar to how the data is copied
    if (!copied_) {
      labels_.Copy(info.labels_);
      if (info.weights_.size() > 0)
        weights_.Copy(info.weights_);
      copied_ = true;
    }

    // run the kernel
#pragma omp parallel for schedule(static, 1) if (devices_.Size() > 1)
    for (int i = 0; i < devices_.Size(); ++i) {
      int d = devices_[i];
      dh::safe_cuda(cudaSetDevice(d));
      const int block = 256;
      size_t n = preds->DeviceSize(d);
      if (n > 0) {
        get_gradient_k<Loss><<<dh::DivRoundUp(n, block), block>>>
          (out_gpair->DevicePointer(d), label_correct_.DevicePointer(d),
           preds->DevicePointer(d), labels_.DevicePointer(d),
           info.weights_.size() > 0 ? weights_.DevicePointer(d) : nullptr,
           n, param_.scale_pos_weight);
        dh::safe_cuda(cudaGetLastError());
      }
      dh::safe_cuda(cudaDeviceSynchronize());
    }

    // copy "label correct" flags back to host
    std::vector<unsigned int>& label_correct_h = label_correct_.HostVector();
    for (int i = 0; i < devices_.Size(); ++i) {
      if (label_correct_h[i] == 0)
        LOG(FATAL) << Loss::LabelErrorMsg();
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) override {
    io_preds->Reshard(devices_);
    size_t ndata = io_preds->Size();
    PredTransformDevice(io_preds);
  }

  void PredTransformDevice(HostDeviceVector<float>* preds) {
#pragma omp parallel for schedule(static, 1) if (devices_.Size() > 1)
    for (int i = 0; i < devices_.Size(); ++i) {
      int d = devices_[i];
      dh::safe_cuda(cudaSetDevice(d));
      const int block = 256;
      size_t n = preds->DeviceSize(d);
      if (n > 0) {
        pred_transform_k<Loss><<<dh::DivRoundUp(n, block), block>>>(preds->DevicePointer(d), n);
        dh::safe_cuda(cudaGetLastError());
      }
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

 protected:
  GPURegLossParam param_;
  GPUSet devices_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GPURegLossParam);

XGBOOST_REGISTER_OBJECTIVE(GPULinearRegression, "gpu:reg:linear")
.describe("Linear regression (computed on GPU).")
.set_body([]() { return new GPURegLossObj<LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticRegression, "gpu:reg:logistic")
.describe("Logistic regression for probability regression task (computed on GPU).")
.set_body([]() { return new GPURegLossObj<LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticClassification, "gpu:binary:logistic")
.describe("Logistic regression for binary classification task (computed on GPU).")
.set_body([]() { return new GPURegLossObj<LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticRaw, "gpu:binary:logitraw")
.describe("Logistic regression for classification, output score "
          "before logistic transformation (computed on GPU)")
.set_body([]() { return new GPURegLossObj<LogisticRaw>(); });

}  // namespace obj
}  // namespace xgboost
