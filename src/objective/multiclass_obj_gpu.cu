/*!
 * Copyright 2018 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <xgboost/base.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"

#include "./regression_loss.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj_gpu);

struct GPUSoftmaxMultiClassParam
    : public dmlc::Parameter<GPUSoftmaxMultiClassParam> {
  int num_class;
  int n_gpus;
  int gpu_id;

  DMLC_DECLARE_PARAMETER(GPUSoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1).describe(
        "Number of output class in the multi-class classification.");
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(-1).describe(
        "Number of GPUs to use for multi-gpu algorithms (NOT IMPLEMENTED)");
    DMLC_DECLARE_FIELD(gpu_id).set_lower_bound(0).set_default(0).describe(
        "gpu to use for objective function evaluation");
  }
};

// Prediction transformation without prob.
__global__ void multiclass_max_transform_k(float *__restrict__ preds,
                                           float *__restrict__ max_data,
                                           size_t ndata, size_t nclass, bool prob) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= ndata)
    return;
  float *rec = &preds[i * nclass];
  max_data[i] =
      static_cast<float>(common::FindMaxIndex(rec, rec + nclass) - rec);
}

// Prediction transformation with prob.
__global__ void multiclass_prob_transform_k(float *__restrict__ preds,
                                            size_t ndata, size_t nclass) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= ndata)
    return;
  float *start = preds + i * nclass;
  common::Softmax(start, start + nclass);
}

// Kernel for gradient and approximated hessian
__global__ void multiclass_gradient_k(GradientPair *__restrict__ gpair,
                                      float *__restrict__ labels,
                                      float *__restrict__ preds,
                                      float *__restrict__ weights,
                                      size_t ndata, size_t nclass,
                                      int *__restrict__ label_error) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= ndata)
    return;

  float *start = &preds[i * nclass];
  common::Softmax(start, start + nclass);

  auto label = labels[i];
  if (label < 0 || label >= nclass) {
    // If there is an incorrect label, the host code will know. No need to
    // sync the threads.
    *label_error = label;
    label = 0;
  }
  bst_float wt = weights == nullptr ? 1.0f : weights[i];
  for (int k = 0; k < nclass; ++k) {
    bst_float p = *(start + k);
    const float eps = 1e-16f;
    const bst_float h = fmax(2.0f * p * (1.0f - p) * wt, eps);
    if (label == k) {
      gpair[i * nclass + k] = GradientPair((p - 1.0f) * wt, h);
    } else {
      gpair[i * nclass + k] = GradientPair(p * wt, h);
    }
  }
}

class GPUSoftmaxMultiClassObj : public ObjFunction {
 protected:
  void LazyResize(size_t n, size_t n_weights) {
    if (labels_.Size() == n && weights_.Size() == n_weights) {
      return;
    }
    copied_ = false;

    labels_.Reshard(devices_);
    weights_.Reshard(devices_);

    if (labels_.Size() != n) {
      labels_.Resize(n);
    }
    if (weights_.Size() != n_weights) {
      weights_.Resize(n_weights);
    }
  }

 public:
  explicit GPUSoftmaxMultiClassObj(bool output_prob)
  : output_prob_(output_prob), copied_{false}, h_label_error_{0} {
      dh::safe_cuda(cudaMalloc(&d_label_error_, sizeof(int)));
      dh::safe_cuda(cudaMemset(d_label_error_, 0, sizeof(int)));
    }

  ~GPUSoftmaxMultiClassObj() override {
    dh::safe_cuda(cudaFree(d_label_error_));
  }

  void Configure(
      const std::vector<std::pair<std::string, std::string>> &args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    devices_ = GPUSet::Range(param_.gpu_id, dh::NDevicesAll(param_.n_gpus));

    max_data_.Reshard(devices_);
  };

  void GetGradient(HostDeviceVector<bst_float> *preds, const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK(preds->Size() ==
          (static_cast<size_t>(param_.num_class) * info.labels_.size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match";

    const int nclass = param_.num_class;
    size_t n = preds->Size();
    auto ndata = static_cast<size_t>(n / nclass);  // number of data points

    preds->Reshard(devices_);
    out_gpair->Reshard(devices_);
    out_gpair->Resize(preds->Size());

    LazyResize(info.labels_.size(), info.weights_.size());

    if (!copied_) {
      labels_.Copy(info.labels_);
      if (info.weights_.size() > 0) {
        weights_.Copy(info.weights_);
      } else {
        copied_ = true;
      }
    }

#pragma omp parallel for schedule(static, 1) if (devices_.Size() > 1)
    for (int i = 0; i < devices_.Size(); ++i) {
      int d = devices_[i];
      dh::safe_cuda(cudaSetDevice(d));

      const int block = 256;

      multiclass_gradient_k<<<dh::DivRoundUp(ndata, block), block>>>
        (out_gpair->DevicePointer(d), labels_.DevicePointer(d),
         preds->DevicePointer(d),
         info.weights_.size() > 0 ? weights_.DevicePointer(d) : nullptr,
         ndata, nclass, d_label_error_);
      dh::safe_cuda(cudaGetLastError());
    }

    dh::safe_cuda(cudaDeviceSynchronize());
    dh::safe_cuda(cudaMemcpy(&h_label_error_, d_label_error_, sizeof(int),
                             cudaMemcpyDeviceToHost));

    CHECK(h_label_error_ >= 0 && h_label_error_ < nclass)
        << "SoftmaxMultiClassObj: label must be in [0, num_class),"
        << " num_class=" << nclass << " but found " << h_label_error_
        << " in label.";

    dh::safe_cuda(cudaMemset(d_label_error_, 0, sizeof(int)));
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    this->Transform(io_preds, output_prob_);
  }

  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    this->Transform(io_preds, true);
  }

  const char *DefaultEvalMetric() const override { return "merror"; }

 private:
  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) {
    const int nclass = param_.num_class;

    const auto ndata = static_cast<int>(io_preds->Size() / nclass);
    io_preds->Reshard(devices_);
    if (!prob)
      max_data_.Resize(ndata);

#pragma omp parallel for schedule(static, 1) if (devices_.Size() > 1)
    for (int i = 0; i < devices_.Size(); ++i) {
      int d = devices_[i];

      dh::safe_cuda(cudaSetDevice(d));
      const int block = 256;

      size_t n = io_preds->DeviceSize(d);
      size_t ndata = n / nclass;

      if (n > 0) {
        if (prob)
          multiclass_prob_transform_k<<<dh::DivRoundUp(ndata, block), block>>>(
              io_preds->DevicePointer(d), ndata, nclass);
        else
          multiclass_max_transform_k<<<dh::DivRoundUp(ndata, block), block>>>(
              io_preds->DevicePointer(d), max_data_.DevicePointer(d), ndata, nclass,
              prob);
        dh::safe_cuda(cudaGetLastError());
      }
      dh::safe_cuda(cudaDeviceSynchronize());
    }
    if (!prob) {
      io_preds->HostVector() = max_data_.HostVector();
    }
  }

  int h_label_error_;
  int *d_label_error_;
  bool copied_;

  bool output_prob_;
  GPUSoftmaxMultiClassParam param_;

  GPUSet devices_;

  HostDeviceVector<float> max_data_;
  HostDeviceVector<float> labels_, weights_;
};

DMLC_REGISTER_PARAMETER(GPUSoftmaxMultiClassParam);

XGBOOST_REGISTER_OBJECTIVE(GPUSoftmaxMultiClass, "gpu:multi:softmax")
    .describe("Softmax for multi-class classification, output class index "
              "(computed on GPU).")
    .set_body([]() { return new GPUSoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "gpu:multi:softprob")
    .describe("Softmax for multi-class classification, output probability"
              "distribution (computed on GPU).")
    .set_body([]() { return new GPUSoftmaxMultiClassObj(true); });

}  // namespace obj
}  // namespace xgboost
