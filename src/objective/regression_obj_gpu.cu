// GPU implementation of objective function.  Necessary as evaluating the objective
// function takes significant time when training on the GPU.

#include <dmlc/omp.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <memory>
#include <vector>

#include "../common/device_helpers.cuh"
#include "./regression_loss.h"

using namespace dh;

namespace xgboost {
namespace obj {


DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);

struct GPURegLossParam : public dmlc::Parameter<GPURegLossParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPURegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
        .describe("Scale the weight of positive examples by this factor");
  }
};

// GPU kernel for gradient computation
template<typename Loss>
__global__ void get_gradient_k
(bst_gpair *__restrict__ out_gpair,  uint *__restrict__ label_correct,
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
  out_gpair[i] = bst_gpair
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

  // manages device data
  struct DeviceData {
    dvec<float> preds, labels, weights;
    dvec<uint> label_correct;
    dvec<bst_gpair> out_gpair;
    bulk_allocator<memory_type::DEVICE> ba;

    // allocate everything on device
    DeviceData(size_t n) {
      ba.allocate(0, true,
                  &preds, n,
                  &labels, n,
                  &weights, n,
                  &label_correct, 1,
                  &out_gpair, n);
    }
    size_t size() const { return preds.size(); }
  };

  
  bool copied_;
  std::unique_ptr<DeviceData> data_;

  // allocate device data for n elements, do nothing if enough memory is allocated already
  void LazyResize(int n) {
    if (data_ != NULL && data_->size() >= n)
      return;
    copied_ = false;
    // free the old data and allocate the new data
    data_.reset(nullptr_t());
    std::unique_ptr<DeviceData> new_data(new DeviceData(n));
    data_.swap(new_data);
  }
  
 public:
  GPURegLossObj() : copied_(false) {}
  
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "labels are not correctly provided"
        << "preds.size=" << preds.size() << ", label.size=" << info.labels.size();
    out_gpair->resize(preds.size());

    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size());

    // always use GPU 0
    safe_cuda(cudaSetDevice(0));
    LazyResize(ndata);

    DeviceData& d = *data_;

    d.label_correct.fill(1);

    // only copy the labels and weights once, similar to how the data is copied
    if (!copied_) {
      thrust::copy(info.labels.begin(), info.labels.end(), d.labels.tbegin());
      if (info.weights.size() > 0)
        thrust::copy(info.weights.begin(), info.weights.end(), d.weights.tbegin());
      copied_ = true;
    }
    
    // copy input data to the GPU
    thrust::copy(preds.begin(), preds.end(), d.preds.tbegin());
    
    // run the kernel
    const int block = 256;
    get_gradient_k<Loss><<<div_round_up(ndata, block), block>>>
      (d.out_gpair.data(), d.label_correct.data(), d.preds.data(),
       d.labels.data(), info.weights.size() > 0 ? d.weights.data() : nullptr,
       ndata, param_.scale_pos_weight);
    safe_cuda(cudaGetLastError());
    safe_cuda(cudaDeviceSynchronize());

    // copy output data from the GPU
    thrust::copy_n(d.out_gpair.tbegin(), out_gpair->size(), out_gpair->begin());
    uint label_correct_uint = 0;
    thrust::copy_n(d.label_correct.tbegin(), 1, &label_correct_uint);
    bool label_correct = label_correct_uint != 0;
    
    if (!label_correct) {
      LOG(FATAL) << Loss::LabelErrorMsg();
    }
  }
  
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }
  
  void PredTransform(std::vector<float> *io_preds) override {
    std::vector<float> &preds = *io_preds;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
    
    // always use GPU 0
    safe_cuda(cudaSetDevice(0));
    LazyResize(ndata);

    DeviceData& d = *data_;

    thrust::copy(preds.begin(), preds.end(), d.preds.tbegin());

    const int block = 256;
    pred_transform_k<Loss><<<div_round_up(ndata, block), block>>>(d.preds.data(), ndata);
    safe_cuda(cudaGetLastError());
    safe_cuda(cudaDeviceSynchronize());
    thrust::copy_n(d.preds.tbegin(), preds.size(), preds.begin());
  }
  
  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

 protected:
  GPURegLossParam param_;
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

}
}
