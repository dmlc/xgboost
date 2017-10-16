// GPU implementation of objective function.  Necessary as evaluating the objective
// function takes significant time when training on the GPU.

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <vector>

#include "../common/device_helpers.cuh"
#include "./loss.h"

using namespace dh;

namespace xgboost {
namespace obj {


DMLC_REGISTRY_FILE_TAG(objective_gpu);

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
(bst_gpair *__restrict__ out_gpair,  bool *__restrict__ label_correct,
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
    *label_correct = false;
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
  
  float *preds_d_, *labels_d_, *weights_d_;
  bool *label_correct_d_;
  bst_gpair *out_gpair_d_;
  int max_n_;
  std::vector<cudaStream_t> streams_;
  bool copied_;

  // allocate device data for n elements
  void AllocDevice(int n) {
    copied_ = false;
    size_t sz = sizeof(float) * n;
    safe_cuda(cudaMalloc((void **)&preds_d_, sz));
    safe_cuda(cudaMalloc((void **)&labels_d_, sz));
    safe_cuda(cudaMalloc((void **)&weights_d_, sz));
    safe_cuda(cudaMalloc((void **)&out_gpair_d_, 2*sz));
    safe_cuda(cudaMalloc((void **)&label_correct_d_, sizeof(bool)));
    max_n_ = n;
  }

  // free device data
  void FreeDevice() {
    if (max_n_ == 0)
      return;
    // free the GPU data
    safe_cuda(cudaFree(preds_d_));
    safe_cuda(cudaFree(labels_d_));
    safe_cuda(cudaFree(weights_d_));
    safe_cuda(cudaFree(out_gpair_d_));
    safe_cuda(cudaFree(label_correct_d_));
    preds_d_ = nullptr;
    labels_d_ = nullptr;
    weights_d_ = nullptr;
    out_gpair_d_ = nullptr;
    label_correct_d_ = nullptr;   
    max_n_ = 0;
  }

  // allocate device data for n elements, do nothing if enough memory is allocated already
  void LazyReallocDevice(int n) {
    if (max_n_ >= n)
      return;
    FreeDevice();
    AllocDevice(n);
  }
  
 public:
  GPURegLossObj() : preds_d_(nullptr), labels_d_(nullptr), weights_d_(nullptr),
                    label_correct_d_(nullptr), out_gpair_d_(nullptr), max_n_(0), copied_(false) {}

  virtual ~GPURegLossObj() {
    FreeDevice();
  }
  
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
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size());
    const float *weights = info.weights.size() > 0 ? &info.weights[0] : nullptr;
    
    // always use GPU 0
    safe_cuda(cudaSetDevice(0));
    LazyReallocDevice(ndata);

    safe_cuda(cudaMemset(label_correct_d_, 255, sizeof(bool)));

    size_t sz = ndata *sizeof(float);
    int ndata_per_stream = div_round_up(ndata, streams_.size());

    // only copy the labels and weights once, similar to how the data is copied
    if (!copied_) {
      safe_cuda(cudaMemcpy(labels_d_, &info.labels[0], sz, cudaMemcpyDefault));
      if (weights)
        safe_cuda(cudaMemcpy(weights_d_, weights, sz, cudaMemcpyDefault));
      copied_ = true;
    }
    
    // copy input data to the GPU
    safe_cuda(cudaMemcpy(preds_d_, &preds[0], sz, cudaMemcpyDefault));
    
    // run the kernel
    const int block = 256;
    get_gradient_k<Loss><<<div_round_up(ndata, block), block>>>
      (out_gpair_d_, label_correct_d_, preds_d_, labels_d_,
       weights ? weights_d_ : nullptr, ndata, param_.scale_pos_weight);
    safe_cuda(cudaGetLastError());
    safe_cuda(cudaDeviceSynchronize());
    // copy output data from the GPU
    safe_cuda(cudaMemcpy(&(*out_gpair)[0], out_gpair_d_,  2 * sz, cudaMemcpyDefault));
    safe_cuda(cudaMemcpy(&label_correct, label_correct_d_, sizeof(bool), cudaMemcpyDefault));
    
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
    
    // currently, always use GPU 0
    safe_cuda(cudaSetDevice(0));
    LazyReallocDevice(ndata);

    size_t sz = ndata * sizeof(float);
    safe_cuda(cudaMemcpy(preds_d_, &preds[0], sz, cudaMemcpyDefault));

    const int block = 256;
    pred_transform_k<Loss><<<div_round_up(ndata, block), block>>>(preds_d_, ndata);
    safe_cuda(cudaGetLastError());
    safe_cuda(cudaDeviceSynchronize());
    safe_cuda(cudaMemcpy(&preds[0], preds_d_, sz, cudaMemcpyDefault));
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
