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
#include "../common/dhvec.h"
#include "./regression_loss.h"

using namespace dh;

namespace xgboost {
  namespace obj {


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
        dvec<float> labels, weights;
        dvec<uint> label_correct;

        // allocate everything on device
        DeviceData(bulk_allocator<memory_type::DEVICE>& ba, int device_idx, size_t n) {
          ba.allocate(device_idx, false,
                      &labels, n,
                      &weights, n,
                      &label_correct, 1);
        }
        size_t size() const { return labels.size(); }
      };


      bool copied_;
      std::unique_ptr<bulk_allocator<memory_type::DEVICE>> ba_;
      std::unique_ptr<DeviceData> data_;
      dhvec<bst_float> preds_d_;
      dhvec<bst_gpair> out_gpair_d_;

      // allocate device data for n elements, do nothing if enough memory is allocated already
      void LazyResize(int n) {
        if (data_.get() != nullptr && data_->size() >= n)
          return;
        copied_ = false;
        // free the old data and allocate the new data
        ba_.reset(new bulk_allocator<memory_type::DEVICE>());
        data_.reset(new DeviceData(*ba_, 0, n));
        preds_d_.resize(n, param_.gpu_id);
        out_gpair_d_.resize(n, param_.gpu_id);
      }

    public:
      GPURegLossObj() : copied_(false), preds_d_(0, -1), out_gpair_d_(0, -1) {}

      void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
        param_.InitAllowUnknown(args);
        CHECK(param_.n_gpus != 0) << "Must have at least one device";
      }
      void GetGradient(const std::vector<float> &preds,
                       const MetaInfo &info,
                       int iter,
                       std::vector<bst_gpair> *out_gpair) override {
        CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
        CHECK_EQ(preds.size(), info.labels.size())
          << "labels are not correctly provided"
          << "preds.size=" << preds.size() << ", label.size=" << info.labels.size();

        size_t ndata = preds.size();
        out_gpair->resize(ndata);
        LazyResize(ndata);
        thrust::copy(preds.begin(), preds.end(), preds_d_.tbegin(param_.gpu_id));
        GetGradientDevice(preds_d_.ptr_d(param_.gpu_id), info, iter,
                          out_gpair_d_.ptr_d(param_.gpu_id), ndata);
        thrust::copy_n(out_gpair_d_.tbegin(param_.gpu_id), ndata, out_gpair->begin());
      }

      void GetGradient(dhvec<float>& preds,
                       const MetaInfo &info,
                       int iter,
                       dhvec<bst_gpair>* out_gpair) override {
        CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
        CHECK_EQ(preds.size(), info.labels.size())
          << "labels are not correctly provided"
          << "preds.size=" << preds.size() << ", label.size=" << info.labels.size();
        size_t ndata = preds.size();
        out_gpair->resize(ndata, param_.gpu_id);
        LazyResize(ndata);
        GetGradientDevice(preds.ptr_d(param_.gpu_id), info, iter,
                          out_gpair->ptr_d(param_.gpu_id), ndata);
      }

    private:

      void GetGradientDevice(float* preds,
                             const MetaInfo &info,
                             int iter,
                             bst_gpair* out_gpair, size_t n) {

        safe_cuda(cudaSetDevice(param_.gpu_id));
        DeviceData& d = *data_;
        d.label_correct.fill(1);
        // only copy the labels and weights once, similar to how the data is copied
        if (!copied_) {
          thrust::copy(info.labels.begin(), info.labels.begin() + n,
                       d.labels.tbegin());
          if (info.weights.size() > 0) {
            thrust::copy(info.weights.begin(), info.weights.begin() + n,
                         d.weights.tbegin());
          }
          copied_ = true;
        }

        // run the kernel
        const int block = 256;
        get_gradient_k<Loss><<<div_round_up(n, block), block>>>
          (out_gpair, d.label_correct.data(), preds,
           d.labels.data(), info.weights.size() > 0 ? d.weights.data() : nullptr,
           n, param_.scale_pos_weight);
        safe_cuda(cudaGetLastError());

        // copy output data from the GPU
        uint label_correct_h;
        thrust::copy_n(d.label_correct.tbegin(), 1, &label_correct_h);

        bool label_correct = label_correct_h != 0;
        if (!label_correct) {
          LOG(FATAL) << Loss::LabelErrorMsg();
        }
      }

    public:

      const char* DefaultEvalMetric() const override {
        return Loss::DefaultEvalMetric();
      }

      void PredTransform(std::vector<float> *io_preds) override {
        LazyResize(io_preds->size());
        thrust::copy(io_preds->begin(), io_preds->end(), preds_d_.tbegin(param_.gpu_id));
        PredTransformDevice(preds_d_.ptr_d(param_.gpu_id), io_preds->size());
        thrust::copy_n(preds_d_.tbegin(param_.gpu_id), io_preds->size(), io_preds->begin());
      }

      void PredTransform(dhvec<float> *io_preds) override {
        PredTransformDevice(io_preds->ptr_d(param_.gpu_id), io_preds->size());
      }

      void PredTransformDevice(float* preds, size_t n) {
        safe_cuda(cudaSetDevice(param_.gpu_id));
        const int block = 256;
        pred_transform_k<Loss><<<div_round_up(n, block), block>>>(preds, n);
        safe_cuda(cudaGetLastError());
        safe_cuda(cudaDeviceSynchronize());
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
