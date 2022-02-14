/*!
 * Copyright 2015-2021 by Contributors
 * \file multiclass_obj_oneapi.cc
 * \brief Definition of multi-class classification objectives.
 */
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#include "xgboost/parameter.h"
#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/json.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj_oneapi);

/*!
 * \brief Do inplace softmax transformaton on start to end
 *
 * \tparam Iterator Input iterator type
 *
 * \param start Start iterator of input
 * \param end end iterator of input
 */
template <typename Iterator>
inline void SoftmaxOneAPI(Iterator start, Iterator end) {
  bst_float wmax = *start;
  for (Iterator i = start+1; i != end; ++i) {
    wmax = sycl::max(*i, wmax);
  }
  float wsum = 0.0f;
  for (Iterator i = start; i != end; ++i) {
    *i = sycl::exp(*i - wmax);
    wsum += *i;
  }
  for (Iterator i = start; i != end; ++i) {
    *i /= static_cast<float>(wsum);
  }
}

/*!
 * \brief Find the maximum iterator within the iterators
 * \param begin The begining iterator.
 * \param end The end iterator.
 * \return the iterator point to the maximum value.
 * \tparam Iterator The type of the iterator.
 */
template<typename Iterator>
inline Iterator FindMaxIndexOneAPI(Iterator begin, Iterator end) {
  Iterator maxit = begin;
  for (Iterator it = begin; it != end; ++it) {
    if (*it > *maxit) maxit = it;
  }
  return maxit;
}

struct SoftmaxMultiClassParamOneAPI : public XGBoostParameter<SoftmaxMultiClassParamOneAPI> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParamOneAPI) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassObjOneAPI : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObjOneAPI(bool output_prob)
  : output_prob_(output_prob) {}

  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);

    sycl::default_selector selector;
    qu_ = sycl::queue(selector);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels.Size() == 0) {
      return;
    }
    CHECK(preds.Size() == (static_cast<size_t>(param_.num_class) * info.labels.Size()))
        << "SoftmaxMultiClassObjOneAPI: label size and pred size does not match.\n"
        << "label.Size() * num_class: "
        << info.labels.Size() * static_cast<size_t>(param_.num_class) << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << preds.Size();

    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(preds.Size() / nclass);

    out_gpair->Resize(preds.Size());

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    sycl::buffer<bst_float, 1> preds_buf(preds.HostPointer(), preds.Size());
    sycl::buffer<bst_float, 1> labels_buf(info.labels.Data()->HostPointer(), info.labels.Size());
    sycl::buffer<GradientPair, 1> out_gpair_buf(out_gpair->HostPointer(), out_gpair->Size());
    sycl::buffer<bst_float, 1> weights_buf(is_null_weight ? NULL : info.weights_.HostPointer(),
                                               is_null_weight ? 1 : info.weights_.Size());

    sycl::buffer<int, 1> additional_input_buf(1);
	{
		auto additional_input_acc = additional_input_buf.template get_access<sycl::access::mode::write>();
		additional_input_acc[0] = 1; // Fill the label_correct flag
	}

    qu_.submit([&](sycl::handler& cgh) {
      auto preds_acc            = preds_buf.template get_access<sycl::access::mode::read>(cgh);
      auto labels_acc           = labels_buf.template get_access<sycl::access::mode::read>(cgh);
      auto weights_acc          = weights_buf.template get_access<sycl::access::mode::read>(cgh);
      auto out_gpair_acc        = out_gpair_buf.template get_access<sycl::access::mode::write>(cgh);
      auto additional_input_acc = additional_input_buf.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<>(sycl::range<1>(ndata), [=](sycl::id<1> pid) {
        int idx = pid[0];

        bst_float const * point = &preds_acc[idx * nclass];

        // Part of Softmax function
        bst_float wmax = std::numeric_limits<bst_float>::min();
        for (int k = 0; k < nclass; k++) { wmax = sycl::max(point[k], wmax); }
        float wsum = 0.0f;
        for (int k = 0; k < nclass; k++) { wsum += sycl::exp(point[k] - wmax); }
        auto label = labels_acc[idx];
        if (label < 0 || label >= nclass) {
          additional_input_acc[0] = 0;
          label = 0;
        }
        bst_float wt = is_null_weight ? 1.0f : weights_acc[idx];
        for (int k = 0; k < nclass; ++k) {
          bst_float p = expf(point[k] - wmax) / static_cast<float>(wsum);
          const float eps = 1e-16f;
          const bst_float h = sycl::max(2.0f * p * (1.0f - p) * wt, eps);
          p = label == k ? p - 1.0f : p;
          out_gpair_acc[idx * nclass + k] = GradientPair(p * wt, h);
        }
      });
    }).wait();

    int flag = 1;
	{
		auto additional_input_acc = additional_input_buf.template get_access<sycl::access::mode::read>();
		flag = additional_input_acc[0];
	}

    if (flag == 0) {
      LOG(FATAL) << "SoftmaxMultiClassObjOneAPI: label must be in [0, num_class).";
    }
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "mlogloss";
  }

  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) const {
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);
    max_preds_.Resize(ndata);

    {
      sycl::buffer<bst_float, 1> io_preds_buf(io_preds->HostPointer(), io_preds->Size());

      if (prob) {
        qu_.submit([&](sycl::handler& cgh) {
          auto io_preds_acc = io_preds_buf.template get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(sycl::range<1>(ndata), [=](sycl::id<1> pid) {
            int idx = pid[0];
            bst_float * point = &io_preds_acc[idx * nclass];
            SoftmaxOneAPI(point, point + nclass);
          });
        }).wait();
      } else {
        sycl::buffer<bst_float, 1> max_preds_buf(max_preds_.HostPointer(), max_preds_.Size());

        qu_.submit([&](sycl::handler& cgh) {
          auto io_preds_acc = io_preds_buf.template get_access<sycl::access::mode::read>(cgh);
          auto max_preds_acc = max_preds_buf.template get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(sycl::range<1>(ndata), [=](sycl::id<1> pid) {
            int idx = pid[0];
            bst_float const * point = &io_preds_acc[idx * nclass];
            max_preds_acc[idx] = FindMaxIndexOneAPI(point, point + nclass) - point;
          });
        }).wait();
      }
    }

    if (!prob) {
      io_preds->Resize(max_preds_.Size());
      io_preds->Copy(max_preds_);
    }
  }

  struct ObjInfo Task() const override {return {ObjInfo::kClassification, false}; } 

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob");
    } else {
      out["name"] = String("multi:softmax");
    }
    out["softmax_multiclass_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["softmax_multiclass_param"], &param_);
  }

 private:
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParamOneAPI param_;
  // Cache for max_preds
  mutable HostDeviceVector<bst_float> max_preds_;

  mutable sycl::queue qu_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParamOneAPI);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClassOneAPI, "multi:softmax_oneapi")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassObjOneAPI(false); });
// XGBOOST_REGISTERATE_DEVICEID_KERNEL("multi:softmax", DeviceType::kOneAPI_CPU, "multi:softmax_oneapi");
XGBOOST_REGISTERATE_DEVICEID_KERNEL("multi:softmax", DeviceType::kOneAPI_GPU, "multi:softmax_oneapi");

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClassOneAPI, "multi:softprob_oneapi")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassObjOneAPI(true); });
// XGBOOST_REGISTERATE_DEVICEID_KERNEL("multi:softprob", DeviceType::kOneAPI_CPU, "multi:softprob_oneapi");
XGBOOST_REGISTERATE_DEVICEID_KERNEL("multi:softprob", DeviceType::kOneAPI_GPU, "multi:softprob_oneapi");

}  // namespace obj
}  // namespace xgboost
