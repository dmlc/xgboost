#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <memory>
#include <vector>

#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

#include "../../src/common/transform.h"
#include "../../src/common/common.h"
#include "./regression_loss_oneapi.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj_oneapi);

struct RegLossParamOneAPI : public XGBoostParameter<RegLossParamOneAPI> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParamOneAPI) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
  }
};

template<typename Loss>
class RegLossObjOneAPI : public ObjFunction {
 protected:
  HostDeviceVector<int> label_correct_;

 public:
  RegLossObjOneAPI() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);

    cl::sycl::default_selector selector;
    qu_ = cl::sycl::queue(selector);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels_.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << " " << "labels are not correctly provided"
        << "preds.size=" << preds.Size() << ", label.size=" << info.labels_.Size() << ", "
        << "Loss: " << Loss::Name();

    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);

    // TODO: add label_correct check
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;

    cl::sycl::buffer<bst_float, 1> preds_buf(preds.HostPointer(), preds.Size());
    cl::sycl::buffer<bst_float, 1> labels_buf(info.labels_.HostPointer(), info.labels_.Size());
    cl::sycl::buffer<GradientPair, 1> out_gpair_buf(out_gpair->HostPointer(), out_gpair->Size());
    cl::sycl::buffer<bst_float, 1> weights_buf(is_null_weight ? NULL : info.weights_.HostPointer(),
                                               is_null_weight ? 1 : info.weights_.Size());

	cl::sycl::buffer<int, 1> additional_input_buf(1);
	{
		auto additional_input_acc = additional_input_buf.get_access<cl::sycl::access::mode::write>();
		additional_input_acc[0] = 1; // Fill the label_correct flag
	}

    auto scale_pos_weight = param_.scale_pos_weight;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
        << "Number of weights should be equal to number of data points.";
    }

    qu_.submit([&](cl::sycl::handler& cgh) {
      auto preds_acc            = preds_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto labels_acc           = labels_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto weights_acc          = weights_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto out_gpair_acc        = out_gpair_buf.get_access<cl::sycl::access::mode::write>(cgh);
      auto additional_input_acc = additional_input_buf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<>(cl::sycl::range<1>(ndata), [=](cl::sycl::id<1> pid) {
        int idx = pid[0];
        bst_float p = Loss::PredTransform(preds_acc[idx]);
        bst_float w = is_null_weight ? 1.0f : weights_acc[idx];
        bst_float label = labels_acc[idx];
        if (label == 1.0f) {
          w *= scale_pos_weight;
        }
        if (!Loss::CheckLabel(label)) {
          // If there is an incorrect label, the host code will know.
          additional_input_acc[0] = 0;
        }
        out_gpair_acc[idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                          Loss::SecondOrderGradient(p, label) * w);
      });
    }).wait();

    int flag = 1;
	{
		auto additional_input_acc = additional_input_buf.get_access<cl::sycl::access::mode::read>();
		flag = additional_input_acc[0];
	}

    if (flag == 0) {
      LOG(FATAL) << Loss::LabelErrorMsg();
    }
  
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) override {
    size_t const ndata = io_preds->Size();

    cl::sycl::buffer<bst_float, 1> io_preds_buf(io_preds->HostPointer(), io_preds->Size());

    qu_.submit([&](cl::sycl::handler& cgh) {
      auto io_preds_acc = io_preds_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<>(cl::sycl::range<1>(ndata), [=](cl::sycl::id<1> pid) {
        int idx = pid[0];
        io_preds_acc[idx] = Loss::PredTransform(io_preds_acc[idx]);
      });
    }).wait();
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["reg_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["reg_loss_param"], &param_);
  }

 protected:
  RegLossParamOneAPI param_;

  cl::sycl::queue qu_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParamOneAPI);

// TODO: Find a better way to dispatch names of DPC++ kernels with various template parameters of loss function
XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegressionOneAPI, LinearSquareLossOneAPI::Name())
.describe("Regression with squared error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LinearSquareLossOneAPI>(); });
XGBOOST_REGISTER_OBJECTIVE(SquareLogErrorOneAPI, SquaredLogErrorOneAPI::Name())
.describe("Regression with root mean squared logarithmic error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<SquaredLogErrorOneAPI>(); });
XGBOOST_REGISTER_OBJECTIVE(LogisticRegressionOneAPI, LogisticRegressionOneAPI::Name())
.describe("Logistic regression for probability regression task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticRegressionOneAPI>(); });
XGBOOST_REGISTER_OBJECTIVE(LogisticClassificationOneAPI, LogisticClassificationOneAPI::Name())
.describe("Logistic regression for binary classification task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticClassificationOneAPI>(); });
XGBOOST_REGISTER_OBJECTIVE(LogisticRawOneAPI, LogisticRawOneAPI::Name())
.describe("Logistic regression for classification, output score "
          "before logistic transformation with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticRawOneAPI>(); });

}  // namespace obj
}  // namespace xgboost
