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
#include "regression_loss_oneapi.h"
#include "device_manager_oneapi.h"

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
 private:
  DeviceManagerOneAPI device_manager;
  std::unique_ptr<ObjFunction> objective_backend_;
  RegLossParamOneAPI param_;

 public:
  RegLossObjOneAPI() = default;

  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);

    const DeviceSelector::Specification& device_spec = tparam_->device_selector.Fit();
    sycl::device device = device_manager.GetDevice(device_spec);
    bool is_cpu = device.is_cpu() || device.is_host();

    LOG(INFO) << "device_spec = " << device_spec << ", is_cpu = " << int(is_cpu);

    if (is_cpu) {
      objective_backend_.reset(ObjFunction::Create(Loss::Name(), tparam_));
    } else {
      objective_backend_.reset(ObjFunction::Create(Loss::BackendName(), tparam_));
    }
    objective_backend_->Configure(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    objective_backend_->GetGradient(preds, info, iter, out_gpair);
  }

  const char* DefaultEvalMetric() const override {
    return objective_backend_->DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    objective_backend_->PredTransform(io_preds);
  }

  float ProbToMargin(float base_score) const override {
    return objective_backend_->ProbToMargin(base_score);
  }

  struct ObjInfo Task() const override {
    return objective_backend_->Task();
  }

  uint32_t Targets(MetaInfo const& info) const override {
    return objective_backend_->Targets(info);
  }

  void SaveConfig(Json* p_out) const override {
    objective_backend_->SaveConfig(p_out);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["reg_loss_param"], &param_);
  }
};

template<typename Loss>
class RegLossObjOneAPIBackend : public ObjFunction {
 private:
  DeviceManagerOneAPI device_manager;

 protected:
  HostDeviceVector<int> label_correct_;

 public:
  RegLossObjOneAPIBackend() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);

    const DeviceSelector::Specification& device_spec = tparam_->device_selector.Fit();
    qu_ = device_manager.GetQueue(device_spec);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels.Size())
        << " " << "labels are not correctly provided"
        << "preds.size=" << preds.Size() << ", label.size=" << info.labels.Size() << ", "
        << "Loss: " << Loss::Name();

    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);

    // TODO: add label_correct check
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;

    sycl::buffer<bst_float, 1> preds_buf(preds.HostPointer(), preds.Size());
    sycl::buffer<bst_float, 1> labels_buf(info.labels.Data()->HostPointer(), info.labels.Size());
    sycl::buffer<GradientPair, 1> out_gpair_buf(out_gpair->HostPointer(), out_gpair->Size());
    sycl::buffer<bst_float, 1> weights_buf(is_null_weight ? NULL : info.weights_.HostPointer(),
                                               is_null_weight ? 1 : info.weights_.Size());

    const size_t n_targets = std::max(info.labels.Shape(1), static_cast<size_t>(1));

    sycl::buffer<int, 1> additional_input_buf(1);
    {
      auto additional_input_acc = additional_input_buf.get_access<sycl::access::mode::write>();
      additional_input_acc[0] = 1; // Fill the label_correct flag
    }

    auto scale_pos_weight = param_.scale_pos_weight;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), info.labels.Shape(0))
        << "Number of weights should be equal to number of data points.";
    }

    qu_.submit([&](sycl::handler& cgh) {
      auto preds_acc            = preds_buf.get_access<sycl::access::mode::read>(cgh);
      auto labels_acc           = labels_buf.get_access<sycl::access::mode::read>(cgh);
      auto weights_acc          = weights_buf.get_access<sycl::access::mode::read>(cgh);
      auto out_gpair_acc        = out_gpair_buf.get_access<sycl::access::mode::write>(cgh);
      auto additional_input_acc = additional_input_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<>(sycl::range<1>(ndata), [=](sycl::id<1> pid) {
        int idx = pid[0];
        bst_float p = Loss::PredTransform(preds_acc[idx]);
        bst_float w = is_null_weight ? 1.0f : weights_acc[idx/n_targets];
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
		auto additional_input_acc = additional_input_buf.get_access<sycl::access::mode::read>();
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

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    size_t const ndata = io_preds->Size();
    sycl::buffer<bst_float, 1> io_preds_buf(io_preds->HostPointer(), io_preds->Size());

    qu_.submit([&](sycl::handler& cgh) {
      auto io_preds_acc = io_preds_buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<>(sycl::range<1>(ndata), [=](sycl::id<1> pid) {
        int idx = pid[0];
        io_preds_acc[idx] = Loss::PredTransform(io_preds_acc[idx]);
      });
    }).wait();
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

  struct ObjInfo Task() const override {
    return Loss::Info();
  };

  uint32_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
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

  mutable sycl::queue qu_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParamOneAPI);

// TODO: Find a better way to dispatch names of DPC++ kernels with various template parameters of loss function
XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegressionOneAPI, LinearSquareLossOneAPI::KernelName())
.describe("Regression with squared error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LinearSquareLossOneAPI>(); });
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LinearSquareLossOneAPI::Name(), DeviceType::kOneAPI_Auto, LinearSquareLossOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LinearSquareLossOneAPI::Name(), DeviceType::kOneAPI_CPU,  LinearSquareLossOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LinearSquareLossOneAPI::Name(), DeviceType::kOneAPI_GPU,  LinearSquareLossOneAPI::KernelName());
XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegressionOneAPIBackend, LinearSquareLossOneAPI::BackendName())
.describe("Regression with squared error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPIBackend<LinearSquareLossOneAPI>(); });

XGBOOST_REGISTER_OBJECTIVE(SquareLogErrorOneAPI, SquaredLogErrorOneAPI::KernelName())
.describe("Regression with root mean squared logarithmic error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<SquaredLogErrorOneAPI>(); });
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(SquaredLogErrorOneAPI::Name(), DeviceType::kOneAPI_Auto, SquaredLogErrorOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(SquaredLogErrorOneAPI::Name(), DeviceType::kOneAPI_CPU,  SquaredLogErrorOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(SquaredLogErrorOneAPI::Name(), DeviceType::kOneAPI_GPU,  SquaredLogErrorOneAPI::KernelName());
XGBOOST_REGISTER_OBJECTIVE(SquareLogErrorOneAPIBackend, SquaredLogErrorOneAPI::BackendName())
.describe("Regression with root mean squared logarithmic error with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPIBackend<SquaredLogErrorOneAPI>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRegressionOneAPI, LogisticRegressionOneAPI::KernelName())
.describe("Logistic regression for probability regression task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticRegressionOneAPI>(); });
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRegressionOneAPI::Name(), DeviceType::kOneAPI_Auto, LogisticRegressionOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRegressionOneAPI::Name(), DeviceType::kOneAPI_CPU,  LogisticRegressionOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRegressionOneAPI::Name(), DeviceType::kOneAPI_GPU,  LogisticRegressionOneAPI::KernelName());
XGBOOST_REGISTER_OBJECTIVE(LogisticRegressionOneAPIBackend, LogisticRegressionOneAPI::BackendName())
.describe("Logistic regression for probability regression task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPIBackend<LogisticRegressionOneAPI>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticClassificationOneAPI, LogisticClassificationOneAPI::KernelName())
.describe("Logistic regression for binary classification task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticClassificationOneAPI>(); });
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticClassificationOneAPI::Name(), DeviceType::kOneAPI_Auto, LogisticClassificationOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticClassificationOneAPI::Name(), DeviceType::kOneAPI_CPU,  LogisticClassificationOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticClassificationOneAPI::Name(), DeviceType::kOneAPI_GPU,  LogisticClassificationOneAPI::KernelName());
XGBOOST_REGISTER_OBJECTIVE(LogisticClassificationOneAPIBackend, LogisticClassificationOneAPI::BackendName())
.describe("Logistic regression for binary classification task with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPIBackend<LogisticClassificationOneAPI>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRawOneAPI, LogisticRawOneAPI::KernelName())
.describe("Logistic regression for classification, output score "
          "before logistic transformation with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPI<LogisticRawOneAPI>(); });
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRawOneAPI::Name(), DeviceType::kOneAPI_Auto, LogisticRawOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRawOneAPI::Name(), DeviceType::kOneAPI_CPU,  LogisticRawOneAPI::KernelName());
XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(LogisticRawOneAPI::Name(), DeviceType::kOneAPI_GPU,  LogisticRawOneAPI::KernelName());
XGBOOST_REGISTER_OBJECTIVE(LogisticRawOneAPIBackend, LogisticRawOneAPI::BackendName())
.describe("Logistic regression for classification, output score "
          "before logistic transformation with DPC++ backend.")
.set_body([]() { return new RegLossObjOneAPIBackend<LogisticRawOneAPI>(); });

}  // namespace obj
}  // namespace xgboost
