/*!
 * Copyright 2014-2022 by Contributors
 * \file device_selector.cc
 */

#include <xgboost/device_selector.h>
#include <dmlc/registry.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/json.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::KernelsReg);
DMLC_REGISTRY_ENABLE(::xgboost::DeviceReg);
}  // namespace dmlc

namespace xgboost {

DeviceSelector::Specification& DeviceSelector::Fit() {
  return fit;
}

DeviceSelector::Specification& DeviceSelector::Predict() {
  return predict;
}

DeviceSelector::Specification DeviceSelector::Fit() const {
  return fit;
}

DeviceSelector::Specification DeviceSelector::Predict() const {
  return predict;
}

DeviceSelector::DeviceSelector() {
  std::stringstream ss;
  ss << fit.Prefix() << fit << "; " << predict.Prefix() << predict;
  user_input = ss.str();
}

void DeviceSelector::Init(const std::string& user_input_device_selector) {
  int fit_position = user_input_device_selector.find(fit.Prefix());
  int predict_position = user_input_device_selector.find(predict.Prefix());

  CHECK((fit_position == std::string::npos) == (predict_position == std::string::npos))
    <<  "Both " << fit.Prefix() << " and " << predict.Prefix() << " or neither of them should be specified";

  if ((fit_position == std::string::npos) && (predict_position == std::string::npos)) {
    // user_input looks like: device_selector='oneapi:cpu:0'
    fit.Init(user_input_device_selector);
    predict.Init(user_input_device_selector);
  } else {
    int separator_position = user_input_device_selector.find(';');
    CHECK(separator_position != std::string::npos)
      <<  fit.Prefix() << " and " << predict.Prefix() << " shuold be separated by \';\'";

    int fit_specification_begin = fit_position + fit.Prefix().size();
    int predict_specification_begin = predict_position + predict.Prefix().size();
    if (fit_position < predict_position) {
      // user_input looks like: device_selector='fit:oneapi:gpu:0; predict:oneapi:cpu:0'
      int fit_specification_lenght = separator_position - fit_specification_begin;
      fit.Init(user_input_device_selector.substr(fit_specification_begin, fit_specification_lenght));
      predict.Init(user_input_device_selector.substr(predict_specification_begin));
    } else if (fit_position > predict_position) {
      // user_input looks like: device_selector='predict:oneapi:gpu:0; fit:oneapi:cpu:0'
      int predict_specification_length = separator_position - predict_specification_begin;
      fit.Init(user_input_device_selector.substr(fit_specification_begin));
      predict.Init(user_input_device_selector.substr(predict_specification_begin, predict_specification_length));
    }
  }

  /* Check constrains on fit and predict
   * Currently, only oneapi devices support differing specifications for fitting and prediction.
   * For cuda one still can specify predictor manually.
   */
  if (fit != predict) {
    bool is_fit_oneapi_device = (fit.Type() == DeviceType::kOneAPI_CPU) ||
                                (fit.Type() == DeviceType::kOneAPI_GPU);
    bool is_predict_oneapi_device = (predict.Type() == DeviceType::kOneAPI_CPU) ||
                                    (predict.Type() == DeviceType::kOneAPI_GPU);
    CHECK(is_fit_oneapi_device && is_predict_oneapi_device)
      <<  "Currently, only oneapi devices support differing specifications for fitting and prediction. " <<
          "For cuda one still can specify predictor manually. " <<
          "fit = " << fit << "; predict = " << predict << ";";
  }

  // Save the user input device_selector
  std::stringstream ss;
  ss << *this;
  user_input = ss.str();
}

bool operator != (const DeviceSelector::Specification& lhs, const DeviceSelector::Specification& rhs) {
  bool ans = (lhs.Type() != rhs.Type()) || (lhs.Index() != rhs.Index());
  return ans;
}

std::istream& operator >> (std::istream& is, DeviceSelector& device_selector) {
    std::string input;
    std::getline(is, input);

    device_selector.Init(input);
    return is;
}

std::ostream& operator << (std::ostream& os, const DeviceSelector& device_selector) {
  os << device_selector.GetUserInput();

  return os;
}

std::string DeviceSelector::GetUserInput() const {
  return user_input;
}

void DeviceSelector::SaveConfig(Json* p_out) const {
  auto& out = *p_out;

  out["name"] = String(user_input);
}

std::string DeviceSelector::Specification::Prefix() const {
  return prefix_;
}

void DeviceSelector::Specification::Init(const std::string& specification) {
  int position = specification.find(':');

  std::string device_name;
  std::string index_name;
  if (position == std::string::npos) {
    device_name = specification;
  } else {
    device_name = specification.substr(0,position);
    std::string spec_tail = specification.substr(position+1);

    position = spec_tail.find(':');
    if (position == std::string::npos) {
      // Check if spec_tail is a number
      auto it = spec_tail.begin();
      while (it != spec_tail.end() && std::isdigit(*it)) ++it;
      bool is_number = (it == spec_tail.end());

      if (is_number) {
        index_name = spec_tail;
      } else {
        device_name += ":" + spec_tail;
      }
    } else {
      device_name += ":" + spec_tail.substr(0,position);
      index_name = spec_tail.substr(position+1);

      // Check if index_name is a number
      auto it = index_name.begin();
      while (it != index_name.end() && std::isdigit(*it)) ++it;
      bool is_number = (it == index_name.end());
      CHECK(is_number)
        << "Incorrect device specification format: " << specification;
    }
  }

  auto *e = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->Find(device_name);
  CHECK(e != nullptr)
    << "Specified device: " << specification.substr(0,position) << " is unknown.";

  type_ = e->body;

  if (index_name.empty()) {
    index_ = -1;
  } else {
    index_ = std::stoi(index_name);
  }
}

DeviceType DeviceSelector::Specification::Type() const {
  return type_;
}

int DeviceSelector::Specification::Index() const {
  return index_;
}

std::string DeviceSelector::Specification::GetKernelName(const std::string& method_name) const {
  /*
   * Replace the method name, if a specific one is registrated for the current device_selector
   */
  auto *e = ::dmlc::Registry< ::xgboost::KernelsReg>::Get()->Find(method_name);
  if (e != nullptr) {
    auto& register_page = e->body;
    if (register_page.count(type_) > 0) {
      return register_page.at(type_);
    }
  }
  return method_name;
}

std::ostream& operator << (std::ostream& os, const DeviceSelector::Specification& specification) {
  std::vector<std::string> known_device_names = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->ListAllNames();
  for (const std::string& device_name : known_device_names) {
    auto device_type = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->Find(device_name)->body;
    if (device_type == specification.Type()) {
      os << device_name;
      if (specification.Index() != DeviceSelector::kDefaultIndex) {
        os << ':' << specification.Index();
      }
      return os;
    }
  }
  CHECK(false)
    << "Can't find device name for type enumerated as " << static_cast<int> (specification.Type());

  return os;
}

void DeviceSelector::UpdateByGPUId(int gpu_id) {
  fit.UpdateByGPUId(gpu_id);
  predict.UpdateByGPUId(gpu_id);
}

int DeviceSelector::GetGPUId() {
  return fit.GetGPUId();
}

void DeviceSelector::Specification::UpdateByGPUId(int gpu_id) {
  if (gpu_id != GenericParameter::kCpuId) {
    type_ = DeviceType::kCUDA;
    index_ = gpu_id;
  }
}

int DeviceSelector::Specification::GetGPUId() {
  if (type_ == DeviceType::kCUDA) {
    return index_;
  } else {
    return GenericParameter::kCpuId;
  }
}

std::string DeviceSelector::Specification::Name() const {
  std::stringstream ss;
  ss << *this;

  return ss.str();
}

XGBOOST_REGISTERATE_DEVICE("cpu")
.set_body(DeviceType::kDefault);

XGBOOST_REGISTERATE_DEVICE("cuda")
.set_body(DeviceType::kCUDA);


} // namespace xgboost