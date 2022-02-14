/*!
 * Copyright 2014-2022 by Contributors
 * \file device_id.cc
 */

#include <xgboost/device_id.h>
#include <dmlc/registry.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/json.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::KernelsReg);
DMLC_REGISTRY_ENABLE(::xgboost::DeviceReg);
}  // namespace dmlc

namespace xgboost {

void DeviceId::Init(const std::string& user_input_device_id) {
    size_t position = user_input_device_id.find_last_of(':');

    const std::string device_name = user_input_device_id.substr(0,position);
    const std::string index_name = user_input_device_id.substr(position+1);
    auto *e = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->Find(device_name);
    CHECK(e != nullptr)
      << "Specified device: " << user_input_device_id.substr(0,position) << " is unknown.";

    type_ = e->body;
    index_ = std::stoi(index_name);
}

DeviceType DeviceId::Type() const {
  return type_;
}

int DeviceId::Index() const {
  return index_;
}

std::string DeviceId::GetKernelName(const std::string& method_name) const {
  /*
   * Replace the method name, if a specific one is registrated for the current device_id
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

void DeviceId::SaveConfig(Json* p_out) const {
  auto& out = *p_out;

  std::stringstream ss;
  ss << *this;

  std::string name;
  ss >> name;
  out["name"] = String(name);
}

std::istream& operator >> (std::istream& is, DeviceId& device_id) {
    std::string input;
    is >> input;
    device_id.Init(input);
    return is;
}

std::ostream& operator << (std::ostream& os, const DeviceId& device_id) {
  std::vector<std::string> known_device_names = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->ListAllNames();
  for (const std::string& device_name : known_device_names) {
    auto device_type = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->Find(device_name)->body;
    if (device_type == device_id.Type()) {
      os << device_name << ':' << device_id.Index();
      return os;
    }
  }
  CHECK(false)
    << "Can't find device name for type enumerated as " << static_cast<int> (device_id.Type());

  return os;
}

void DeviceId::UpdateByGPUId(int gpu_id) {
  if (gpu_id != GenericParameter::kCpuId) {
    type_ = DeviceType::kCUDA;
    index_ = gpu_id;
  }
}

int DeviceId::GetGPUId() {
  if (type_ == DeviceType::kCUDA) {
    return index_;
  } else {
    return GenericParameter::kCpuId;
  }
}

XGBOOST_REGISTERATE_DEVICE("cpu")
.set_body(DeviceType::kDefault);

XGBOOST_REGISTERATE_DEVICE("cuda")
.set_body(DeviceType::kCUDA);


} // namespace xgboost