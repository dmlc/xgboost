/*!
 * Copyright 2014-2022 by Contributors
 * \file device_id.h
 */
#ifndef XGBOOST_DEVICE_ID_H_
#define XGBOOST_DEVICE_ID_H_

#include <dmlc/registry.h>
#include <xgboost/parameter.h>
#include <xgboost/json.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace xgboost {

/*!
* Enum of all supported devices.
* kDefault    for user entry device_id = "cpu:*" or for not specified device_id
* kCUDA       for user entry device_id = "cuda:*"
* kOneAPI_CPU for user entry device_id = "oneapi:cpu:*"
* kOneAPI_GPU for user entry device_id = "oneapi:gpu:*"
* 
*/
enum class DeviceType : size_t {
kDefault = 0, kCUDA = 1, kOneAPI_CPU = 2, kOneAPI_GPU = 3
};

using KernelsRegisterEntry_t = std::unordered_map<DeviceType, std::string>;

class DeviceId {
 public:
  static int constexpr kDefaultIndex = -1;

  void Init(const std::string& user_input_device_id);

  DeviceType Type() const;

  int Index() const;

  std::string GetKernelName(const std::string& method_name) const;

  void SaveConfig(Json* p_out) const;

  void UpdateByGPUId(int gpu_id);

 private:
  DeviceType type_ = DeviceType::kDefault;
  int index_ = kDefaultIndex;
};

std::istream& operator >> (std::istream& is, DeviceId& device_id);

std::ostream& operator << (std::ostream& os, const DeviceId& device_id);

#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define UNIQUE_KERNEL_REGISTRAR_NAME_ CAT(__registrate_device_id_kernel,__COUNTER__)
/*!
 * \brief Macro to register kernel names for specific device.
 */

struct KernelsReg
    : public dmlc::FunctionRegEntryBase<KernelsReg,
                                        ::xgboost::KernelsRegisterEntry_t> {
};

#define XGBOOST_REGISTERATE_DEVICEID_KERNEL(method_name, device_type, kernel_name)      \
  static DMLC_ATTRIBUTE_UNUSED auto&& UNIQUE_KERNEL_REGISTRAR_NAME_ = ::dmlc::Registry< ::xgboost::KernelsReg>::Get()->__REGISTER__(method_name) \
  .body.insert({device_type, kernel_name}) \


#define UNIQUE_DEVICE_REGISTRAR_NAME_ CAT(__registrate_device_,__COUNTER__)
/*!
 * \brief Macro to register kernel names for specific device.
 */

struct DeviceReg
    : public dmlc::FunctionRegEntryBase<DeviceReg,
                                        ::xgboost::DeviceType> {
};

#define XGBOOST_REGISTERATE_DEVICE(device_name)      \
  static DMLC_ATTRIBUTE_UNUSED auto UNIQUE_DEVICE_REGISTRAR_NAME_ = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->__REGISTER__(device_name) \

}  // namespace xgboost

#endif  // XGBOOST_DEVICE_ID_H_