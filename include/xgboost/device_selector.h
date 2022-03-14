/*!
 * Copyright 2014-2022 by Contributors
 * \file device_selector.h
 */
#ifndef XGBOOST_DEVICE_SELECTOR_H_
#define XGBOOST_DEVICE_SELECTOR_H_

#include <dmlc/registry.h>
#include <xgboost/parameter.h>
#include <xgboost/json.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace xgboost {

/*!
* Enum of all supported devices.
* kDefault     for user entry device_selector = "cpu:*" or for not specified device_selector
* kCUDA        for user entry device_selector = "cuda:*"
* kOneAPI_Auto for user entry device_selector = "oneapi:*"
* kOneAPI_CPU  for user entry device_selector = "oneapi:cpu:*"
* kOneAPI_GPU  for user entry device_selector = "oneapi:gpu:*"
* 
*/
enum class DeviceType : size_t {
kDefault = 0, kCUDA = 1, kOneAPI_Auto = 2, kOneAPI_CPU = 3, kOneAPI_GPU = 4
};

using KernelsRegisterEntry_t = std::unordered_map<DeviceType, std::string>;

class DeviceSelector {
 public:
  static int constexpr kDefaultIndex = -1;

  DeviceSelector();

  void Init(const std::string& user_input_device_selector);

  class Specification {
    public:
      Specification(const std::string& prefix) : prefix_(prefix + ':') {}

      void Init(const std::string& specification);

      DeviceType Type() const;

      int Index() const;

      std::string GetKernelName(const std::string& method_name) const;
      
      void UpdateByGPUId(int gpu_id);

      int GetGPUId();

      std::string Prefix() const;

      std::string Name() const;

    private:
      std::string prefix_;

      DeviceType type_ = DeviceType::kDefault;
      int index_ = DeviceSelector::kDefaultIndex;
  };

  void SaveConfig(Json* p_out) const;

  void UpdateByGPUId(int gpu_id);

  int GetGPUId();

  Specification& Fit();

  Specification& Predict();

  Specification Fit() const;

  Specification Predict() const;

  std::string GetUserInput() const;

 private:
  Specification fit = Specification("fit");
  Specification predict = Specification("predict");

  /* As far as device_selector can be changed during learner configuration, 
   * we save the initial version of user input
   */
  std::string user_input;
};

std::istream& operator >> (std::istream& is, DeviceSelector& device_selector);

std::ostream& operator << (std::ostream& os, const DeviceSelector& device_selector);

bool operator != (const DeviceSelector::Specification& lhs, const DeviceSelector::Specification& rhs);

std::ostream& operator << (std::ostream& os, const DeviceSelector::Specification& specification);

#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define UNIQUE_KERNEL_REGISTRAR_NAME_ CAT(__registrate_device_selector_kernel,__COUNTER__)

/*!
 * \brief Macro to register kernel names for specific device.
 *
 * \code
 * // example of registering a device
 * XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL("grow_quantile_histmaker", DeviceType::kOneAPI_CPU, "grow_quantile_histmaker_oneapi");
 * \endcode
 */
struct KernelsReg
    : public dmlc::FunctionRegEntryBase<KernelsReg,
                                        ::xgboost::KernelsRegisterEntry_t> {
};

#define XGBOOST_REGISTERATE_DEVICE_SELECTOR_KERNEL(method_name, device_type, kernel_name)      \
  static DMLC_ATTRIBUTE_UNUSED auto&& UNIQUE_KERNEL_REGISTRAR_NAME_ = ::dmlc::Registry< ::xgboost::KernelsReg>::Get()->__REGISTER__(method_name) \
  .body.insert({device_type, kernel_name}) \


#define UNIQUE_DEVICE_REGISTRAR_NAME_ CAT(__registrate_device_,__COUNTER__)

/*!
 * \brief Macro to register kernel names for specific device.
 *
 * \code
 * // example of registering a device
 * XGBOOST_REGISTERATE_DEVICE("cpu")
 * .set_body(DeviceType::kDefault);
 * \endcode
 */
struct DeviceReg
    : public dmlc::FunctionRegEntryBase<DeviceReg,
                                        ::xgboost::DeviceType> {
};

#define XGBOOST_REGISTERATE_DEVICE(device_name)      \
  static DMLC_ATTRIBUTE_UNUSED auto UNIQUE_DEVICE_REGISTRAR_NAME_ = ::dmlc::Registry< ::xgboost::DeviceReg>::Get()->__REGISTER__(device_name) \

}  // namespace xgboost

#endif  // XGBOOST_DEVICE_SELECTOR_H_