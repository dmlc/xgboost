/**
 * Copyright 2021-2025, XGBoost Contributors
 * \file context_helper.h
 */
#ifndef PLUGIN_SYCL_CONTEXT_HELPER_H_
#define PLUGIN_SYCL_CONTEXT_HELPER_H_

#include <xgboost/context.h>

namespace xgboost {
namespace sycl {

DeviceOrd DeviceFP64(const DeviceOrd& device);

}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_CONTEXT_HELPER_H_
