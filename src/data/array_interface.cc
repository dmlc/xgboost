/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include "array_interface.h"

#if !defined(XGBOOST_USE_CUDA)

#include "../common/common.h"  // for AssertGPUSupport

#endif  // !defined(XGBOOST_USE_CUDA)

namespace xgboost {
std::string ArrayInterfaceHandler::TypeStr(Type type) {
  auto name_fn = [](std::int32_t bits, char t) {
    return std::to_string(bits) + "-bit " + ArrayInterfaceErrors::TypeStr(t);
  };
  switch (type) {
    case kF2:
      return name_fn(16, 'f');
    case kF4:
      return name_fn(32, 'f');
    case kF8:
      return name_fn(64, 'f');
    case kF16:
      return name_fn(128, 'f');
    case kI1:
      return name_fn(8, 'i');
    case kI2:
      return name_fn(16, 'i');
    case kI4:
      return name_fn(32, 'i');
    case kI8:
      return name_fn(64, 'i');
    case kU1:
      return name_fn(8, 'u');
    case kU2:
      return name_fn(16, 'u');
    case kU4:
      return name_fn(32, 'u');
    case kU8:
      return name_fn(64, 'u');
  }
  LOG(FATAL) << "unreachable";
  return {};
}

#if !defined(XGBOOST_USE_CUDA)
void ArrayInterfaceHandler::SyncCudaStream(int64_t) { common::AssertGPUSupport(); }
bool ArrayInterfaceHandler::IsCudaPtr(void const *) { return false; }
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
