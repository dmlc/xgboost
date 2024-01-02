/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include "array_interface.h"

#include "../common/common.h"  // for AssertGPUSupport

namespace xgboost {
#if !defined(XGBOOST_USE_CUDA)
void ArrayInterfaceHandler::SyncCudaStream(int64_t) { common::AssertGPUSupport(); }
bool ArrayInterfaceHandler::IsCudaPtr(void const *) { return false; }
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
