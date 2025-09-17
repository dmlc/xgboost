/**
 * Copyright 2023 by XGBoost Contributors
 */

// Dummy file to enable the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(quantile_obj);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "quantile_obj.cu"
#endif  // !defined(XBGOOST_USE_CUDA)
