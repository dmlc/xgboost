/*!
 * Copyright 2019 XGBoost contributors
 */

// Dummy file to keep the CUDA conditional compile trick.
#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(rank_obj);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "rank_obj.cu"
#endif  // XGBOOST_USE_CUDA
