/*!
 * Copyright 2018 XGBoost contributors
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(hinge_obj);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "hinge.cu"
#endif
