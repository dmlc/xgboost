/*!
 * Copyright 2018 XGBoost contributors
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>

#include "../common/linalg_op.h"

#include "rabit/rabit.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "regression_obj.cu"
#endif  // XGBOOST_USE_CUDA
