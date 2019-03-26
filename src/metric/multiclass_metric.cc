/*!
 * Copyright 2019 XGBoost contributors
 */
// Dummy file to keep the CUDA conditional compile trick.


#include <rabit/rabit.h>
#include <xgboost/metric/multiclass_metric.h>
#include <cmath>
#include "../common/math.h"

#if !defined(XGBOOST_USE_CUDA)
#include "multiclass_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA)

