/*!
 * Copyright (c) 2021 by Contributors
 * \file data.cuh
 * \brief Dispatching for input data.
 */
#include "xgboost/data.h"
#include "simple_dmatrix.h"

namespace xgboost {
template <typename AdapterT>
DMatrix *DMatrix::CreateFromGPU(AdapterT *adapter, float missing, int nthread) {
  return data::SimpleDMatrix::FromGPUData(adapter, missing, nthread);
}
}  // namespace xgboost
