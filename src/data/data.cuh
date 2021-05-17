/*!
 * Copyright (c) 2021 by Contributors
 * \file data.cuh
 * \brief Dispatching for input data.
 */
#include "xgboost/data.h"
#include "simple_dmatrix.h"

namespace xgboost {
template <typename AdapterT>
DMatrix *DMatrix::CreateFromGPU(AdapterT *adapter, float missing, int nthread,
                                const std::string &cache_prefix,
                                size_t page_size) {
  CHECK_EQ(cache_prefix.size(), 0)
      << "Device memory construction is not currently supported with external "
         "memory.";
  return data::SimpleDMatrix::FromGPUData(adapter, missing, nthread);
}
}  // namespace xgboost
