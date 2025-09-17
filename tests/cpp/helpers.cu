/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <xgboost/c_api.h>

#include "helpers.h"

namespace xgboost {
CudaArrayIterForTest::CudaArrayIterForTest(float sparsity, size_t rows,
                                           size_t cols, size_t batches)
    : ArrayIterForTest{sparsity, rows, cols, batches} {
  rng_->Device(FstCU());
  std::tie(batches_, interface_) = rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
  this->Reset();
}

int CudaArrayIterForTest::Next() {
  if (iter_ == n_batches_) {
    return 0;
  }
  XGProxyDMatrixSetDataCudaArrayInterface(proxy_, batches_[iter_].c_str());
  iter_++;
  return 1;
}
}  // namespace xgboost
