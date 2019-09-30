/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <xgboost/data.h>

// dummy implementation of ELlpackPage in case CUDA is not used
namespace xgboost {

class EllpackPageImpl {};

EllpackPage::EllpackPage(DMatrix* dmat, const BatchParam& param) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

EllpackPage::~EllpackPage() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

size_t EllpackPage::Size() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
  return 0;
}

void EllpackPage::Clear() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

void EllpackPage::Push(const SparsePage& batch) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

size_t EllpackPage::MemCostBytes() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
  return 0;
}


}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
