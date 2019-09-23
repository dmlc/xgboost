/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <xgboost/data.h>

// dummy implementation of ELlpackPage in case CUDA is not used
namespace xgboost {

class EllpackPageImpl {};

EllpackPage::EllpackPage() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

EllpackPage::EllpackPage(DMatrix* dmat) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

EllpackPage::~EllpackPage() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but EllpackPage is required";
}

}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
