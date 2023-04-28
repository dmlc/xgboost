/**
 * Copyright 2019-2023, XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <xgboost/data.h>

// dummy implementation of EllpackPage in case CUDA is not used
namespace xgboost {

class EllpackPageImpl {};

EllpackPage::EllpackPage() = default;

EllpackPage::EllpackPage(Context const*, DMatrix*, const BatchParam&) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}

EllpackPage::~EllpackPage() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}

void EllpackPage::SetBaseRowId(std::size_t) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}
size_t EllpackPage::Size() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
  return 0;
}

}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
