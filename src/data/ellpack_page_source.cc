/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <xgboost/data.h>
#include "ellpack_page_source.h"
namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false) {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
}

void EllpackPageSource::BeforeFirst() {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
}

bool EllpackPageSource::Next() {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
  return false;
}

EllpackPage& EllpackPageSource::Value() {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
  EllpackPage* page;
  return *page;
}

const EllpackPage& EllpackPageSource::Value() const {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
  EllpackPage* page;
  return *page;
}

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
