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

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
