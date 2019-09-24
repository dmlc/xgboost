/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include "ellpack_page_source.h"

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(const std::string& cache_info) noexcept(false)
    : SparsePageSource(cache_info, ".ellpack.page") {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
}

void EllpackPageSource::CreateEllpackPage(DMatrix* src, const std::string& cache_info) {
  LOG(FATAL) << "Internal Error: "
                "XGBoost is not compiled with CUDA but EllpackPageSource is required";
}

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
