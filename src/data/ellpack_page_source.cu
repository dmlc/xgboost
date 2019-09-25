/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false)
    : page_(dmat, param) {}

void EllpackPageSource::CreateEllpackPage(DMatrix* dmat, const std::string& cache_info) {}

}  // namespace data
}  // namespace xgboost
