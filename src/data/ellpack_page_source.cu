/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* dmat, const std::string& cache_info) noexcept(false)
    : page_(dmat) {}

void EllpackPageSource::CreateEllpackPage(DMatrix* dmat, const std::string& cache_info) {}

}  // namespace data
}  // namespace xgboost
