/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(const std::string& cache_info) noexcept(false)
    : SparsePageSource(cache_info, ".ellpack.page") {}

void EllpackPageSource::CreateEllpackPage(DMatrix* src, const std::string& cache_info) {}

}  // namespace data
}  // namespace xgboost
