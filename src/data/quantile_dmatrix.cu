/**
 * Copyright 2024, XGBoost Contributors
 */
#include "ellpack_page.cuh"

namespace xgboost::data {
void GetCutsFromEllpack(EllpackPage const& page, common::HistogramCuts* cuts) {
  *cuts = page.Impl()->Cuts();
}
}  // namespace xgboost::data
