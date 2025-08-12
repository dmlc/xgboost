/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#pragma once
#include <memory>  // for shared_ptr

#include "../data/proxy_dmatrix.h"  // for DMatrixProxy
#include "xgboost/data.h"           // for DMatrix
#include "xgboost/learner.h"        // LearnerModelParam

namespace xgboost::predictor {
template <typename Adapter>
void CheckProxyDMatrix(std::shared_ptr<Adapter> m, data::DMatrixProxy const* proxy,
                       LearnerModelParam const* p) {
  CHECK(proxy);
  CHECK(!proxy->Info().IsColumnSplit())
      << "Inplace predict support for column-wise data split is not yet implemented.";
  auto n_features_data = m->NumColumns();
  auto n_features_model = p->num_feature;
  CHECK_EQ(n_features_data, n_features_model)
      << "Number of columns in data must equal to the trained model.";
  CHECK_EQ(proxy->Info().num_row_, m->NumRows());
  CHECK_EQ(proxy->Info().num_col_, m->NumColumns());
  CHECK_EQ(proxy->Info().num_nonzero_, 0);  // unknown
}
}  // namespace xgboost::predictor
