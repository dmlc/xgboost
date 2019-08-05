/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file ellpack_page.cc
 */
#ifndef XGBOOST_USE_CUDA

#include <xgboost/data.h>

// dummy implementation of ELlpackPage in case CUDA is not used
namespace xgboost {

class EllpackPageImpl {};

EllpackPage::EllpackPage(DMatrix* dmat) {
  LOG(FATAL) << "Not implemented.";
}

EllpackPage::~EllpackPage() {
  LOG(FATAL) << "Not implemented.";
}

}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
