/**
 * Copyright 2019-2024, XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include "ellpack_page.h"

#include <xgboost/data.h>

#include <memory>  // for shared_ptr

// dummy implementation of EllpackPage in case CUDA is not used
namespace xgboost {

class EllpackPageImpl {
  std::shared_ptr<common::HistogramCuts> cuts_;

 public:
  [[nodiscard]] common::HistogramCuts const& Cuts() const { return *cuts_; }
  [[nodiscard]] std::shared_ptr<common::HistogramCuts const> CutsShared() const { return cuts_; }
};

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
bst_idx_t EllpackPage::Size() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
  return 0;
}

[[nodiscard]] common::HistogramCuts const& EllpackPage::Cuts() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
  return impl_->Cuts();
}
}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
