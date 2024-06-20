/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <memory>

#include "ellpack_page.cuh"
#include "ellpack_page.h"  // for EllpackPage
#include "ellpack_page_source.h"

namespace xgboost::data {
void EllpackPageSource::Fetch() {
  dh::safe_cuda(cudaSetDevice(device_.ordinal));
  if (!this->ReadCache()) {
    if (count_ != 0 && !sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(count_, source_->Iter());
    auto const &csr = source_->Page();
    this->page_.reset(new EllpackPage{});
    auto *impl = this->page_->Impl();
    *impl = EllpackPageImpl(device_, cuts_, *csr, is_dense_, row_stride_, feature_types_);
    page_->SetBaseRowId(csr->base_rowid);
    this->WriteCache();
  }
}
}  // namespace xgboost::data
