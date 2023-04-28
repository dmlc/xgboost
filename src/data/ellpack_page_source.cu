/**
 * Copyright 2019-2023, XGBoost contributors
 */
#include <memory>
#include <utility>

#include "ellpack_page.cuh"
#include "ellpack_page_source.h"

namespace xgboost {
namespace data {
void EllpackPageSource::Fetch() {
  dh::safe_cuda(cudaSetDevice(device_));
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
    *impl = EllpackPageImpl(device_, *cuts_, *csr, is_dense_, row_stride_, feature_types_);
    page_->SetBaseRowId(csr->base_rowid);
    this->WriteCache();
  }
}
}  // namespace data
}  // namespace xgboost
