/*!
 * Copyright 2019-2021 XGBoost contributors
 */
#include <memory>
#include <utility>

#include "ellpack_page.cuh"
#include "ellpack_page_source.h"

namespace xgboost {
namespace data {
void EllpackPageSource::Fetch() {
  dh::safe_cuda(cudaSetDevice(param_.gpu_id));
  if (!this->ReadCache()) {
    auto const &csr = source_->Page();
    this->page_.reset(new EllpackPage{});
    auto *impl = this->page_->Impl();
    *impl = EllpackPageImpl(param_.gpu_id, *cuts_, *csr, is_dense_, row_stride_,
                            feature_types_);
    page_->SetBaseRowId(csr->base_rowid);
    this->WriteCache();
  }
}
}  // namespace data
}  // namespace xgboost
