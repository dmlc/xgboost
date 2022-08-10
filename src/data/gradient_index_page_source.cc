/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#include "gradient_index_page_source.h"

namespace xgboost {
namespace data {
void GradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    if (count_ != 0 && !sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(count_, source_->Iter());
    auto const& csr = source_->Page();
    CHECK_NE(cuts_.Values().size(), 0);
    this->page_.reset(new GHistIndexMatrix(*csr, feature_types_, cuts_, max_bin_per_feat_,
                                           is_dense_, sparse_thresh_, nthreads_));
    this->WriteCache();
  }
}
}  // namespace data
}  // namespace xgboost
