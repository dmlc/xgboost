/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include "gradient_index_page_source.h"

#include <memory>   // for make_shared
#include <utility>  // for move

#include "../common/hist_util.h"  // for HistogramCuts
#include "gradient_index.h"       // for GHistIndexMatrix

namespace xgboost::data {
void GradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    if (count_ != 0 && !sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      //
      // The mixin doesn't sync the source if `sync_` is false, we need to sync it
      // ourselves.
      ++(*source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(count_, source_->Iter());
    auto const& csr = source_->Page();
    CHECK_NE(cuts_.Values().size(), 0);
    this->page_.reset(new GHistIndexMatrix{*csr, feature_types_, cuts_, max_bin_per_feat_,
                                           is_dense_, sparse_thresh_, nthreads_});
    this->WriteCache();
  }
}

void ExtGradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    CHECK_EQ(count_, source_->Iter());
    ++(*source_);
    CHECK_GE(source_->Iter(), 1);
    CHECK_NE(cuts_.Values().size(), 0);
    HostAdapterDispatch(proxy_, [this](auto const& value) {
      // This does three things:
      // - Generate CSR matrix for gradient index.
      // - Generate the column matrix for gradient index.
      // - Concatenate the meta info.
      common::HistogramCuts cuts{this->cuts_};
      this->page_.reset();
      // The external iterator has the data when the `next` method is called. Therefore,
      // it's one step ahead of this source.

      // FIXME(jiamingy): For now, we use the `info->IsDense()` to represent all batches
      // similar to the sparse DMatrix source. We should use per-batch property with proxy
      // DMatrix info instead. This requires more fine-grained tests.
      this->page_ = std::make_shared<GHistIndexMatrix>(
          value.NumRows(), this->base_rows_.at(source_->Iter() - 1), std::move(cuts),
          this->p_.max_bin, info_->IsDense());
      bst_idx_t prev_sum = 0;
      bst_idx_t rbegin = 0;
      // Use `value.NumRows()` for the size of a single batch. Unlike the
      // `IterativeDMatrix`, external memory doesn't concatenate the pages.
      this->page_->PushAdapterBatch(ctx_, rbegin, prev_sum, value, this->missing_,
                                    this->feature_types_, this->p_.sparse_thresh, value.NumRows());
      this->page_->PushAdapterBatchColumns(ctx_, value, this->missing_, rbegin);
      this->info_->Extend(proxy_->Info(), false, false);
    });
    this->WriteCache();
  }
}
}  // namespace xgboost::data
