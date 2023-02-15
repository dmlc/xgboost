/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <memory>  // std::unique_ptr

#include "../common/column_matrix.h"
#include "../common/hist_util.h"  // Index
#include "ellpack_page.cuh"
#include "gradient_index.h"
#include "xgboost/data.h"

namespace xgboost {
// Similar to GHistIndexMatrix::SetIndexData, but without the need for adaptor or bin
// searching. Is there a way to unify the code?
template <typename BinT, typename CompressOffset>
void SetIndexData(Context const* ctx, EllpackPageImpl const* page,
                  std::vector<size_t>* p_hit_count_tloc, CompressOffset&& get_offset,
                  GHistIndexMatrix* out) {
  auto accessor = page->GetHostAccessor();
  auto const kNull = static_cast<bst_bin_t>(accessor.NullValue());

  common::Span<BinT> index_data_span = {out->index.data<BinT>(), out->index.Size()};
  auto n_bins_total = page->Cuts().TotalBins();

  auto& hit_count_tloc = *p_hit_count_tloc;
  hit_count_tloc.clear();
  hit_count_tloc.resize(ctx->Threads() * n_bins_total, 0);

  common::ParallelFor(page->Size(), ctx->Threads(), [&](auto i) {
    auto tid = omp_get_thread_num();
    size_t in_rbegin = page->row_stride * i;
    size_t out_rbegin = out->row_ptr[i];
    auto r_size = out->row_ptr[i + 1] - out->row_ptr[i];
    for (size_t j = 0; j < r_size; ++j) {
      auto bin_idx = accessor.gidx_iter[in_rbegin + j];
      assert(bin_idx != kNull);
      index_data_span[out_rbegin + j] = get_offset(bin_idx, j);
      ++hit_count_tloc[tid * n_bins_total + bin_idx];
    }
  });
}

void GetRowPtrFromEllpack(Context const* ctx, EllpackPageImpl const* page,
                          std::vector<size_t>* p_out) {
  auto& row_ptr = *p_out;
  row_ptr.resize(page->Size() + 1, 0);
  if (page->is_dense) {
    std::fill(row_ptr.begin() + 1, row_ptr.end(), page->row_stride);
  } else {
    auto accessor = page->GetHostAccessor();
    auto const kNull = static_cast<bst_bin_t>(accessor.NullValue());

    common::ParallelFor(page->Size(), ctx->Threads(), [&](auto i) {
      size_t ibegin = page->row_stride * i;
      for (size_t j = 0; j < page->row_stride; ++j) {
        bst_bin_t bin_idx = accessor.gidx_iter[ibegin + j];
        if (bin_idx != kNull) {
          row_ptr[i + 1]++;
        }
      }
    });
  }
  std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
}

GHistIndexMatrix::GHistIndexMatrix(Context const* ctx, MetaInfo const& info,
                                   EllpackPage const& in_page, BatchParam const& p)
    : max_numeric_bins_per_feat{p.max_bin} {
  auto page = in_page.Impl();
  isDense_ = page->is_dense;

  CHECK_EQ(info.num_row_, in_page.Size());

  this->cut = page->Cuts();
  // pull to host early, prevent race condition
  this->cut.Ptrs();
  this->cut.Values();
  this->cut.MinValues();

  this->ResizeIndex(info.num_nonzero_, page->is_dense);
  if (page->is_dense) {
    this->index.SetBinOffset(page->Cuts().Ptrs());
  }

  auto n_bins_total = page->Cuts().TotalBins();
  GetRowPtrFromEllpack(ctx, page, &this->row_ptr);
  if (page->is_dense) {
    common::DispatchBinType(this->index.GetBinTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      ::xgboost::SetIndexData<T>(ctx, page, &hit_count_tloc_, index.MakeCompressor<T>(), this);
    });
  } else {
    // no compression
    ::xgboost::SetIndexData<uint32_t>(
        ctx, page, &hit_count_tloc_, [&](auto bin_idx, auto) { return bin_idx; }, this);
  }

  this->hit_count.resize(n_bins_total, 0);
  this->GatherHitCount(ctx->Threads(), n_bins_total);

  // sanity checks
  CHECK_EQ(this->Features(), info.num_col_);
  CHECK_EQ(this->Size(), info.num_row_);
  CHECK(this->cut.cut_ptrs_.HostCanRead());
  CHECK(this->cut.cut_values_.HostCanRead());
  CHECK(this->cut.min_vals_.HostCanRead());

  this->columns_ = std::make_unique<common::ColumnMatrix>(*this, p.sparse_thresh);
  this->columns_->InitFromGHist(ctx, *this);
}
}  // namespace xgboost
