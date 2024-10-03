/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../common/column_matrix.h"
#include "../common/hist_util.h"  // Index
#include "ellpack_page.cuh"
#include "gradient_index.h"
#include "xgboost/data.h"

namespace xgboost {
// Similar to GHistIndexMatrix::SetIndexData, but without the need for adaptor or bin
// searching. Is there a way to unify the code?
template <typename BinT, typename DecompressOffset>
void SetIndexData(Context const* ctx, EllpackPageImpl const* page,
                  std::vector<size_t>* p_hit_count_tloc, DecompressOffset&& get_offset,
                  GHistIndexMatrix* out) {
  std::vector<common::CompressedByteT> h_gidx_buffer;
  auto accessor = page->GetHostAccessor(ctx, &h_gidx_buffer);
  auto const kNull = static_cast<bst_bin_t>(accessor.NullValue());

  auto index_data_span = common::Span{out->index.data<BinT>(), out->index.Size()};
  auto n_bins_total = page->Cuts().TotalBins();

  auto& hit_count_tloc = *p_hit_count_tloc;
  hit_count_tloc.clear();
  hit_count_tloc.resize(ctx->Threads() * n_bins_total, 0);
  bool dense_compressed = page->IsDenseCompressed() && !page->IsDense();
  common::ParallelFor(page->Size(), ctx->Threads(), [&](auto ridx) {
    auto tid = omp_get_thread_num();
    size_t in_rbegin = page->info.row_stride * ridx;
    size_t out_rbegin = out->row_ptr[ridx];
    if (dense_compressed) {
      for (std::size_t j = 0, k = 0; j < page->info.row_stride; ++j) {
        bst_bin_t bin_idx = accessor.gidx_iter[in_rbegin + j];
        if (XGBOOST_EXPECT((bin_idx != kNull), true)) {  // relatively dense
          bin_idx = get_offset(bin_idx, j);
          index_data_span[out_rbegin + k++] = bin_idx;
          ++hit_count_tloc[tid * n_bins_total + bin_idx];
        }
      }
    } else {
      auto r_size = out->row_ptr[ridx + 1] - out->row_ptr[ridx];
      for (size_t j = 0; j < r_size; ++j) {
        bst_bin_t bin_idx = accessor.gidx_iter[in_rbegin + j];
        assert(bin_idx != kNull);
        index_data_span[out_rbegin + j] = bin_idx;
        ++hit_count_tloc[tid * n_bins_total + get_offset(bin_idx, j)];
      }
    }
  });
}

void GetRowPtrFromEllpack(Context const* ctx, EllpackPageImpl const* page,
                          common::RefResourceView<std::size_t>* p_out) {
  auto& row_ptr = *p_out;
  row_ptr = common::MakeFixedVecWithMalloc(page->Size() + 1, std::size_t{0});
  if (page->IsDense()) {
    std::fill(row_ptr.begin() + 1, row_ptr.end(), page->info.row_stride);
  } else {
    std::vector<common::CompressedByteT> h_gidx_buffer;
    auto accessor = page->GetHostAccessor(ctx, &h_gidx_buffer);
    auto const kNull = static_cast<bst_bin_t>(accessor.NullValue());

    common::ParallelFor(page->Size(), ctx->Threads(), [&](auto i) {
      size_t ibegin = page->info.row_stride * i;
      for (size_t j = 0; j < page->info.row_stride; ++j) {
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
    : cut{in_page.Cuts()},
      max_numeric_bins_per_feat{p.max_bin},
      isDense_{in_page.Impl()->IsDense()},
      base_rowid{in_page.BaseRowId()} {
  auto page = in_page.Impl();
  CHECK_EQ(info.num_row_, in_page.Size());

  // pull to host early, prevent race condition
  this->cut.Ptrs();
  this->cut.Values();
  this->cut.MinValues();

  this->ResizeIndex(info.num_nonzero_, page->IsDense());
  if (page->IsDense()) {
    this->index.SetBinOffset(page->Cuts().Ptrs());
  }

  auto offset = page->Cuts().cut_ptrs_.ConstHostSpan();
  auto n_bins_total = page->Cuts().TotalBins();
  GetRowPtrFromEllpack(ctx, page, &this->row_ptr);
  if (page->IsDenseCompressed()) {
    common::DispatchBinType(this->index.GetBinTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      ::xgboost::SetIndexData<T>(
          ctx, page, &hit_count_tloc_,
          [offset](bst_bin_t bin_idx, bst_feature_t fidx) { return bin_idx + offset[fidx]; }, this);
    });
  } else {
    // no compression
    ::xgboost::SetIndexData<uint32_t>(
        ctx, page, &hit_count_tloc_, [&](auto bin_idx, auto) { return bin_idx; }, this);
  }

  this->hit_count = common::MakeFixedVecWithMalloc(n_bins_total, std::size_t{0});
  this->GatherHitCount(ctx->Threads(), n_bins_total);

  // sanity checks
  CHECK_EQ(this->Features(), in_page.Cuts().NumFeatures());
  CHECK_EQ(this->Size(), info.num_row_);
  CHECK(this->cut.cut_ptrs_.HostCanRead());
  CHECK(this->cut.cut_values_.HostCanRead());
  CHECK(this->cut.min_vals_.HostCanRead());

  this->columns_ = std::make_unique<common::ColumnMatrix>(*this, p.sparse_thresh);
  this->columns_->InitFromGHist(ctx, *this);
}
}  // namespace xgboost
