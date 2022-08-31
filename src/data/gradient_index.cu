#include <memory>  // std::unique_ptr

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "ellpack_page.cuh"
#include "gradient_index.h"
#include "xgboost/data.h"

namespace xgboost {
void CopyEllpackToGHist(Context const* ctx, MetaInfo const& info, EllpackPageImpl const& page,
                        size_t nnz, GHistIndexMatrix* p_out) {
  CHECK_EQ(info.num_row_, page.Size());
  auto cuts = page.Cuts();
  GHistIndexMatrix& out = *p_out;
  out.cut = page.Cuts();

  out.ResizeIndex(info.num_nonzero_, page.is_dense);
  if (page.is_dense) {
    out.index.SetBinOffset(page.Cuts().Ptrs());
  }

  auto n_threads = ctx->Threads();
  auto const& h_gidx_buffer = page.gidx_buffer.ConstHostVector();
  common::CompressedIterator<bst_bin_t> data_buf{h_gidx_buffer.data(), page.NumSymbols()};
  if (page.is_dense) {
    uint32_t const* offsets = out.index.Offset();
    out.row_ptr.resize(page.Size() + 1);
    out.row_ptr.front() = 0;
    for (size_t i = 1; i < out.row_ptr.size(); ++i) {
      out.row_ptr[i] = out.row_ptr[i - 1] + page.row_stride;
    }
    auto n_bins_total = page.Cuts().TotalBins();
    std::vector<size_t> hit_count_tloc(ctx->Threads() * n_bins_total, 0);
    common::DispatchBinType(out.index.GetBinTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto get_offset = [&](auto bin_idx, auto fidx) {
        return static_cast<T>(bin_idx - offsets[fidx]);
      };
      common::Span<T> index_data_span = {out.index.data<T>(), out.index.Size()};
      common::ParallelFor(page.Size(), n_threads, [&](auto i) {
        auto tid = omp_get_thread_num();
        size_t ibegin = page.row_stride * i;
        for (size_t j = 0; j < page.row_stride; ++j) {
          auto bin_idx = data_buf[ibegin + j];
          index_data_span[ibegin + j] = get_offset(bin_idx, j);
          ++hit_count_tloc[tid * n_bins_total + bin_idx];
        }
      });
    });

    out.hit_count.resize(n_bins_total, 0);
    common::ParallelFor(n_bins_total, ctx->Threads(), [&](auto idx) {
      for (int32_t tid = 0; tid < n_threads; ++tid) {
        out.hit_count[idx] += hit_count_tloc[tid * n_bins_total + idx];
        hit_count_tloc[tid * n_bins_total + idx] = 0;  // reset for next batch
      }
    });

    {

    }
  } else {
    LOG(FATAL) << "Not implemented";
  }
  CHECK_EQ(out.Features(), info.num_col_);
  CHECK_EQ(out.Size(), info.num_row_);
}

GHistIndexMatrix::GHistIndexMatrix(Context const* ctx, MetaInfo const& info,
                                   EllpackPage const& in_page, BatchParam const& p) {
  auto page = in_page.Impl();

  CopyEllpackToGHist(ctx, info, *page, info.num_nonzero_, this);

  this->columns_ = std::make_unique<common::ColumnMatrix>(*this, p.sparse_thresh);
  auto n_features = info.num_col_;

  if (page->is_dense) {
    // row index is compressed, we need to dispatch it.
    common::DispatchBinType(index.GetBinTypeSize(),
                            [&, size = page->Size(), n_features = n_features](auto t) {
                              using RowBinIdxT = decltype(t);
                              columns_->SetIndexNoMissing(base_rowid, index.data<RowBinIdxT>(),
                                                          size, n_features, ctx->Threads());
                            });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}
}  // namespace xgboost
