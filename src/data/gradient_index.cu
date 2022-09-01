#include <memory>  // std::unique_ptr

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "ellpack_page.cuh"
#include "gradient_index.h"
#include "xgboost/data.h"

namespace xgboost {
GHistIndexMatrix::GHistIndexMatrix(Context const* ctx, MetaInfo const& info,
                                   EllpackPage const& in_page, BatchParam const& p)
    : max_num_bins{p.max_bin} {
  auto page = in_page.Impl();
  isDense_ = page->is_dense;

  CHECK_EQ(info.num_row_, in_page.Size());

  this->cut = page->Cuts();

  this->cut.Ptrs();
  this->cut.Values();
  this->cut.MinValues();

  this->ResizeIndex(info.num_nonzero_, page->is_dense);
  if (page->is_dense) {
    this->index.SetBinOffset(page->Cuts().Ptrs());
  }

  auto n_threads = ctx->Threads();
  auto const& h_gidx_buffer = page->gidx_buffer.ConstHostVector();
  auto n_bins_total = page->Cuts().TotalBins();
  std::vector<size_t> hit_count_tloc(ctx->Threads() * n_bins_total, 0);

  common::CompressedIterator<bst_bin_t> data_buf{h_gidx_buffer.data(), page->NumSymbols()};
  if (page->is_dense) {
    uint32_t const* offsets = this->index.Offset();
    this->row_ptr.resize(page->Size() + 1);
    this->row_ptr.front() = 0;
    for (size_t i = 1; i < this->row_ptr.size(); ++i) {
      this->row_ptr[i] = this->row_ptr[i - 1] + page->row_stride;
    }
    common::DispatchBinType(this->index.GetBinTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto get_offset = [&](auto bin_idx, auto fidx) {
        return static_cast<T>(bin_idx - offsets[fidx]);
      };
      common::Span<T> index_data_span = {this->index.data<T>(), this->index.Size()};
      common::ParallelFor(page->Size(), n_threads, [&](auto i) {
        auto tid = omp_get_thread_num();
        size_t ibegin = page->row_stride * i;
        for (size_t j = 0; j < page->row_stride; ++j) {
          auto bin_idx = data_buf[ibegin + j];
          index_data_span[ibegin + j] = get_offset(bin_idx, j);
          ++hit_count_tloc[tid * n_bins_total + bin_idx];
        }
      });
    });
  } else {
    std::vector<size_t> row_size(page->Size() + 1, 0);
    auto const kNull = static_cast<bst_bin_t>(page->GetDeviceAccessor(ctx->gpu_id).NullValue());
    common::ParallelFor(page->Size(), ctx->Threads(), [&](auto i) {
      size_t ibegin = page->row_stride * i;
      for (size_t j = 0; j < page->row_stride; ++j) {
        auto bin_idx = data_buf[ibegin + j];
        if (bin_idx != kNull) {
          row_size[i + 1]++;
        }
      }
    });
    std::partial_sum(row_size.begin(), row_size.end(), row_size.begin());
    this->row_ptr = std::move(row_size);
    common::Span<uint32_t> index_data_span = {this->index.data<uint32_t>(), this->index.Size()};
    common::ParallelFor(page->Size(), n_threads, [&](auto i) {
      auto tid = omp_get_thread_num();
      size_t in_rbegin = page->row_stride * i;
      size_t out_rbegin = this->row_ptr[i];
      auto r_size = this->row_ptr[i + 1] - this->row_ptr[i];
      for (size_t j = 0; j < r_size; ++j) {
        auto bin_idx = data_buf[in_rbegin + j];
        assert(bin_idx != kNull);
        index_data_span[out_rbegin + j] = bin_idx;
        ++hit_count_tloc[tid * n_bins_total + bin_idx];
      }
    });
  }

  this->hit_count.resize(n_bins_total, 0);
  common::ParallelFor(n_bins_total, ctx->Threads(), [&](auto idx) {
    for (int32_t tid = 0; tid < n_threads; ++tid) {
      this->hit_count[idx] += hit_count_tloc[tid * n_bins_total + idx];
      hit_count_tloc[tid * n_bins_total + idx] = 0;  // reset for next batch
    }
  });

  CHECK_EQ(this->Features(), info.num_col_);
  CHECK_EQ(this->Size(), info.num_row_);
  CHECK(this->cut.cut_ptrs_.HostCanRead());
  CHECK(this->cut.cut_values_.HostCanRead());
  CHECK(this->cut.min_vals_.HostCanRead());

  this->columns_ = std::make_unique<common::ColumnMatrix>(*this, p.sparse_thresh);
  this->columns_->PushBatch(ctx->Threads(), *this);
}
}  // namespace xgboost
