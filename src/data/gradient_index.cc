/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \brief Data type for fast histogram aggregation.
 */
#include "gradient_index.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>  // std::forward

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "../common/numeric.h"
#include "../common/threading_utils.h"

namespace xgboost {

GHistIndexMatrix::GHistIndexMatrix() : columns_{std::make_unique<common::ColumnMatrix>()} {}

GHistIndexMatrix::GHistIndexMatrix(DMatrix *p_fmat, bst_bin_t max_bins_per_feat,
                                   double sparse_thresh, bool sorted_sketch, int32_t n_threads,
                                   common::Span<float> hess) {
  CHECK(p_fmat->SingleColBlock());
  // We use sorted sketching for approx tree method since it's more efficient in
  // computation time (but higher memory usage).
  cut = common::SketchOnDMatrix(p_fmat, max_bins_per_feat, n_threads, sorted_sketch, hess);

  max_num_bins = max_bins_per_feat;
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(n_threads * nbins, 0);

  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;
  auto ft = p_fmat->Info().feature_types.ConstHostSpan();

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    this->PushBatch(batch, ft, n_threads);
  }
  this->columns_ = std::make_unique<common::ColumnMatrix>();

  // hessian is empty when hist tree method is used or when dataset is empty
  if (hess.empty() && !std::isnan(sparse_thresh)) {
    // hist
    CHECK(!sorted_sketch);
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      this->columns_->InitFromSparse(page, *this, sparse_thresh, n_threads);
    }
  }
}

GHistIndexMatrix::GHistIndexMatrix(MetaInfo const &info, common::HistogramCuts &&cuts,
                                   bst_bin_t max_bin_per_feat)
    : row_ptr(info.num_row_ + 1, 0),
      hit_count(cuts.TotalBins(), 0),
      cut{std::forward<common::HistogramCuts>(cuts)},
      max_num_bins(max_bin_per_feat),
      isDense_{info.num_col_ * info.num_row_ == info.num_nonzero_} {}

#if !defined(XGBOOST_USE_CUDA)
GHistIndexMatrix::GHistIndexMatrix(Context const *, MetaInfo const &, EllpackPage const &,
                                   BatchParam const &) {
  common::AssertGPUSupport();
}
#endif  // defined(XGBOOST_USE_CUDA)

GHistIndexMatrix::~GHistIndexMatrix() = default;

void GHistIndexMatrix::PushBatch(SparsePage const &batch, common::Span<FeatureType const> ft,
                                 int32_t n_threads) {
  auto page = batch.GetView();
  auto it = common::MakeIndexTransformIter([&](size_t ridx) { return page[ridx].size(); });
  common::PartialSum(n_threads, it, it + page.Size(), static_cast<size_t>(0), row_ptr.begin());
  data::SparsePageAdapterBatch adapter_batch{page};
  auto is_valid = [](auto) { return true; };  // SparsePage always contains valid entries
  PushBatchImpl(n_threads, adapter_batch, 0, is_valid, ft);
}

GHistIndexMatrix::GHistIndexMatrix(SparsePage const &batch, common::Span<FeatureType const> ft,
                                   common::HistogramCuts const &cuts, int32_t max_bins_per_feat,
                                   bool isDense, double sparse_thresh, int32_t n_threads) {
  CHECK_GE(n_threads, 1);
  base_rowid = batch.base_rowid;
  isDense_ = isDense;
  cut = cuts;
  max_num_bins = max_bins_per_feat;
  CHECK_EQ(row_ptr.size(), 0);
  // The number of threads is pegged to the batch size. If the OMP
  // block is parallelized on anything other than the batch/block size,
  // it should be reassigned
  row_ptr.resize(batch.Size() + 1, 0);
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(n_threads * nbins, 0);

  this->PushBatch(batch, ft, n_threads);
  this->columns_ = std::make_unique<common::ColumnMatrix>();
  if (!std::isnan(sparse_thresh)) {
    this->columns_->InitFromSparse(batch, *this, sparse_thresh, n_threads);
  }
}

template <typename Batch>
void GHistIndexMatrix::PushAdapterBatchColumns(Context const *ctx, Batch const &batch,
                                               float missing, size_t rbegin) {
  CHECK(columns_);
  this->columns_->PushBatch(ctx->Threads(), batch, missing, *this, rbegin);
}

#define INSTANTIATION_PUSH(BatchT)                                 \
  template void GHistIndexMatrix::PushAdapterBatchColumns<BatchT>( \
      Context const *ctx, BatchT const &batch, float missing, size_t rbegin);

INSTANTIATION_PUSH(data::CSRArrayAdapterBatch)
INSTANTIATION_PUSH(data::ArrayAdapterBatch)
INSTANTIATION_PUSH(data::SparsePageAdapterBatch)

#undef INSTANTIATION_PUSH

void GHistIndexMatrix::ResizeIndex(const size_t n_index, const bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    // compress dense index to uint8
    index.SetBinTypeSize(common::kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max()) &&
              max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) &&
             isDense) {
    // compress dense index to uint16
    index.SetBinTypeSize(common::kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(common::kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}

common::ColumnMatrix const &GHistIndexMatrix::Transpose() const {
  CHECK(columns_);
  return *columns_;
}

float GHistIndexMatrix::GetFvalue(size_t ridx, size_t fidx, bool is_cat) const {
  auto const &values = cut.Values();
  auto const &mins = cut.MinValues();
  auto const &ptrs = cut.Ptrs();
  if (is_cat) {
    auto f_begin = ptrs[fidx];
    auto f_end = ptrs[fidx + 1];
    auto begin = RowIdx(ridx);
    auto end = RowIdx(ridx + 1);
    auto gidx = BinarySearchBin(begin, end, index, f_begin, f_end);
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return values[gidx];
  }

  auto get_bin_val = [&](auto &column) {
    auto bin_idx = column[ridx];
    if (bin_idx == common::DenseColumnIter<uint8_t, true>::kMissingId) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return common::HistogramCuts::NumericBinValue(ptrs, values, mins, fidx, bin_idx);
  };

  if (columns_->GetColumnType(fidx) == common::kDenseColumn) {
    if (columns_->AnyMissing()) {
      return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
        auto column = columns_->DenseColumn<decltype(dtype), true>(fidx);
        return get_bin_val(column);
      });
    } else {
      return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
        auto column = columns_->DenseColumn<decltype(dtype), false>(fidx);
        return get_bin_val(column);
      });
    }
  } else {
    return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
      auto column = columns_->SparseColumn<decltype(dtype)>(fidx, 0);
      return get_bin_val(column);
    });
  }

  SPAN_CHECK(false);
  return std::numeric_limits<float>::quiet_NaN();
}

bool GHistIndexMatrix::ReadColumnPage(dmlc::SeekStream *fi) {
  return this->columns_->Read(fi, this->cut.Ptrs().data());
}

size_t GHistIndexMatrix::WriteColumnPage(dmlc::Stream *fo) const {
  return this->columns_->Write(fo);
}
}  // namespace xgboost
