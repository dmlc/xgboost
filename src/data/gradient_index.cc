/**
 * Copyright 2017-2025, XGBoost Contributors
 * \brief Data type for fast histogram aggregation.
 */
#include "gradient_index.h"

#include <limits>
#include <memory>
#include <utility>  // for forward

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "../common/numeric.h"
#include "../common/transform_iterator.h"  // for MakeIndexTransformIter

namespace xgboost {

GHistIndexMatrix::GHistIndexMatrix() : columns_{std::make_unique<common::ColumnMatrix>()} {}

GHistIndexMatrix::GHistIndexMatrix(Context const *ctx, DMatrix *p_fmat, bst_bin_t max_bins_per_feat,
                                   double sparse_thresh, bool sorted_sketch,
                                   common::Span<float const> hess)
    : max_numeric_bins_per_feat{max_bins_per_feat} {
  CHECK(p_fmat->SingleColBlock());
  // We use sorted sketching for approx tree method since it's more efficient in
  // computation time (but higher memory usage).
  cut = common::SketchOnDMatrix(ctx, p_fmat, max_bins_per_feat, sorted_sketch, hess);

  const uint32_t nbins = cut.Ptrs().back();
  hit_count = common::MakeFixedVecWithMalloc(nbins, std::size_t{0});
  hit_count_tloc_.resize(ctx->Threads() * nbins, 0);

  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr = common::MakeFixedVecWithMalloc(new_size, std::size_t{0});

  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;
  auto ft = p_fmat->Info().feature_types.ConstHostSpan();

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    this->PushBatch(batch, ft, ctx->Threads());
  }
  this->columns_ = std::make_unique<common::ColumnMatrix>();

  // hessian is empty when hist tree method is used or when dataset is empty
  if (hess.empty() && !std::isnan(sparse_thresh)) {
    // hist
    CHECK(!sorted_sketch);
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      this->columns_->InitFromSparse(page, *this, sparse_thresh, ctx->Threads());
    }
  }
}

GHistIndexMatrix::GHistIndexMatrix(MetaInfo const &info, common::HistogramCuts &&cuts,
                                   bst_bin_t max_bin_per_feat)
    : row_ptr{common::MakeFixedVecWithMalloc(info.num_row_ + 1, std::size_t{0})},
      hit_count{common::MakeFixedVecWithMalloc(cuts.TotalBins(), std::size_t{0})},
      cut{std::forward<common::HistogramCuts>(cuts)},
      max_numeric_bins_per_feat(max_bin_per_feat),
      isDense_{info.IsDense()} {}

GHistIndexMatrix::GHistIndexMatrix(bst_idx_t n_samples, bst_idx_t base_rowid,
                                   common::HistogramCuts &&cuts, bst_bin_t max_bin_per_feat,
                                   bool is_dense)
    : row_ptr{common::MakeFixedVecWithMalloc(n_samples + 1, std::size_t{0})},
      hit_count{common::MakeFixedVecWithMalloc(cuts.TotalBins(), std::size_t{0})},
      cut{std::forward<common::HistogramCuts>(cuts)},
      max_numeric_bins_per_feat(max_bin_per_feat),
      base_rowid{base_rowid},
      isDense_{is_dense} {}

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
  auto it = common::MakeIndexTransformIter([&](std::size_t ridx) { return page[ridx].size(); });
  common::PartialSum(n_threads, it, it + page.Size(), static_cast<size_t>(0), row_ptr.begin());
  data::SparsePageAdapterBatch adapter_batch{page};
  auto is_valid = [](auto) { return true; };  // SparsePage always contains valid entries
  PushBatchImpl(n_threads, adapter_batch, 0, is_valid, ft);
}

GHistIndexMatrix::GHistIndexMatrix(SparsePage const &batch, common::Span<FeatureType const> ft,
                                   common::HistogramCuts cuts, bst_bin_t max_bins_per_feat,
                                   bool is_dense, double sparse_thresh, std::int32_t n_threads)
    : cut{std::move(cuts)},
      max_numeric_bins_per_feat{max_bins_per_feat},
      base_rowid{batch.base_rowid},
      isDense_{is_dense} {
  CHECK_GE(n_threads, 1);
  CHECK_EQ(row_ptr.size(), 0);
  row_ptr = common::MakeFixedVecWithMalloc(batch.Size() + 1, std::size_t{0});

  const uint32_t nbins = cut.Ptrs().back();
  hit_count = common::MakeFixedVecWithMalloc(nbins, std::size_t{0});
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
INSTANTIATION_PUSH(data::ColumnarAdapterBatch)
#undef INSTANTIATION_PUSH

void GHistIndexMatrix::ResizeColumns(double sparse_thresh) {
  CHECK(!std::isnan(sparse_thresh));
  this->columns_ = std::make_unique<common::ColumnMatrix>(*this, sparse_thresh);
}

void GHistIndexMatrix::ResizeIndex(const size_t n_index, const bool isDense) {
  auto make_index = [this, n_index](auto t, common::BinTypeSize t_size) {
    // Must resize instead of allocating a new one. This function is called everytime a
    // new batch is pushed, and we grow the size accordingly without loosing the data in
    // the previous batches.
    using T = decltype(t);
    std::size_t n_bytes = sizeof(T) * n_index;
    CHECK_GE(n_bytes, this->data.size());

    auto resource = this->data.Resource();
    decltype(this->data) new_vec;
    if (!resource) {
      CHECK(this->data.empty());
      new_vec = common::MakeFixedVecWithMalloc(n_bytes, std::uint8_t{0});
    } else {
      CHECK(resource->Type() == common::ResourceHandler::kMalloc);
      auto malloc_resource = std::dynamic_pointer_cast<common::MallocResource>(resource);
      CHECK(malloc_resource);
      malloc_resource->Resize(n_bytes);

      // gcc-11.3 doesn't work if DataAs is used.
      std::uint8_t *new_ptr = reinterpret_cast<std::uint8_t *>(malloc_resource->Data());
      new_vec = {new_ptr, n_bytes / sizeof(std::uint8_t), malloc_resource};
    }
    this->data = std::move(new_vec);
    this->index = common::Index{common::Span{data.data(), static_cast<size_t>(data.size())},
        t_size};
  };

  if ((MaxNumBinPerFeat() - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) &&
      isDense) {
    // compress dense index to uint8
    make_index(std::uint8_t{}, common::kUint8BinsTypeSize);
  } else if ((MaxNumBinPerFeat() - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max()) &&
              MaxNumBinPerFeat() - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) &&
             isDense) {
    // compress dense index to uint16
    make_index(std::uint16_t{}, common::kUint16BinsTypeSize);
  } else {
    // no compression
    make_index(std::uint32_t{}, common::kUint32BinsTypeSize);
  }
}

common::ColumnMatrix const &GHistIndexMatrix::Transpose() const {
  CHECK(columns_);
  return *columns_;
}

bst_bin_t GHistIndexMatrix::GetGindex(size_t ridx, size_t fidx) const {
  auto begin = RowIdx(ridx);
  if (IsDense()) {
    return static_cast<bst_bin_t>(this->index[begin + fidx]);
  }
  auto end = RowIdx(ridx + 1);
  auto const& cut_ptrs = cut.Ptrs();
  auto f_begin = cut_ptrs[fidx];
  auto f_end = cut_ptrs[fidx + 1];
  return BinarySearchBin(begin, end, this->index, f_begin, f_end);
}

float GHistIndexMatrix::GetFvalue(size_t ridx, size_t fidx, bool is_cat) const {
  auto const &values = cut.Values();
  auto const &mins = cut.MinValues();
  auto const &ptrs = cut.Ptrs();
  return this->GetFvalue(ptrs, values, mins, ridx, fidx, is_cat);
}

float GetFvalueImpl(std::vector<std::uint32_t> const &ptrs, std::vector<float> const &values,
                    std::vector<float> const &mins, bst_idx_t ridx, bst_feature_t fidx,
                    bst_idx_t base_rowid, std::unique_ptr<common::ColumnMatrix> const &columns_) {
  auto get_bin_val = [&](auto &column) {
    auto bin_idx = column[ridx - base_rowid];
    if (bin_idx == common::DenseColumnIter<uint8_t, true>::kMissingId) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return common::HistogramCuts::NumericBinValue(ptrs, values, mins, fidx, bin_idx);
  };
  switch (columns_->GetColumnType(fidx)) {
    case common::kDenseColumn: {
      if (columns_->AnyMissing()) {
        return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
          auto column = columns_->DenseColumn<decltype(dtype), true>(fidx);
          return get_bin_val(column);
        });
      } else {
        return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
          auto column = columns_->DenseColumn<decltype(dtype), false>(fidx);
          auto bin_idx = column[ridx - base_rowid];
          return common::HistogramCuts::NumericBinValue(ptrs, values, mins, fidx, bin_idx);
        });
      }
    }
    case common::kSparseColumn: {
      return common::DispatchBinType(columns_->GetTypeSize(), [&](auto dtype) {
        auto column = columns_->SparseColumn<decltype(dtype)>(fidx, 0);
        return get_bin_val(column);
      });
    }
  }

  SPAN_CHECK(false);
  return std::numeric_limits<float>::quiet_NaN();
}

bool GHistIndexMatrix::ReadColumnPage(common::AlignedResourceReadStream *fi) {
  return this->columns_->Read(fi, this->cut.Ptrs().data());
}

std::size_t GHistIndexMatrix::WriteColumnPage(common::AlignedFileWriteStream *fo) const {
  return this->columns_->Write(fo);
}
}  // namespace xgboost
