/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \brief Data type for fast histogram aggregation.
 */
#include "gradient_index.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "../common/threading_utils.h"

namespace xgboost {

GHistIndexMatrix::GHistIndexMatrix() : columns_{std::make_unique<common::ColumnMatrix>()} {}

GHistIndexMatrix::GHistIndexMatrix(DMatrix *x, int32_t max_bin, double sparse_thresh,
                                   bool sorted_sketch, int32_t n_threads,
                                   common::Span<float> hess) {
  this->Init(x, max_bin, sparse_thresh, sorted_sketch, n_threads, hess);
}

GHistIndexMatrix::~GHistIndexMatrix() = default;

void GHistIndexMatrix::PushBatch(SparsePage const &batch,
                                 common::Span<FeatureType const> ft,
                                 size_t rbegin, size_t prev_sum, uint32_t nbins,
                                 int32_t n_threads) {
  // The number of threads is pegged to the batch size. If the OMP
  // block is parallelized on anything other than the batch/block size,
  // it should be reassigned
  const size_t batch_threads =
      std::max(static_cast<size_t>(1), std::min(batch.Size(), static_cast<size_t>(n_threads)));
  auto page = batch.GetView();
  common::MemStackAllocator<size_t, 128> partial_sums(batch_threads);

  size_t block_size = batch.Size() / batch_threads;

  dmlc::OMPException exc;
#pragma omp parallel num_threads(batch_threads)
  {
#pragma omp for
    for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
      exc.Run([&]() {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? batch.Size()
                                                  : (block_size * (tid + 1)));

        size_t running_sum = 0;
        for (size_t ridx = ibegin; ridx < iend; ++ridx) {
          running_sum += page[ridx].size();
          row_ptr[rbegin + 1 + ridx] = running_sum;
        }
      });
    }

#pragma omp single
    {
      exc.Run([&]() {
        partial_sums[0] = prev_sum;
        for (size_t i = 1; i < batch_threads; ++i) {
          partial_sums[i] = partial_sums[i - 1] + row_ptr[rbegin + i * block_size];
        }
      });
    }

#pragma omp for
    for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
      exc.Run([&]() {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? batch.Size()
                                                  : (block_size * (tid + 1)));

        for (size_t i = ibegin; i < iend; ++i) {
          row_ptr[rbegin + 1 + i] += partial_sums[tid];
        }
      });
    }
  }
  exc.Rethrow();

  const size_t n_index = row_ptr[rbegin + batch.Size()];  // number of entries in this page
  ResizeIndex(n_index, isDense_);

  CHECK_GT(cut.Values().size(), 0U);

  if (isDense_) {
    index.SetBinOffset(cut.Ptrs());
  }
  uint32_t const *offsets = index.Offset();

  if (isDense_) {
    // Inside the lambda functions, bin_idx is the index for cut value across all
    // features. By subtracting it with starting pointer of each feature, we can reduce
    // it to smaller value and compress it to smaller types.
    common::BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == common::kUint8BinsTypeSize) {
      common::Span<uint8_t> index_data_span = {index.data<uint8_t>(), n_index};
      SetIndexData(index_data_span, ft, batch_threads, batch, rbegin, nbins,
                   [offsets](auto bin_idx, auto fidx) {
                     return static_cast<uint8_t>(bin_idx - offsets[fidx]);
                   });
    } else if (curent_bin_size == common::kUint16BinsTypeSize) {
      common::Span<uint16_t> index_data_span = {index.data<uint16_t>(), n_index};
      SetIndexData(index_data_span, ft, batch_threads, batch, rbegin, nbins,
                   [offsets](auto bin_idx, auto fidx) {
                     return static_cast<uint16_t>(bin_idx - offsets[fidx]);
                   });
    } else {
      CHECK_EQ(curent_bin_size, common::kUint32BinsTypeSize);
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
      SetIndexData(index_data_span, ft, batch_threads, batch, rbegin, nbins,
                   [offsets](auto bin_idx, auto fidx) {
                     return static_cast<uint32_t>(bin_idx - offsets[fidx]);
                   });
    }
  } else {
    /* For sparse DMatrix we have to store index of feature for each bin
       in index field to chose right offset. So offset is nullptr and index is
       not reduced */
    common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
    SetIndexData(index_data_span, ft, batch_threads, batch, rbegin, nbins,
                 [](auto idx, auto) { return idx; });
  }

  common::ParallelFor(nbins, n_threads, [&](bst_omp_uint idx) {
    for (int32_t tid = 0; tid < n_threads; ++tid) {
      hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
      hit_count_tloc_[tid * nbins + idx] = 0;  // reset for next batch
    }
  });
}

void GHistIndexMatrix::Init(DMatrix *p_fmat, int max_bins, double sparse_thresh, bool sorted_sketch,
                            int32_t n_threads, common::Span<float> hess) {
  // We use sorted sketching for approx tree method since it's more efficient in
  // computation time (but higher memory usage).
  cut = common::SketchOnDMatrix(p_fmat, max_bins, n_threads, sorted_sketch, hess);

  max_num_bins = max_bins;
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(n_threads * nbins, 0);

  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;
  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;
  auto ft = p_fmat->Info().feature_types.ConstHostSpan();

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    this->PushBatch(batch, ft, rbegin, prev_sum, nbins, n_threads);
    prev_sum = row_ptr[rbegin + batch.Size()];
    rbegin += batch.Size();
  }
  this->columns_ = std::make_unique<common::ColumnMatrix>();

  // hessian is empty when hist tree method is used or when dataset is empty
  if (hess.empty() && !std::isnan(sparse_thresh)) {
    // hist
    CHECK(!sorted_sketch);
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      this->columns_->Init(page, *this, sparse_thresh, n_threads);
    }
  }
}

void GHistIndexMatrix::Init(SparsePage const &batch, common::Span<FeatureType const> ft,
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

  size_t rbegin = 0;
  size_t prev_sum = 0;

  this->PushBatch(batch, ft, rbegin, prev_sum, nbins, n_threads);
  this->columns_ = std::make_unique<common::ColumnMatrix>();
  if (!std::isnan(sparse_thresh)) {
    this->columns_->Init(batch, *this, sparse_thresh, n_threads);
  }
}

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

bool GHistIndexMatrix::ReadColumnPage(dmlc::SeekStream *fi) {
  return this->columns_->Read(fi, this->cut.Ptrs().data());
}

size_t GHistIndexMatrix::WriteColumnPage(dmlc::Stream *fo) const {
  return this->columns_->Write(fo);
}
}  // namespace xgboost
