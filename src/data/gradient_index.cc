/*!
 * Copyright 2017-2021 by Contributors
 * \brief Data type for fast histogram aggregation.
 */
#include <algorithm>
#include <limits>
#include "gradient_index.h"
#include "../common/hist_util.h"

namespace xgboost {

void GHistIndexMatrix::PushBatch(SparsePage const &batch, size_t rbegin,
                                 size_t prev_sum, uint32_t nbins,
                                 int32_t n_threads) {
  // The number of threads is pegged to the batch size. If the OMP
  // block is parallelized on anything other than the batch/block size,
  // it should be reassigned
  const size_t batch_threads =
      std::max(size_t(1), std::min(batch.Size(),
                                   static_cast<size_t>(n_threads)));
  auto page = batch.GetView();
  common::MemStackAllocator<size_t, 128> partial_sums(batch_threads);
  size_t *p_part = partial_sums.Get();

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

        size_t sum = 0;
        for (size_t i = ibegin; i < iend; ++i) {
          sum += page[i].size();
          row_ptr[rbegin + 1 + i] = sum;
        }
      });
    }

#pragma omp single
    {
      exc.Run([&]() {
        p_part[0] = prev_sum;
        for (size_t i = 1; i < batch_threads; ++i) {
          p_part[i] = p_part[i - 1] + row_ptr[rbegin + i * block_size];
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
          row_ptr[rbegin + 1 + i] += p_part[tid];
        }
      });
    }
  }
  exc.Rethrow();

  const size_t n_offsets = cut.Ptrs().size() - 1;
  const size_t n_index = row_ptr[rbegin + batch.Size()];
  ResizeIndex(n_index, isDense_);

  CHECK_GT(cut.Values().size(), 0U);

  uint32_t *offsets = nullptr;
  if (isDense_) {
    index.ResizeOffset(n_offsets);
    offsets = index.Offset();
    for (size_t i = 0; i < n_offsets; ++i) {
      offsets[i] = cut.Ptrs()[i];
    }
  }

  if (isDense_) {
    common::BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == common::kUint8BinsTypeSize) {
      common::Span<uint8_t> index_data_span = {index.data<uint8_t>(), n_index};
      SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                   [offsets](auto idx, auto j) {
                     return static_cast<uint8_t>(idx - offsets[j]);
                   });

    } else if (curent_bin_size == common::kUint16BinsTypeSize) {
      common::Span<uint16_t> index_data_span = {index.data<uint16_t>(),
                                                n_index};
      SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                   [offsets](auto idx, auto j) {
                     return static_cast<uint16_t>(idx - offsets[j]);
                   });
    } else {
      CHECK_EQ(curent_bin_size, common::kUint32BinsTypeSize);
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(),
                                                n_index};
      SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                   [offsets](auto idx, auto j) {
                     return static_cast<uint32_t>(idx - offsets[j]);
                   });
    }

    /* For sparse DMatrix we have to store index of feature for each bin
       in index field to chose right offset. So offset is nullptr and index is
       not reduced */
  } else {
    common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
    SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                 [](auto idx, auto) { return idx; });
  }

  common::ParallelFor(bst_omp_uint(nbins), n_threads, [&](bst_omp_uint idx) {
    for (int32_t tid = 0; tid < n_threads; ++tid) {
      hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
      hit_count_tloc_[tid * nbins + idx] = 0;  // reset for next batch
    }
  });
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_bins, common::Span<float> hess) {
  cut = common::SketchOnDMatrix(p_fmat, max_bins, hess);

  max_num_bins = max_bins;
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);

  this->p_fmat = p_fmat;
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

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    this->PushBatch(batch, rbegin, prev_sum, nbins, nthread);
    prev_sum = row_ptr[rbegin + batch.Size()];
    rbegin += batch.Size();
  }
}

void GHistIndexMatrix::Init(SparsePage const &batch,
                            common::HistogramCuts const &cuts,
                            int32_t max_bins_per_feat, bool isDense,
                            int32_t n_threads) {
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

  this->PushBatch(batch, rbegin, prev_sum, nbins, n_threads);
}

void GHistIndexMatrix::ResizeIndex(const size_t n_index,
                                   const bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    index.SetBinTypeSize(common::kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max())  &&
    max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) && isDense) {
    index.SetBinTypeSize(common::kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(common::kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}
}  // namespace xgboost
