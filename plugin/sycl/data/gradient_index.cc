/*!
 * Copyright 2017-2024 by Contributors
 * \file gradient_index.cc
 */
#include <vector>
#include <limits>
#include <algorithm>

#include "gradient_index.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

uint32_t SearchBin(const bst_float* cut_values, const uint32_t* cut_ptrs, Entry const& e) {
  auto beg = cut_ptrs[e.index];
  auto end = cut_ptrs[e.index + 1];
  auto it = std::upper_bound(cut_values + beg, cut_values + end, e.fvalue);
  uint32_t idx = it - cut_values;
  if (idx == end) {
    idx -= 1;
  }
  return idx;
}

template <typename BinIdxType>
void mergeSort(BinIdxType* begin, BinIdxType* end, BinIdxType* buf) {
  const size_t total_len = end - begin;
  for (size_t block_len = 1; block_len < total_len; block_len <<= 1) {
    for (size_t cur_block = 0; cur_block + block_len < total_len; cur_block += 2 * block_len) {
      size_t start = cur_block;
      size_t mid = start + block_len;
      size_t finish = mid + block_len < total_len ? mid + block_len : total_len;
      size_t left_pos = start;
      size_t right_pos = mid;
      size_t pos = start;
      while (left_pos < mid || right_pos < finish) {
        if (left_pos < mid && (right_pos == finish || begin[left_pos] < begin[right_pos])) {
          buf[pos++] = begin[left_pos++];
        } else {
          buf[pos++] = begin[right_pos++];
        }
      }
      for (size_t i = start; i < finish; i++) begin[i] = buf[i];
    }
  }
}

template <typename BinIdxType, bool isDense>
void GHistIndexMatrix::SetIndexData(::sycl::queue* qu,
                                    BinIdxType* index_data,
                                    DMatrix *dmat,
                                    size_t nbins,
                                    size_t row_stride) {
  if (nbins == 0) return;
  const bst_float* cut_values = cut.cut_values_.ConstDevicePointer();
  const uint32_t* cut_ptrs = cut.cut_ptrs_.ConstDevicePointer();
  size_t* hit_count_ptr = hit_count.DevicePointer();

  BinIdxType* sort_data = reinterpret_cast<BinIdxType*>(sort_buff.Data());

  ::sycl::event event;
  for (auto &batch : dmat->GetBatches<SparsePage>()) {
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const xgboost::Entry *data_ptr = batch.data.ConstDevicePointer();
      const bst_idx_t *offset_vec = batch.offset.ConstDevicePointer();
      size_t batch_size = batch.Size();
      if (batch_size > 0) {
        const auto base_rowid = batch.base_rowid;
        event = qu->submit([&](::sycl::handler& cgh) {
          cgh.depends_on(event);
          cgh.parallel_for<>(::sycl::range<1>(batch_size), [=](::sycl::item<1> pid) {
          const size_t i = pid.get_id(0);
          const size_t ibegin = offset_vec[i];
          const size_t iend = offset_vec[i + 1];
          const size_t size = iend - ibegin;
          const size_t start = (i + base_rowid) * row_stride;
          for (bst_uint j = 0; j < size; ++j) {
            uint32_t idx = SearchBin(cut_values, cut_ptrs, data_ptr[ibegin + j]);
            index_data[start + j] = isDense ? idx - cut_ptrs[j] : idx;
            AtomicRef<size_t> hit_count_ref(hit_count_ptr[idx]);
            hit_count_ref.fetch_add(1);
          }
          if constexpr (!isDense) {
            // Sparse case only
            mergeSort<BinIdxType>(index_data + start, index_data + start + size, sort_data + start);
            for (bst_uint j = size; j < row_stride; ++j) {
              index_data[start + j] = nbins;
            }
          }
        });
      });
      }
    }
  }
  qu->wait();
}

void GHistIndexMatrix::ResizeIndex(size_t n_index, bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    index.SetBinTypeSize(BinTypeSize::kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max())  &&
    max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) && isDense) {
    index.SetBinTypeSize(BinTypeSize::kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(BinTypeSize::kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}

void GHistIndexMatrix::Init(::sycl::queue* qu,
                            Context const * ctx,
                            DMatrix *dmat,
                            int max_bins) {
  nfeatures = dmat->Info().num_col_;

  cut = xgboost::common::SketchOnDMatrix(ctx, dmat, max_bins);
  cut.SetDevice(ctx->Device());

  max_num_bins = max_bins;
  const uint32_t nbins = cut.Ptrs().back();
  this->nbins = nbins;

  hit_count.SetDevice(ctx->Device());
  hit_count.Resize(nbins, 0);

  this->p_fmat = dmat;
  const bool isDense = dmat->IsDense();
  this->isDense_ = isDense;

  index.setQueue(qu);

  row_stride = 0;
  size_t n_rows = 0;
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    batch.data.SetDevice(ctx->Device());
    batch.offset.SetDevice(ctx->Device());
    n_rows += batch.Size();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }

  const size_t n_offsets = cut.cut_ptrs_.Size() - 1;
  const size_t n_index = n_rows * row_stride;
  ResizeIndex(n_index, isDense);

  CHECK_GT(cut.cut_values_.Size(), 0U);

  if (isDense) {
    BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == BinTypeSize::kUint8BinsTypeSize) {
      SetIndexData<uint8_t, true>(qu, index.data<uint8_t>(), dmat, nbins, row_stride);

    } else if (curent_bin_size == BinTypeSize::kUint16BinsTypeSize) {
      SetIndexData<uint16_t, true>(qu, index.data<uint16_t>(), dmat, nbins, row_stride);
    } else {
      CHECK_EQ(curent_bin_size, BinTypeSize::kUint32BinsTypeSize);
      SetIndexData<uint32_t, true>(qu, index.data<uint32_t>(), dmat, nbins, row_stride);
    }
  /* For sparse DMatrix we have to store index of feature for each bin
     in index field to chose right offset. So offset is nullptr and index is not reduced */
  } else {
    sort_buff.Resize(qu, n_rows * row_stride * sizeof(uint32_t));
    SetIndexData<uint32_t, false>(qu, index.data<uint32_t>(), dmat, nbins, row_stride);
  }
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
