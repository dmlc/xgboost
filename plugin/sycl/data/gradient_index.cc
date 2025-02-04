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

template <typename BinIdxType>
void GHistIndexMatrix::SetIndexData(::sycl::queue qu,
                                    BinIdxType* index_data,
                                    const DeviceMatrix &dmat,
                                    size_t nbins,
                                    size_t row_stride,
                                    uint32_t* offsets) {
  if (nbins == 0) return;
  const xgboost::Entry *data_ptr = dmat.data.DataConst();
  const bst_idx_t *offset_vec = dmat.row_ptr.DataConst();
  const size_t num_rows = dmat.row_ptr.Size() - 1;
  const bst_float* cut_values = cut_device.Values().DataConst();
  const uint32_t* cut_ptrs = cut_device.Ptrs().DataConst();
  size_t* hit_count_ptr = hit_count_buff.Data();

  // Sparse case only
  if (!offsets) {
    // sort_buff has type uint8_t
    sort_buff.Resize(&qu, num_rows * row_stride * sizeof(BinIdxType));
  }
  BinIdxType* sort_data = reinterpret_cast<BinIdxType*>(sort_buff.Data());

  auto event = qu.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      const size_t ibegin = offset_vec[i];
      const size_t iend = offset_vec[i + 1];
      const size_t size = iend - ibegin;
      const size_t start = i * row_stride;
      for (bst_uint j = 0; j < size; ++j) {
        uint32_t idx = SearchBin(cut_values, cut_ptrs, data_ptr[ibegin + j]);
        index_data[start + j] = offsets ? idx - offsets[j] : idx;
        AtomicRef<size_t> hit_count_ref(hit_count_ptr[idx]);
        hit_count_ref.fetch_add(1);
      }
      if (!offsets) {
        // Sparse case only
        mergeSort<BinIdxType>(index_data + start, index_data + start + size, sort_data + start);
        for (bst_uint j = size; j < row_stride; ++j) {
          index_data[start + j] = nbins;
        }
      }
    });
  });
  qu.memcpy(hit_count.data(), hit_count_ptr, nbins * sizeof(size_t), event);
  qu.wait();
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

void GHistIndexMatrix::Init(::sycl::queue qu,
                            Context const * ctx,
                            const DeviceMatrix& p_fmat_device,
                            int max_bins) {
  nfeatures = p_fmat_device.p_mat->Info().num_col_;

  cut = xgboost::common::SketchOnDMatrix(ctx, p_fmat_device.p_mat, max_bins);
  cut_device.Init(qu, cut);

  max_num_bins = max_bins;
  const uint32_t nbins = cut.Ptrs().back();
  this->nbins = nbins;
  hit_count.resize(nbins, 0);
  hit_count_buff.Resize(&qu, nbins, 0);

  this->p_fmat = p_fmat_device.p_mat;
  const bool isDense = p_fmat_device.p_mat->IsDense();
  this->isDense_ = isDense;

  index.setQueue(qu);

  row_stride = 0;
  for (const auto& batch : p_fmat_device.p_mat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }

  const size_t n_offsets = cut_device.Ptrs().Size() - 1;
  const size_t n_rows = p_fmat_device.row_ptr.Size() - 1;
  const size_t n_index = n_rows * row_stride;
  ResizeIndex(n_index, isDense);

  CHECK_GT(cut_device.Values().Size(), 0U);

  uint32_t* offsets = nullptr;
  if (isDense) {
    index.ResizeOffset(n_offsets);
    offsets = index.Offset();
    qu.memcpy(offsets, cut_device.Ptrs().DataConst(),
              sizeof(uint32_t) * n_offsets).wait_and_throw();
  }

  if (isDense) {
    BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == BinTypeSize::kUint8BinsTypeSize) {
      SetIndexData(qu, index.data<uint8_t>(), p_fmat_device, nbins, row_stride, offsets);

    } else if (curent_bin_size == BinTypeSize::kUint16BinsTypeSize) {
      SetIndexData(qu, index.data<uint16_t>(), p_fmat_device, nbins, row_stride, offsets);
    } else {
      CHECK_EQ(curent_bin_size, BinTypeSize::kUint32BinsTypeSize);
      SetIndexData(qu, index.data<uint32_t>(), p_fmat_device, nbins, row_stride, offsets);
    }
  /* For sparse DMatrix we have to store index of feature for each bin
     in index field to chose right offset. So offset is nullptr and index is not reduced */
  } else {
    SetIndexData(qu, index.data<uint32_t>(), p_fmat_device, nbins, row_stride, offsets);
  }
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
