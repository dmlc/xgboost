/*!
 * Copyright 2017-2024 by Contributors
 * \file gradient_index.h
 */
#ifndef PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_
#define PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_

#include <vector>

#include "../data.h"
#include "../../src/common/hist_util.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

using BinTypeSize = ::xgboost::common::BinTypeSize;

/*!
 * \brief Index data and offsets stored in USM buffers to provide access from device kernels
 */
struct Index {
  Index() {
    SetBinTypeSize(binTypeSize_);
  }
  Index(const Index& i) = delete;
  Index& operator=(Index i) = delete;
  Index(Index&& i) = delete;
  Index& operator=(Index&& i) = delete;
  void SetBinTypeSize(BinTypeSize binTypeSize) {
    binTypeSize_ = binTypeSize;
    CHECK(binTypeSize == BinTypeSize::kUint8BinsTypeSize  ||
          binTypeSize == BinTypeSize::kUint16BinsTypeSize ||
          binTypeSize == BinTypeSize::kUint32BinsTypeSize);
  }
  BinTypeSize GetBinTypeSize() const {
    return binTypeSize_;
  }

  template<typename T>
  T* data() {
    return reinterpret_cast<T*>(data_.Data());
  }

  template<typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_.DataConst());
  }

  size_t Size() const {
    return data_.Size() / (binTypeSize_);
  }

  void Resize(::sycl::queue* qu, const size_t nBytesData) {
    data_.Resize(qu, nBytesData);
  }

  uint8_t* begin() const {
    return data_.Begin();
  }

  uint8_t* end() const {
    return data_.End();
  }

 private:
  USMVector<uint8_t, MemoryType::on_device> data_;
  BinTypeSize binTypeSize_ {BinTypeSize::kUint8BinsTypeSize};
};

/*!
 * \brief Preprocessed global index matrix, in CSR format, stored in USM buffers
 *
 *  Transform floating values to integer index in histogram
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  /*! \brief The index data */
  Index index;
  /*! \brief hit count of each index */
  HostDeviceVector<size_t> hit_count;

  USMVector<uint8_t, MemoryType::on_device> sort_buff;
  /*! \brief The corresponding cuts */
  xgboost::common::HistogramCuts cut;
  size_t max_num_bins;
  size_t min_num_bins;
  size_t nbins;
  size_t nfeatures;
  size_t row_stride;

  // Create a global histogram matrix based on a given DMatrix device wrapper
  void Init(::sycl::queue* qu, Context const * ctx,
            DMatrix *dmat, int max_num_bins);

  template <typename BinIdxType, bool isDense>
  void SetIndexData(::sycl::queue* qu, Context const * ctx, BinIdxType* index_data,
                    DMatrix *dmat);

  void ResizeIndex(::sycl::queue* qu, size_t n_index);

  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.cut_ptrs_.Size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.cut_ptrs_.ConstHostVector()[fid];
      auto iend = cut.cut_ptrs_.ConstHostVector()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        *(counts + fid) += hit_count.ConstHostVector()[i];
      }
    }
  }
  inline bool IsDense() const {
    return isDense_;
  }

 private:
  bool isDense_;
};

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_
