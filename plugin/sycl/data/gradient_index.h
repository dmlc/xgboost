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
    switch (binTypeSize) {
      case BinTypeSize::kUint8BinsTypeSize:
        func_ = &GetValueFromUint8;
        break;
      case BinTypeSize::kUint16BinsTypeSize:
        func_ = &GetValueFromUint16;
        break;
      case BinTypeSize::kUint32BinsTypeSize:
        func_ = &GetValueFromUint32;
        break;
      default:
        CHECK(binTypeSize == BinTypeSize::kUint8BinsTypeSize  ||
              binTypeSize == BinTypeSize::kUint16BinsTypeSize ||
              binTypeSize == BinTypeSize::kUint32BinsTypeSize);
    }
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

  void Resize(const size_t nBytesData) {
    data_.Resize(qu_, nBytesData);
  }

  uint8_t* begin() const {
    return data_.Begin();
  }

  uint8_t* end() const {
    return data_.End();
  }

  void setQueue(::sycl::queue* qu) {
    qu_ = qu;
  }

 private:
  static uint32_t GetValueFromUint8(const uint8_t* t, size_t i) {
    return reinterpret_cast<const uint8_t*>(t)[i];
  }
  static uint32_t GetValueFromUint16(const uint8_t* t, size_t i) {
    return reinterpret_cast<const uint16_t*>(t)[i];
  }
  static uint32_t GetValueFromUint32(const uint8_t* t, size_t i) {
    return reinterpret_cast<const uint32_t*>(t)[i];
  }

  using Func = uint32_t (*)(const uint8_t*, size_t);

  USMVector<uint8_t, MemoryType::on_device> data_;
  BinTypeSize binTypeSize_ {BinTypeSize::kUint8BinsTypeSize};
  Func func_;

  ::sycl::queue* qu_;
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
  DMatrix* p_fmat;
  size_t max_num_bins;
  size_t nbins;
  size_t nfeatures;
  size_t row_stride;

  // Create a global histogram matrix based on a given DMatrix device wrapper
  void Init(::sycl::queue* qu, Context const * ctx,
            DMatrix *dmat, int max_num_bins);

  template <typename BinIdxType, bool isDense>
  void SetIndexData(::sycl::queue* qu, BinIdxType* index_data,
                    DMatrix *dmat,
                    size_t nbins, size_t row_stride);

  void ResizeIndex(size_t n_index, bool isDense);

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
