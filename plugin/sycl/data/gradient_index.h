/*!
 * Copyright 2017-2024 by Contributors
 * \file gradient_index.h
 */
#ifndef PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_
#define PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_

#include <vector>

#include "../data.h"
#include "../../src/common/hist_util.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

/*!
 * \brief SYCL implementation of HistogramCuts stored in USM buffers to provide access from device kernels
 */
class HistogramCuts {
 protected:
  using BinIdx = uint32_t;

 public:
  HistogramCuts() {}

  explicit HistogramCuts(::sycl::queue* qu) {}

  ~HistogramCuts() {
  }

  void Init(::sycl::queue* qu, xgboost::common::HistogramCuts const& cuts) {
    qu_ = qu;
    cut_values_.Init(qu_, cuts.cut_values_.HostVector());
    cut_ptrs_.Init(qu_, cuts.cut_ptrs_.HostVector());
    min_vals_.Init(qu_, cuts.min_vals_.HostVector());
  }

  // Getters for USM buffers to pass pointers into device kernels
  const USMVector<uint32_t>& Ptrs()      const { return cut_ptrs_;   }
  const USMVector<float>&    Values()    const { return cut_values_; }
  const USMVector<float>&    MinValues() const { return min_vals_;   }

 private:
  USMVector<bst_float> cut_values_;
  USMVector<uint32_t> cut_ptrs_;
  USMVector<float> min_vals_;
  ::sycl::queue* qu_;
};

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
  uint32_t operator[](size_t i) const {
    if (!offset_.Empty()) {
      return func_(data_.DataConst(), i) + offset_[i%p_];
    } else {
      return func_(data_.DataConst(), i);
    }
  }
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

  uint32_t* Offset() {
    return offset_.Data();
  }

  const uint32_t* Offset() const {
    return offset_.DataConst();
  }

  size_t Size() const {
    return data_.Size() / (binTypeSize_);
  }

  void Resize(const size_t nBytesData) {
    data_.Resize(qu_, nBytesData);
  }

  void ResizeOffset(const size_t nDisps) {
    offset_.Resize(qu_, nDisps);
    p_ = nDisps;
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
  // size of this field is equal to number of features
  USMVector<uint32_t, MemoryType::on_device> offset_;
  BinTypeSize binTypeSize_ {BinTypeSize::kUint8BinsTypeSize};
  size_t p_ {1};
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
  std::vector<size_t> hit_count;
  /*! \brief buffers for calculations */
  USMVector<size_t, MemoryType::on_device> hit_count_buff;
  USMVector<uint8_t, MemoryType::on_device> sort_buff;
  /*! \brief The corresponding cuts */
  xgboost::common::HistogramCuts cut;
  HistogramCuts cut_device;
  DMatrix* p_fmat;
  size_t max_num_bins;
  size_t nbins;
  size_t nfeatures;
  size_t row_stride;

  // Create a global histogram matrix based on a given DMatrix device wrapper
  void Init(::sycl::queue* qu, Context const * ctx,
            const sycl::DeviceMatrix& p_fmat_device, int max_num_bins);

  template <typename BinIdxType>
  void SetIndexData(::sycl::queue* qu, BinIdxType* index_data,
                    const sycl::DeviceMatrix &dmat_device,
                    size_t nbins, size_t row_stride, uint32_t* offsets);

  void ResizeIndex(size_t n_index, bool isDense);

  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut_device.Ptrs().Size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut_device.Ptrs()[fid];
      auto iend = cut_device.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        *(counts + fid) += hit_count[i];
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
