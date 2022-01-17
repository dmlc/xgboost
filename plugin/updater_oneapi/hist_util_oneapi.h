/*!
 * Copyright 2017-2021 by Contributors
 * \file hist_util_oneapi.h
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_
#define XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_

#include <vector>

#include "data_oneapi.h"
#include "row_set_oneapi.h"

#include "../../src/common/hist_util.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace common {

template<typename GradientSumT>
using GHistRowOneAPI = USMVector<xgboost::detail::GradientPairInternal<GradientSumT> >;

/*!
 * \brief OneAPI implementation of HistogramCuts stored in USM buffers to provide access from device kernels
 */
class HistogramCutsOneAPI {
protected:
  using BinIdx = uint32_t;

public:
  HistogramCutsOneAPI() {}

  HistogramCutsOneAPI(sycl::queue qu) {
    cut_ptrs_.Resize(qu_, 1, 0);
  }

  ~HistogramCutsOneAPI() {
  }

  void Init(sycl::queue qu, HistogramCuts const& cuts) {
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
  sycl::queue qu_;
};

/*!
 * \brief Index data and offsets stored in USM buffers to provide access from device kernels
 */
struct IndexOneAPI {
  IndexOneAPI() {
    SetBinTypeSize(binTypeSize_);
  }
  IndexOneAPI(const IndexOneAPI& i) = delete;
  IndexOneAPI& operator=(IndexOneAPI i) = delete;
  IndexOneAPI(IndexOneAPI&& i) = delete;
  IndexOneAPI& operator=(IndexOneAPI&& i) = delete;
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
      case kUint8BinsTypeSize:
        func_ = &GetValueFromUint8;
        break;
      case kUint16BinsTypeSize:
        func_ = &GetValueFromUint16;
        break;
      case kUint32BinsTypeSize:
        func_ = &GetValueFromUint32;
        break;
      default:
        CHECK(binTypeSize == kUint8BinsTypeSize  ||
              binTypeSize == kUint16BinsTypeSize ||
              binTypeSize == kUint32BinsTypeSize);
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

  void setQueue(sycl::queue qu) {
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

  USMVector<uint8_t> data_;
  USMVector<uint32_t> offset_;  // size of this field is equal to number of features
  BinTypeSize binTypeSize_ {kUint8BinsTypeSize};
  size_t p_ {1};
  Func func_;

  sycl::queue qu_;
};


/*!
 * \brief Preprocessed global index matrix, in CSR format, stored in USM buffers
 *
 *  Transform floating values to integer index in histogram
 */
struct GHistIndexMatrixOneAPI {
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  USMVector<size_t> row_ptr_device;
  /*! \brief The index data */
  IndexOneAPI index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  HistogramCuts cut;
  HistogramCutsOneAPI cut_device;
  DMatrix* p_fmat;
  size_t max_num_bins;
  size_t nbins;
  size_t nfeatures;
  size_t row_stride;

  // Create a global histogram matrix based on a given DMatrix device wrapper
  void Init(sycl::queue qu, const DeviceMatrixOneAPI& p_fmat_device, int max_num_bins);

  template <typename BinIdxType>
  void SetIndexData(sycl::queue qu, common::Span<BinIdxType> index_data_span,
                    const DeviceMatrixOneAPI &dmat_device,
                    size_t nbins, size_t row_stride, uint32_t* offsets);

  void ResizeIndex(const size_t n_offsets, const size_t n_index,
                   const bool isDense);

  inline void GetFeatureCounts(std::vector<size_t>& counts) const {
    auto nfeature = cut_device.Ptrs().Size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut_device.Ptrs()[fid];
      auto iend = cut_device.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }
  inline bool IsDense() const {
    return isDense_;
  }

 private:
  bool isDense_;
};

class ColumnMatrixOneAPI;

/*!
 * \brief Fill histogram with zeroes
 */
template<typename GradientSumT>
void InitHist(sycl::queue qu,
              GHistRowOneAPI<GradientSumT>& hist,
              size_t size);

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(sycl::queue qu,
              GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src,
              size_t size);

/*!
 * \brief Compute subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
void SubtractionHist(sycl::queue qu,
                     GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src1,
                     const GHistRowOneAPI<GradientSumT>& src2,
                     size_t size);

/*!
 * \brief Histograms of gradient statistics for multiple nodes
 */
template<typename GradientSumT>
class HistCollectionOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  // Access histogram for i-th node
  GHistRowT& operator[](bst_uint nid) {
    return data_[nid];
  }

  const GHistRowT& operator[](bst_uint nid) const {
    return data_[nid];
  }

  // Initialize histogram collection
  void Init(sycl::queue qu, uint32_t nbins) {
    qu_ = qu;
    if (nbins_ != nbins) {
      nbins_ = nbins;
      data_.clear();
    }
  }

  // Create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
    if (nid >= data_.size()) {
      data_.resize(nid + 1);
    }
    data_[nid].Resize(qu_, nbins_, xgboost::detail::GradientPairInternal<GradientSumT>(0, 0));
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ = 0;

  std::vector<GHistRowT> data_;

  sycl::queue qu_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 */
template<typename GradientSumT>
class ParallelGHistBuilderOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  void Init(sycl::queue qu, size_t nbins) {
    qu_ = qu;
    if (nbins != nbins_) {
      hist_buffer_.Init(qu_, nbins);
      nbins_ = nbins;
    }
  }

  void Reset(size_t nblocks) {
    hist_device_buffer_.Resize(qu_, nblocks * nbins_ * 2);
  }

  GHistRowT& GetDeviceBuffer() {
    return hist_device_buffer_;
  }

 protected:
  /*! \brief Number of bins in each histogram */
  size_t nbins_ = 0;
  /*! \brief Buffers for histograms for all nodes processed */
  HistCollectionOneAPI<GradientSumT> hist_buffer_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  GHistRowT hist_device_buffer_;

  sycl::queue qu_;
};

/*!
 * \brief Builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilderOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  GHistBuilderOneAPI() = default;
  GHistBuilderOneAPI(sycl::queue qu, uint32_t nbins) : qu_{qu}, nbins_{nbins} {}

  // Construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const USMVector<GradientPair>& gpair_device,
                 const RowSetCollectionOneAPI::Elem& row_indices,
                 const GHistIndexMatrixOneAPI& gmat,
                 GHistRowT& hist,
                 bool isDense,
                 GHistRowT& hist_buffer);

  // Construct a histogram via subtraction trick
  void SubtractionTrick(GHistRowT& self,
                        GHistRowT& sibling,
                        GHistRowT& parent);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ { 0 };

  sycl::queue qu_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_
