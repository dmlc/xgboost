/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_uti_oneapi.h
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_
#define XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_

#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <map>

#include "data_oneapi.h"
#include "row_set_oneapi.h"

#include "../../src/common/row_set.h"
#include "../../src/common/threading_utils.h"
#include "../../src/tree/param.h"
#include "../../src/common/quantile.h"
#include "../../src/common/timer.h"
#include "../../src/common/hist_util.h"
#include "../../src/common/common.h"
#include "../include/rabit/rabit.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace common {

using GHistIndexRow = Span<uint32_t const>;

template<typename GradientSumT>
using GHistRowOneAPI = USMVector<xgboost::detail::GradientPairInternal<GradientSumT> >;

class HistogramCutsOneAPI {
protected:
  using BinIdx = uint32_t;

public:
  HistogramCutsOneAPI() {}

  HistogramCutsOneAPI(cl::sycl::queue qu) {
    cut_ptrs_.Resize(qu_, 1, 0);
  }

  ~HistogramCutsOneAPI() {
  }

  void Init(cl::sycl::queue qu, HistogramCuts const& cuts) {
    qu_ = qu;
    cut_values_.Init(qu_, cuts.cut_values_.HostVector());
    cut_ptrs_.Init(qu_, cuts.cut_ptrs_.HostVector());
    min_vals_.Init(qu_, cuts.min_vals_.HostVector());
  }

  uint32_t FeatureBins(uint32_t feature) const {
    return cut_ptrs_[feature + 1] - cut_ptrs_[feature];
  }

  // Getters.  Cuts should be of no use after building histogram indices, but currently
  // it's deeply linked with quantile_hist, gpu sketcher and gpu_hist.  So we preserve
  // these for now.
  const USMVector<uint32_t>& Ptrs()      const { return cut_ptrs_;   }
  const USMVector<float>&    Values()    const { return cut_values_; }
  const USMVector<float>&    MinValues() const { return min_vals_;   }

  size_t TotalBins() const { return cut_ptrs_[cut_ptrs_.Size() - 1]; }

  // Return the index of a cut point that is strictly greater than the input
  // value, or the last available index if none exists

private:
  USMVector<bst_float> cut_values_;
  USMVector<uint32_t> cut_ptrs_;
  USMVector<float> min_vals_;
  cl::sycl::queue qu_;
};

uint32_t SearchBin(bst_float* cut_values, uint32_t* cut_ptrs, float value, uint32_t column_id);

uint32_t SearchBin(bst_float* cut_values, uint32_t* cut_ptrs, EntryOneAPI const& e);

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

  size_t OffsetSize() const {
    return offset_.Size();
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
  uint8_t* begin() const {  // NOLINT
    return data_.Begin();
  }
  uint8_t* end() const {  // NOLINT
    return data_.End();
  }

  void setQueue(cl::sycl::queue qu) {
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

  cl::sycl::queue qu_;
};


/*!
 * \brief preprocessed global index matrix, in CSR format
 *
 *  Transform floating values to integer index in histogram This is a global histogram
 *  index for CPU histogram.  On GPU ellpack page is used.
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
  // Create a global histogram matrix, given cut
  void Init(cl::sycl::queue qu, const DeviceMatrixOneAPI& p_fmat_device, int max_num_bins);

  template <typename BinIdxType>
  void SetIndexData(cl::sycl::queue qu, common::Span<BinIdxType> index_data_span,
                    const DeviceMatrixOneAPI &dmat_device,
                    size_t nbins, uint32_t* offsets);

  void ResizeIndex(const size_t n_offsets, const size_t n_index,
                   const bool isDense);

  inline void GetFeatureCounts(size_t* counts) const {
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
  std::vector<size_t> hit_count_tloc_;
  bool isDense_;
};

class ColumnMatrixOneAPI;

/*!
 * \brief fill a histogram by zeros
 */
template<typename GradientSumT>
void InitializeHistByZeroes(GHistRowOneAPI<GradientSumT>& hist, size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(cl::sycl::queue qu,
                     GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src1,
                     const GHistRowOneAPI<GradientSumT>& src2,
                     size_t size);

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
template<typename GradientSumT>
class HistCollectionOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  // access histogram for i-th node
  GHistRowT& operator[](bst_uint nid) {
    return data_[nid];
  }

  const GHistRowT& operator[](bst_uint nid) const {
    return data_[nid];
  }

  // have we computed a histogram for i-th node?
  bool RowExists(bst_uint nid) const {
    return (nid < data_.size() && !data_[nid].Empty());
  }

  // initialize histogram collection
  void Init(cl::sycl::queue qu, uint32_t nbins) {
    qu_ = qu;
    if (nbins_ != nbins) {
      nbins_ = nbins;
      // quite expensive operation, so let's do this only once
      data_.clear();
    }
    n_nodes_added_ = 0;
  }

  // create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
    if (nid >= data_.size()) {
      data_.resize(nid + 1);
    }
    data_[nid].Resize(qu_, nbins_, xgboost::detail::GradientPairInternal<GradientSumT>(0, 0));
    n_nodes_added_++;
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  /*! \brief amount of active nodes in hist collection */
  uint32_t n_nodes_added_ = 0;

  std::vector<GHistRowT> data_;

  cl::sycl::queue qu_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 * Supports processing multiple tree-nodes for nested parallelism
 * Able to reduce histograms across threads in efficient way
 */
template<typename GradientSumT>
class ParallelGHistBuilderOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  void Init(cl::sycl::queue qu, size_t nbins) {
    qu_ = qu;
    if (nbins != nbins_) {
      hist_buffer_.Init(qu_, nbins);
      nbins_ = nbins;
    }
  }

  void Reset(size_t nthreads) {
    hist_device_buffer_.Resize(qu_, nthreads * nbins_ * 2);
  }

  GHistRowT& GetDeviceBuffer() {
    return hist_device_buffer_;
  }

  // Add new elements if needed, mark all hists as unused
  // targeted_hists - already allocated hists which should contain final results after Reduce() call
  void Reset(size_t nthreads, size_t nodes, const BlockedSpace2d& space,
             const std::vector<GHistRowT>& targeted_hists) {
    hist_buffer_.Init(qu_, nbins_);
    tid_nid_to_hist_.clear();
    hist_memory_.clear();
    threads_to_nids_map_.clear();

    targeted_hists_ = targeted_hists;

    CHECK_EQ(nodes, targeted_hists.size());

    nodes_    = nodes;
    nthreads_ = nthreads;

    MatchThreadsToNodes(space);
    AllocateAdditionalHistograms();
    MatchNodeNidPairToHist();

    hist_was_used_.resize(nthreads * nodes_);
    std::fill(hist_was_used_.begin(), hist_was_used_.end(), static_cast<int>(false));
  }

  // Get specified hist, initialize hist by zeros if it wasn't used before
  GHistRowT& GetInitializedHist(size_t tid, size_t nid) {
    CHECK_LT(nid, nodes_);
    CHECK_LT(tid, nthreads_);

    size_t idx = tid_nid_to_hist_.at({tid, nid});
    GHistRowT& hist = hist_memory_[idx];

    if (!hist_was_used_[tid * nodes_ + nid]) {
      InitializeHistByZeroes(hist, 0, hist.Size());
      hist_was_used_[tid * nodes_ + nid] = static_cast<int>(true);
    }

    return hist;
  }

  // Reduce following bins (begin, end] for nid-node in dst across threads
  void ReduceHist(size_t nid, size_t begin, size_t end) {
    CHECK_GT(end, begin);
    CHECK_LT(nid, nodes_);

    GHistRowT& dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;
        const size_t idx = tid_nid_to_hist_.at({tid, nid});
        GHistRowT& src = hist_memory_[idx];

        if (dst.Data() != src.Data()) {
          IncrementHist(dst, src, begin, end);
        }
      }
    }
    if (!is_updated) {
      // In distributed mode - some tree nodes can be empty on local machines,
      // So we need just set local hist by zeros in this case
      InitializeHistByZeroes(dst, begin, end);
    }
  }

 protected:
  void MatchThreadsToNodes(const BlockedSpace2d& space) {
    const size_t space_size = space.Size();
    const size_t chunck_size = space_size / nthreads_ + !!(space_size % nthreads_);

    threads_to_nids_map_.resize(nthreads_ * nodes_, false);

    for (size_t tid = 0; tid < nthreads_; ++tid) {
      size_t begin = chunck_size * tid;
      size_t end   = std::min(begin + chunck_size, space_size);

      if (begin < space_size) {
        size_t nid_begin = space.GetFirstDimension(begin);
        size_t nid_end   = space.GetFirstDimension(end-1);

        for (size_t nid = nid_begin; nid <= nid_end; ++nid) {
          // true - means thread 'tid' will work to compute partial hist for node 'nid'
          threads_to_nids_map_[tid * nodes_ + nid] = true;
        }
      }
    }
  }

  void AllocateAdditionalHistograms() {
    size_t hist_allocated_additionally = 0;

    for (size_t nid = 0; nid < nodes_; ++nid) {
      int nthreads_for_nid = 0;

      for (size_t tid = 0; tid < nthreads_; ++tid) {
        if (threads_to_nids_map_[tid * nodes_ + nid]) {
          nthreads_for_nid++;
        }
      }

      // In distributed mode - some tree nodes can be empty on local machines,
      // set nthreads_for_nid to 0 in this case.
      // In another case - allocate additional (nthreads_for_nid - 1) histograms,
      // because one is already allocated externally (will store final result for the node).
      hist_allocated_additionally += std::max<int>(0, nthreads_for_nid - 1);
    }

    for (size_t i = 0; i < hist_allocated_additionally; ++i) {
      hist_buffer_.AddHistRow(i);
    }
  }

  void MatchNodeNidPairToHist() {
    size_t hist_total = 0;
    size_t hist_allocated_additionally = 0;

    for (size_t nid = 0; nid < nodes_; ++nid) {
      bool first_hist = true;
      for (size_t tid = 0; tid < nthreads_; ++tid) {
        if (threads_to_nids_map_[tid * nodes_ + nid]) {
          if (first_hist) {
            hist_memory_.push_back(targeted_hists_[nid]);
            first_hist = false;
          } else {
            hist_memory_.push_back(hist_buffer_[hist_allocated_additionally]);
            hist_allocated_additionally++;
          }
          // map pair {tid, nid} to index of allocated histogram from hist_memory_
          tid_nid_to_hist_[{tid, nid}] = hist_total++;
          CHECK_EQ(hist_total, hist_memory_.size());
        }
      }
    }
  }

  /*! \brief number of bins in each histogram */
  size_t nbins_ = 0;
  /*! \brief number of threads for parallel computation */
  size_t nthreads_ = 0;
  /*! \brief number of nodes which will be processed in parallel  */
  size_t nodes_ = 0;
  /*! \brief Buffer for additional histograms for Parallel processing  */
  HistCollectionOneAPI<GradientSumT> hist_buffer_;
  /*!
   * \brief Marks which hists were used, it means that they should be merged.
   * Contains only {true or false} values
   * but 'int' is used instead of 'bool', because std::vector<bool> isn't thread safe
   */
  std::vector<int> hist_was_used_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  std::vector<bool> threads_to_nids_map_;
  /*! \brief Contains histograms for final results  */
  std::vector<GHistRowT> targeted_hists_;
  /*! \brief Allocated memory for histograms used for construction  */
  std::vector<GHistRowT> hist_memory_;
  /*! \brief map pair {tid, nid} to index of allocated histogram from hist_memory_  */
  std::map<std::pair<size_t, size_t>, size_t> tid_nid_to_hist_;

  GHistRowT hist_device_buffer_;

  cl::sycl::queue qu_;
};

/*!
 * \brief builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilderOneAPI {
 public:
  using GHistRowT = GHistRowOneAPI<GradientSumT>;

  GHistBuilderOneAPI() = default;
  GHistBuilderOneAPI(cl::sycl::queue qu, size_t nthread, uint32_t nbins) : qu_{qu}, nthread_{nthread}, nbins_{nbins} {}

  // construct a histogram via histogram aggregation
  void BuildHist(common::Monitor& builder_monitor_, const std::vector<GradientPair>& gpair,
                 const USMVector<GradientPair>& gpair_device,
                 const RowSetCollectionOneAPI::Elem& row_indices,
                 const GHistIndexMatrixOneAPI& gmat,
                 GHistRowT& hist,
                 bool isDense,
                 GHistRowT& hist_buffer);

  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRowT& self,
                        GHistRowT& sibling,
                        GHistRowT& parent);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_ { 0 };
  /*! \brief number of all bins over all features */
  uint32_t nbins_ { 0 };

  cl::sycl::queue qu_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_ONEAPI_H_
