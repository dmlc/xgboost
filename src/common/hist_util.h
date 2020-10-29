/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.h
 * \brief Utility for fast histogram aggregation
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_H_
#define XGBOOST_COMMON_HIST_UTIL_H_

#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <map>

#include "row_set.h"
#include "common.h"
#include "threading_utils.h"
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"

namespace xgboost {
namespace common {
/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;

// A CSC matrix representing histogram cuts, used in CPU quantile hist.
// The cut values represent upper bounds of bins containing approximately equal numbers of elements
class HistogramCuts {
 protected:
  using BinIdx = uint32_t;

 public:
  HostDeviceVector<bst_float> cut_values_;  // NOLINT
  HostDeviceVector<uint32_t> cut_ptrs_;     // NOLINT
  // storing minimum value in a sketch set.
  HostDeviceVector<float> min_vals_;  // NOLINT

  HistogramCuts();
  HistogramCuts(HistogramCuts const& that) {
    cut_values_.Resize(that.cut_values_.Size());
    cut_ptrs_.Resize(that.cut_ptrs_.Size());
    min_vals_.Resize(that.min_vals_.Size());
    cut_values_.Copy(that.cut_values_);
    cut_ptrs_.Copy(that.cut_ptrs_);
    min_vals_.Copy(that.min_vals_);
  }

  HistogramCuts(HistogramCuts&& that) noexcept(true) {
    *this = std::forward<HistogramCuts&&>(that);
  }

  HistogramCuts& operator=(HistogramCuts const& that) {
    cut_values_.Resize(that.cut_values_.Size());
    cut_ptrs_.Resize(that.cut_ptrs_.Size());
    min_vals_.Resize(that.min_vals_.Size());
    cut_values_.Copy(that.cut_values_);
    cut_ptrs_.Copy(that.cut_ptrs_);
    min_vals_.Copy(that.min_vals_);
    return *this;
  }

  HistogramCuts& operator=(HistogramCuts&& that) noexcept(true) {
    cut_ptrs_ = std::move(that.cut_ptrs_);
    cut_values_ = std::move(that.cut_values_);
    min_vals_ = std::move(that.min_vals_);
    return *this;
  }

  uint32_t FeatureBins(uint32_t feature) const {
    return cut_ptrs_.ConstHostVector().at(feature + 1) -
           cut_ptrs_.ConstHostVector()[feature];
  }

  // Getters.  Cuts should be of no use after building histogram indices, but currently
  // it's deeply linked with quantile_hist, gpu sketcher and gpu_hist.  So we preserve
  // these for now.
  std::vector<uint32_t> const& Ptrs()      const { return cut_ptrs_.ConstHostVector();   }
  std::vector<float>    const& Values()    const { return cut_values_.ConstHostVector(); }
  std::vector<float>    const& MinValues() const { return min_vals_.ConstHostVector();   }

  size_t TotalBins() const { return cut_ptrs_.ConstHostVector().back(); }

  // Return the index of a cut point that is strictly greater than the input
  // value, or the last available index if none exists
  BinIdx SearchBin(float value, uint32_t column_id) const {
    auto beg = cut_ptrs_.ConstHostVector().at(column_id);
    auto end = cut_ptrs_.ConstHostVector().at(column_id + 1);
    const auto &values = cut_values_.ConstHostVector();
    auto it = std::upper_bound(values.cbegin() + beg, values.cbegin() + end, value);
    BinIdx idx = it - values.cbegin();
    if (idx == end) {
      idx -= 1;
    }
    return idx;
  }

  BinIdx SearchBin(Entry const& e) const {
    return SearchBin(e.fvalue, e.index);
  }
};

inline HistogramCuts SketchOnDMatrix(DMatrix *m, int32_t max_bins) {
  HistogramCuts out;
  auto const& info = m->Info();
  const auto threads = omp_get_max_threads();
  std::vector<std::vector<bst_row_t>> column_sizes(threads);
  for (auto& column : column_sizes) {
    column.resize(info.num_col_, 0);
  }
  std::vector<bst_row_t> reduced(info.num_col_, 0);
  for (auto const& page : m->GetBatches<SparsePage>()) {
    auto const &entries_per_column =
        HostSketchContainer::CalcColumnSize(page, info.num_col_, threads);
    for (size_t i = 0; i < entries_per_column.size(); ++i) {
      reduced[i] += entries_per_column[i];
    }
  }
  HostSketchContainer container(reduced, max_bins,
                                HostSketchContainer::UseGroup(info));
  for (auto const &page : m->GetBatches<SparsePage>()) {
    container.PushRowPage(page, info);
  }
  container.MakeCuts(&out);
  return out;
}

enum BinTypeSize {
  kUint8BinsTypeSize  = 1,
  kUint16BinsTypeSize = 2,
  kUint32BinsTypeSize = 4
};

struct Index {
  Index() {
    SetBinTypeSize(binTypeSize_);
  }
  Index(const Index& i) = delete;
  Index& operator=(Index i) = delete;
  Index(Index&& i) = delete;
  Index& operator=(Index&& i) = delete;
  uint32_t operator[](size_t i) const {
    if (offset_ptr_ != nullptr) {
      return func_(data_ptr_, i) + offset_ptr_[i%p_];
    } else {
      return func_(data_ptr_, i);
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
  T* data() const {  // NOLINT
    return static_cast<T*>(data_ptr_);
  }
  uint32_t* Offset() const {
    return offset_ptr_;
  }
  size_t OffsetSize() const {
    return offset_.size();
  }
  size_t Size() const {
    return data_.size() / (binTypeSize_);
  }
  void Resize(const size_t nBytesData) {
    data_.resize(nBytesData);
    data_ptr_ = reinterpret_cast<void*>(data_.data());
  }
  void ResizeOffset(const size_t nDisps) {
    offset_.resize(nDisps);
    offset_ptr_ = offset_.data();
    p_ = nDisps;
  }
  std::vector<uint8_t>::const_iterator begin() const {  // NOLINT
    return data_.begin();
  }
  std::vector<uint8_t>::const_iterator end() const {  // NOLINT
    return data_.end();
  }

 private:
  static uint32_t GetValueFromUint8(void *t, size_t i) {
    return reinterpret_cast<uint8_t*>(t)[i];
  }
  static uint32_t GetValueFromUint16(void* t, size_t i) {
    return reinterpret_cast<uint16_t*>(t)[i];
  }
  static uint32_t GetValueFromUint32(void* t, size_t i) {
    return reinterpret_cast<uint32_t*>(t)[i];
  }

  using Func = uint32_t (*)(void*, size_t);

  std::vector<uint8_t> data_;
  std::vector<uint32_t> offset_;  // size of this field is equal to number of features
  void* data_ptr_;
  BinTypeSize binTypeSize_ {kUint8BinsTypeSize};
  size_t p_ {1};
  uint32_t* offset_ptr_ {nullptr};
  Func func_;
};


/*!
 * \brief preprocessed global index matrix, in CSR format
 *
 *  Transform floating values to integer index in histogram This is a global histogram
 *  index for CPU histogram.  On GPU ellpack page is used.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  Index index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  HistogramCuts cut;
  DMatrix* p_fmat;
  size_t max_num_bins;
  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_num_bins);

  // specific method for sparse data as no posibility to reduce allocated memory
  template <typename BinIdxType, typename GetOffset>
  void SetIndexData(common::Span<BinIdxType> index_data_span,
                    size_t batch_threads, const SparsePage &batch,
                    size_t rbegin, size_t nbins, GetOffset get_offset) {
    const xgboost::Entry *data_ptr = batch.data.HostVector().data();
    const std::vector<bst_row_t> &offset_vec = batch.offset.HostVector();
    const size_t batch_size = batch.Size();
    CHECK_LT(batch_size, offset_vec.size());
    BinIdxType* index_data = index_data_span.data();
#pragma omp parallel for num_threads(batch_threads) schedule(static)
    for (omp_ulong i = 0; i < batch_size; ++i) {
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      SparsePage::Inst inst = {data_ptr + offset_vec[i], size};
      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        uint32_t idx = cut.SearchBin(inst[j]);
        index_data[ibegin + j] = get_offset(idx, j);
        ++hit_count_tloc_[tid * nbins + idx];
      }
    }
  }

  void ResizeIndex(const size_t n_index,
                   const bool isDense);

  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.Ptrs().size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.Ptrs()[fid];
      auto iend = cut.Ptrs()[fid + 1];
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

template <typename GradientIndex>
int32_t XGBOOST_HOST_DEV_INLINE BinarySearchBin(bst_uint begin, bst_uint end,
                                                GradientIndex const &data,
                                                uint32_t const fidx_begin,
                                                uint32_t const fidx_end) {
  uint32_t previous_middle = std::numeric_limits<uint32_t>::max();
  while (end != begin) {
    auto middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      return static_cast<int32_t>(gidx);
    } else if (gidx < fidx_begin) {
      begin = middle;
    } else {
      end = middle;
    }
  }
  // Value is missing
  return -1;
}

struct GHistIndexBlock {
  const size_t* row_ptr;
  const uint32_t* index;

  inline GHistIndexBlock(const size_t* row_ptr, const uint32_t* index)
    : row_ptr(row_ptr), index(index) {}

  // get i-th row
  inline GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i], row_ptr[i + 1] - row_ptr[i]};
  }
};

class ColumnMatrix;

class GHistIndexBlockMatrix {
 public:
  void Init(const GHistIndexMatrix& gmat,
            const ColumnMatrix& colmat,
            const tree::TrainParam& param);

  inline GHistIndexBlock operator[](size_t i) const {
    return {blocks_[i].row_ptr_begin, blocks_[i].index_begin};
  }

  inline size_t GetNumBlock() const {
    return blocks_.size();
  }

 private:
  std::vector<size_t> row_ptr_;
  std::vector<uint32_t> index_;
  const HistogramCuts* cut_;
  struct Block {
    const size_t* row_ptr_begin;
    const size_t* row_ptr_end;
    const uint32_t* index_begin;
    const uint32_t* index_end;
  };
  std::vector<Block> blocks_;
};

template<typename GradientSumT>
using GHistRow = Span<xgboost::detail::GradientPairInternal<GradientSumT> >;

/*!
 * \brief fill a histogram by zeros
 */
template<typename GradientSumT>
void InitilizeHistByZeroes(GHistRow<GradientSumT> hist, size_t begin, size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
template<typename GradientSumT>
void IncrementHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> add,
                   size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
template<typename GradientSumT>
void CopyHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src,
              size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src1,
                     const GHistRow<GradientSumT> src2,
                     size_t begin, size_t end);

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
template<typename GradientSumT>
class HistCollection {
 public:
  using GHistRowT = GHistRow<GradientSumT>;
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;

  // access histogram for i-th node
  GHistRowT operator[](bst_uint nid) const {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    CHECK_NE(row_ptr_[nid], kMax);
    GradientPairT* ptr =
        const_cast<GradientPairT*>(dmlc::BeginPtr(data_) + row_ptr_[nid]);
    return {ptr, nbins_};
  }

  // have we computed a histogram for i-th node?
  bool RowExists(bst_uint nid) const {
    const uint32_t k_max = std::numeric_limits<uint32_t>::max();
    return (nid < row_ptr_.size() && row_ptr_[nid] != k_max);
  }

  // initialize histogram collection
  void Init(uint32_t nbins) {
    if (nbins_ != nbins) {
      nbins_ = nbins;
      // quite expensive operation, so let's do this only once
      data_.clear();
    }
    row_ptr_.clear();
    n_nodes_added_ = 0;
  }

  // create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    if (nid >= row_ptr_.size()) {
      row_ptr_.resize(nid + 1, kMax);
    }
    CHECK_EQ(row_ptr_[nid], kMax);

    if (data_.size() < nbins_ * (nid + 1)) {
      data_.resize(nbins_ * (nid + 1));
    }

    row_ptr_[nid] = nbins_ * n_nodes_added_;
    n_nodes_added_++;
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  /*! \brief amount of active nodes in hist collection */
  uint32_t n_nodes_added_ = 0;

  std::vector<GradientPairT> data_;

  /*! \brief row_ptr_[nid] locates bin for histogram of node nid */
  std::vector<size_t> row_ptr_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 * Supports processing multiple tree-nodes for nested parallelism
 * Able to reduce histograms across threads in efficient way
 */
template<typename GradientSumT>
class ParallelGHistBuilder {
 public:
  using GHistRowT = GHistRow<GradientSumT>;

  void Init(size_t nbins) {
    if (nbins != nbins_) {
      hist_buffer_.Init(nbins);
      nbins_ = nbins;
    }
  }

  // Add new elements if needed, mark all hists as unused
  // targeted_hists - already allocated hists which should contain final results after Reduce() call
  void Reset(size_t nthreads, size_t nodes, const BlockedSpace2d& space,
             const std::vector<GHistRowT>& targeted_hists) {
    hist_buffer_.Init(nbins_);
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
  GHistRowT GetInitializedHist(size_t tid, size_t nid) {
    CHECK_LT(nid, nodes_);
    CHECK_LT(tid, nthreads_);

    size_t idx = tid_nid_to_hist_.at({tid, nid});
    GHistRowT hist = hist_memory_[idx];

    if (!hist_was_used_[tid * nodes_ + nid]) {
      InitilizeHistByZeroes(hist, 0, hist.size());
      hist_was_used_[tid * nodes_ + nid] = static_cast<int>(true);
    }

    return hist;
  }

  // Reduce following bins (begin, end] for nid-node in dst across threads
  void ReduceHist(size_t nid, size_t begin, size_t end) {
    CHECK_GT(end, begin);
    CHECK_LT(nid, nodes_);

    GHistRowT dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;
        const size_t idx = tid_nid_to_hist_.at({tid, nid});
        GHistRowT src = hist_memory_[idx];

        if (dst.data() != src.data()) {
          IncrementHist(dst, src, begin, end);
        }
      }
    }
    if (!is_updated) {
      // In distributed mode - some tree nodes can be empty on local machines,
      // So we need just set local hist by zeros in this case
      InitilizeHistByZeroes(dst, begin, end);
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
  HistCollection<GradientSumT> hist_buffer_;
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
};

/*!
 * \brief builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilder {
 public:
  using GHistRowT = GHistRow<GradientSumT>;

  GHistBuilder() = default;
  GHistBuilder(size_t nthread, uint32_t nbins) : nthread_{nthread}, nbins_{nbins} {}

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 GHistRowT hist,
                 bool isDense);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      GHistRowT hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRowT self,
                        GHistRowT sibling,
                        GHistRowT parent);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_ { 0 };
  /*! \brief number of all bins over all features */
  uint32_t nbins_ { 0 };
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
