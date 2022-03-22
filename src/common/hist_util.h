/*!
 * Copyright 2017-2022 by XGBoost Contributors
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

#include "categorical.h"
#include "common.h"
#include "quantile.h"
#include "row_set.h"
#include "threading_utils.h"
#include "timer.h"

namespace xgboost {
class GHistIndexMatrix;

namespace common {
/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;

// A CSC matrix representing histogram cuts.
// The cut values represent upper bounds of bins containing approximately equal numbers of elements
class HistogramCuts {
  bool has_categorical_{false};
  float max_cat_{-1.0f};

 protected:
  using BinIdx = uint32_t;

  void Swap(HistogramCuts&& that) noexcept(true) {
    std::swap(cut_values_, that.cut_values_);
    std::swap(cut_ptrs_, that.cut_ptrs_);
    std::swap(min_vals_, that.min_vals_);

    std::swap(has_categorical_, that.has_categorical_);
    std::swap(max_cat_, that.max_cat_);
  }

  void Copy(HistogramCuts const& that) {
    cut_values_.Resize(that.cut_values_.Size());
    cut_ptrs_.Resize(that.cut_ptrs_.Size());
    min_vals_.Resize(that.min_vals_.Size());
    cut_values_.Copy(that.cut_values_);
    cut_ptrs_.Copy(that.cut_ptrs_);
    min_vals_.Copy(that.min_vals_);
    has_categorical_ = that.has_categorical_;
    max_cat_ = that.max_cat_;
  }

 public:
  HostDeviceVector<float> cut_values_;   // NOLINT
  HostDeviceVector<uint32_t> cut_ptrs_;  // NOLINT
  // storing minimum value in a sketch set.
  HostDeviceVector<float> min_vals_;  // NOLINT

  HistogramCuts();
  HistogramCuts(HistogramCuts const& that) { this->Copy(that); }

  HistogramCuts(HistogramCuts&& that) noexcept(true) {
    this->Swap(std::forward<HistogramCuts>(that));
  }

  HistogramCuts& operator=(HistogramCuts const& that) {
    this->Copy(that);
    return *this;
  }

  HistogramCuts& operator=(HistogramCuts&& that) noexcept(true) {
    this->Swap(std::forward<HistogramCuts>(that));
    return *this;
  }

  uint32_t FeatureBins(bst_feature_t feature) const {
    return cut_ptrs_.ConstHostVector().at(feature + 1) - cut_ptrs_.ConstHostVector()[feature];
  }

  std::vector<uint32_t> const& Ptrs()      const { return cut_ptrs_.ConstHostVector();   }
  std::vector<float>    const& Values()    const { return cut_values_.ConstHostVector(); }
  std::vector<float>    const& MinValues() const { return min_vals_.ConstHostVector();   }

  bool HasCategorical() const { return has_categorical_; }
  float MaxCategory() const { return max_cat_; }
  /**
   * \brief Set meta info about categorical features.
   *
   * \param has_cat Do we have categorical feature in the data?
   * \param max_cat The maximum categorical value in all features.
   */
  void SetCategorical(bool has_cat, float max_cat) {
    has_categorical_ = has_cat;
    max_cat_ = max_cat;
  }

  size_t TotalBins() const { return cut_ptrs_.ConstHostVector().back(); }

  // Return the index of a cut point that is strictly greater than the input
  // value, or the last available index if none exists
  BinIdx SearchBin(float value, bst_feature_t column_id, std::vector<uint32_t> const& ptrs,
                   std::vector<float> const& values) const {
    auto end = ptrs[column_id + 1];
    auto beg = ptrs[column_id];
    auto it = std::upper_bound(values.cbegin() + beg, values.cbegin() + end, value);
    BinIdx idx = it - values.cbegin();
    idx -= !!(idx == end);
    return idx;
  }

  BinIdx SearchBin(float value, bst_feature_t column_id) const {
    return this->SearchBin(value, column_id, Ptrs(), Values());
  }

  /**
   * \brief Search the bin index for numerical feature.
   */
  BinIdx SearchBin(Entry const& e) const {
    return SearchBin(e.fvalue, e.index);
  }

  /**
   * \brief Search the bin index for categorical feature.
   */
  BinIdx SearchCatBin(Entry const &e) const {
    auto const &ptrs = this->Ptrs();
    auto const &vals = this->Values();
    auto end = ptrs.at(e.index + 1) + vals.cbegin();
    auto beg = ptrs[e.index] + vals.cbegin();
    // Truncates the value in case it's not perfectly rounded.
    auto v  = static_cast<float>(common::AsCat(e.fvalue));
    auto bin_idx = std::lower_bound(beg, end, v) - vals.cbegin();
    if (bin_idx == ptrs.at(e.index + 1)) {
      bin_idx -= 1;
    }
    return bin_idx;
  }
};

/**
 * \brief Run CPU sketching on DMatrix.
 *
 * \param use_sorted Whether should we use SortedCSC for sketching, it's more efficient
 *                   but consumes more memory.
 */
inline HistogramCuts SketchOnDMatrix(DMatrix* m, int32_t max_bins, int32_t n_threads,
                                     bool use_sorted = false, Span<float> const hessian = {}) {
  HistogramCuts out;
  auto const& info = m->Info();
  std::vector<std::vector<bst_row_t>> column_sizes(n_threads);
  for (auto& column : column_sizes) {
    column.resize(info.num_col_, 0);
  }
  std::vector<bst_row_t> reduced(info.num_col_, 0);
  for (auto const& page : m->GetBatches<SparsePage>()) {
    auto const& entries_per_column =
        HostSketchContainer::CalcColumnSize(page, info.num_col_, n_threads);
    for (size_t i = 0; i < entries_per_column.size(); ++i) {
      reduced[i] += entries_per_column[i];
    }
  }

  if (!use_sorted) {
    HostSketchContainer container(max_bins, m->Info(), reduced, HostSketchContainer::UseGroup(info),
                                  hessian, n_threads);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      container.PushRowPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  } else {
    SortedSketchContainer container{
        max_bins, m->Info(), reduced, HostSketchContainer::UseGroup(info), hessian, n_threads};
    for (auto const& page : m->GetBatches<SortedCSCPage>()) {
      container.PushColPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  }

  return out;
}

enum BinTypeSize : uint32_t {
  kUint8BinsTypeSize  = 1,
  kUint16BinsTypeSize = 2,
  kUint32BinsTypeSize = 4
};

/**
 * \brief Optionally compressed gradient index. The compression works only with dense
 *        data.
 *
 *   The main body of construction code is in gradient_index.cc, this struct is only a
 *   storage class.
 */
struct Index {
  Index() { SetBinTypeSize(binTypeSize_); }
  Index(const Index& i) = delete;
  Index& operator=(Index i) = delete;
  Index(Index&& i) = delete;
  Index& operator=(Index&& i) = delete;
  uint32_t operator[](size_t i) const {
    if (!bin_offset_.empty()) {
      // dense, compressed
      auto fidx = i % bin_offset_.size();
      // restore the index by adding back its feature offset.
      return func_(data_.data(), i) + bin_offset_[fidx];
    } else {
      return func_(data_.data(), i);
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
        CHECK(binTypeSize == kUint8BinsTypeSize || binTypeSize == kUint16BinsTypeSize ||
              binTypeSize == kUint32BinsTypeSize);
    }
  }
  BinTypeSize GetBinTypeSize() const {
    return binTypeSize_;
  }
  template <typename T>
  T const* data() const {  // NOLINT
    return reinterpret_cast<T const*>(data_.data());
  }
  template <typename T>
  T* data() {  // NOLINT
    return reinterpret_cast<T*>(data_.data());
  }
  uint32_t const* Offset() const { return bin_offset_.data(); }
  size_t OffsetSize() const { return bin_offset_.size(); }
  size_t Size() const { return data_.size() / (binTypeSize_); }

  void Resize(const size_t n_bytes) {
    data_.resize(n_bytes);
  }
  // set the offset used in compression, cut_ptrs is the CSC indptr in HistogramCuts
  void SetBinOffset(std::vector<uint32_t> const& cut_ptrs) {
    bin_offset_.resize(cut_ptrs.size() - 1);  // resize to number of features.
    std::copy_n(cut_ptrs.begin(), bin_offset_.size(), bin_offset_.begin());
  }
  std::vector<uint8_t>::const_iterator begin() const {  // NOLINT
    return data_.begin();
  }
  std::vector<uint8_t>::const_iterator end() const {  // NOLINT
    return data_.end();
  }

  std::vector<uint8_t>::iterator begin() {  // NOLINT
    return data_.begin();
  }
  std::vector<uint8_t>::iterator end() {  // NOLINT
    return data_.end();
  }

 private:
  // Functions to decompress the index.
  static uint32_t GetValueFromUint8(uint8_t const* t, size_t i) { return t[i]; }
  static uint32_t GetValueFromUint16(uint8_t const* t, size_t i) {
    return reinterpret_cast<uint16_t const*>(t)[i];
  }
  static uint32_t GetValueFromUint32(uint8_t const* t, size_t i) {
    return reinterpret_cast<uint32_t const*>(t)[i];
  }

  using Func = uint32_t (*)(uint8_t const*, size_t);

  std::vector<uint8_t> data_;
  // starting position of each feature inside the cut values (the indptr of the CSC cut matrix
  // HistogramCuts without the last entry.) Used for bin compression.
  std::vector<uint32_t> bin_offset_;

  BinTypeSize binTypeSize_ {kUint8BinsTypeSize};
  Func func_;
};

template <typename GradientIndex>
int32_t XGBOOST_HOST_DEV_INLINE BinarySearchBin(size_t begin, size_t end,
                                                GradientIndex const &data,
                                                uint32_t const fidx_begin,
                                                uint32_t const fidx_end) {
  size_t previous_middle = std::numeric_limits<size_t>::max();
  while (end != begin) {
    size_t middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    // index into all the bins
    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      // Found the intersection.
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

class ColumnMatrix;

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
    const size_t id = row_ptr_.at(nid);
    CHECK_NE(id, kMax);
    GradientPairT* ptr = nullptr;
    if (contiguous_allocation_) {
      ptr = const_cast<GradientPairT*>(data_[0].data() + nbins_*id);
    } else {
      ptr = const_cast<GradientPairT*>(data_[id].data());
    }
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

    if (data_.size() < (nid + 1)) {
      data_.resize((nid + 1));
    }

    row_ptr_[nid] = n_nodes_added_;
    n_nodes_added_++;
  }
  // allocate thread local memory i-th node
  void AllocateData(bst_uint nid) {
    if (data_[row_ptr_[nid]].size() == 0) {
      data_[row_ptr_[nid]].resize(nbins_, {0, 0});
    }
  }
  // allocate common buffer contiguously for all nodes, need for single Allreduce call
  void AllocateAllData() {
    const size_t new_size = nbins_*data_.size();
    contiguous_allocation_ = true;
    if (data_[0].size() != new_size) {
      data_[0].resize(new_size);
    }
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  /*! \brief amount of active nodes in hist collection */
  uint32_t n_nodes_added_ = 0;
  /*! \brief flag to identify contiguous memory allocation */
  bool contiguous_allocation_ = false;

  std::vector<std::vector<GradientPairT>> data_;

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

    int idx = tid_nid_to_hist_.at({tid, nid});
    if (idx >= 0) {
      hist_buffer_.AllocateData(idx);
    }
    GHistRowT hist = idx == -1 ? targeted_hists_[nid] : hist_buffer_[idx];

    if (!hist_was_used_[tid * nodes_ + nid]) {
      InitilizeHistByZeroes(hist, 0, hist.size());
      hist_was_used_[tid * nodes_ + nid] = static_cast<int>(true);
    }

    return hist;
  }

  // Reduce following bins (begin, end] for nid-node in dst across threads
  void ReduceHist(size_t nid, size_t begin, size_t end) const {
    CHECK_GT(end, begin);
    CHECK_LT(nid, nodes_);

    GHistRowT dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;

        int idx = tid_nid_to_hist_.at({tid, nid});
        GHistRowT src = idx == -1 ? targeted_hists_[nid] : hist_buffer_[idx];

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

 private:
  void MatchNodeNidPairToHist() {
    size_t hist_allocated_additionally = 0;

    for (size_t nid = 0; nid < nodes_; ++nid) {
      bool first_hist = true;
      for (size_t tid = 0; tid < nthreads_; ++tid) {
        if (threads_to_nids_map_[tid * nodes_ + nid]) {
          if (first_hist) {
            tid_nid_to_hist_[{tid, nid}] = -1;
            first_hist = false;
          } else {
            tid_nid_to_hist_[{tid, nid}] = hist_allocated_additionally++;
          }
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
  /*!
   * \brief map pair {tid, nid} to index of allocated histogram from hist_buffer_ and targeted_hists_,
   * -1 is reserved for targeted_hists_
   */
  std::map<std::pair<size_t, size_t>, int> tid_nid_to_hist_;
};

/*!
 * \brief builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilder {
 public:
  using GHistRowT = GHistRow<GradientSumT>;

  GHistBuilder() = default;
  explicit GHistBuilder(uint32_t nbins): nbins_{nbins} {}

  // construct a histogram via histogram aggregation
  template <bool any_missing>
  void BuildHist(const std::vector<GradientPair> &gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix &gmat, GHistRowT hist) const;
  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ { 0 };
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
