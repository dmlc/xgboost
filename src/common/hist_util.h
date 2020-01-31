/*!
 * Copyright 2017 by Contributors
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
#include "threading_utils.h"
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"

namespace xgboost {
namespace common {

/*
 * \brief A thin wrapper around dynamically allocated C-style array.
 * Make sure to call resize() before use.
 */
template<typename T>
struct SimpleArray {
  ~SimpleArray() {
    std::free(ptr_);
    ptr_ = nullptr;
  }

  void resize(size_t n) {
    T* ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
    CHECK(ptr) << "Failed to allocate memory";
    if (ptr_) {
      std::memcpy(ptr, ptr_, n_ * sizeof(T));
      std::free(ptr_);
    }
    ptr_ = ptr;
    n_ = n;
  }

  T& operator[](size_t idx) {
    return ptr_[idx];
  }

  T& operator[](size_t idx) const {
    return ptr_[idx];
  }

  size_t size() const {
    return n_;
  }

  T back() const {
    return ptr_[n_-1];
  }

  T* data() {
    return ptr_;
  }

  const T* data() const {
    return ptr_;
  }


  T* begin() {
    return ptr_;
  }

  const T* begin() const {
    return ptr_;
  }

  T* end() {
    return ptr_ + n_;
  }

  const T* end() const {
    return ptr_ + n_;
  }

 private:
  T* ptr_ = nullptr;
  size_t n_ = 0;
};

/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;

// A CSC matrix representing histogram cuts, used in CPU quantile hist.
class HistogramCuts {
  // Using friends to avoid creating a virtual class, since HistogramCuts is used as value
  // object in many places.
  friend class SparseCuts;
  friend class DenseCuts;
  friend class CutsBuilder;

 protected:
  using BinIdx = uint32_t;
  common::Monitor monitor_;

  std::vector<bst_float> cut_values_;
  std::vector<uint32_t> cut_ptrs_;
  std::vector<float> min_vals_;  // storing minimum value in a sketch set.

 public:
  HistogramCuts();
  HistogramCuts(HistogramCuts const& that) = delete;
  HistogramCuts(HistogramCuts&& that) noexcept(true) {
    *this = std::forward<HistogramCuts&&>(that);
  }
  HistogramCuts& operator=(HistogramCuts const& that) = delete;
  HistogramCuts& operator=(HistogramCuts&& that) noexcept(true) {
    monitor_ = std::move(that.monitor_);
    cut_ptrs_ = std::move(that.cut_ptrs_);
    cut_values_ = std::move(that.cut_values_);
    min_vals_ = std::move(that.min_vals_);
    return *this;
  }

  /* \brief Build histogram cuts. */
  void Build(DMatrix* dmat, uint32_t const max_num_bins);
  /* \brief How many bins a feature has. */
  uint32_t FeatureBins(uint32_t feature) const {
    return cut_ptrs_.at(feature+1) - cut_ptrs_[feature];
  }

  // Getters.  Cuts should be of no use after building histogram indices, but currently
  // it's deeply linked with quantile_hist, gpu sketcher and gpu_hist.  So we preserve
  // these for now.
  std::vector<uint32_t> const& Ptrs()      const { return cut_ptrs_;   }
  std::vector<float>    const& Values()    const { return cut_values_; }
  std::vector<float>    const& MinValues() const { return min_vals_;   }

  size_t TotalBins() const { return cut_ptrs_.back(); }

  BinIdx SearchBin(float value, uint32_t column_id) {
    auto beg = cut_ptrs_.at(column_id);
    auto end = cut_ptrs_.at(column_id + 1);
    auto it = std::upper_bound(cut_values_.cbegin() + beg, cut_values_.cbegin() + end, value);
    if (it == cut_values_.cend()) {
      it = cut_values_.cend() - 1;
    }
    BinIdx idx = it - cut_values_.cbegin();
    return idx;
  }

  BinIdx SearchBin(Entry const& e) {
    return SearchBin(e.fvalue, e.index);
  }
};

/* \brief An interface for building quantile cuts.
 *
 * `DenseCuts' always assumes there are `max_bins` for each feature, which makes it not
 * suitable for sparse dataset.  On the other hand `SparseCuts' uses `GetColumnBatches',
 * which doubles the memory usage, hence can not be applied to dense dataset.
 */
class CutsBuilder {
 public:
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;

 protected:
  HistogramCuts* p_cuts_;
  /* \brief return whether group for ranking is used. */
  static bool UseGroup(DMatrix* dmat);

 public:
  explicit CutsBuilder(HistogramCuts* p_cuts) : p_cuts_{p_cuts} {}
  virtual ~CutsBuilder() = default;

  static uint32_t SearchGroupIndFromRow(
      std::vector<bst_uint> const& group_ptr, size_t const base_rowid) {
    using KIt = std::vector<bst_uint>::const_iterator;
    KIt res = std::lower_bound(group_ptr.cbegin(), group_ptr.cend() - 1, base_rowid);
    // Cannot use CHECK_NE because it will try to print the iterator.
    bool const found = res != group_ptr.cend() - 1;
    if (!found) {
      LOG(FATAL) << "Row " << base_rowid << " does not lie in any group!";
    }
    uint32_t group_ind = std::distance(group_ptr.cbegin(), res);
    return group_ind;
  }

  void AddCutPoint(WXQSketch::SummaryContainer const& summary) {
    if (summary.size > 1 && summary.size <= 16) {
      /* specialized code categorial / ordinal data -- use midpoints */
      for (size_t i = 1; i < summary.size; ++i) {
        bst_float cpt = (summary.data[i].value + summary.data[i - 1].value) / 2.0f;
        if (i == 1 || cpt > p_cuts_->cut_values_.back()) {
          p_cuts_->cut_values_.push_back(cpt);
        }
      }
    } else {
      for (size_t i = 2; i < summary.size; ++i) {
        bst_float cpt = summary.data[i - 1].value;
        if (i == 2 || cpt > p_cuts_->cut_values_.back()) {
          p_cuts_->cut_values_.push_back(cpt);
        }
      }
    }
  }

  /* \brief Build histogram indices. */
  virtual void Build(DMatrix* dmat, uint32_t const max_num_bins) = 0;
};

/*! \brief Cut configuration for sparse dataset. */
class SparseCuts : public CutsBuilder {
  /* \brief Distrbute columns to each thread according to number of entries. */
  static std::vector<size_t> LoadBalance(SparsePage const& page, size_t const nthreads);
  Monitor monitor_;

 public:
  explicit SparseCuts(HistogramCuts* container) :
      CutsBuilder(container) {
    monitor_.Init(__FUNCTION__);
  }

  /* \brief Concatonate the built cuts in each thread. */
  void Concat(std::vector<std::unique_ptr<SparseCuts>> const& cuts, uint32_t n_cols);
  /* \brief Build histogram indices in single thread. */
  void SingleThreadBuild(SparsePage const& page, MetaInfo const& info,
                         uint32_t max_num_bins,
                         bool const use_group_ind,
                         uint32_t beg, uint32_t end, uint32_t thread_id);
  void Build(DMatrix* dmat, uint32_t const max_num_bins) override;
};

/*! \brief Cut configuration for dense dataset. */
class DenseCuts  : public CutsBuilder {
 protected:
  Monitor monitor_;

 public:
  explicit DenseCuts(HistogramCuts* container) :
      CutsBuilder(container) {
    monitor_.Init(__FUNCTION__);
  }
  void Init(std::vector<WXQSketch>* sketchs, uint32_t max_num_bins);
  void Build(DMatrix* p_fmat, uint32_t max_num_bins) override;
};

// FIXME(trivialfis): Merge this into generic cut builder.
/*! \brief Builds the cut matrix on the GPU.
 *
 *  \return The row stride across the entire dataset.
 */
size_t DeviceSketch(int device,
                    int max_bin,
                    int gpu_batch_nrows,
                    DMatrix* dmat,
                    HistogramCuts* hmat);

/*!
 * \brief preprocessed global index matrix, in CSR format
 *  Transform floating values to integer index in histogram
 *  This is a global histogram index.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  std::vector<uint32_t> index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  HistogramCuts cut;
  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_num_bins);
  // get i-th row
  inline GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i],
            static_cast<GHistIndexRow::index_type>(
                row_ptr[i + 1] - row_ptr[i])};
  }
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

 private:
  std::vector<size_t> hit_count_tloc_;
};

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

/*!
 * \brief histogram of gradient statistics for a single node.
 *  Consists of multiple GradStats, each entry showing total gradient statistics
 *     for that particular bin
 *  Uses global bin id so as to represent all features simultaneously
 */
using GHistRow = Span<tree::GradStats>;

/*!
 * \brief fill a histogram by zeros
 */
void InitilizeHistByZeroes(GHistRow hist, size_t begin, size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, const GHistRow add, size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHist(GHistRow dst, const GHistRow src, size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHist(GHistRow dst, const GHistRow src1, const GHistRow src2,
                     size_t begin, size_t end);

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollection {
 public:
  // access histogram for i-th node
  GHistRow operator[](bst_uint nid) const {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    CHECK_NE(row_ptr_[nid], kMax);
    tree::GradStats* ptr =
        const_cast<tree::GradStats*>(dmlc::BeginPtr(data_) + row_ptr_[nid]);
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

  std::vector<tree::GradStats> data_;

  /*! \brief row_ptr_[nid] locates bin for histogram of node nid */
  std::vector<size_t> row_ptr_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 * Supports processing multiple tree-nodes for nested parallelism
 * Able to reduce histograms across threads in efficient way
 */
class ParallelGHistBuilder {
 public:
  void Init(size_t nbins) {
    if (nbins != nbins_) {
      hist_buffer_.Init(nbins);
      nbins_ = nbins;
    }
  }

  // Add new elements if needed, mark all hists as unused
  // targeted_hists - already allocated hists which should contain final results after Reduce() call
  void Reset(size_t nthreads, size_t nodes, const BlockedSpace2d& space,
             const std::vector<GHistRow>& targeted_hists) {
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
  GHistRow GetInitializedHist(size_t tid, size_t nid) {
    CHECK_LT(nid, nodes_);
    CHECK_LT(tid, nthreads_);

    size_t idx = tid_nid_to_hist_.at({tid, nid});
    GHistRow hist = hist_memory_[idx];

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

    GHistRow dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;
        const size_t idx = tid_nid_to_hist_.at({tid, nid});
        GHistRow src = hist_memory_[idx];

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
  HistCollection hist_buffer_;
  /*!
   * \brief Marks which hists were used, it means that they should be merged.
   * Contains only {true or false} values
   * but 'int' is used instead of 'bool', because std::vector<bool> isn't thread safe
   */
  std::vector<int> hist_was_used_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  std::vector<bool> threads_to_nids_map_;
  /*! \brief Contains histograms for final results  */
  std::vector<GHistRow> targeted_hists_;
  /*! \brief Allocated memory for histograms used for construction  */
  std::vector<GHistRow> hist_memory_;
  /*! \brief map pair {tid, nid} to index of allocated histogram from hist_memory_  */
  std::map<std::pair<size_t, size_t>, size_t> tid_nid_to_hist_;
};

/*!
 * \brief builder for histograms of gradient statistics
 */
class GHistBuilder {
 public:
  // initialize builder
  inline void Init(size_t nthread, uint32_t nbins) {
    nthread_ = nthread;
    nbins_ = nbins;
  }

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 GHistRow hist);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      GHistRow hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

  uint32_t GetNumBins() {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  uint32_t nbins_;
};


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
