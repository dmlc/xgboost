/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.h
 * \brief Utility for fast histogram aggregation
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_SYCL_H_
#define XGBOOST_COMMON_HIST_UTIL_SYCL_H_

#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <map>

#include "../../src/common/row_set.h"
#include "../../src/common/threading_utils.h"
#include "../../src/tree/param.h"
#include "../../src/common/quantile.h"
#include "../../src/common/timer.h"
#include "../../src/common/hist_util.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace common {
/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;
using GHistRowSycl = Span<tree::GradStats>; // Span containing USM pointer

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollectionSycl {
 public:
  ~HistCollectionSycl() {
    for (size_t i = 0; i < data_.size(); i++) {
      if (data_[i]) {
        cl::sycl::free(data_[i], qu_);
      }
    }
  }
  // access histogram for i-th node
  GHistRowSycl operator[](bst_uint nid) const {
    tree::GradStats* missed_ptr = nullptr;
    CHECK_NE(data_[nid], missed_ptr);
    tree::GradStats* ptr = data_[nid];
    return {ptr, nbins_};
  }

  // have we computed a histogram for i-th node?
  bool RowExists(bst_uint nid) const {
    return (nid < data_.size() && data_[nid] != 0);
  }

  // initialize histogram collection
  void Init(cl::sycl::queue qu, uint32_t nbins) {
    if (nbins_ != nbins) {
      nbins_ = nbins;
      // quite expensive operation, so let's do this only once
      for (size_t i = 0; i < data_.size(); i++) {
      	if (data_[i]) {
          cl::sycl::free(data_[i], qu_);
        }
      }
      data_.clear();
    }
    n_nodes_added_ = 0;
    qu_ = qu;
  }

  // create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
    if (data_.size() < nid + 1) {
      data_.resize(nid + 1, nullptr);
    }
    if (!data_[nid]) {
      data_[nid] = cl::sycl::malloc_device<tree::GradStats>(nbins_, qu_);
      n_nodes_added_++;
    }
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  /*! \brief amount of active nodes in hist collection */
  uint32_t n_nodes_added_ = 0;

  std::vector<tree::GradStats*> data_;

  cl::sycl::queue qu_;
};

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl add, size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl src, size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl src1, const GHistRowSycl src2,
                     size_t begin, size_t end);

/*!
 * \brief Stores temporary histograms to compute them in parallel
 * Supports processing multiple tree-nodes for nested parallelism
 * Able to reduce histograms across threads in efficient way
 */
class ParallelGHistBuilderSycl {
 public:
  void Init(cl::sycl::queue qu, size_t nbins) {
    if (nbins != nbins_) {
      hist_buffer_.Init(qu, nbins);
      nbins_ = nbins;
    }
  }

  // Add new elements if needed, mark all hists as unused
  // targeted_hists - already allocated hists which should contain final results after Reduce() call
  void Reset(cl::sycl::queue qu, size_t nthreads, size_t nodes, const BlockedSpace2d& space,
             const std::vector<GHistRowSycl>& targeted_hists) {
    hist_buffer_.Init(qu, nbins_);
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
  GHistRowSycl GetInitializedHist(size_t tid, size_t nid) {
    CHECK_LT(nid, nodes_);
    CHECK_LT(tid, nthreads_);

    size_t idx = tid_nid_to_hist_.at({tid, nid});
    GHistRowSycl hist = hist_memory_[idx];

    if (!hist_was_used_[tid * nodes_ + nid]) {
      InitilizeHistByZeroes(hist, 0, hist.size());
      hist_was_used_[tid * nodes_ + nid] = static_cast<int>(true);
    }

    return hist;
  }

  void ReduceHistSycl(cl::sycl::queue qu, size_t nid, size_t begin, size_t end) {
    CHECK_GT(end, begin);
    CHECK_LT(nid, nodes_);

    GHistRowSycl dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;
        const size_t idx = tid_nid_to_hist_.at({tid, nid});
        GHistRowSycl src = hist_memory_[idx];

        if (dst.data() != src.data()) {
          IncrementHistSycl(qu, dst, src, begin, end);
        }
      }
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

  const size_t preferableGroupSize_ = 256;
  const size_t maxLocalHistograms_ = 256;

  /*! \brief number of bins in each histogram */
  size_t nbins_ = 0;
  /*! \brief number of threads for parallel computation */
  size_t nthreads_ = 0;
  /*! \brief number of nodes which will be processed in parallel  */
  size_t nodes_ = 0;
  /*! \brief Buffer for additional histograms for Parallel processing  */
  HistCollectionSycl hist_buffer_;
  /*!
   * \brief Marks which hists were used, it means that they should be merged.
   * Contains only {true or false} values
   * but 'int' is used instead of 'bool', because std::vector<bool> isn't thread safe
   */
  std::vector<int> hist_was_used_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  std::vector<bool> threads_to_nids_map_;
  /*! \brief Contains histograms for final results  */
  std::vector<GHistRowSycl> targeted_hists_;
  /*! \brief Allocated memory for histograms used for construction  */
  std::vector<GHistRowSycl> hist_memory_;
  /*! \brief map pair {tid, nid} to index of allocated histogram from hist_memory_  */
  std::map<std::pair<size_t, size_t>, size_t> tid_nid_to_hist_;
};

/*!
 * \brief builder for histograms of gradient statistics
 */
class GHistBuilder {
 public:
  GHistBuilderSycl() = default;
  GHistBuilderSycl(size_t nthread, uint32_t nbins) : nthread_{nthread}, nbins_{nbins} {}

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 GHistRowSycl hist,
                 bool isDense);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      GHistRowSycl hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRowSycl self, GHistRowSycl sibling, GHistRowSycl parent);

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
#endif  // XGBOOST_COMMON_HIST_UTIL_SYCL_H_
