/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>  // for fill, min, max, copy_n
#include <cinttypes>  // for int32_t, int64_t
#include <cstddef>    // for size_t
#include <map>        // for map
#include <queue>      // for queue
#include <utility>    // for as_const, pair
#include <vector>     // for vector

#include "../../collective/communicator-inl.h"
#include "../../common/hist_util.h"
#include "../../common/io.h"  // for MallocResource
#include "../../data/gradient_index.h"
#include "expand_entry.h"
#include "param.h"               // for HistMakerTrainParam
#include "xgboost/base.h"        // for bst_node_t, bst_bin_t
#include "xgboost/context.h"     // for Context
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
// fixme: restrict it to single thread instead.
/**
 * @brief CPU storage for node histogram.
 */
class HistogramStorage {
  std::vector<GradientPairPrecise> data_;
  // maps node index to histogram buffer offset in data_
  std::map<bst_node_t, std::size_t> nidx_map_;
  bst_bin_t n_bins_{0};
  std::size_t size_{0};

  std::queue<std::size_t> free_slots_;

  void AllocNode(bst_node_t nidx, std::size_t offset) {
    this->nidx_map_[nidx] = offset;
    this->size_ += n_bins_;
    auto hist = common::Span{data_.data(), data_.size()}.subspan(offset, n_bins_);
    common::InitilizeHistByZeroes(hist, 0, hist.size());
  }

  void MarkFree(typename decltype(nidx_map_)::const_iterator it) {
    if (it != nidx_map_.cend()) {
      auto offset = it->second;
      free_slots_.push(offset);
      auto hist = common::Span{data_.data(), data_.size()}.subspan(offset, n_bins_);
      nidx_map_.erase(it);
      CHECK_GE(this->size_, n_bins_);
      this->size_ -= n_bins_;
    }
  }

 public:
  HistogramStorage() = default;
  explicit HistogramStorage(bst_bin_t n_total_bins) { this->Reset(n_total_bins); }
  [[nodiscard]] std::size_t Bytes() const {
    return common::Span{data_.data(), data_.size()}.size_bytes();
  }
  void Reset(bst_bin_t n_total_bins) {
    if (n_bins_ != 0) {
      CHECK_EQ(n_bins_, n_total_bins);
    }
    n_bins_ = n_total_bins;

    while (!nidx_map_.empty()) {
      this->MarkFree(nidx_map_.cbegin());
    }
    CHECK_EQ(this->size_, 0);
  }
  [[nodiscard]] bst_bin_t TotalBins() const { return n_bins_; }

  void AllocateHistograms(common::Span<bst_node_t const> nodes) {
    bst_node_t n_nodes = nodes.size();
    if (n_nodes > 16) {  // fixme: for testing only
      this->Reset(n_bins_);
    }

    std::cout << "n_nodes: " << n_nodes << std::endl;
    bst_node_t nidx_in_set = 0;
    // reuse previously allocated histograms
    while (!this->free_slots_.empty() && nidx_in_set < n_nodes) {
      auto offset = free_slots_.front();
      this->free_slots_.pop();
      this->AllocNode(nodes[nidx_in_set], offset);
      nidx_in_set++;
    }

    // handle nodes that don't have reusable histogram buffer

    // grow-only cache
    CHECK_GE(this->data_.size(), this->size_);
    auto n_remaining_nodes = n_nodes - nidx_in_set;
    CHECK_GE(n_remaining_nodes, 0);
    auto n_alloc_bins = n_remaining_nodes * this->n_bins_;

    auto new_size = n_alloc_bins + this->size_;
    auto prev = this->size_;
    if (new_size > this->data_.size()) {
      CHECK_EQ(prev, this->data_.size());
      this->data_.resize(new_size);
    }
    for (; nidx_in_set < n_nodes; ++nidx_in_set) {
      auto nidx = nodes[nidx_in_set];
      this->AllocNode(nidx, prev);
      prev += this->n_bins_;
    }
    std::cout << "data.size:" << data_.size()
              << " kbytes:" << common::Span{data_.data(), data_.size()}.size_bytes() / 1024
              << std::endl;
    // At this point, the number of nodes can exceed the maximum number of nodes specified
    // as we need to guarantee all histogram buffers are available.
  }
  void AllocateHistograms(std::vector<bst_node_t> const &nodes) {
    this->AllocateHistograms(common::Span{nodes.data(), nodes.size()});
  }

  [[nodiscard]] bool HistogramExist(bst_node_t nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend();
  }
  // fixme: const
  [[nodiscard]] common::GHistRow GetHist(bst_node_t nidx) {
    auto offset = nidx_map_.at(nidx);
    auto hist = common::Span{data_.data(), data_.size()}.subspan(offset, n_bins_);
    return hist;
  }
  [[nodiscard]] auto GetHist(bst_node_t nidx) const {
    auto offset = nidx_map_.at(nidx);
    auto hist = common::Span{data_.data(), data_.size()}.subspan(offset, n_bins_);
    return hist;
  }
  [[nodiscard]] common::GHistRow operator[](std::size_t i) { return this->GetHist(i); }
  [[nodiscard]] auto operator[](std::size_t i) const { return this->GetHist(i); }

  void MarkFree(bst_node_t nidx) {
    auto it = nidx_map_.find(nidx);
    this->MarkFree(it);
  }

  [[nodiscard]] bst_bin_t NumTotalBins() const { return n_bins_; }
  [[nodiscard]] std::size_t NodeCapacity() const { return nidx_map_.size(); }
  [[nodiscard]] std::size_t FreeSlots() const { return free_slots_.size(); }
};

class ParallelGHistCollection {
 public:
  void Init(bst_bin_t n_total_bins) {
    if (n_total_bins != nbins_) {
      hist_buffer_.Init(n_total_bins);
      nbins_ = n_total_bins;
    }
  }

  // Add new elements if needed, mark all hists as unused
  // targeted_hists - already allocated hists which should contain final results after Reduce() call
  void Reset(Context const *ctx, std::size_t n_nodes, const common::BlockedSpace2d &space,
             std::vector<common::GHistRow> const &targeted_hists) {
    hist_buffer_.Init(nbins_);
    tid_nid_to_hist_.clear();
    threads_to_nids_map_.clear();

    targeted_hists_ = targeted_hists;

    CHECK_EQ(n_nodes, targeted_hists.size());

    nodes_ = n_nodes;
    nthreads_ = ctx->Threads();

    MatchThreadsToNodes(space);
    AllocateAdditionalHistograms();
    MatchNodeNidPairToHist();

    hist_was_used_.resize(nthreads_ * nodes_);
    std::fill(hist_was_used_.begin(), hist_was_used_.end(), static_cast<int>(false));
  }

  // Get specified hist, initialize hist by zeros if it wasn't used before
  common::GHistRow GetInitializedHist(size_t tid, size_t nid) {
    CHECK_LT(nid, nodes_);
    CHECK_LT(tid, nthreads_);

    int idx = tid_nid_to_hist_.at({tid, nid});
    if (idx >= 0) {
      hist_buffer_.AllocateData(idx);
    }
    common::GHistRow hist = idx == -1 ? targeted_hists_[nid] : hist_buffer_[idx];

    if (!hist_was_used_[tid * nodes_ + nid]) {
      common::InitilizeHistByZeroes(hist, 0, hist.size());
      hist_was_used_[tid * nodes_ + nid] = static_cast<int>(true);
    }

    return hist;
  }

  // Reduce following bins (begin, end] for nid-node in dst across threads
  void ReduceHist(size_t nid, size_t begin, size_t end) const {
    CHECK_GT(end, begin);
    CHECK_LT(nid, nodes_);

    common::GHistRow dst = targeted_hists_[nid];

    bool is_updated = false;
    for (size_t tid = 0; tid < nthreads_; ++tid) {
      if (hist_was_used_[tid * nodes_ + nid]) {
        is_updated = true;

        int idx = tid_nid_to_hist_.at({tid, nid});
        common::ConstGHistRow src = idx == -1 ? targeted_hists_[nid] : hist_buffer_[idx];

        if (dst.data() != src.data()) {
          common::IncrementHist(dst, src, begin, end);
        }
      }
    }
    if (!is_updated) {
      // In distributed mode - some tree nodes can be empty on local machines,
      // So we need just set local hist by zeros in this case
      common::InitilizeHistByZeroes(dst, begin, end);
    }
  }

  void MatchThreadsToNodes(common::BlockedSpace2d const &space) {
    const size_t space_size = space.Size();
    const size_t chunck_size = space_size / nthreads_ + !!(space_size % nthreads_);

    threads_to_nids_map_.resize(nthreads_ * nodes_, false);

    for (size_t tid = 0; tid < nthreads_; ++tid) {
      size_t begin = chunck_size * tid;
      size_t end = std::min(begin + chunck_size, space_size);

      if (begin < space_size) {
        size_t nid_begin = space.GetFirstDimension(begin);
        size_t nid_end = space.GetFirstDimension(end - 1);

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

  [[nodiscard]] bst_bin_t TotalBins() const { return nbins_; }

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
  bst_bin_t nbins_ = 0;
  /*! \brief number of threads for parallel computation */
  size_t nthreads_ = 0;
  /*! \brief number of nodes which will be processed in parallel  */
  size_t nodes_ = 0;
  /*! \brief Buffer for additional histograms for Parallel processing  */
  common::HistCollection hist_buffer_;
  /*!
   * \brief Marks which hists were used, it means that they should be merged.
   * Contains only {true or false} values
   * but 'int' is used instead of 'bool', because std::vector<bool> isn't thread safe
   */
  std::vector<int> hist_was_used_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  std::vector<bool> threads_to_nids_map_;
  /*! \brief Contains histograms for final results  */
  std::vector<common::GHistRow> targeted_hists_;
  /*!
   * \brief map pair {tid, nid} to index of allocated histogram from hist_buffer_ and
   * targeted_hists_, -1 is reserved for targeted_hists_
   */
  std::map<std::pair<size_t, size_t>, int> tid_nid_to_hist_;
};

class HistAllreduceSpace {
  std::vector<GradientPairPrecise> data_;
  // Number of bins for each node.
  bst_bin_t n_bins_{0};
  bst_node_t n_nodes_;

 public:
  void ResizeNodes(std::size_t n_nodes) {
    auto n_items = n_nodes * n_bins_;
    data_.resize(n_items);
    n_nodes_ = n_nodes;
  }

  void Allreduce(Context const *ctx, HistogramStorage *p_hist,
                 common::Span<bst_node_t const> nodes_to_build,
                 common::Span<bst_node_t const> nodes_to_sub) {
    CHECK_GE(n_bins_, 0);
    auto n_candidates = this->n_nodes_;
    auto contiguous = common::Span{data_.data(), data_.size()};
    auto &hist_collection_ = *p_hist;
    common::ParallelFor(n_nodes_, ctx->Threads(), [&](auto nidx_in_set) {
      auto dst = contiguous.subspan(n_bins_ * nidx_in_set, static_cast<std::size_t>(n_bins_));
      common::GHistRow src;
      if (nidx_in_set < n_candidates) {
        src = hist_collection_.GetHist(nodes_to_build[nidx_in_set]);
      } else {
        src = hist_collection_.GetHist(nodes_to_sub[nidx_in_set - n_candidates]);
      }
      std::copy_n(src.data(), src.size(), dst.data());
    });
    static_assert(sizeof(GradientPairPrecise) == sizeof(double) * 2);
    collective::Allreduce<collective::Operation::kSum>(
        reinterpret_cast<double *>(contiguous.data()), contiguous.size() * 2);
    common::ParallelFor(n_nodes_, ctx->Threads(), [&](auto nidx_in_set) {
      common::GHistRow dest;
      if (nidx_in_set < n_candidates) {
        dest = hist_collection_.GetHist(nodes_to_build[nidx_in_set]);
      } else {
        dest = hist_collection_.GetHist(nodes_to_sub[nidx_in_set - n_candidates]);
      }
      auto src = common::ConstGHistRow{contiguous.data() + n_bins_ * nidx_in_set,
                                       static_cast<std::size_t>(n_bins_)};
      std::copy_n(src.data(), src.size(), dest.data());
    });
  }
};

constexpr std::size_t DefaultHistSpaceGran() { return 512; }
constexpr std::size_t DefaultHistSubSpaceGran() { return 2048; }

// Construct a work space for building histogram.  Eventually we should move this
// function into histogram builder once hist tree method supports external memory.
template <typename Partitioner>
common::BlockedSpace2d ConstructHistSpace(Partitioner const &partitioners,
                                          common::Span<bst_node_t const> nodes_to_build) {
  std::vector<size_t> partition_size(nodes_to_build.size(), 0);
  for (auto const &partition : partitioners) {
    std::size_t k = 0;
    for (auto node : nodes_to_build) {
      auto n_rows_in_node = partition.Partitions()[node].Size();
      partition_size[k] = std::max(partition_size[k], n_rows_in_node);
      k++;
    }
  }
  common::BlockedSpace2d space{nodes_to_build.size(),
                               [&](std::size_t nidx_in_set) { return partition_size[nidx_in_set]; },
                               DefaultHistSpaceGran()};
  return space;
}

template <typename ExpandEntry>
class HistogramBuilder {
  common::Monitor monitor_;  // fixme: use the one from the updater.

  /*! \brief culmulative histogram of gradients. */
  HistogramStorage hist_collection_;
  ParallelGHistCollection hist_buffer_;
  HistAllreduceSpace allreduce_space_;  // fixme: replace this with target hist

  BatchParam param_;
  /** @brief Number of threads used to build histogram. */
  std::int32_t n_threads_{-1};
  /** @brief Whether XGBoost is running in distributed environment. */
  bool is_distributed_{false};
  /** @brief Whether data is split by column. */
  bool is_col_split_{false};

 public:
  /**
   * @brief Decide which node as the build node. Used for multi-target trees.
   */
  static void AssignNodes(RegTree const *p_tree,
                          std::vector<MultiExpandEntry> const &valid_candidates,
                          common::Span<bst_node_t> nodes_to_build,
                          common::Span<bst_node_t> nodes_to_sub) {
    CHECK_EQ(nodes_to_build.size(), valid_candidates.size());

    std::size_t n_idx = 0;
    for (auto const &c : valid_candidates) {
      auto left_nidx = p_tree->LeftChild(c.nid);
      auto right_nidx = p_tree->RightChild(c.nid);
      CHECK_NE(left_nidx, c.nid);
      CHECK_NE(right_nidx, c.nid);

      auto build_nidx = left_nidx;
      auto subtract_nidx = right_nidx;
      auto lit =
          common::MakeIndexTransformIter([&](auto i) { return c.split.left_sum[i].GetHess(); });
      auto left_sum = std::accumulate(lit, lit + c.split.left_sum.size(), .0);
      auto rit =
          common::MakeIndexTransformIter([&](auto i) { return c.split.right_sum[i].GetHess(); });
      auto right_sum = std::accumulate(rit, rit + c.split.right_sum.size(), .0);
      auto fewer_right = right_sum < left_sum;
      if (fewer_right) {
        std::swap(build_nidx, subtract_nidx);
      }
      nodes_to_build[n_idx] = build_nidx;
      nodes_to_sub[n_idx] = subtract_nidx;
      ++n_idx;
    }
  }

  /**
   * @brief Decide which node as the build node.
   */
  static void AssignNodes(RegTree const *p_tree, std::vector<CPUExpandEntry> const &candidates,
                          common::Span<bst_node_t> nodes_to_build,
                          common::Span<bst_node_t> nodes_to_sub) {
    std::size_t n_idx = 0;
    for (auto const &c : candidates) {
      auto left_nidx = (*p_tree)[c.nid].LeftChild();
      auto right_nidx = (*p_tree)[c.nid].RightChild();
      auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();

      auto build_nidx = left_nidx;
      auto subtract_nidx = right_nidx;
      if (fewer_right) {
        std::swap(build_nidx, subtract_nidx);
      }
      nodes_to_build[n_idx] = build_nidx;
      nodes_to_sub[n_idx] = subtract_nidx;
      n_idx++;
    }
  }

 public:
  /**
   * \param total_bins       Total number of bins across all features
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */
  void Reset(Context const *ctx, bst_bin_t total_bins, BatchParam p, bool is_distributed,
             bool is_col_split) {
    n_threads_ = ctx->Threads();
    param_ = p;
    hist_collection_.Reset(total_bins);
    hist_buffer_.Init(total_bins);

    is_distributed_ = is_distributed;
    is_col_split_ = is_col_split;
  }

  template <bool any_missing>
  void BuildLocalHist(Context const *ctx, common::BlockedSpace2d space,
                      GHistIndexMatrix const &gidx, common::Span<bst_node_t const> nodes_to_build,
                      common::RowSetCollection const &row_set_collection,
                      linalg::VectorView<GradientPair const> gpair, bool force_read_by_column) {
    monitor_.Start(__func__);
    CHECK_EQ(ctx->Threads(), n_threads_);

    common::ParallelFor2d(space, ctx->Threads(), [&](std::size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      bst_node_t nidx = nodes_to_build[nidx_in_set];
      auto elem = row_set_collection[nidx];
      auto start_of_row_set = std::min(r.begin(), elem.Size());
      auto end_of_row_set = std::min(r.end(), elem.Size());
      auto rid_set = common::RowSetCollection::Elem(elem.begin + start_of_row_set,
                                                    elem.begin + end_of_row_set, nidx);
      auto hist = hist_buffer_.GetInitializedHist(tidx, nidx_in_set);

      if (rid_set.Size() != 0) {
        common::BuildHist<any_missing>(gpair.Values(), rid_set, gidx, hist, force_read_by_column);
      }
    });
    monitor_.Stop(__func__);
  }

  /**
   * @brief Build histogram for the root node.
   */
  template <bool any_missing>
  void BuildRootHist(Context const *ctx, std::size_t page_idx, common::BlockedSpace2d space,
                     GHistIndexMatrix const &gidx, linalg::VectorView<GradientPair const> gpair,
                     common::RowSetCollection const &collection,
                     std::vector<bst_node_t> const &nodes, bool force_read_by_column = false) {
    CHECK_EQ(nodes.size(), 1);
    CHECK_EQ(nodes.front(), RegTree::kRoot);
    // Add histogram node storage
    CHECK_EQ(ctx->Threads(), n_threads_);
    if (page_idx == 0) {
      CHECK_EQ(hist_collection_.NodeCapacity(), 0);
      hist_collection_.AllocateHistograms(nodes);
      std::vector<common::GHistRow> target_hist;
      for (auto nidx : nodes) {
        target_hist.emplace_back(hist_collection_[nidx]);
      }
      hist_buffer_.Reset(ctx, nodes.size(), space, target_hist);
    }
    this->BuildLocalHist<any_missing>(ctx, space, gidx, nodes, collection, gpair,
                                      force_read_by_column);
  }

  template <bool any_missing>
  void BuildHistLeftRightNodes(Context const *ctx, std::size_t page_idx,
                               common::BlockedSpace2d space, GHistIndexMatrix const &gidx,
                               common::RowSetCollection const &collection,
                               common::Span<bst_node_t const> all_nodes,
                               common::Span<bst_node_t const> nodes_to_build,
                               linalg::VectorView<GradientPair const> gpair,
                               bool force_read_by_column = false) {
    monitor_.Start(__func__);
    if (page_idx == 0) {
      hist_collection_.AllocateHistograms(all_nodes);
      std::vector<common::GHistRow> target_hist{};
      for (auto nidx : nodes_to_build) {
        target_hist.emplace_back(hist_collection_[nidx]);
      }
      hist_buffer_.Reset(ctx, nodes_to_build.size(), space, target_hist);
    }
    this->BuildLocalHist<any_missing>(ctx, space, gidx, nodes_to_build, collection, gpair,
                                      force_read_by_column);
    monitor_.Stop(__func__);
  }

  void ReduceHist(common::Span<bst_node_t const> nodes_to_build) {
    common::BlockedSpace2d space(
        nodes_to_build.size(), [&](std::size_t) { return hist_collection_.TotalBins(); },
        DefaultHistSubSpaceGran());
    common::ParallelFor2d(space, n_threads_, [&](std::size_t nidx_in_set, common::Range1d r) {
      // Merging histograms from each thread into one
      this->hist_buffer_.ReduceHist(nidx_in_set, r.begin(), r.end());
    });
  }

  void SubtractHist(Context const *ctx, RegTree const *p_tree,
                    common::Span<ExpandEntry const> candidates,
                    common::Span<bst_node_t const> nodes_to_build,
                    common::Span<bst_node_t const> nodes_to_sub) {
    monitor_.Start("sub");
    auto n_bins = this->hist_collection_.TotalBins();
    common::BlockedSpace2d sub_space(
        nodes_to_sub.size(), [&](std::size_t) { return n_bins; }, DefaultHistSubSpaceGran());
    common::ParallelFor2d(
        sub_space, ctx->Threads(), [&](std::size_t nidx_in_set, common::Range1d r) {
          auto nidx = p_tree->Parent(nodes_to_sub[nidx_in_set]);
          auto node_hist = hist_collection_.GetHist(nidx);

          bst_node_t sub_nidx = nodes_to_sub[nidx_in_set];
          auto subtract_hist = hist_collection_.GetHist(sub_nidx);

          auto built_nidx = nodes_to_build[nidx_in_set];
          auto built_hist = hist_collection_.GetHist(built_nidx);

          common::SubtractionHist(subtract_hist, node_hist, built_hist, r.begin(), r.end());
        });
    monitor_.Stop("sub");

    for (auto const &v : candidates) {
      this->hist_collection_.MarkFree(v.nid);
    }
  }

  void AllreduceHist(Context const *ctx, common::Span<ExpandEntry const> candidates,
                     common::Span<bst_node_t const> nodes_to_build,
                     common::Span<bst_node_t const> nodes_to_sub) {
    if (is_distributed_ && !is_col_split_) {
      auto n_nodes = candidates.size() * 2;
      allreduce_space_.ResizeNodes(n_nodes);
      // fixme: no need for nodes_to_sub
      this->allreduce_space_.Allreduce(ctx, &hist_collection_, nodes_to_build, nodes_to_sub);
    }
  }

 public:
  /* Getters for tests. */
  [[nodiscard]] HistogramStorage const &Histogram() const { return hist_collection_; }
  [[nodiscard]] HistogramStorage &Histogram() { return hist_collection_; }
};

template <typename ExpandEntry>
class MultiHistogramBuilder {
  std::vector<HistogramBuilder<ExpandEntry>> target_builders_;
  TrainParam const *param_;
  Context const *ctx_;
  bool is_distributed_{false};

 public:
  template <typename Partitioner>
  void BuildRootHist(DMatrix *p_fmat, RegTree const *p_tree,
                     std::vector<Partitioner> const &partitioners,
                     linalg::MatrixView<GradientPair const> gpair, ExpandEntry const &best,
                     BatchParam const &param) {
    auto n_targets = p_tree->NumTargets();
    CHECK_EQ(gpair.Shape(1), n_targets);
    CHECK_EQ(p_fmat->Info().num_row_, gpair.Shape(0));
    auto is_dense = p_fmat->IsDense() && !is_distributed_;
    std::vector<bst_node_t> nodes{best.nid};
    auto space = ConstructHistSpace(partitioners, nodes);
    std::size_t page_idx{0};
    for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
      for (bst_target_t t{0}; t < n_targets; ++t) {
        auto t_gpair = gpair.Slice(linalg::All(), t);
        if (is_dense) {
          this->target_builders_[t].template BuildRootHist<false>(
              ctx_, page_idx, space, gidx, t_gpair, partitioners[page_idx].Partitions(), nodes);
        } else {
          this->target_builders_[t].template BuildRootHist<true>(
              ctx_, page_idx, space, gidx, t_gpair, partitioners[page_idx].Partitions(), nodes);
        }
      }
      ++page_idx;
    }

    for (auto &v : target_builders_) {
      v.ReduceHist(nodes);
    }
  }

  template <typename Partitioner>
  void BuildHistLeftRight(DMatrix *p_fmat, RegTree const *p_tree,
                          std::vector<Partitioner> const &partitioners,
                          std::vector<ExpandEntry> const &valid_candidates,
                          linalg::MatrixView<GradientPair const> gpair, BatchParam const &param) {
    bool is_dense = p_fmat->IsDense() && !is_distributed_;
    std::vector<bst_node_t> all_nodes(valid_candidates.size() * 2);
    auto nodes_to_build = common::Span{all_nodes.data(), valid_candidates.size()};
    auto nodes_to_sub =
        common::Span{all_nodes.data(), all_nodes.size()}.subspan(valid_candidates.size());
    HistogramBuilder<ExpandEntry>::AssignNodes(p_tree, valid_candidates, nodes_to_build,
                                               nodes_to_sub);

    auto build_impl = [&](common::Span<bst_node_t> alloc_nodes,
                          common::Span<bst_node_t> nodes_to_build) {
      auto space = ConstructHistSpace(partitioners, nodes_to_build);
      std::size_t page_idx{0};
      for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
        CHECK_EQ(gpair.Shape(1), p_tree->NumTargets());
        for (bst_target_t t = 0; t < p_tree->NumTargets(); ++t) {
          auto t_gpair = gpair.Slice(linalg::All(), t);
          CHECK_EQ(t_gpair.Shape(0), p_fmat->Info().num_row_);
          if (is_dense) {
            this->target_builders_[t].template BuildHistLeftRightNodes<false>(
                ctx_, page_idx, space, page, partitioners[page_idx].Partitions(), alloc_nodes,
                nodes_to_build, t_gpair);
          } else {
            this->target_builders_[t].template BuildHistLeftRightNodes<true>(
                ctx_, page_idx, space, page, partitioners[page_idx].Partitions(), alloc_nodes,
                nodes_to_build, t_gpair);
          }
        }
        page_idx++;
      }

      for (auto &b : target_builders_) {
        b.ReduceHist(nodes_to_build);
      }
    };
    build_impl(all_nodes, nodes_to_build);

    std::vector<bst_node_t> additional_build;
    std::vector<ExpandEntry> avail_cand;
    std::vector<bst_node_t> avail_build;
    std::vector<bst_node_t> avail_sub;
    for (std::size_t i = 0; i < valid_candidates.size(); ++i) {
      if (target_builders_.front().Histogram().HistogramExist(valid_candidates[i].nid)) {
        avail_build.push_back(nodes_to_build[i]);
        avail_sub.push_back(nodes_to_sub[i]);
        avail_cand.push_back(valid_candidates[i]);
      } else {
        additional_build.push_back(nodes_to_sub[i]);
      }
    }
    for (auto &b : target_builders_) {
      b.SubtractHist(ctx_, p_tree, avail_cand, avail_build, avail_sub);
    }
    build_impl({}, additional_build);
  }

  [[nodiscard]] auto const &Histogram(bst_target_t t) const {
    return target_builders_[t].Histogram();
  }
  [[nodiscard]] auto &Histogram(bst_target_t t) { return target_builders_[t].Histogram(); }

  void Reset(Context const *ctx, bst_bin_t total_bins, bst_target_t n_targets, BatchParam const &p,
             bool is_distributed, bool is_col_split) {
    ctx_ = ctx;
    is_distributed_ = is_distributed;
    target_builders_.resize(n_targets);
    for (auto &v : target_builders_) {
      v.Reset(ctx, total_bins, p, is_distributed, is_col_split);
    }
  }
};
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
