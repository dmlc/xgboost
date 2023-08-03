/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>
#include <limits>
#include <vector>

#include "../../collective/communicator-inl.h"
#include "../../common/hist_util.h"
#include "../../data/gradient_index.h"
#include "expand_entry.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
template <typename ExpandEntry>
class HistogramBuilder {
  /*! \brief culmulative histogram of gradients. */
  common::HistCollection hist_;
  common::ParallelGHistBuilder buffer_;
  BatchParam param_;
  int32_t n_threads_{-1};
  size_t n_batches_{0};
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_{false};
  bool is_col_split_{false};

 public:
  /**
   * \param total_bins       Total number of bins across all features
   * \param max_bin_per_feat Maximum number of bins per feature, same as the `max_bin`
   *                         training parameter.
   * \param n_threads        Number of threads.
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */
  void Reset(uint32_t total_bins, BatchParam p, int32_t n_threads, size_t n_batches,
             bool is_distributed, bool is_col_split) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    n_batches_ = n_batches;
    param_ = p;
    hist_.Init(total_bins);
    buffer_.Init(total_bins);
    is_distributed_ = is_distributed;
    is_col_split_ = is_col_split;
  }

  template <bool any_missing>
  void BuildLocalHistograms(size_t page_idx, common::BlockedSpace2d space,
                            GHistIndexMatrix const &gidx,
                            std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                            common::RowSetCollection const &row_set_collection,
                            common::Span<GradientPair const> gpair_h, bool force_read_by_column) {
    const size_t n_nodes = nodes_for_explicit_hist_build.size();
    CHECK_GT(n_nodes, 0);

    std::vector<common::GHistRow> target_hists(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
      auto const nidx = nodes_for_explicit_hist_build[i].nid;
      target_hists[i] = hist_[nidx];
    }
    if (page_idx == 0) {
      // FIXME(jiamingy): Handle different size of space.  Right now we use the maximum
      // partition size for the buffer, which might not be efficient if partition sizes
      // has significant variance.
      buffer_.Reset(this->n_threads_, n_nodes, space, target_hists);
    }

    // Parallel processing by nodes and data in each node
    common::ParallelFor2d(space, this->n_threads_, [&](size_t nid_in_set, common::Range1d r) {
      const auto tid = static_cast<unsigned>(omp_get_thread_num());
      const int32_t nid = nodes_for_explicit_hist_build[nid_in_set].nid;
      auto elem = row_set_collection[nid];
      auto start_of_row_set = std::min(r.begin(), elem.Size());
      auto end_of_row_set = std::min(r.end(), elem.Size());
      auto rid_set = common::RowSetCollection::Elem(elem.begin + start_of_row_set,
                                                    elem.begin + end_of_row_set, nid);
      auto hist = buffer_.GetInitializedHist(tid, nid_in_set);
      if (rid_set.Size() != 0) {
        common::BuildHist<any_missing>(gpair_h, rid_set, gidx, hist, force_read_by_column);
      }
    });
  }

  void AddHistRows(int *starting_index,
                   std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                   std::vector<ExpandEntry> const &nodes_for_subtraction_trick) {
    for (auto const &entry : nodes_for_explicit_hist_build) {
      int nid = entry.nid;
      this->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
    }

    for (auto const &node : nodes_for_subtraction_trick) {
      this->hist_.AddHistRow(node.nid);
    }
    this->hist_.AllocateAllData();
  }

  /** Main entry point of this class, build histogram for tree nodes. */
  void BuildHist(size_t page_id, common::BlockedSpace2d space, GHistIndexMatrix const &gidx,
                 RegTree const *p_tree, common::RowSetCollection const &row_set_collection,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 common::Span<GradientPair const> gpair, bool force_read_by_column = false) {
    int starting_index = std::numeric_limits<int>::max();
    if (page_id == 0) {
      this->AddHistRows(&starting_index, nodes_for_explicit_hist_build,
                        nodes_for_subtraction_trick);
    }
    if (gidx.IsDense()) {
      this->BuildLocalHistograms<false>(page_id, space, gidx, nodes_for_explicit_hist_build,
                                        row_set_collection, gpair, force_read_by_column);
    } else {
      this->BuildLocalHistograms<true>(page_id, space, gidx, nodes_for_explicit_hist_build,
                                       row_set_collection, gpair, force_read_by_column);
    }

    CHECK_GE(n_batches_, 1);
    if (page_id != n_batches_ - 1) {
      return;
    }

    this->SyncHistogram(p_tree, nodes_for_explicit_hist_build,
                                   nodes_for_subtraction_trick, starting_index);
  }
  /** same as the other build hist but handles only single batch data (in-core) */
  void BuildHist(size_t page_id, GHistIndexMatrix const &gidx, RegTree *p_tree,
                 common::RowSetCollection const &row_set_collection,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 common::Span<GradientPair const> gpair, bool force_read_by_column = false) {
    const size_t n_nodes = nodes_for_explicit_hist_build.size();
    // create space of size (# rows in each node)
    common::BlockedSpace2d space(
        n_nodes,
        [&](size_t nidx_in_set) {
          const int32_t nidx = nodes_for_explicit_hist_build[nidx_in_set].nid;
          return row_set_collection[nidx].Size();
        },
        256);
    this->BuildHist(page_id, space, gidx, p_tree, row_set_collection, nodes_for_explicit_hist_build,
                    nodes_for_subtraction_trick, gpair, force_read_by_column);
  }

  void SyncHistogram(RegTree const *p_tree,
                     std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                     std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                     int starting_index) {
    auto n_bins = buffer_.TotalBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return n_bins; }, 1024);
    CHECK(hist_.IsContiguous());
    common::ParallelFor2d(space, this->n_threads_, [&](size_t node, common::Range1d r) {
      const auto &entry = nodes_for_explicit_hist_build[node];
      auto this_hist = this->hist_[entry.nid];
      // Merging histograms from each thread into once
      this->buffer_.ReduceHist(node, r.begin(), r.end());
    });

    if (is_distributed_ && !is_col_split_) {
      collective::Allreduce<collective::Operation::kSum>(
          reinterpret_cast<double *>(this->hist_[starting_index].data()),
          n_bins * nodes_for_explicit_hist_build.size() * 2);
    }

    common::ParallelFor2d(space, this->n_threads_, [&](std::size_t nidx_in_set, common::Range1d r) {
      const auto &entry = nodes_for_explicit_hist_build[nidx_in_set];
      auto this_hist = this->hist_[entry.nid];
      if (!p_tree->IsRoot(entry.nid)) {
        auto const parent_id = p_tree->Parent(entry.nid);
        auto const subtraction_node_id = nodes_for_subtraction_trick[nidx_in_set].nid;
        auto parent_hist = this->hist_[parent_id];
        auto sibling_hist = this->hist_[subtraction_node_id];
        common::SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
      }
    });
  }

 public:
  /* Getters for tests. */
  common::HistCollection const &Histogram() { return hist_; }
  auto &Buffer() { return buffer_; }
};

// Construct a work space for building histogram.  Eventually we should move this
// function into histogram builder once hist tree method supports external memory.
template <typename Partitioner, typename ExpandEntry = CPUExpandEntry>
common::BlockedSpace2d ConstructHistSpace(Partitioner const &partitioners,
                                          std::vector<ExpandEntry> const &nodes_to_build) {
  std::vector<size_t> partition_size(nodes_to_build.size(), 0);
  for (auto const &partition : partitioners) {
    size_t k = 0;
    for (auto node : nodes_to_build) {
      auto n_rows_in_node = partition.Partitions()[node.nid].Size();
      partition_size[k] = std::max(partition_size[k], n_rows_in_node);
      k++;
    }
  }
  common::BlockedSpace2d space{
      nodes_to_build.size(), [&](size_t nidx_in_set) { return partition_size[nidx_in_set]; }, 256};
  return space;
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
