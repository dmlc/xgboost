/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>
#include <limits>
#include <vector>

#include "../../common/hist_util.h"
#include "../../data/gradient_index.h"
#include "expand_entry.h"
#include "rabit/rabit.h"
#include "xgboost/tree_model.h"
#include "../../common/hist_builder.h"
#include "../../common/opt_partition_builder.h"

namespace xgboost {
namespace tree {
template <typename ExpandEntry>
class HistogramBuilder {
  /*! \brief culmulative histogram of gradients. */
  common::HistCollection hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  common::HistCollection hist_local_worker_;
  common::GHistBuilder builder_;
  common::ParallelGHistBuilder buffer_;
  BatchParam param_;
  int32_t n_threads_{-1};
  size_t n_batches_{0};
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_{false};

 public:
  /**
   * \param total_bins       Total number of bins across all features
   * \param max_bin_per_feat Maximum number of bins per feature, same as the `max_bin`
   *                         training parameter.
   * \param n_threads        Number of threads.
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */

  void Reset(uint32_t total_bins, int32_t max_bin_per_feat, int32_t n_threads, size_t n_batches,
             int32_t max_depth,
             bool is_distributed = rabit::IsDistributed()) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    n_batches_ = n_batches;
    hist_.Init(total_bins);
    hist_local_worker_.Init(total_bins);
    buffer_.Init(total_bins);
    buffer_.AllocateHistBufer(max_depth, n_threads);
    builder_ = common::GHistBuilder();
    is_distributed_ = is_distributed;
    // Workaround s390x gcc 7.5.0
    auto DMLC_ATTRIBUTE_UNUSED __force_instantiation = &GradientPairPrecise::Reduce;
  }

  template <typename BinIdxType, bool any_missing, bool is_root, typename PartitionType>
  void
  BuildLocalHistograms(size_t page_idx,
                       GHistIndexMatrix const &gidx,
                       std::vector<ExpandEntry> nodes_for_explicit_hist_build,
                       const std::vector<GradientPair> &gpair_h,
                       const PartitionType* p_opt_partition_builder,
                       // template?
                       std::vector<uint16_t>* p_node_ids) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    std::vector<uint16_t>& node_ids = *p_node_ids;
    int nthreads = this->n_threads_;

    #pragma omp parallel num_threads(nthreads)
    {
      size_t tid = omp_get_thread_num();
      const BinIdxType* numa = gidx.index.data<BinIdxType>();
      const std::vector<common::Slice>& local_slices =
        opt_partition_builder.GetSlices(tid);
      buffer_.AllocateHistForLocalThread(
        opt_partition_builder.GetNodes(tid), tid);
      for (const common::Slice& slice : local_slices) {
        const uint32_t* rows = slice.addr;
        // CHECK(rows != nullptr);
        builder_.template BuildHist<BinIdxType, any_missing, is_root>(
          gpair_h, rows, slice.b, slice.e, gidx, numa, node_ids.data(),
          &buffer_.histograms_buffer[tid], buffer_.local_threads_mapping[tid].data(),
          opt_partition_builder.base_rowid);
      }
    }
  }

  void
  AddHistRows(int *starting_index, int *sync_count,
              std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
              std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
              RegTree *p_tree) {
    if (is_distributed_) {
      this->AddHistRowsDistributed(starting_index, sync_count,
                                   nodes_for_explicit_hist_build,
                                   nodes_for_subtraction_trick, p_tree);
    } else {
      this->AddHistRowsLocal(starting_index, sync_count,
                             nodes_for_explicit_hist_build,
                             nodes_for_subtraction_trick);
    }
  }

  /** Main entry point of this class, build histogram for tree nodes. */
  template <typename BinIdxType, bool is_root, typename PartitionType>
  void BuildHist(size_t page_id, GHistIndexMatrix const &gidx,
                 RegTree *p_tree,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 std::vector<GradientPair> const &gpair,
                 const PartitionType* p_opt_partition_builder,
                 std::vector<uint16_t>* p_node_ids,
                 const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    if (page_id == 0) {
      this->AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build,
                        nodes_for_subtraction_trick, p_tree);
    }
    if (gidx.IsDense()) {
      this->BuildLocalHistograms<BinIdxType, false, is_root>(page_id, gidx,
                                        nodes_for_explicit_hist_build,
                                        gpair, p_opt_partition_builder, p_node_ids);
    } else {
      this->BuildLocalHistograms<uint32_t, true, is_root>(page_id, gidx,
                                       nodes_for_explicit_hist_build,
                                       gpair, p_opt_partition_builder, p_node_ids);
    }

    CHECK_GE(n_batches_, 1);
    if (page_id != n_batches_ - 1) {
      return;
    }

    if (is_distributed_) {
      this->SyncHistogramDistributed(p_tree, nodes_for_explicit_hist_build,
                                     nodes_for_subtraction_trick,
                                     starting_index, sync_count,
                                     p_opt_partition_builder, merged_thread_ids);
    } else {
      this->SyncHistogramLocal(p_tree, nodes_for_explicit_hist_build,
                               nodes_for_subtraction_trick, starting_index,
                               sync_count, p_opt_partition_builder, merged_thread_ids);
    }
  }

  template <typename PartitionType>
  void SyncHistogramDistributed(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count,
      const PartitionType* p_opt_partition_builder,
      const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    const size_t nbins = hist_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);
    common::ParallelFor2d(
        space, n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          if (merged_thread_ids) {
            this->buffer_.ReduceHist(reinterpret_cast<double*>(this_hist.data()),
                                    (*merged_thread_ids)[node],
                                    entry.nid, r.begin(), r.end());
          } else {
            this->buffer_.ReduceHist(reinterpret_cast<double*>(this_hist.data()),
                                    opt_partition_builder.GetThreadIdsForNode(entry.nid),
                                    entry.nid, r.begin(), r.end());
          }
          // Store posible parent node
          auto this_local = hist_local_worker_[entry.nid];
          common::CopyHist(this_local, this_hist, r.begin(), r.end());

          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_local_worker_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
            // Store posible parent node
            auto sibling_local = hist_local_worker_[subtraction_node_id];
            common::CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
          }
        });

    rabit::Allreduce<rabit::op::Sum>(reinterpret_cast<double*>(this->hist_[starting_index].data()),
                                     hist_.GetNumBins() * sync_count * 2);

    ParallelSubtractionHist(space, nodes_for_explicit_hist_build,
                            nodes_for_subtraction_trick, p_tree);

    common::BlockedSpace2d space2(
        nodes_for_subtraction_trick.size(), [&](size_t) { return nbins; },
        1024);
    ParallelSubtractionHist(space2, nodes_for_subtraction_trick,
                            nodes_for_explicit_hist_build, p_tree);
  }

// <<<<<<< HEAD
  template <typename PartitionType>
  void SyncHistogramLocal(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count,
      const PartitionType* p_opt_partition_builder,
      const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    const size_t nbins = this->hist_.GetNumBins();
// =======
//   void SyncHistogramLocal(RegTree *p_tree,
//                           std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
//                           std::vector<ExpandEntry> const &nodes_for_subtraction_trick) {
//     const size_t nbins = this->builder_.GetNumBins();
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);

    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          if (merged_thread_ids) {
            this->buffer_.ReduceHist(reinterpret_cast<double*>(this_hist.data()),
                                    (*merged_thread_ids)[node],
                                    entry.nid, r.begin(), r.end());
          } else {
            this->buffer_.ReduceHist(reinterpret_cast<double*>(this_hist.data()),
                                    opt_partition_builder.GetThreadIdsForNode(entry.nid),
                                    entry.nid, r.begin(), r.end());
          }
          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
          }
        });
  }

 public:
  /* Getters for tests. */
  common::HistCollection const &Histogram() { return hist_; }
  auto& Buffer() { return buffer_; }

 private:
  void
  ParallelSubtractionHist(const common::BlockedSpace2d &space,
                          const std::vector<ExpandEntry> &nodes,
                          const std::vector<ExpandEntry> &subtraction_nodes,
                          const RegTree *p_tree) {
    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes[node];
          if (!((*p_tree)[entry.nid].IsLeftChild())) {
            auto this_hist = this->hist_[entry.nid];

            if (!(*p_tree)[entry.nid].IsRoot()) {
              const int subtraction_node_id = subtraction_nodes[node].nid;
              auto parent_hist = hist_[(*p_tree)[entry.nid].Parent()];
              auto sibling_hist = hist_[subtraction_node_id];
              common::SubtractionHist(this_hist, parent_hist, sibling_hist,
                                      r.begin(), r.end());
            }
          }
        });
  }

  // Add a tree node to histogram buffer in local training environment.
  void AddHistRowsLocal(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick) {
    for (auto const &entry : nodes_for_explicit_hist_build) {
      int nid = entry.nid;
      this->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
    }
    (*sync_count) = nodes_for_explicit_hist_build.size();

    for (auto const &node : nodes_for_subtraction_trick) {
      this->hist_.AddHistRow(node.nid);
    }
    this->hist_.AllocateAllData();
  }

  void AddHistRowsDistributed(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      RegTree *p_tree) {
    const size_t explicit_size = nodes_for_explicit_hist_build.size();
    const size_t subtaction_size = nodes_for_subtraction_trick.size();
    std::vector<int> merged_node_ids(explicit_size + subtaction_size);
    for (size_t i = 0; i < explicit_size; ++i) {
      merged_node_ids[i] = nodes_for_explicit_hist_build[i].nid;
    }
    for (size_t i = 0; i < subtaction_size; ++i) {
      merged_node_ids[explicit_size + i] = nodes_for_subtraction_trick[i].nid;
    }
    std::sort(merged_node_ids.begin(), merged_node_ids.end());
    int n_left = 0;
    for (auto const &nid : merged_node_ids) {
      if ((*p_tree)[nid].IsLeftChild()) {
        this->hist_.AddHistRow(nid);
        (*starting_index) = std::min(nid, (*starting_index));
        n_left++;
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    for (auto const &nid : merged_node_ids) {
      if (!((*p_tree)[nid].IsLeftChild())) {
        this->hist_.AddHistRow(nid);
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    this->hist_.AllocateAllData();
    this->hist_local_worker_.AllocateAllData();
    (*sync_count) = std::max(1, n_left);
  }
};

// Construct a work space for building histogram.  Eventually we should move this
// function into histogram builder once hist tree method supports external memory.
template <typename Partitioner>
common::BlockedSpace2d ConstructHistSpace(Partitioner const &partitioners,
                                          std::vector<CPUExpandEntry> const &nodes_to_build) {
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
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
