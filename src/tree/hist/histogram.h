/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>
#include <limits>
#include <vector>

#include "rabit/rabit.h"
#include "xgboost/tree_model.h"
#include "../../common/hist_util.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT, typename ExpandEntry> class HistogramBuilder {
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
  using GHistRowT = common::GHistRow<GradientSumT>;

  /*! \brief culmulative histogram of gradients. */
  common::HistCollection<GradientSumT> hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  common::HistCollection<GradientSumT> hist_local_worker_;
  common::GHistBuilder<GradientSumT> builder_;
  common::ParallelGHistBuilder<GradientSumT> buffer_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> reducer_;
  int32_t max_bin_ {-1};
  int32_t n_threads_ {-1};
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_ {false};

 public:
  /**
   * \param total_bins       Total number of bins across all features
   * \param max_bin_per_feat Maximum number of bins per feature, same as the `max_bin`
   *                         training parameter.
   * \param n_threads        Number of threads.
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */
  void Reset(uint32_t total_bins, int32_t max_bin_per_feat, int32_t n_threads,
             bool is_distributed = rabit::IsDistributed()) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    CHECK_GE(max_bin_per_feat, 2);
    max_bin_ = max_bin_per_feat;
    hist_.Init(total_bins);
    hist_local_worker_.Init(total_bins);
    buffer_.Init(total_bins);
    builder_ = common::GHistBuilder<GradientSumT>(n_threads, total_bins);
    is_distributed_ = is_distributed;
  }

  template <bool any_missing>
  void
  BuildLocalHistograms(DMatrix *p_fmat,
                       std::vector<ExpandEntry> nodes_for_explicit_hist_build,
                       common::RowSetCollection const &row_set_collection,
                       const std::vector<GradientPair> &gpair_h) {
    const size_t n_nodes = nodes_for_explicit_hist_build.size();

    // create space of size (# rows in each node)
    common::BlockedSpace2d space(
        n_nodes,
        [&](size_t node) {
          const int32_t nid = nodes_for_explicit_hist_build[node].nid;
          return row_set_collection[nid].Size();
        },
        256);

    std::vector<GHistRowT> target_hists(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes_for_explicit_hist_build[i].nid;
      target_hists[i] = hist_[nid];
    }
    buffer_.Reset(this->n_threads_, n_nodes, space, target_hists);

    // Parallel processing by nodes and data in each node
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(
             BatchParam{GenericParameter::kCpuId, max_bin_})) {
      common::ParallelFor2d(
          space, this->n_threads_, [&](size_t nid_in_set, common::Range1d r) {
            const auto tid = static_cast<unsigned>(omp_get_thread_num());
            const int32_t nid = nodes_for_explicit_hist_build[nid_in_set].nid;

            auto start_of_row_set = row_set_collection[nid].begin;
            auto rid_set = common::RowSetCollection::Elem(
                start_of_row_set + r.begin(), start_of_row_set + r.end(), nid);
            builder_.template BuildHist<any_missing>(
                gpair_h, rid_set, gmat,
                buffer_.GetInitializedHist(tid, nid_in_set));
          });
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

  /* Main entry point of this class, build histogram for tree nodes. */
  void BuildHist(DMatrix *p_fmat, RegTree *p_tree,
                 common::RowSetCollection const &row_set_collection,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 std::vector<GradientPair> const &gpair) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    this->AddHistRows(&starting_index, &sync_count,
                      nodes_for_explicit_hist_build,
                      nodes_for_subtraction_trick, p_tree);
    if (p_fmat->IsDense()) {
      BuildLocalHistograms<false>(p_fmat, nodes_for_explicit_hist_build,
                                  row_set_collection, gpair);
    } else {
      BuildLocalHistograms<true>(p_fmat, nodes_for_explicit_hist_build,
                                 row_set_collection, gpair);
    }
    if (is_distributed_) {
      this->SyncHistogramDistributed(p_tree, nodes_for_explicit_hist_build,
                                     nodes_for_subtraction_trick,
                                     starting_index, sync_count);
    } else {
      this->SyncHistogramLocal(p_tree, nodes_for_explicit_hist_build,
                               nodes_for_subtraction_trick, starting_index,
                               sync_count);
    }
  }

  void SyncHistogramDistributed(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count) {
    const size_t nbins = builder_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);
    common::ParallelFor2d(
        space, n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          buffer_.ReduceHist(node, r.begin(), r.end());
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

    reducer_.Allreduce(this->hist_[starting_index].data(),
                       builder_.GetNumBins() * sync_count);

    ParallelSubtractionHist(space, nodes_for_explicit_hist_build,
                            nodes_for_subtraction_trick, p_tree);

    common::BlockedSpace2d space2(
        nodes_for_subtraction_trick.size(), [&](size_t) { return nbins; },
        1024);
    ParallelSubtractionHist(space2, nodes_for_subtraction_trick,
                            nodes_for_explicit_hist_build, p_tree);
  }

  void SyncHistogramLocal(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count) {
    const size_t nbins = this->builder_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);

    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          this->buffer_.ReduceHist(node, r.begin(), r.end());

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
  common::HistCollection<GradientSumT> const& Histogram() {
    return hist_;
  }
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
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
