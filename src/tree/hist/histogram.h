/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <utility>    // for move
#include <variant>    // for variant
#include <vector>     // for vector

#include "../../collective/allreduce.h"    // for Allreduce
#include "../../common/hist_util.h"        // for GHistRow, ParallelGHi...
#include "../../common/row_set.h"          // for RowSetCollection
#include "../../common/threading_utils.h"  // for ParallelFor2d, Range1d, BlockedSpace2d
#include "../../data/gradient_index.h"     // for GHistIndexMatrix
#include "expand_entry.h"                  // for MultiExpandEntry, CPUExpandEntry
#include "hist_cache.h"                    // for BoundedHistCollection
#include "param.h"                         // for HistMakerTrainParam
#include "xgboost/base.h"                  // for bst_node_t, bst_target_t, bst_bin_t
#include "xgboost/context.h"               // for Context
#include "xgboost/data.h"                  // for BatchIterator, BatchSet
#include "xgboost/linalg.h"                // for MatrixView, All, Vect...
#include "xgboost/logging.h"               // for CHECK_GE
#include "xgboost/span.h"                  // for Span
#include "xgboost/tree_model.h"            // for RegTree

#if defined(XGBOOST_USE_FEDERATED)
#include "../../../plugin/federated/federated_hist.h"  // for FederataedHistPolicy
#endif

namespace xgboost::tree {
/**
 * @brief Decide which node as the build node for multi-target trees.
 */
void AssignNodes(RegTree const *p_tree, std::vector<MultiExpandEntry> const &valid_candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub);

/**
 * @brief Decide which node as the build node.
 */
void AssignNodes(RegTree const *p_tree, std::vector<CPUExpandEntry> const &candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub);

/**
 * @brief Policy class container for building histogram.
 *
 * This is an algorithm template, it requires the specific policy to fill in the
 * `DoBuildLocalHistograms`, `DoSyncHistogram`, and the `Reset` methods. See the default
 * implementation for an example.
 */
template <typename BuildPolicy>
class HistogramPolicyContainer : public BuildPolicy {
  // Culmulative histogram of gradients.
  BoundedHistCollection hist_;
  // Histogram buffers for threads.
  common::ParallelGHistBuilder buffer_;
  BatchParam param_;
  std::int32_t n_threads_{-1};

 public:
  HistogramPolicyContainer() = default;
  HistogramPolicyContainer(HistogramPolicyContainer const &) = delete;
  HistogramPolicyContainer(HistogramPolicyContainer &&) = default;
  HistogramPolicyContainer &operator=(HistogramPolicyContainer const &) = delete;
  HistogramPolicyContainer &operator=(HistogramPolicyContainer &&) = default;

  /**
   * @brief Reset the builder, should be called before growing a new tree.
   *
   * @param total_bins       Total number of bins across all features
   * @param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */
  void Reset(Context const *ctx, bst_bin_t total_bins, BatchParam const &p, bool is_distributed,
             bool is_col_split, HistMakerTrainParam const *param) {
    n_threads_ = ctx->Threads();
    param_ = p;
    hist_.Reset(total_bins, param->max_cached_hist_node);
    buffer_.Init(total_bins);

    BuildPolicy::Reset(ctx, is_distributed, is_col_split);
  }

  /**
   * @brief Allocate histogram, rearrange the nodes if `rearrange` is true and the tree
   *        has reached the cache size limit.
   */
  void AddHistRows(RegTree const *p_tree, std::vector<bst_node_t> *p_nodes_to_build,
                   std::vector<bst_node_t> *p_nodes_to_sub, bool rearrange) {
    CHECK(p_nodes_to_build);
    auto &nodes_to_build = *p_nodes_to_build;
    CHECK(p_nodes_to_sub);
    auto &nodes_to_sub = *p_nodes_to_sub;

    // We first check whether the cache size is already exceeded or about to be exceeded.
    // If not, then we can allocate histograms without clearing the cache and without
    // worrying about missing parent histogram.
    //
    // Otherwise, we need to rearrange the nodes before the allocation to make sure the
    // resulting buffer is contiguous. This is to facilitate efficient allreduce.

    bool can_host = this->hist_.CanHost(nodes_to_build, nodes_to_sub);
    // True if the tree is still within the size of cache limit. Allocate histogram as
    // usual.
    auto cache_is_valid = can_host && !this->hist_.HasExceeded();

    if (!can_host) {
      this->hist_.Clear(true);
    }

    if (!rearrange || cache_is_valid) {
      // If not rearrange, we allocate the histogram as usual, assuming the nodes have
      // been properly arranged by other builders.
      this->hist_.AllocateHistograms(nodes_to_build, nodes_to_sub);
      if (rearrange) {
        CHECK(!this->hist_.HasExceeded());
      }
      return;
    }

    // The cache is full, parent histogram might be removed in previous iterations to
    // saved memory.
    std::vector<bst_node_t> can_subtract;
    for (auto const &v : nodes_to_sub) {
      if (this->hist_.HistogramExists(p_tree->Parent(v))) {
        // We can still use the subtraction trick for this node
        can_subtract.push_back(v);
      } else {
        // This node requires a full build
        nodes_to_build.push_back(v);
      }
    }

    nodes_to_sub = std::move(can_subtract);
    this->hist_.AllocateHistograms(nodes_to_build, nodes_to_sub);
  }

  /** Main entry point of this class, build histogram for tree nodes. */
  void BuildHist(std::size_t page_idx, common::BlockedSpace2d const &space,
                 GHistIndexMatrix const &gidx, common::RowSetCollection const &row_set_collection,
                 std::vector<bst_node_t> const &nodes_to_build,
                 linalg::VectorView<GradientPair const> gpair, bool force_read_by_column = false) {
    CHECK(gpair.Contiguous());

    if (page_idx == 0) {
      // Add the local histogram cache to the parallel buffer before processing the first page.
      auto n_nodes = nodes_to_build.size();
      std::vector<common::GHistRow> target_hists(n_nodes);
      for (size_t i = 0; i < n_nodes; ++i) {
        auto const nidx = nodes_to_build[i];
        target_hists[i] = hist_[nidx];
      }
      buffer_.Reset(this->n_threads_, n_nodes, space, target_hists);
    }

    if (gidx.IsDense()) {
      this->template BuildLocalHistograms<false>(space, gidx, nodes_to_build, row_set_collection,
                                                 gpair.Values(), force_read_by_column);
    } else {
      this->template BuildLocalHistograms<true>(space, gidx, nodes_to_build, row_set_collection,
                                                gpair.Values(), force_read_by_column);
    }
  }

  template <bool any_missing>
  void BuildLocalHistograms(common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
                            std::vector<bst_node_t> const &nodes_to_build,
                            common::RowSetCollection const &row_set_collection,
                            common::Span<GradientPair const> gpair_h, bool force_read_by_column) {
    BuildPolicy::template DoBuildLocalHistograms<any_missing>(
        space, gidx, nodes_to_build, row_set_collection, gpair_h, force_read_by_column, &buffer_);
  }

  void SyncHistogram(Context const *ctx, RegTree const *p_tree,
                     std::vector<bst_node_t> const &nodes_to_build,
                     std::vector<bst_node_t> const &nodes_to_trick) {
    BuildPolicy::DoSyncHistogram(ctx, p_tree, nodes_to_build, nodes_to_trick, &buffer_, &hist_);
  }

 public:
  /* Getters for tests. */
  [[nodiscard]] BoundedHistCollection const &Histogram() const { return hist_; }
  [[nodiscard]] BoundedHistCollection &Histogram() { return hist_; }
  auto &Buffer() { return buffer_; }
};

/**
 * @brief Histogram build policy for normal cases.
 */
class DefaultHistPolicy {
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_{false};
  bool is_col_split_{false};
  std::int32_t n_threads_{-1};

 public:
 public:
  void Reset(Context const *ctx, bool is_distributed, bool is_col_split) {
    this->is_distributed_ = is_distributed;
    this->n_threads_ = ctx->Threads();
    this->is_col_split_ = is_col_split;
  }

  template <bool any_missing>
  void DoBuildLocalHistograms(common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
                              std::vector<bst_node_t> const &nodes_to_build,
                              common::RowSetCollection const &row_set_collection,
                              common::Span<GradientPair const> gpair_h, bool force_read_by_column,
                              common::ParallelGHistBuilder *buffer) {
    common::BuildSampleHistograms<any_missing>(this->n_threads_, space, gidx, nodes_to_build,
                                               row_set_collection, gpair_h, force_read_by_column,
                                               buffer);
  }

  void DoSyncHistogram(Context const *ctx, RegTree const *p_tree,
                       std::vector<bst_node_t> const &nodes_to_build,
                       std::vector<bst_node_t> const &nodes_to_trick,
                       common::ParallelGHistBuilder *buffer, BoundedHistCollection *p_hist) {
    auto n_total_bins = buffer->TotalBins();
    auto &hist = *p_hist;
    common::BlockedSpace2d space(
        nodes_to_build.size(), [&](std::size_t) { return n_total_bins; }, 1024);
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node, common::Range1d r) {
      // Merging histograms from each thread.
      buffer->ReduceHist(node, r.begin(), r.end());
    });
    // Sync the histogram if it's not column split
    if (is_distributed_ && !is_col_split_) {
      // The cache is contiguous, we can perform allreduce for all nodes in one go.
      CHECK(!nodes_to_build.empty());
      auto first_nidx = nodes_to_build.front();
      std::size_t n = n_total_bins * nodes_to_build.size() * 2;
      auto rc = collective::Allreduce(
          ctx, linalg::MakeVec(reinterpret_cast<double *>(hist[first_nidx].data()), n),
          collective::Op::kSum);
      SafeColl(rc);
    }

    common::BlockedSpace2d const &subspace =
        nodes_to_trick.size() == nodes_to_build.size()
            ? space
            : common::BlockedSpace2d{nodes_to_trick.size(),
                                     [&](std::size_t) { return n_total_bins; }, 1024};
    common::ParallelFor2d(
        subspace, ctx->Threads(), [&](std::size_t nidx_in_set, common::Range1d r) {
          auto subtraction_nidx = nodes_to_trick[nidx_in_set];
          auto parent_id = p_tree->Parent(subtraction_nidx);
          auto sibling_nidx = p_tree->IsLeftChild(subtraction_nidx) ? p_tree->RightChild(parent_id)
                                                                    : p_tree->LeftChild(parent_id);
          auto sibling_hist = hist[sibling_nidx];
          auto parent_hist = hist[parent_id];
          auto subtract_hist = hist[subtraction_nidx];
          common::SubtractionHist(subtract_hist, parent_hist, sibling_hist, r.begin(), r.end());
        });
  }
};

using HistogramBuilder = HistogramPolicyContainer<DefaultHistPolicy>;
#if defined(XGBOOST_USE_FEDERATED)
using FedHistogramBuilder = HistogramPolicyContainer<FederataedHistPolicy>;
#endif  // defined(XGBOOST_USE_FEDERATED)

// Construct a work space for building histogram.  Eventually we should move this
// function into histogram builder once hist tree method supports external memory.
template <typename Partitioner>
common::BlockedSpace2d ConstructHistSpace(Partitioner const &partitioners,
                                          std::vector<bst_node_t> const &nodes_to_build) {
  // FIXME(jiamingy): Handle different size of space.  Right now we use the maximum
  // partition size for the buffer, which might not be efficient if partition sizes
  // has significant variance.
  std::vector<std::size_t> partition_size(nodes_to_build.size(), 0);
  for (auto const &partition : partitioners) {
    size_t k = 0;
    for (auto nidx : nodes_to_build) {
      auto n_rows_in_node = partition.Partitions()[nidx].Size();
      partition_size[k] = std::max(partition_size[k], n_rows_in_node);
      k++;
    }
  }
  common::BlockedSpace2d space{
      nodes_to_build.size(), [&](size_t nidx_in_set) { return partition_size[nidx_in_set]; }, 256};
  return space;
}

/**
 * @brief Histogram builder that can handle multiple targets.
 */
class MultiHistogramBuilder {
#if defined(XGBOOST_USE_FEDERATED)
  std::variant<std::vector<HistogramBuilder>, std::vector<FedHistogramBuilder>> target_builders_;
#else
  std::variant<std::vector<HistogramBuilder>> target_builders_;
#endif

  Context const *ctx_;

 public:
  /**
   * @brief Build the histogram for root node.
   */
  template <typename Partitioner, typename ExpandEntry>
  void BuildRootHist(DMatrix *p_fmat, RegTree const *p_tree,
                     std::vector<Partitioner> const &partitioners,
                     linalg::MatrixView<GradientPair const> gpair, ExpandEntry const &best,
                     BatchParam const &param, bool force_read_by_column = false) {
    auto n_targets = p_tree->NumTargets();
    CHECK_EQ(gpair.Shape(1), n_targets);
    CHECK_EQ(p_fmat->Info().num_row_, gpair.Shape(0));

    std::visit(
        [&](auto &target_builders) {
          CHECK_EQ(target_builders.size(), n_targets);
          std::vector<bst_node_t> nodes{best.nid};
          std::vector<bst_node_t> dummy_sub;

          auto space = ConstructHistSpace(partitioners, nodes);
          for (bst_target_t t{0}; t < n_targets; ++t) {
            target_builders[t].AddHistRows(p_tree, &nodes, &dummy_sub, false);
          }
          CHECK(dummy_sub.empty());

          std::size_t page_idx{0};
          for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
            for (bst_target_t t{0}; t < n_targets; ++t) {
              auto t_gpair = gpair.Slice(linalg::All(), t);
              target_builders[t].BuildHist(page_idx, space, gidx,
                                           partitioners[page_idx].Partitions(), nodes, t_gpair,
                                           force_read_by_column);
            }
            ++page_idx;
          }

          for (bst_target_t t = 0; t < p_tree->NumTargets(); ++t) {
            target_builders[t].SyncHistogram(ctx_, p_tree, nodes, dummy_sub);
          }
        },
        target_builders_);
  }
  /**
   * @brief Build histogram for left and right child of valid candidates
   */
  template <typename Partitioner, typename ExpandEntry>
  void BuildHistLeftRight(Context const *ctx, DMatrix *p_fmat, RegTree const *p_tree,
                          std::vector<Partitioner> const &partitioners,
                          std::vector<ExpandEntry> const &valid_candidates,
                          linalg::MatrixView<GradientPair const> gpair, BatchParam const &param,
                          bool force_read_by_column = false) {
    std::vector<bst_node_t> nodes_to_build(valid_candidates.size());
    std::vector<bst_node_t> nodes_to_sub(valid_candidates.size());
    AssignNodes(p_tree, valid_candidates, nodes_to_build, nodes_to_sub);

    std::visit(
        [&](auto &target_builders) {
          // Use the first builder for getting the number of valid nodes.
          target_builders.front().AddHistRows(p_tree, &nodes_to_build, &nodes_to_sub, true);
          CHECK_GE(nodes_to_build.size(), nodes_to_sub.size());
          CHECK_EQ(nodes_to_sub.size() + nodes_to_build.size(), valid_candidates.size() * 2);

          // allocate storage for the rest of the builders
          for (bst_target_t t = 1; t < target_builders.size(); ++t) {
            target_builders[t].AddHistRows(p_tree, &nodes_to_build, &nodes_to_sub, false);
          }

          auto space = ConstructHistSpace(partitioners, nodes_to_build);
          std::size_t page_idx{0};
          // Build one histogram per target per page.
          for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
            CHECK_EQ(gpair.Shape(1), p_tree->NumTargets());
            for (bst_target_t t = 0; t < p_tree->NumTargets(); ++t) {
              auto t_gpair = gpair.Slice(linalg::All(), t);
              CHECK_EQ(t_gpair.Shape(0), p_fmat->Info().num_row_);
              target_builders[t].BuildHist(page_idx, space, page,
                                           partitioners[page_idx].Partitions(), nodes_to_build,
                                           t_gpair, force_read_by_column);
            }
            page_idx++;
          }

          for (bst_target_t t = 0; t < p_tree->NumTargets(); ++t) {
            target_builders[t].SyncHistogram(ctx, p_tree, nodes_to_build, nodes_to_sub);
          }
        },
        target_builders_);
  }

  [[nodiscard]] decltype(auto) Histogram(bst_target_t t) const {
    return std::visit(
        [t](auto const &target_builders) -> decltype(auto) {
          return target_builders[t].Histogram();
        },
        target_builders_);
  }
  [[nodiscard]] decltype(auto) Histogram(bst_target_t t) {
    return std::visit(
        [t](auto const &target_builders) -> decltype(auto) {
          return target_builders[t].Histogram();
        },
        target_builders_);
  }

  void Reset(Context const *ctx, bst_bin_t total_bins, bst_target_t n_targets, BatchParam const &p,
             bool is_distributed, bool is_col_split, bool is_secure,
             HistMakerTrainParam const *param) {
    ctx_ = ctx;
#if defined(XGBOOST_USE_FEDERATED)
    if (is_secure && !std::get_if<std::vector<FedHistogramBuilder>>(&target_builders_)) {
      target_builders_.emplace<std::vector<FedHistogramBuilder>>(n_targets);
    }
#else
    CHECK(!is_secure);
#endif

    std::visit(
        [&](auto &&target_builders) {
          target_builders.resize(n_targets);
          CHECK_GE(n_targets, 1);
          for (auto &v : target_builders) {
            v.Reset(ctx, total_bins, p, is_distributed, is_col_split, param);
          }
        },
        this->target_builders_);
  }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
