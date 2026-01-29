/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>   // for max
#include <cstddef>     // for size_t
#include <cstdint>     // for int32_t
#include <utility>     // for move
#include <vector>      // for vector

#include "../../collective/allreduce.h"    // for Allreduce
#include "../../common/hist_util.h"        // for GHistRow, ParallelGHi...
#include "../../common/row_set.h"          // for RowSetCollection
#include "../../common/threading_utils.h"  // for ParallelFor2d, Range1d, BlockedSpace2d
#include "../../common/cache_manager.h"    // for CacheManager
#include "../../data/gradient_index.h"     // for GHistIndexMatrix
#include "expand_entry.h"                  // for MultiExpandEntry, CPUExpandEntry
#include "hist_cache.h"                    // for BoundedHistCollection
#include "hist_param.h"                    // for HistMakerTrainParam
#include "xgboost/base.h"                  // for bst_node_t, bst_target_t, bst_bin_t
#include "xgboost/context.h"               // for Context
#include "xgboost/data.h"                  // for BatchIterator, BatchSet
#include "xgboost/linalg.h"                // for MatrixView, All, Vect...
#include "xgboost/logging.h"               // for CHECK_GE
#include "xgboost/span.h"                  // for Span
#include "xgboost/tree_model.h"            // for RegTree

namespace xgboost::tree {
/**
 * @brief Decide which node as the build node for multi-target trees.
 */
void AssignNodes(MultiTargetTreeView const &tree,
                 std::vector<MultiExpandEntry> const &valid_candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub);

/**
 * @brief Decide which node as the build node.
 */
void AssignNodes(ScalarTreeView const &tree, std::vector<CPUExpandEntry> const &candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub);

class HistogramBuilder {
  /*! \brief culmulative histogram of gradients. */
  common::Monitor monitor_;
  BoundedHistCollection hist_;
  common::ParallelGHistBuilder buffer_;
  BatchParam param_;
  std::int32_t n_threads_{-1};
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_{false};
  bool is_col_split_{false};

 public:
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
    hist_.Reset(total_bins, param->MaxCachedHistNodes(ctx->Device()));
    buffer_.Init(total_bins);
    is_distributed_ = is_distributed;
    is_col_split_ = is_col_split;
  }

  template <bool any_missing>
  void BuildLocalHistograms(common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
                            std::vector<bst_node_t> const &nodes_to_build,
                            common::RowSetCollection const &row_set_collection,
                            common::Span<GradientPair const> gpair_h, bool read_by_column) {
    // Parallel processing by nodes and data in each node
    common::ParallelFor2d(space, this->n_threads_, [&](size_t nid_in_set, common::Range1d r) {
      const auto tid = static_cast<unsigned>(omp_get_thread_num());
      bst_node_t const nidx = nodes_to_build[nid_in_set];
      auto const& elem = row_set_collection[nidx];
      auto start_of_row_set = std::min(r.begin(), elem.Size());
      auto end_of_row_set = std::min(r.end(), elem.Size());
      auto rid_set = common::Span<bst_idx_t const>{elem.begin() + start_of_row_set,
                                                   elem.begin() + end_of_row_set};
      auto hist = buffer_.GetInitializedHist(tid, nid_in_set);
      if (rid_set.size() != 0) {
        common::BuildHist<any_missing>(gpair_h, rid_set, gidx, hist, read_by_column);
      }
    });
  }

  /**
   * @brief Allocate histogram, rearrange the nodes if `rearrange` is true and the tree
   *        has reached the cache size limit.
   */
  template <typename TreeView>
  void AddHistRows(TreeView const &tree, std::vector<bst_node_t> *p_nodes_to_build,
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
      if (this->hist_.HistogramExists(tree.Parent(v))) {
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
                 linalg::VectorView<GradientPair const> gpair, bool read_by_column) {
    monitor_.Start(__func__);
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
      this->BuildLocalHistograms<false>(space, gidx, nodes_to_build, row_set_collection,
                                        gpair.Values(), read_by_column);
    } else {
      this->BuildLocalHistograms<true>(space, gidx, nodes_to_build, row_set_collection,
                                       gpair.Values(), read_by_column);
    }
    monitor_.Stop(__func__);
  }

  template <typename TreeView>
  void SyncHistogram(Context const *ctx, TreeView const &tree,
                     std::vector<bst_node_t> const &nodes_to_build,
                     std::vector<bst_node_t> const &nodes_to_trick) {
    auto n_total_bins = buffer_.TotalBins();
    common::BlockedSpace2d space(
        nodes_to_build.size(), [&](std::size_t) { return n_total_bins; }, 1024);
    common::ParallelFor2d(space, this->n_threads_, [&](size_t node, common::Range1d r) {
      // Merging histograms from each thread.
      this->buffer_.ReduceHist(node, r.begin(), r.end());
    });
    if (is_distributed_ && !is_col_split_) {
      // The cache is contiguous, we can perform allreduce for all nodes in one go.
      CHECK(!nodes_to_build.empty());
      auto first_nidx = nodes_to_build.front();
      std::size_t n = n_total_bins * nodes_to_build.size() * 2;
      auto rc = collective::Allreduce(
          ctx, linalg::MakeVec(reinterpret_cast<double *>(this->hist_[first_nidx].data()), n),
          collective::Op::kSum);
      SafeColl(rc);
    }

    common::BlockedSpace2d const &subspace =
        nodes_to_trick.size() == nodes_to_build.size()
            ? space
            : common::BlockedSpace2d{nodes_to_trick.size(),
                                     [&](std::size_t) { return n_total_bins; }, 1024};
    common::ParallelFor2d(
        subspace, this->n_threads_, [&](std::size_t nidx_in_set, common::Range1d r) {
          auto subtraction_nidx = nodes_to_trick[nidx_in_set];
          auto parent_id = tree.Parent(subtraction_nidx);
          auto sibling_nidx = tree.IsLeftChild(subtraction_nidx) ? tree.RightChild(parent_id)
                                                                 : tree.LeftChild(parent_id);
          auto sibling_hist = this->hist_[sibling_nidx];
          auto parent_hist = this->hist_[parent_id];
          auto subtract_hist = this->hist_[subtraction_nidx];
          common::SubtractionHist(subtract_hist, parent_hist, sibling_hist, r.begin(), r.end());
        });
  }

 public:
  /* Getters for tests. */
  [[nodiscard]] BoundedHistCollection const &Histogram() const { return hist_; }
  [[nodiscard]] BoundedHistCollection &Histogram() { return hist_; }
  auto &Buffer() { return buffer_; }
};

// Construct a work space for building histogram.  Eventually we should move this
// function into histogram builder once hist tree method supports external memory.
template <typename Partitioner>
common::BlockedSpace2d ConstructHistSpace(Partitioner const &partitioners,
                                          std::vector<bst_node_t> const &nodes_to_build,
                                          const GHistIndexMatrix &gidx,
                                          std::size_t l1_size, bst_bin_t max_bin,
                                          bool read_by_column) {
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

  // Estimate the size of each data block based on model parameters and L1 capacity
  // The general idea is to keep as much working-set data in L1 as possible.
  /* Each processed row occupies ~32 bytes in L1:
   * - gradient pair (p_gpair): sizeof(GradientPair)
   * - row index (rid[i]): sizeof(size_t)
   * - icol_start and icol_end: 2 * sizeof(size_t)
   */
  std::size_t l1_row_foot_print = (sizeof(GradientPair) + 3 * sizeof(size_t));
  double usable_l1_size = 0.8 * l1_size;

  std::size_t space_in_l1_for_rows;
  if (read_by_column) {
   /* In this case, an accurate block_size estimate is performance-critical.
    * For column-wise histogram construction, each column is processed over the
    * same block of rows. If the block fits in L1, the row data are loaded once
    * and reused across all columns; otherwise, the cache must be refilled for
    * each column.
    */

    /* First step: determine whether one histogram column fits into L1.
     * Note: column-wise kernel is used for dense data only.
     */
    std::size_t hist_col_size = 2 * sizeof(GradientPairPrecise) * max_bin;
    bool hist_col_fit_to_l1 = hist_col_size < usable_l1_size;

    /* Second step: compute available L1 space for row data. */
    space_in_l1_for_rows = usable_l1_size - (hist_col_fit_to_l1 ? hist_col_size : 0);
  } else {
    /* In this case, block_size is less critical.
    * For row-wise histogram construction, columns are processed for each row.
    * Rows do not need to remain in L1 across iterations, but choosing a
    * reasonable block_size allows the histogram buffer and offsets to stay in L1,
    * which gives a small performance benefit.
    */

    /* First step: estimate the size of the histogram and the offsets vector. */
    std::size_t n_bins = gidx.cut.Ptrs().back();
    std::size_t n_columns = gidx.cut.Ptrs().size() - 1;
    bool any_missing = !gidx.IsDense();
    std::size_t hist_size = 2 * sizeof(GradientPairPrecise) * n_bins;
    std::size_t offsets_size = any_missing ? 0 : n_columns * sizeof(uint32_t);

    /* Second step: estimate the extra L1 footprint caused by prefetching.
     * Prefetching is not always active, so the estimate is intentionally conservative.
     */
    l1_row_foot_print += sizeof(GradientPair);
    std::size_t idx_bin_size = n_columns * sizeof(uint32_t);

    bool hist_fit_to_l1 = (hist_size + offsets_size + idx_bin_size) < usable_l1_size;

    /* Third step: compute available L1 space for row data. */
    std::size_t occupied_space = (hist_fit_to_l1 ? hist_size : 0) + offsets_size + idx_bin_size;
    space_in_l1_for_rows = usable_l1_size > occupied_space ? usable_l1_size - occupied_space : 0;
  }
  std::size_t block_size = space_in_l1_for_rows / l1_row_foot_print;

  /* Minimum block size = 8 rows.
   * This ensures that a full cache line is utilized when loading gradient pairs.
   */
  constexpr std::size_t kCacheLineSize = 64;
  constexpr std::size_t kMinBlockSize = kCacheLineSize / sizeof(GradientPair);
  block_size = std::max<std::size_t>(kMinBlockSize, block_size);

  common::BlockedSpace2d space{
      nodes_to_build.size(), [&](size_t nidx_in_set) {
                                return partition_size[nidx_in_set];
                              }, block_size};
  return space;
}

/**
 * @brief Histogram builder that can handle multiple targets.
 */
class MultiHistogramBuilder {
  std::vector<HistogramBuilder> target_builders_;
  Context const *ctx_;
  common::CacheManager cache_manager_;

  bool ReadByColumn(const GHistIndexMatrix &gidx, bool force_read_by_column) const {
    if (force_read_by_column) return true;

    auto nbins = gidx.cut.Ptrs().back();
    size_t hist_size = 2 * sizeof(double) * nbins;

    double l3_per_thread = static_cast<double>(cache_manager_.L3Size()) / ctx_->Threads();
    double usable_cache_size =  0.8 * (cache_manager_.L2Size() + l3_per_thread);
    const bool hist_fit_to_l2 = usable_cache_size > hist_size;

    /* In row-wise histogram construction, each iteration of the outer (row-wise) loop
     * accesses bins across the entire histogram; the bins are not localized.
     * If the histogram is too large to fit in L2 cache, random access becomes a major performance bottleneck.
     *
     * or dense data, using column-wise histogram construction,
     * each iteration of the outer (column-wise) loop accesses only a localized portion of the histogram:
     * idx_bin = gradient_index(row_id, col_id) + offset[col_id].
     * This improves cache locality, so the column-wise kernel outperforms the row-wise kernel in this case.
     */
    bool read_by_column = !hist_fit_to_l2 && gidx.IsDense();
    return read_by_column;
  }

 public:
  /**
   * @brief Build the histogram for root node.
   */
  template <typename Partitioner, typename ExpandEntry, typename TreeView>
  void BuildRootHist(DMatrix *p_fmat, TreeView const &tree,
                     std::vector<Partitioner> const &partitioners,
                     linalg::MatrixView<GradientPair const> gpair, ExpandEntry const &best,
                     BatchParam const &param, bool force_read_by_column = false) {
    auto n_targets = gpair.Shape(1);
    CHECK_EQ(p_fmat->Info().num_row_, gpair.Shape(0));
    CHECK_EQ(target_builders_.size(), n_targets);
    std::vector<bst_node_t> nodes{best.nid};
    std::vector<bst_node_t> dummy_sub;

    for (bst_target_t t{0}; t < n_targets; ++t) {
      this->target_builders_[t].AddHistRows(tree, &nodes, &dummy_sub, false);
    }
    CHECK(dummy_sub.empty());

    std::size_t page_idx{0};
    for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
      bool read_by_column = ReadByColumn(gidx, force_read_by_column);

      auto space = ConstructHistSpace(partitioners, nodes, gidx,
                                      cache_manager_.L1Size(), param.max_bin, read_by_column);
      for (bst_target_t t{0}; t < n_targets; ++t) {
        auto t_gpair = gpair.Slice(linalg::All(), t);
        this->target_builders_[t].BuildHist(page_idx, space, gidx,
                                            partitioners[page_idx].Partitions(), nodes, t_gpair,
                                            read_by_column);
      }
      ++page_idx;
    }

    for (bst_target_t t = 0; t < n_targets; ++t) {
      this->target_builders_[t].SyncHistogram(ctx_, tree, nodes, dummy_sub);
    }
  }
  /**
   * @brief Build histogram for left and right child of valid candidates
   */
  template <typename Partitioner, typename ExpandEntry, typename TreeView>
  void BuildHistLeftRight(Context const *ctx, DMatrix *p_fmat, TreeView const &tree,
                          std::vector<Partitioner> const &partitioners,
                          std::vector<ExpandEntry> const &valid_candidates,
                          linalg::MatrixView<GradientPair const> gpair, BatchParam const &param,
                          bool force_read_by_column = false) {
    std::vector<bst_node_t> nodes_to_build(valid_candidates.size());
    std::vector<bst_node_t> nodes_to_sub(valid_candidates.size());
    AssignNodes(tree, valid_candidates, nodes_to_build, nodes_to_sub);

    // use the first builder for getting number of valid nodes.
    target_builders_.front().AddHistRows(tree, &nodes_to_build, &nodes_to_sub, true);
    CHECK_GE(nodes_to_build.size(), nodes_to_sub.size());
    CHECK_EQ(nodes_to_sub.size() + nodes_to_build.size(), valid_candidates.size() * 2);

    // allocate storage for the rest of the builders
    for (bst_target_t t = 1; t < target_builders_.size(); ++t) {
      target_builders_[t].AddHistRows(tree, &nodes_to_build, &nodes_to_sub, false);
    }

    std::size_t page_idx{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, param)) {
      bool read_by_column = ReadByColumn(page, force_read_by_column);

      auto space = ConstructHistSpace(partitioners, nodes_to_build, page,
                                      cache_manager_.L1Size(), param.max_bin, read_by_column);

      auto n_targets = gpair.Shape(1);
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto t_gpair = gpair.Slice(linalg::All(), t);
        CHECK_EQ(t_gpair.Shape(0), p_fmat->Info().num_row_);
        this->target_builders_[t].BuildHist(page_idx, space, page,
                                            partitioners[page_idx].Partitions(), nodes_to_build,
                                            t_gpair, read_by_column);
      }
      page_idx++;
    }

    auto n_targets = gpair.Shape(1);
    for (bst_target_t t = 0; t < n_targets; ++t) {
      this->target_builders_[t].SyncHistogram(ctx, tree, nodes_to_build, nodes_to_sub);
    }
  }

  [[nodiscard]] auto const &Histogram(bst_target_t t) const {
    return target_builders_[t].Histogram();
  }
  [[nodiscard]] auto &Histogram(bst_target_t t) { return target_builders_[t].Histogram(); }
  // Number of targets for histogram building (may differ from tree.NumTargets() for reduced grad)
  [[nodiscard]] bst_target_t NumTargets() const { return target_builders_.size(); }

  void Reset(Context const *ctx, bst_bin_t total_bins, bst_target_t n_targets, BatchParam const &p,
             bool is_distributed, bool is_col_split, HistMakerTrainParam const *param) {
    ctx_ = ctx;
    target_builders_.resize(n_targets);
    CHECK_GE(n_targets, 1);
    for (auto &v : target_builders_) {
      v.Reset(ctx, total_bins, p, is_distributed, is_col_split, param);
    }
  }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
