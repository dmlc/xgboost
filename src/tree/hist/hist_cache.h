/**
 * Copyright 2023-2024 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HIST_CACHE_H_
#define XGBOOST_TREE_HIST_HIST_CACHE_H_
#include <cstddef>  // for size_t
#include <map>      // for map
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../../common/hist_util.h"          // for GHistRow, ConstGHistRow
#include "../../common/ref_resource_view.h"  // for ReallocVector
#include "xgboost/base.h"                    // for bst_node_t, bst_bin_t
#include "xgboost/logging.h"                 // for CHECK_EQ
#include "xgboost/span.h"                    // for Span

namespace xgboost::tree {
/**
 * @brief A persistent cache for CPU histogram.
 *
 *   The size of the cache is first bounded by the `Driver` class then by this cache
 *   implementaiton. The former limits the number of nodes that can be built for each node
 *   batch, while this cache limits the number of all nodes up to the size of
 *   max(|node_batch|, n_cached_node).
 *
 *   The caller is responsible for clearing up the cache as it needs to rearrange the
 *   nodes before making overflowed allocations. The strcut only reports whether the size
 *   limit has benn reached.
 */
class BoundedHistCollection {
  // maps node index to offset in `data_`.
  std::map<bst_node_t, std::size_t> node_map_;
  // currently allocated bins, used for tracking consistentcy.
  std::size_t current_size_{0};

  // stores the histograms in a contiguous buffer
  using Vec = common::ReallocVector<GradientPairPrecise>;
  std::unique_ptr<Vec> data_{new Vec{}};  // nvcc 12.1 trips over std::make_unique

  // number of histogram bins across all features
  bst_bin_t n_total_bins_{0};
  // limits the number of nodes that can be in the cache for each tree
  std::size_t max_cached_nodes_{0};
  // whether the tree has grown beyond the cache limit
  bool has_exceeded_{false};

 public:
  BoundedHistCollection() = default;
  common::GHistRow operator[](std::size_t idx) {
    auto offset = node_map_.at(idx);
    return common::Span{data_->data(), static_cast<size_t>(data_->size())}.subspan(
        offset, n_total_bins_);
  }
  common::ConstGHistRow operator[](std::size_t idx) const {
    auto offset = node_map_.at(idx);
    return common::Span{data_->data(), static_cast<size_t>(data_->size())}.subspan(
        offset, n_total_bins_);
  }
  void Reset(bst_bin_t n_total_bins, std::size_t n_cached_nodes) {
    n_total_bins_ = n_total_bins;
    max_cached_nodes_ = n_cached_nodes;
    this->Clear(false);
  }
  /**
   * @brief Clear the cache, mark whether the cache is exceeded the limit.
   */
  void Clear(bool exceeded) {
    node_map_.clear();
    current_size_ = 0;
    has_exceeded_ = exceeded;
  }

  [[nodiscard]] bool CanHost(common::Span<bst_node_t const> nodes_to_build,
                             common::Span<bst_node_t const> nodes_to_sub) const {
    auto n_new_nodes = nodes_to_build.size() + nodes_to_sub.size();
    return n_new_nodes + node_map_.size() <= max_cached_nodes_;
  }

  /**
   * @brief Allocate histogram buffers for all nodes.
   *
   *   The resulting histogram buffer is contiguous for all nodes in the order of
   *   allocation.
   */
  void AllocateHistograms(common::Span<bst_node_t const> nodes_to_build,
                          common::Span<bst_node_t const> nodes_to_sub) {
    auto n_new_nodes = nodes_to_build.size() + nodes_to_sub.size();
    auto alloc_size = n_new_nodes * n_total_bins_;
    auto new_size = alloc_size + current_size_;
    if (new_size > data_->size()) {
      data_->Resize(new_size);
    }
    for (auto nidx : nodes_to_build) {
      node_map_[nidx] = current_size_;
      current_size_ += n_total_bins_;
    }
    for (auto nidx : nodes_to_sub) {
      node_map_[nidx] = current_size_;
      current_size_ += n_total_bins_;
    }
    CHECK_EQ(current_size_, new_size);
  }
  void AllocateHistograms(std::vector<bst_node_t> const& nodes) {
    this->AllocateHistograms(common::Span<bst_node_t const>{nodes},
                             common::Span<bst_node_t const>{});
  }

  [[nodiscard]] bool HasExceeded() const { return has_exceeded_; }
  [[nodiscard]] bool HistogramExists(bst_node_t nidx) const {
    return node_map_.find(nidx) != node_map_.cend();
  }
  [[nodiscard]] std::size_t Size() const { return current_size_; }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_HIST_CACHE_H_
