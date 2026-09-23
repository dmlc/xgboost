/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXACT_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_EXACT_HISTOGRAM_H_

#include <algorithm>  // for fill
#include <cstddef>    // for size_t
#include <map>        // for map
#include <memory>     // for unique_ptr
#include <vector>     // for vector

#include "../../common/exact_multinomial/packed_stats.h"  // for PackedMultinomialStats
#include "../../common/ref_resource_view.h"               // for ReallocVector
#include "../../common/threading_utils.h"                 // for ParallelFor
#include "../../data/gradient_index.h"                    // for GHistIndexMatrix
#include "xgboost/base.h"                                 // for bst_node_t, bst_bin_t
#include "xgboost/context.h"                              // for Context
#include "xgboost/gradient.h"                             // for ExactHessian
#include "xgboost/linalg.h"                               // for MatrixView
#include "xgboost/logging.h"                              // for CHECK_EQ
#include "xgboost/span.h"                                 // for Span

namespace xgboost::tree {
/**
 * @brief Exact multinomial histogram statistics.
 *
 * A scalar objective stores one `GradientPairPrecise` per bin. The multi-target path stores
 * one such pair per bin *per target*, in `n_targets` separate histograms. Exact multinomial
 * mode instead stores a single record per bin holding the `K-1` free-class gradients
 * followed by the packed lower triangle of the dense Hessian:
 *
 *   [ G_0 .. G_{d-1} | H_00 | H_10 H_11 | H_20 H_21 H_22 | .. ]      d = K - 1
 *
 * which is exactly @ref common::PackedMultinomialStats in double precision, so the layout
 * and index convention are shared with the per-row transport and the leaf solver rather
 * than redefined here.
 *
 * Accumulation is in double while the per-row transport is float, mirroring the existing
 * `GradientPair` / `GradientPairPrecise` split.
 */

/** @brief Number of doubles per bin for `n_free` free classes. */
[[nodiscard]] inline std::size_t ExactHistRecordSize(bst_target_t n_free) {
  return common::PackedStatsStride(n_free);
}

/**
 * @brief Bytes held by an exact histogram cache.
 *
 * The existing cache budget (`max_cached_hist_node`) counts nodes rather than bytes and is
 * blind to the per-bin record size -- multi-target training already consumes `n_targets`
 * times the budgeted memory for the same node count. This helper makes the exact cost
 * explicit so a caller can report or bound it.
 */
[[nodiscard]] inline std::size_t ExactHistMemoryBytes(bst_bin_t n_total_bins, std::size_t n_nodes,
                                                      bst_target_t n_free) {
  return sizeof(double) * (static_cast<std::size_t>(n_total_bins) + 1) * n_nodes *
         ExactHistRecordSize(n_free);
}

/**
 * @brief A persistent cache of exact multinomial histograms, one record per bin.
 *
 * The node bookkeeping deliberately mirrors @ref BoundedHistCollection rather than sharing
 * an abstraction with it. The scalar collection's `operator[]` returns a
 * `Span<GradientPairPrecise>` that is load bearing across the tree code, so generalizing it
 * would ripple through the scalar path this milestone must leave untouched. Factoring out
 * the shared node map is worth revisiting once exact mode is established.
 *
 * Storage is one contiguous buffer for all nodes, in allocation order, so a node's
 * histogram is a flat `double` span that can be reduced as a plain numerical buffer.
 */
class ExactHistCollection {
  // Maps node index to its offset in `data_`, counted in doubles.
  std::map<bst_node_t, std::size_t> node_map_;
  std::size_t current_size_{0};

  using Vec = common::ReallocVector<double>;
  std::unique_ptr<Vec> data_{new Vec{}};

  bst_bin_t n_total_bins_{0};
  bst_target_t n_free_{0};
  std::size_t record_size_{0};
  std::size_t max_cached_nodes_{0};
  bool has_exceeded_{false};

  [[nodiscard]] std::size_t NodeStride() const {
    return this->RecordsPerNode() * record_size_;
  }

 public:
  ExactHistCollection() = default;

  void Reset(bst_bin_t n_total_bins, bst_target_t n_free, std::size_t n_cached_nodes) {
    n_total_bins_ = n_total_bins;
    n_free_ = n_free;
    record_size_ = ExactHistRecordSize(n_free);
    max_cached_nodes_ = n_cached_nodes;
    this->Clear(false);
  }

  void Clear(bool exceeded) {
    node_map_.clear();
    current_size_ = 0;
    has_exceeded_ = exceeded;
  }

  /**
   * @brief Records stored per node: one per bin, plus one for the node total.
   *
   * The node total cannot be recovered by summing bins, because every row contributes to one
   * bin *per feature* -- a bin sum over all features counts each row `n_features` times. It
   * also cannot be recovered from a single feature's bins when that feature has missing
   * values. Carrying it as an extra record keeps it exact, and means the existing
   * all-reduce and sibling subtraction cover it with no special handling.
   */
  [[nodiscard]] std::size_t RecordsPerNode() const {
    return static_cast<std::size_t>(n_total_bins_) + 1;
  }
  /** @brief Index of the node-total record. */
  [[nodiscard]] bst_bin_t TotalIndex() const { return n_total_bins_; }

  [[nodiscard]] bst_target_t NumFree() const { return n_free_; }
  [[nodiscard]] std::size_t RecordSize() const { return record_size_; }
  [[nodiscard]] bst_bin_t TotalBins() const { return n_total_bins_; }
  [[nodiscard]] std::size_t Size() const { return current_size_; }
  [[nodiscard]] bool HasExceeded() const { return has_exceeded_; }
  [[nodiscard]] bool HistogramExists(bst_node_t nidx) const {
    return node_map_.find(nidx) != node_map_.cend();
  }

  [[nodiscard]] bool CanHost(common::Span<bst_node_t const> nodes_to_build,
                             common::Span<bst_node_t const> nodes_to_sub) const {
    auto n_new_nodes = nodes_to_build.size() + nodes_to_sub.size();
    return n_new_nodes + node_map_.size() <= max_cached_nodes_;
  }

  /** @brief Bytes currently held, for reporting against a memory budget. */
  [[nodiscard]] std::size_t MemoryBytes() const { return current_size_ * sizeof(double); }

  /**
   * @brief Allocate histograms for all nodes, contiguously and in allocation order.
   */
  void AllocateHistograms(common::Span<bst_node_t const> nodes_to_build,
                          common::Span<bst_node_t const> nodes_to_sub) {
    auto n_new_nodes = nodes_to_build.size() + nodes_to_sub.size();
    auto alloc_size = n_new_nodes * this->NodeStride();
    auto new_size = alloc_size + current_size_;
    if (new_size > data_->size()) {
      data_->Resize(new_size);
    }
    for (auto nidx : nodes_to_build) {
      node_map_[nidx] = current_size_;
      current_size_ += this->NodeStride();
    }
    for (auto nidx : nodes_to_sub) {
      node_map_[nidx] = current_size_;
      current_size_ += this->NodeStride();
    }
    CHECK_EQ(current_size_, new_size);
  }
  void AllocateHistograms(std::vector<bst_node_t> const& nodes) {
    this->AllocateHistograms(common::Span<bst_node_t const>{nodes},
                             common::Span<bst_node_t const>{});
  }

  /** @brief The whole histogram of a node as a flat buffer. */
  common::Span<double> operator[](bst_node_t nidx) {
    auto offset = node_map_.at(nidx);
    return common::Span<double>{data_->data(), static_cast<std::size_t>(data_->size())}.subspan(
        offset, this->NodeStride());
  }
  common::Span<double const> operator[](bst_node_t nidx) const {
    auto offset = node_map_.at(nidx);
    return common::Span<double const>{data_->data(),
                                      static_cast<std::size_t>(data_->size())}
        .subspan(offset, this->NodeStride());
  }

  /** @brief One bin's statistics, interpreted through the shared packed layout. */
  [[nodiscard]] common::PackedMultinomialStats<double> RecordAt(bst_node_t nidx, bst_bin_t bin) {
    return common::PackedStatsAtBin((*this)[nidx], n_free_, static_cast<std::size_t>(bin));
  }
  [[nodiscard]] common::PackedMultinomialStats<double const> RecordAt(bst_node_t nidx,
                                                                      bst_bin_t bin) const {
    return common::PackedStatsAtBin((*this)[nidx], n_free_, static_cast<std::size_t>(bin));
  }
};

/**
 * @brief Zero a node histogram.
 */
inline void ZeroExactHist(common::Span<double> hist) {
  std::fill(hist.begin(), hist.end(), 0.0);
}

/**
 * @brief `dst = parent - sibling`, elementwise over the flat record buffer.
 *
 * Gradients and Hessian entries live in one contiguous record with identical layout in all
 * three histograms, so the sibling trick is a single linear pass and cannot lose or
 * transpose an entry. Negative off-diagonal Hessian terms are preserved exactly, being
 * ordinary elements of the same buffer.
 */
inline void SubtractExactHist(common::Span<double> dst, common::Span<double const> parent,
                              common::Span<double const> sibling) {
  CHECK_EQ(dst.size(), parent.size());
  CHECK_EQ(dst.size(), sibling.size());
  for (std::size_t i = 0, n = dst.size(); i < n; ++i) {
    dst[i] = parent[i] - sibling[i];
  }
}

/**
 * @brief Accumulate one row into the bin records it touches.
 *
 * The free-class gradients are read from the existing gpair -- the first `d` of the `K`
 * columns are exactly the free-class gradients, so nothing is recomputed and no probability
 * is formed here. The Hessian triangle is copied from the per-row transport. Neither the
 * softmax nor the Hessian is reconstructed.
 */
inline void AddRowToExactHist(common::Span<double> hist, std::size_t record_size,
                              bst_target_t n_free, bst_bin_t bin,
                              linalg::VectorView<GradientPair const> gpair_row,
                              common::Span<float const> hessian_row) {
  auto* record = hist.data() + static_cast<std::size_t>(bin) * record_size;
  for (bst_target_t i = 0; i < n_free; ++i) {
    record[i] += static_cast<double>(gpair_row(i).GetGrad());
  }
  auto* triangle = record + n_free;
  for (std::size_t k = 0, n = hessian_row.size(); k < n; ++k) {
    triangle[k] += static_cast<double>(hessian_row[k]);
  }
}

/**
 * @brief Thread-local accumulation buffers for the exact histogram.
 *
 * One contiguous block per thread, allocated once per build and reused, so neither a row
 * nor a bin ever triggers an allocation. Threads never share a block, so the accumulation
 * needs no atomics.
 */
class ExactHistThreadBuffer {
  std::vector<double> data_;
  std::size_t stride_{0};
  std::int32_t n_threads_{0};

 public:
  /** @param n_records Records per node: one per bin plus the node total. */
  void Reset(std::int32_t n_threads, std::size_t n_records, std::size_t record_size) {
    n_threads_ = n_threads;
    stride_ = n_records * record_size;
    data_.assign(static_cast<std::size_t>(n_threads) * stride_, 0.0);
  }

  [[nodiscard]] std::size_t Stride() const { return stride_; }
  [[nodiscard]] std::int32_t NumThreads() const { return n_threads_; }

  [[nodiscard]] common::Span<double> ThreadSpan(std::int32_t tid) {
    return common::Span<double>{data_.data() + static_cast<std::size_t>(tid) * stride_, stride_};
  }

  void Zero() { std::fill(data_.begin(), data_.end(), 0.0); }

  /** @brief Sum every thread's block into @p out. */
  void ReduceTo(common::Span<double> out) const {
    CHECK_EQ(out.size(), stride_);
    for (std::int32_t tid = 0; tid < n_threads_; ++tid) {
      auto const* block = data_.data() + static_cast<std::size_t>(tid) * stride_;
      for (std::size_t i = 0; i < stride_; ++i) {
        out[i] += block[i];
      }
    }
  }
};

/**
 * @brief Build one node's exact histogram from the rows it owns.
 *
 * Each row is visited once and every bin it touches receives one contiguous record update,
 * rather than one traversal per Hessian entry.
 *
 * @param hist     Destination, already zeroed, `n_total_bins * record_size` doubles.
 * @param n_free   `K - 1`.
 * @param gmat     Binned feature matrix.
 * @param rows     Row indices belonging to the node.
 * @param gpair    Per-row gradients, `(n_samples, K)`.
 * @param hessian  Per-row packed exact Hessian.
 */
inline void BuildExactHist(Context const* ctx, common::Span<double> hist, bst_target_t n_free,
                           GHistIndexMatrix const& gmat, common::Span<bst_idx_t const> rows,
                           linalg::MatrixView<GradientPair const> gpair,
                           ExactHessian const& hessian, ExactHistThreadBuffer* buffer) {
  auto record_size = ExactHistRecordSize(n_free);
  auto total_index = static_cast<bst_bin_t>(gmat.cut.TotalBins());
  CHECK_EQ(hist.size(), (static_cast<std::size_t>(total_index) + 1) * record_size);
  CHECK_EQ(hessian.n_free, n_free);
  CHECK_GE(gpair.Shape(1), static_cast<std::size_t>(n_free));

  auto n_features = gmat.Features();
  auto n_threads = ctx->Threads();
  buffer->Reset(n_threads, static_cast<std::size_t>(total_index) + 1, record_size);

  common::ParallelFor(rows.size(), n_threads, [&](std::size_t i) {
    auto ridx = rows[i];
    auto local = buffer->ThreadSpan(omp_get_thread_num());
    auto gpair_row = gpair.Slice(ridx, linalg::All());
    auto hessian_row = hessian.HostRow(ridx);
    for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
      auto bin = gmat.GetGindex(ridx, fidx);
      if (bin < 0) {
        continue;  // missing value
      }
      AddRowToExactHist(local, record_size, n_free, bin, gpair_row, hessian_row);
    }
    // Once per row, independent of how many features it has values for.
    AddRowToExactHist(local, record_size, n_free, total_index, gpair_row, hessian_row);
  });

  buffer->ReduceTo(hist);
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXACT_HISTOGRAM_H_
