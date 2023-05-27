/**
 * Copyright 2021-2023 XGBoost contributors
 * \file common_row_partitioner.h
 * \brief Common partitioner logic for hist and approx methods.
 */
#ifndef XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_
#define XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_

#include <algorithm>  // std::all_of
#include <cinttypes>  // std::uint32_t
#include <limits>  // std::numeric_limits
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/linalg_op.h"  // cbegin
#include "../common/numeric.h"  // Iota
#include "../common/partition_builder.h"
#include "hist/expand_entry.h"  // CPUExpandEntry
#include "xgboost/base.h"
#include "xgboost/context.h"    // Context
#include "xgboost/linalg.h"       // TensorView

namespace xgboost::tree {

static constexpr size_t kPartitionBlockSize = 2048;

class ColumnSplitHelper {
 public:
  ColumnSplitHelper() = default;

  ColumnSplitHelper(bst_row_t num_row,
                    common::PartitionBuilder<kPartitionBlockSize>* partition_builder,
                    common::RowSetCollection* row_set_collection)
      : partition_builder_{partition_builder}, row_set_collection_{row_set_collection} {
    decision_storage_.resize(num_row);
    decision_bits_ = BitVector(common::Span<BitVector::value_type>(decision_storage_));
    missing_storage_.resize(num_row);
    missing_bits_ = BitVector(common::Span<BitVector::value_type>(missing_storage_));
  }

  template <typename BinIdxType, bool any_missing, bool any_cat, typename ExpandEntry>
  void Partition(common::BlockedSpace2d const& space, std::int32_t n_threads,
                 GHistIndexMatrix const& gmat, common::ColumnMatrix const& column_matrix,
                 std::vector<ExpandEntry> const& nodes,
                 std::vector<int32_t> const& split_conditions, RegTree const* p_tree) {
    // When data is split by column, we don't have all the feature values in the local worker, so
    // we first collect all the decisions and whether the feature is missing into bit vectors.
    std::fill(decision_storage_.begin(), decision_storage_.end(), 0);
    std::fill(missing_storage_.begin(), missing_storage_.end(), 0);
    common::ParallelFor2d(space, n_threads, [&](size_t node_in_set, common::Range1d r) {
      const int32_t nid = nodes[node_in_set].nid;
      bst_bin_t split_cond = column_matrix.IsInitialized() ? split_conditions[node_in_set] : 0;
      partition_builder_->MaskRows<BinIdxType, any_missing, any_cat>(
          node_in_set, nodes, r, split_cond, gmat, column_matrix, *p_tree,
          (*row_set_collection_)[nid].begin, &decision_bits_, &missing_bits_);
    });

    // Then aggregate the bit vectors across all the workers.
    collective::Allreduce<collective::Operation::kBitwiseOR>(decision_storage_.data(),
                                                             decision_storage_.size());
    collective::Allreduce<collective::Operation::kBitwiseAND>(missing_storage_.data(),
                                                              missing_storage_.size());

    // Finally use the bit vectors to partition the rows.
    common::ParallelFor2d(space, n_threads, [&](size_t node_in_set, common::Range1d r) {
      size_t begin = r.begin();
      const int32_t nid = nodes[node_in_set].nid;
      const size_t task_id = partition_builder_->GetTaskIdx(node_in_set, begin);
      partition_builder_->AllocateForTask(task_id);
      partition_builder_->PartitionByMask(node_in_set, nodes, r, gmat, *p_tree,
                                          (*row_set_collection_)[nid].begin, decision_bits_,
                                          missing_bits_);
    });
  }

 private:
  using BitVector = RBitField8;
  std::vector<BitVector::value_type> decision_storage_{};
  BitVector decision_bits_{};
  std::vector<BitVector::value_type> missing_storage_{};
  BitVector missing_bits_{};
  common::PartitionBuilder<kPartitionBlockSize>* partition_builder_;
  common::RowSetCollection* row_set_collection_;
};

class CommonRowPartitioner {
 public:
  bst_row_t base_rowid = 0;

  CommonRowPartitioner() = default;
  CommonRowPartitioner(Context const* ctx, bst_row_t num_row, bst_row_t _base_rowid,
                       bool is_col_split)
      : base_rowid{_base_rowid}, is_col_split_{is_col_split} {
    row_set_collection_.Clear();
    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(num_row);

    std::size_t* p_row_indices = row_indices.data();
    common::Iota(ctx, p_row_indices, p_row_indices + row_indices.size(), base_rowid);
    row_set_collection_.Init();

    if (is_col_split_) {
      column_split_helper_ = ColumnSplitHelper{num_row, &partition_builder_, &row_set_collection_};
    }
  }

  template <typename ExpandEntry>
  void FindSplitConditions(const std::vector<ExpandEntry>& nodes, const RegTree& tree,
                           const GHistIndexMatrix& gmat, std::vector<int32_t>* split_conditions) {
    auto const& ptrs = gmat.cut.Ptrs();
    auto const& vals = gmat.cut.Values();

    for (std::size_t i = 0; i < nodes.size(); ++i) {
      bst_node_t const nidx = nodes[i].nid;
      bst_feature_t const fidx = tree.SplitIndex(nidx);
      float const split_pt = tree.SplitCond(nidx);
      std::uint32_t const lower_bound = ptrs[fidx];
      std::uint32_t const upper_bound = ptrs[fidx + 1];
      bst_bin_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (auto bound = lower_bound; bound < upper_bound; ++bound) {
        if (split_pt == vals[bound]) {
          split_cond = static_cast<bst_bin_t>(bound);
        }
      }
      (*split_conditions)[i] = split_cond;
    }
  }

  template <typename ExpandEntry>
  void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes, RegTree const* p_tree) {
    const size_t n_nodes = nodes.size();
    for (unsigned int i = 0; i < n_nodes; ++i) {
      const int32_t nidx = nodes[i].nid;
      const size_t n_left = partition_builder_.GetNLeftElems(i);
      const size_t n_right = partition_builder_.GetNRightElems(i);
      CHECK_EQ(p_tree->LeftChild(nidx) + 1, p_tree->RightChild(nidx));
      row_set_collection_.AddSplit(nidx, p_tree->LeftChild(nidx), p_tree->RightChild(nidx), n_left,
                                   n_right);
    }
  }

  template <typename ExpandEntry>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      std::vector<ExpandEntry> const& nodes, RegTree const* p_tree) {
    auto const& column_matrix = gmat.Transpose();
    if (column_matrix.IsInitialized()) {
      if (gmat.cut.HasCategorical()) {
        this->template UpdatePosition<true>(ctx, gmat, column_matrix, nodes, p_tree);
      } else {
        this->template UpdatePosition<false>(ctx, gmat, column_matrix, nodes, p_tree);
      }
    } else {
      /* ColumnMatrix is not initilized.
       * It means that we use 'approx' method.
       * any_missing and any_cat don't metter in this case.
       * Jump directly to the main method.
       */
      this->template UpdatePosition<uint8_t, true, true>(ctx, gmat, column_matrix, nodes, p_tree);
    }
  }

  template <bool any_cat, typename ExpandEntry>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<ExpandEntry> const& nodes, RegTree const* p_tree) {
    if (column_matrix.AnyMissing()) {
      this->template UpdatePosition<true, any_cat>(ctx, gmat, column_matrix, nodes, p_tree);
    } else {
      this->template UpdatePosition<false, any_cat>(ctx, gmat, column_matrix, nodes, p_tree);
    }
  }

  template <bool any_missing, bool any_cat, typename ExpandEntry>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<ExpandEntry> const& nodes, RegTree const* p_tree) {
    common::DispatchBinType(column_matrix.GetTypeSize(), [&](auto t) {
      using T = decltype(t);
      this->template UpdatePosition<T, any_missing, any_cat>(ctx, gmat, column_matrix, nodes,
                                                             p_tree);
    });
  }

  template <typename BinIdxType, bool any_missing, bool any_cat, typename ExpandEntry>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<ExpandEntry> const& nodes, RegTree const* p_tree) {
    // 1. Find split condition for each split
    size_t n_nodes = nodes.size();

    std::vector<int32_t> split_conditions;
    if (column_matrix.IsInitialized()) {
      split_conditions.resize(n_nodes);
      FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
    }

    // 2.1 Create a blocked space of size SUM(samples in each node)
    common::BlockedSpace2d space(
        n_nodes,
        [&](size_t node_in_set) {
          int32_t nid = nodes[node_in_set].nid;
          return row_set_collection_[nid].Size();
        },
        kPartitionBlockSize);

    // 2.2 Initialize the partition builder
    // allocate buffers for storage intermediate results by each thread
    partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
      const int32_t nid = nodes[node_in_set].nid;
      const size_t size = row_set_collection_[nid].Size();
      const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
      return n_tasks;
    });
    CHECK_EQ(base_rowid, gmat.base_rowid);

    // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
    // Store results in intermediate buffers from partition_builder_
    if (is_col_split_) {
      column_split_helper_.Partition<BinIdxType, any_missing, any_cat>(
          space, ctx->Threads(), gmat, column_matrix, nodes, split_conditions, p_tree);
    } else {
      common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
        size_t begin = r.begin();
        const int32_t nid = nodes[node_in_set].nid;
        const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
        partition_builder_.AllocateForTask(task_id);
        bst_bin_t split_cond = column_matrix.IsInitialized() ? split_conditions[node_in_set] : 0;
        partition_builder_.template Partition<BinIdxType, any_missing, any_cat>(
            node_in_set, nodes, r, split_cond, gmat, column_matrix, *p_tree,
            row_set_collection_[nid].begin);
      });
    }

    // 3. Compute offsets to copy blocks of row-indexes
    // from partition_builder_ to row_set_collection_
    partition_builder_.CalculateRowOffsets();

    // 4. Copy elements from partition_builder_ to row_set_collection_ back
    // with updated row-indexes for each tree-node
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
      const int32_t nid = nodes[node_in_set].nid;
      partition_builder_.MergeToArray(node_in_set, r.begin(),
                                      const_cast<size_t*>(row_set_collection_[nid].begin));
    });

    // 5. Add info about splits into row_set_collection_
    AddSplitsToRowSet(nodes, p_tree);
  }

  [[nodiscard]] auto const& Partitions() const { return row_set_collection_; }

  [[nodiscard]] std::size_t Size() const {
    return std::distance(row_set_collection_.begin(), row_set_collection_.end());
  }

  auto& operator[](bst_node_t nidx) { return row_set_collection_[nidx]; }
  auto const& operator[](bst_node_t nidx) const { return row_set_collection_[nidx]; }

  void LeafPartition(Context const* ctx, RegTree const& tree, common::Span<float const> hess,
                     std::vector<bst_node_t>* p_out_position) const {
    partition_builder_.LeafPartition(ctx, tree, this->Partitions(), p_out_position,
                                     [&](size_t idx) -> bool { return hess[idx] - .0f == .0f; });
  }

  void LeafPartition(Context const* ctx, RegTree const& tree,
                     linalg::TensorView<GradientPair const, 2> gpair,
                     std::vector<bst_node_t>* p_out_position) const {
    if (gpair.Shape(1) > 1) {
      partition_builder_.LeafPartition(
          ctx, tree, this->Partitions(), p_out_position, [&](std::size_t idx) -> bool {
            auto sample = gpair.Slice(idx, linalg::All());
            return std::all_of(linalg::cbegin(sample), linalg::cend(sample),
                               [](GradientPair const& g) { return g.GetHess() - .0f == .0f; });
          });
    } else {
      auto s = gpair.Slice(linalg::All(), 0);
      partition_builder_.LeafPartition(
          ctx, tree, this->Partitions(), p_out_position,
          [&](std::size_t idx) -> bool { return s(idx).GetHess() - .0f == .0f; });
    }
  }
  void LeafPartition(Context const* ctx, RegTree const& tree,
                     common::Span<GradientPair const> gpair,
                     std::vector<bst_node_t>* p_out_position) const {
    partition_builder_.LeafPartition(
        ctx, tree, this->Partitions(), p_out_position,
        [&](std::size_t idx) -> bool { return gpair[idx].GetHess() - .0f == .0f; });
  }

 private:
  common::PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;
  bool is_col_split_;
  ColumnSplitHelper column_split_helper_;
};

}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_
