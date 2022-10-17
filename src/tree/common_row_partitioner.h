/*!
 * Copyright 2021-2022 XGBoost contributors
 * \file common_row_partitioner.h
 * \brief Common partitioner logic for hist and approx methods.
 */
#ifndef XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_
#define XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_

#include <limits>  // std::numeric_limits
#include <vector>

#include "../common/numeric.h"  // Iota
#include "../common/partition_builder.h"
#include "hist/expand_entry.h"           // CPUExpandEntry
#include "xgboost/generic_parameters.h"  // Context

namespace xgboost {
namespace tree {
class CommonRowPartitioner {
  static constexpr size_t kPartitionBlockSize = 2048;
  common::PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;

 public:
  bst_row_t base_rowid = 0;

  CommonRowPartitioner() = default;
  CommonRowPartitioner(Context const* ctx, bst_row_t num_row, bst_row_t _base_rowid)
      : base_rowid{_base_rowid} {
    row_set_collection_.Clear();
    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(num_row);

    std::size_t* p_row_indices = row_indices.data();
    common::Iota(ctx, p_row_indices, p_row_indices + row_indices.size(), base_rowid);
    row_set_collection_.Init();
  }

  void FindSplitConditions(const std::vector<CPUExpandEntry>& nodes, const RegTree& tree,
                           const GHistIndexMatrix& gmat, std::vector<int32_t>* split_conditions) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      const int32_t nid = nodes[i].nid;
      const bst_uint fid = tree[nid].SplitIndex();
      const bst_float split_pt = tree[nid].SplitCond();
      const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
      const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
      bst_bin_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (auto bound = lower_bound; bound < upper_bound; ++bound) {
        if (split_pt == gmat.cut.Values()[bound]) {
          split_cond = static_cast<int32_t>(bound);
        }
      }
      (*split_conditions).at(i) = split_cond;
    }
  }

  void AddSplitsToRowSet(const std::vector<CPUExpandEntry>& nodes, RegTree const* p_tree) {
    const size_t n_nodes = nodes.size();
    for (unsigned int i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes[i].nid;
      const size_t n_left = partition_builder_.GetNLeftElems(i);
      const size_t n_right = partition_builder_.GetNRightElems(i);
      CHECK_EQ((*p_tree)[nid].LeftChild() + 1, (*p_tree)[nid].RightChild());
      row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(), (*p_tree)[nid].RightChild(),
                                   n_left, n_right);
    }
  }

  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree) {
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

  template <bool any_cat>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree) {
    if (column_matrix.AnyMissing()) {
      this->template UpdatePosition<true, any_cat>(ctx, gmat, column_matrix, nodes, p_tree);
    } else {
      this->template UpdatePosition<false, any_cat>(ctx, gmat, column_matrix, nodes, p_tree);
    }
  }

  template <bool any_missing, bool any_cat>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree) {
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        this->template UpdatePosition<uint8_t, any_missing, any_cat>(ctx, gmat, column_matrix,
                                                                     nodes, p_tree);
        break;
      case common::kUint16BinsTypeSize:
        this->template UpdatePosition<uint16_t, any_missing, any_cat>(ctx, gmat, column_matrix,
                                                                      nodes, p_tree);
        break;
      case common::kUint32BinsTypeSize:
        this->template UpdatePosition<uint32_t, any_missing, any_cat>(ctx, gmat, column_matrix,
                                                                      nodes, p_tree);
        break;
      default:
        // no default behavior
        CHECK(false) << column_matrix.GetTypeSize();
    }
  }

  template <typename BinIdxType, bool any_missing, bool any_cat>
  void UpdatePosition(Context const* ctx, GHistIndexMatrix const& gmat,
                      const common::ColumnMatrix& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree) {
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

  auto const& Partitions() const { return row_set_collection_; }

  size_t Size() const {
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
                     common::Span<GradientPair const> gpair,
                     std::vector<bst_node_t>* p_out_position) const {
    partition_builder_.LeafPartition(
        ctx, tree, this->Partitions(), p_out_position,
        [&](size_t idx) -> bool { return gpair[idx].GetHess() - .0f == .0f; });
  }
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_
