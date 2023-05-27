/**
 * Copyright 2021-2023 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_PARTITION_BUILDER_H_
#define XGBOOST_COMMON_PARTITION_BUILDER_H_

#include <xgboost/data.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../tree/hist/expand_entry.h"
#include "categorical.h"
#include "column_matrix.h"
#include "xgboost/context.h"
#include "xgboost/tree_model.h"

namespace xgboost::common {
// The builder is required for samples partition to left and rights children for set of nodes
// Responsible for:
// 1) Effective memory allocation for intermediate results for multi-thread work
// 2) Merging partial results produced by threads into original row set (row_set_collection_)
// BlockSize is template to enable memory alignment easily with C++11 'alignas()' feature
template<size_t BlockSize>
class PartitionBuilder {
  using BitVector = RBitField8;

 public:
  template<typename Func>
  void Init(const size_t n_tasks, size_t n_nodes, Func funcNTask) {
    left_right_nodes_sizes_.resize(n_nodes);
    blocks_offsets_.resize(n_nodes+1);

    blocks_offsets_[0] = 0;
    for (size_t i = 1; i < n_nodes+1; ++i) {
      blocks_offsets_[i] = blocks_offsets_[i-1] + funcNTask(i-1);
    }

    if (n_tasks > max_n_tasks_) {
      mem_blocks_.resize(n_tasks);
      max_n_tasks_ = n_tasks;
    }
  }

  // split row indexes (rid_span) to 2 parts (left_part, right_part) depending
  // on comparison of indexes values (idx_span) and split point (split_cond)
  // Handle dense columns
  // Analog of std::stable_partition, but in no-inplace manner
  template <bool default_left, bool any_missing, typename ColumnType, typename Predicate>
  inline std::pair<size_t, size_t> PartitionKernel(ColumnType* p_column,
                                                   common::Span<const size_t> row_indices,
                                                   common::Span<size_t> left_part,
                                                   common::Span<size_t> right_part,
                                                   size_t base_rowid, Predicate&& pred) {
    auto& column = *p_column;
    size_t* p_left_part = left_part.data();
    size_t* p_right_part = right_part.data();
    size_t nleft_elems = 0;
    size_t nright_elems = 0;

    auto p_row_indices = row_indices.data();
    auto n_samples = row_indices.size();

    for (size_t i = 0; i < n_samples; ++i) {
      auto rid = p_row_indices[i];
      const int32_t bin_id = column[rid - base_rowid];
      if (any_missing && bin_id == ColumnType::kMissingId) {
        if (default_left) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      } else {
        if (pred(rid, bin_id)) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      }
    }

    return {nleft_elems, nright_elems};
  }

  template <typename Pred>
  inline std::pair<size_t, size_t> PartitionRangeKernel(common::Span<const size_t> ridx,
                                                        common::Span<size_t> left_part,
                                                        common::Span<size_t> right_part,
                                                        Pred pred) {
    size_t* p_left_part = left_part.data();
    size_t* p_right_part = right_part.data();
    size_t nleft_elems = 0;
    size_t nright_elems = 0;
    for (auto row_id : ridx) {
      if (pred(row_id)) {
        p_left_part[nleft_elems++] = row_id;
      } else {
        p_right_part[nright_elems++] = row_id;
      }
    }
    return {nleft_elems, nright_elems};
  }

  template <typename BinIdxType, bool any_missing, bool any_cat, typename ExpandEntry>
  void Partition(const size_t node_in_set, std::vector<ExpandEntry> const& nodes,
                 const common::Range1d range, const bst_bin_t split_cond,
                 GHistIndexMatrix const& gmat, const common::ColumnMatrix& column_matrix,
                 const RegTree& tree, const size_t* rid) {
    common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
    common::Span<size_t> left = GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right = GetRightBuffer(node_in_set, range.begin(), range.end());
    std::size_t nid = nodes[node_in_set].nid;
    bst_feature_t fid = tree.SplitIndex(nid);
    bool default_left = tree.DefaultLeft(nid);
    bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
    auto node_cats = tree.NodeCats(nid);
    auto const& cut_values = gmat.cut.Values();

    auto pred_hist = [&](auto ridx, auto bin_id) {
      if (any_cat && is_cat) {
        auto gidx = gmat.GetGindex(ridx, fid);
        bool go_left = default_left;
        if (gidx > -1) {
          go_left = Decision(node_cats, cut_values[gidx]);
        }
        return go_left;
      } else {
        return bin_id <= split_cond;
      }
    };

    auto pred_approx = [&](auto ridx) {
      auto gidx = gmat.GetGindex(ridx, fid);
      bool go_left = default_left;
      if (gidx > -1) {
        if (is_cat) {
          go_left = Decision(node_cats, cut_values[gidx]);
        } else {
          go_left = cut_values[gidx] <= nodes[node_in_set].split.split_value;
        }
      }
      return go_left;
    };

    std::pair<size_t, size_t> child_nodes_sizes;
    if (!column_matrix.IsInitialized()) {
      child_nodes_sizes = PartitionRangeKernel(rid_span, left, right, pred_approx);
    } else {
      if (column_matrix.GetColumnType(fid) == xgboost::common::kDenseColumn) {
        auto column = column_matrix.DenseColumn<BinIdxType, any_missing>(fid);
        if (default_left) {
          child_nodes_sizes = PartitionKernel<true, any_missing>(&column, rid_span, left, right,
                                                                 gmat.base_rowid, pred_hist);
        } else {
          child_nodes_sizes = PartitionKernel<false, any_missing>(&column, rid_span, left, right,
                                                                  gmat.base_rowid, pred_hist);
        }
      } else {
        CHECK_EQ(any_missing, true);
        auto column =
            column_matrix.SparseColumn<BinIdxType>(fid, rid_span.front() - gmat.base_rowid);
        if (default_left) {
          child_nodes_sizes = PartitionKernel<true, any_missing>(&column, rid_span, left, right,
                                                                 gmat.base_rowid, pred_hist);
        } else {
          child_nodes_sizes = PartitionKernel<false, any_missing>(&column, rid_span, left, right,
                                                                  gmat.base_rowid, pred_hist);
        }
      }
    }

    const size_t n_left  = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    SetNLeftElems(node_in_set, range.begin(), n_left);
    SetNRightElems(node_in_set, range.begin(), n_right);
  }

  template <bool any_missing, typename ColumnType, typename Predicate>
  void MaskKernel(ColumnType* p_column, common::Span<const size_t> row_indices, size_t base_rowid,
                  BitVector* decision_bits, BitVector* missing_bits, Predicate&& pred) {
    auto& column = *p_column;
    for (auto const row_id : row_indices) {
      auto const bin_id = column[row_id - base_rowid];
      if (any_missing && bin_id == ColumnType::kMissingId) {
        missing_bits->Set(row_id - base_rowid);
      } else if (pred(row_id, bin_id)) {
        decision_bits->Set(row_id - base_rowid);
      }
    }
  }

  /**
   * @brief When data is split by column, we don't have all the features locally on the current
   * worker, so we go through all the rows and mark the bit vectors on whether the decision is made
   * to go right, or if the feature value used for the split is missing.
   */
  template <typename BinIdxType, bool any_missing, bool any_cat, typename ExpandEntry>
  void MaskRows(const size_t node_in_set, std::vector<ExpandEntry> const& nodes,
                const common::Range1d range, bst_bin_t split_cond, GHistIndexMatrix const& gmat,
                const common::ColumnMatrix& column_matrix, const RegTree& tree, const size_t* rid,
                BitVector* decision_bits, BitVector* missing_bits) {
    common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
    std::size_t nid = nodes[node_in_set].nid;
    bst_feature_t fid = tree.SplitIndex(nid);
    bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
    auto node_cats = tree.NodeCats(nid);
    auto const& cut_values = gmat.cut.Values();

    if (!column_matrix.IsInitialized()) {
      for (auto row_id : rid_span) {
        auto gidx = gmat.GetGindex(row_id, fid);
        if (gidx > -1) {
          bool go_left;
          if (is_cat) {
            go_left = Decision(node_cats, cut_values[gidx]);
          } else {
            go_left = cut_values[gidx] <= nodes[node_in_set].split.split_value;
          }
          if (go_left) {
            decision_bits->Set(row_id - gmat.base_rowid);
          }
        } else {
          missing_bits->Set(row_id - gmat.base_rowid);
        }
      }
    } else {
      auto pred_hist = [&](auto ridx, auto bin_id) {
        if (any_cat && is_cat) {
          auto gidx = gmat.GetGindex(ridx, fid);
          CHECK_GT(gidx, -1);
          return Decision(node_cats, cut_values[gidx]);
        } else {
          return bin_id <= split_cond;
        }
      };

      if (column_matrix.GetColumnType(fid) == xgboost::common::kDenseColumn) {
        auto column = column_matrix.DenseColumn<BinIdxType, any_missing>(fid);
        MaskKernel<any_missing>(&column, rid_span, gmat.base_rowid, decision_bits, missing_bits,
                                pred_hist);
      } else {
        CHECK_EQ(any_missing, true);
        auto column =
            column_matrix.SparseColumn<BinIdxType>(fid, rid_span.front() - gmat.base_rowid);
        MaskKernel<any_missing>(&column, rid_span, gmat.base_rowid, decision_bits, missing_bits,
                                pred_hist);
      }
    }
  }

  /**
   * @brief Once we've aggregated the decision and missing bits from all the workers, we can then
   * use them to partition the rows accordingly.
   */
  template <typename ExpandEntry>
  void PartitionByMask(const size_t node_in_set, std::vector<ExpandEntry> const& nodes,
                       const common::Range1d range, GHistIndexMatrix const& gmat,
                       const RegTree& tree, const size_t* rid, BitVector const& decision_bits,
                       BitVector const& missing_bits) {
    common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
    common::Span<size_t> left = GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right = GetRightBuffer(node_in_set, range.begin(), range.end());
    std::size_t nid = nodes[node_in_set].nid;
    bool default_left = tree.DefaultLeft(nid);

    auto pred = [&](auto ridx) {
      bool go_left = default_left;
      bool is_missing = missing_bits.Check(ridx - gmat.base_rowid);
      if (!is_missing) {
        go_left = decision_bits.Check(ridx - gmat.base_rowid);
      }
      return go_left;
    };

    std::pair<size_t, size_t> child_nodes_sizes;
    child_nodes_sizes = PartitionRangeKernel(rid_span, left, right, pred);

    const size_t n_left  = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    SetNLeftElems(node_in_set, range.begin(), n_left);
    SetNRightElems(node_in_set, range.begin(), n_right);
  }

  // allocate thread local memory, should be called for each specific task
  void AllocateForTask(size_t id) {
    if (mem_blocks_[id].get() == nullptr) {
      BlockInfo* local_block_ptr = new BlockInfo;
      CHECK_NE(local_block_ptr, (BlockInfo*)nullptr);
      mem_blocks_[id].reset(local_block_ptr);
    }
  }

  common::Span<size_t> GetLeftBuffer(int nid, size_t begin, size_t end) {
    const size_t task_idx = GetTaskIdx(nid, begin);
    return { mem_blocks_.at(task_idx)->Left(), end - begin };
  }

  common::Span<size_t> GetRightBuffer(int nid, size_t begin, size_t end) {
    const size_t task_idx = GetTaskIdx(nid, begin);
    return { mem_blocks_.at(task_idx)->Right(), end - begin };
  }

  void SetNLeftElems(int nid, size_t begin, size_t n_left) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx)->n_left = n_left;
  }

  void SetNRightElems(int nid, size_t begin, size_t n_right) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx)->n_right = n_right;
  }


  [[nodiscard]] std::size_t GetNLeftElems(int nid) const {
    return left_right_nodes_sizes_[nid].first;
  }

  [[nodiscard]] std::size_t GetNRightElems(int nid) const {
    return left_right_nodes_sizes_[nid].second;
  }

  // Each thread has partial results for some set of tree-nodes
  // The function decides order of merging partial results into final row set
  void CalculateRowOffsets() {
    for (size_t i = 0; i < blocks_offsets_.size()-1; ++i) {
      size_t n_left = 0;
      for (size_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
        mem_blocks_[j]->n_offset_left = n_left;
        n_left += mem_blocks_[j]->n_left;
      }
      size_t n_right = 0;
      for (size_t j = blocks_offsets_[i]; j < blocks_offsets_[i + 1]; ++j) {
        mem_blocks_[j]->n_offset_right = n_left + n_right;
        n_right += mem_blocks_[j]->n_right;
      }
      left_right_nodes_sizes_[i] = {n_left, n_right};
    }
  }

  void MergeToArray(int nid, size_t begin, size_t* rows_indexes) {
    size_t task_idx = GetTaskIdx(nid, begin);

    size_t* left_result  = rows_indexes + mem_blocks_[task_idx]->n_offset_left;
    size_t* right_result = rows_indexes + mem_blocks_[task_idx]->n_offset_right;

    const size_t* left = mem_blocks_[task_idx]->Left();
    const size_t* right = mem_blocks_[task_idx]->Right();

    std::copy_n(left, mem_blocks_[task_idx]->n_left, left_result);
    std::copy_n(right, mem_blocks_[task_idx]->n_right, right_result);
  }

  size_t GetTaskIdx(int nid, size_t begin) {
    return blocks_offsets_[nid] + begin / BlockSize;
  }

  // Copy row partitions into global cache for reuse in objective
  template <typename Sampledp>
  void LeafPartition(Context const* ctx, RegTree const& tree, RowSetCollection const& row_set,
                     std::vector<bst_node_t>* p_position, Sampledp sampledp) const {
    auto& h_pos = *p_position;
    h_pos.resize(row_set.Data()->size(), std::numeric_limits<bst_node_t>::max());

    auto p_begin = row_set.Data()->data();
    ParallelFor(row_set.Size(), ctx->Threads(), [&](size_t i) {
      auto const& node = row_set[i];
      if (node.node_id < 0) {
        return;
      }
      CHECK(tree.IsLeaf(node.node_id));
      if (node.begin) {  // guard for empty node.
        size_t ptr_offset = node.end - p_begin;
        CHECK_LE(ptr_offset, row_set.Data()->size()) << node.node_id;
        for (auto idx = node.begin; idx != node.end; ++idx) {
          h_pos[*idx] = sampledp(*idx) ? ~node.node_id : node.node_id;
        }
      }
    });
  }

 protected:
  struct BlockInfo{
    size_t n_left;
    size_t n_right;

    size_t n_offset_left;
    size_t n_offset_right;

    size_t* Left() {
      return &left_data_[0];
    }

    size_t* Right() {
      return &right_data_[0];
    }
   private:
    size_t left_data_[BlockSize];
    size_t right_data_[BlockSize];
  };
  std::vector<std::pair<size_t, size_t>> left_right_nodes_sizes_;
  std::vector<size_t> blocks_offsets_;
  std::vector<std::shared_ptr<BlockInfo>> mem_blocks_;
  size_t max_n_tasks_ = 0;
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_PARTITION_BUILDER_H_
