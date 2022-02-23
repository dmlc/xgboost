/*!
 * Copyright 2021-2022 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_PARTITION_BUILDER_H_
#define XGBOOST_COMMON_PARTITION_BUILDER_H_

#include <xgboost/data.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "categorical.h"
#include "column_matrix.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace common {

// The builder is required for samples partition to left and rights children for set of nodes
// Responsible for:
// 1) Effective memory allocation for intermediate results for multi-thread work
// 2) Merging partial results produced by threads into original row set (row_set_collection_)
// BlockSize is template to enable memory alignment easily with C++11 'alignas()' feature
template<size_t BlockSize>
class PartitionBuilder {
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
  inline std::pair<size_t, size_t> PartitionKernel(const ColumnType& column,
                                                   common::Span<const size_t> row_indices,
                                                   common::Span<size_t> left_part,
                                                   common::Span<size_t> right_part,
                                                   size_t base_rowid, Predicate&& pred) {
    size_t* p_left_part = left_part.data();
    size_t* p_right_part = right_part.data();
    size_t nleft_elems = 0;
    size_t nright_elems = 0;
    auto state = column.GetInitialState(row_indices.front() - base_rowid);

    auto p_row_indices = row_indices.data();
    auto n_samples = row_indices.size();

    for (size_t i = 0; i < n_samples; ++i) {
      auto rid = p_row_indices[i];
      const int32_t bin_id = column.GetBinIdx(rid - base_rowid, &state);
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

  template <typename BinIdxType, bool any_missing, bool any_cat>
  void Partition(const size_t node_in_set, const size_t nid, const common::Range1d range,
                 const int32_t split_cond, GHistIndexMatrix const& gmat,
                 const ColumnMatrix& column_matrix, const RegTree& tree, const size_t* rid) {
    common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
    common::Span<size_t> left = GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right = GetRightBuffer(node_in_set, range.begin(), range.end());
    const bst_uint fid = tree[nid].SplitIndex();
    const bool default_left = tree[nid].DefaultLeft();
    const auto column_ptr = column_matrix.GetColumn<BinIdxType, any_missing>(fid);

    bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
    auto node_cats = tree.NodeCats(nid);

    auto const& index = gmat.index;
    auto const& cut_values = gmat.cut.Values();
    auto const& cut_ptrs = gmat.cut.Ptrs();

    auto pred = [&](auto ridx, auto bin_id) {
      if (any_cat && is_cat) {
        auto begin = gmat.RowIdx(ridx);
        auto end = gmat.RowIdx(ridx + 1);
        auto f_begin = cut_ptrs[fid];
        auto f_end = cut_ptrs[fid + 1];
        // bypassing the column matrix as we need the cut value instead of bin idx for categorical
        // features.
        auto gidx = BinarySearchBin(begin, end, index, f_begin, f_end);
        bool go_left;
        if (gidx == -1) {
          go_left = default_left;
        } else {
          go_left = Decision(node_cats, cut_values[gidx], default_left);
        }
        return go_left;
      } else {
        return bin_id <= split_cond;
      }
    };

    std::pair<size_t, size_t> child_nodes_sizes;
    if (column_ptr->GetType() == xgboost::common::kDenseColumn) {
      const common::DenseColumn<BinIdxType, any_missing>& column =
            static_cast<const common::DenseColumn<BinIdxType, any_missing>& >(*(column_ptr.get()));
      if (default_left) {
        child_nodes_sizes = PartitionKernel<true, any_missing>(column, rid_span, left, right,
                                                               gmat.base_rowid, pred);
      } else {
        child_nodes_sizes = PartitionKernel<false, any_missing>(column, rid_span, left, right,
                                                                gmat.base_rowid, pred);
      }
    } else {
      CHECK_EQ(any_missing, true);
      const common::SparseColumn<BinIdxType>& column
        = static_cast<const common::SparseColumn<BinIdxType>& >(*(column_ptr.get()));
      if (default_left) {
        child_nodes_sizes = PartitionKernel<true, any_missing>(column, rid_span, left, right,
                                                               gmat.base_rowid, pred);
      } else {
        child_nodes_sizes = PartitionKernel<false, any_missing>(column, rid_span, left, right,
                                                                gmat.base_rowid, pred);
      }
    }

    const size_t n_left  = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
    SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
  }

  /**
   * \brief Partition tree nodes with specific range of row indices.
   *
   * \tparam Pred       Predicate for whether a row should be partitioned to the left node.
   *
   * \param node_in_set The index of node in current batch of nodes.
   * \param nid         The cannonical node index (node index in the tree).
   * \param range       The range of input row index.
   * \param fidx        Feature index.
   * \param p_row_set_collection Pointer to rows that are  being partitioned.
   * \param pred        A callback function that returns whether current row should be
   *                    partitioned to the left node, it should accept the row index as
   *                    input and returns a boolean value.
   */
  template <typename Pred>
  void PartitionRange(const size_t node_in_set, const size_t nid, common::Range1d range,
                      bst_feature_t fidx, common::RowSetCollection* p_row_set_collection,
                      Pred pred) {
    auto& row_set_collection = *p_row_set_collection;
    const size_t* p_ridx = row_set_collection[nid].begin;
    common::Span<const size_t> ridx(p_ridx + range.begin(), p_ridx + range.end());
    common::Span<size_t> left = this->GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right = this->GetRightBuffer(node_in_set, range.begin(), range.end());
    std::pair<size_t, size_t> child_nodes_sizes = PartitionRangeKernel(ridx, left, right, pred);

    const size_t n_left = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    this->SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
    this->SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
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

  void SetNLeftElems(int nid, size_t begin, size_t end, size_t n_left) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx)->n_left = n_left;
  }

  void SetNRightElems(int nid, size_t begin, size_t end, size_t n_right) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx)->n_right = n_right;
  }


  size_t GetNLeftElems(int nid) const {
    return left_right_nodes_sizes_[nid].first;
  }

  size_t GetNRightElems(int nid) const {
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
      for (size_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
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

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_PARTITION_BUILDER_H_
