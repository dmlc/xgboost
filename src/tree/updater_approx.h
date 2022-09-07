/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_UPDATER_APPROX_H_
#define XGBOOST_TREE_UPDATER_APPROX_H_

#include <limits>
#include <utility>
#include <vector>

#include "../common/partition_builder.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "hist/param.h"
#include "param.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class ApproxRowPartitioner {
  static constexpr size_t kPartitionBlockSize = 2048;
  common::PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;

 public:
  bst_row_t base_rowid = 0;

  static auto SearchCutValue(bst_row_t ridx, bst_feature_t fidx, GHistIndexMatrix const &index,
                             std::vector<uint32_t> const &cut_ptrs,
                             std::vector<float> const &cut_values) {
    int32_t gidx = -1;
    if (index.IsDense()) {
      // RowIdx returns the starting pos of this row
      gidx = index.index[index.RowIdx(ridx) + fidx];
    } else {
      auto begin = index.RowIdx(ridx);
      auto end = index.RowIdx(ridx + 1);
      auto f_begin = cut_ptrs[fidx];
      auto f_end = cut_ptrs[fidx + 1];
      gidx = common::BinarySearchBin(begin, end, index.index, f_begin, f_end);
    }
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return cut_values[gidx];
  }

 public:
  void UpdatePosition(GenericParameter const *ctx, GHistIndexMatrix const &index,
                      std::vector<CPUExpandEntry> const &candidates, RegTree const *p_tree) {
    size_t n_nodes = candidates.size();

    auto const &cut_values = index.cut.Values();
    auto const &cut_ptrs = index.cut.Ptrs();

    common::BlockedSpace2d space{n_nodes,
                                 [&](size_t node_in_set) {
                                   auto candidate = candidates[node_in_set];
                                   int32_t nid = candidate.nid;
                                   return row_set_collection_[nid].Size();
                                 },
                                 kPartitionBlockSize};
    partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
      auto candidate = candidates[node_in_set];
      const int32_t nid = candidate.nid;
      const size_t size = row_set_collection_[nid].Size();
      const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
      return n_tasks;
    });
    auto node_ptr = p_tree->GetCategoriesMatrix().node_ptr;
    auto categories = p_tree->GetCategoriesMatrix().categories;
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
      auto candidate = candidates[node_in_set];
      auto is_cat = candidate.split.is_cat;
      const int32_t nid = candidate.nid;
      auto fidx = candidate.split.SplitIndex();
      const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, r.begin());
      partition_builder_.AllocateForTask(task_id);
      partition_builder_.PartitionRange(
          node_in_set, nid, r, fidx, &row_set_collection_, [&](size_t row_id) {
            auto cut_value = SearchCutValue(row_id, fidx, index, cut_ptrs, cut_values);
            if (std::isnan(cut_value)) {
              return candidate.split.DefaultLeft();
            }
            bst_node_t nidx = candidate.nid;
            auto segment = node_ptr[nidx];
            auto node_cats = categories.subspan(segment.beg, segment.size);
            bool go_left = true;
            if (is_cat) {
              go_left = common::Decision(node_cats, cut_value, candidate.split.DefaultLeft());
            } else {
              go_left = cut_value <= candidate.split.split_value;
            }
            return go_left;
          });
    });

    partition_builder_.CalculateRowOffsets();
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
      auto candidate = candidates[node_in_set];
      const int32_t nid = candidate.nid;
      partition_builder_.MergeToArray(node_in_set, r.begin(),
                                      const_cast<size_t *>(row_set_collection_[nid].begin));
    });
    for (size_t i = 0; i < candidates.size(); ++i) {
      auto const &candidate = candidates[i];
      auto nidx = candidate.nid;
      auto n_left = partition_builder_.GetNLeftElems(i);
      auto n_right = partition_builder_.GetNRightElems(i);
      CHECK_EQ(n_left + n_right, row_set_collection_[nidx].Size());
      bst_node_t left_nidx = (*p_tree)[nidx].LeftChild();
      bst_node_t right_nidx = (*p_tree)[nidx].RightChild();
      row_set_collection_.AddSplit(nidx, left_nidx, right_nidx, n_left, n_right);
    }
  }

  auto const &Partitions() const { return row_set_collection_; }

  auto operator[](bst_node_t nidx) { return row_set_collection_[nidx]; }
  auto const &operator[](bst_node_t nidx) const { return row_set_collection_[nidx]; }

  size_t Size() const {
    return std::distance(row_set_collection_.begin(), row_set_collection_.end());
  }

  ApproxRowPartitioner() = default;
  explicit ApproxRowPartitioner(bst_row_t num_row, bst_row_t _base_rowid)
      : base_rowid{_base_rowid} {
    row_set_collection_.Clear();
    auto p_positions = row_set_collection_.Data();
    p_positions->resize(num_row);
    std::iota(p_positions->begin(), p_positions->end(), base_rowid);
    row_set_collection_.Init();
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_APPROX_H_
