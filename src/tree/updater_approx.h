/*!
 * Copyright 2021-2022 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_UPDATER_APPROX_H_
#define XGBOOST_TREE_UPDATER_APPROX_H_

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "../common/partition_builder.h"
#include "../common/opt_partition_builder.h"
#include "../common/column_matrix.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "param.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class RowPartitioner {
 public:
  using NodeIdListT = std::vector<uint16_t>;
  using NodeMaskListT = std::unordered_map<uint32_t, bool>;
  using SplitFtrListT = std::unordered_map<uint32_t, int32_t>;
  using SplitIndListT = std::unordered_map<uint32_t, uint64_t>;

 private:
  common::OptPartitionBuilder opt_partition_builder_;
  NodeIdListT node_ids_;

 public:
  bst_row_t base_rowid = 0;
  bool is_loss_guided = false;

 private:
  /**
   * \brief Class for storing UpdatePosition template parameters' values for dispatching simplification
   */
    class DispatchParameterSet final {
     public:
        DispatchParameterSet(bool has_missing, common::BinTypeSize bin_type_size,
                             bool loss_guide, bool has_categorical) :
                    has_missing_(has_missing), bin_type_size_(bin_type_size),
                    loss_guide_(loss_guide), has_categorical_(has_categorical) {}

        common::BinTypeSize GetBinTypeSize() const { return bin_type_size_; }

        bool GetHasMissing() const { return has_missing_; }
        bool GetLossGuide() const { return loss_guide_; }
        bool GetHasCategorical() const { return has_categorical_; }

     private:
        bool has_missing_;
        common::BinTypeSize bin_type_size_;
        bool loss_guide_;
        bool has_categorical_;
    };

  /**
   * \brief Class for storing UpdatePosition call params' values
   *   for simplification of dispatching by template parameters
   */
  class UpdatePositionHelper final {
   public:
    UpdatePositionHelper(xgboost::tree::RowPartitioner* row_partitioner,
      GenericParameter const* ctx, GHistIndexMatrix const& gmat,
      // common::ColumnMatrix const& column_matrix,
      std::vector<xgboost::tree::CPUExpandEntry> const& nodes,
      RegTree const* p_tree,
      int depth,
      NodeMaskListT* smalest_nodes_mask_ptr,
      const bool loss_guide,
      SplitFtrListT* split_conditions,
      SplitIndListT* split_ind,
      const size_t max_depth,
      NodeIdListT* child_node_ids,
      bool is_left_small = true,
      bool check_is_left_small = false) :
        row_partitioner_(*row_partitioner),
        ctx_(ctx),
        gmat_(gmat),
        // column_matrix_(gmat.Transpose()),
        nodes_(nodes),
        p_tree_(p_tree),
        depth_(depth),
        smalest_nodes_mask_ptr_(smalest_nodes_mask_ptr),
        loss_guide_(loss_guide),
        split_conditions_(split_conditions),
        split_ind_(split_ind),
        max_depth_(max_depth),
        child_node_ids_(child_node_ids),
        is_left_small_(is_left_small),
        check_is_left_small_(check_is_left_small) { }

    template <bool missing, typename BinType, bool is_loss_guide, bool has_cat>
    void Call() {
      row_partitioner_.template UpdatePosition<missing, BinType, is_loss_guide, has_cat>(
        ctx_,
        gmat_,
        // column_matrix_,
        nodes_,
        p_tree_,
        depth_,
        smalest_nodes_mask_ptr_,
        loss_guide_,
        split_conditions_,
        split_ind_,
        max_depth_,
        child_node_ids_,
        is_left_small_,
        check_is_left_small_);
    }

   private:
    xgboost::tree::RowPartitioner& row_partitioner_;
    GenericParameter const* ctx_;
    GHistIndexMatrix const& gmat_;
    // common::ColumnMatrix const& column_matrix_;
    std::vector<xgboost::tree::CPUExpandEntry> const& nodes_;
    RegTree const* p_tree_;
    int depth_;
    NodeMaskListT* smalest_nodes_mask_ptr_;
    const bool loss_guide_;
    SplitFtrListT* split_conditions_;
    SplitIndListT* split_ind_;
    const size_t max_depth_;
    NodeIdListT* child_node_ids_;
    bool is_left_small_;
    bool check_is_left_small_;
  };

 public:
  template <bool ... switch_values_set>
  void DispatchFromHasMissing(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
    std::integer_sequence<bool, switch_values_set...>) {
      const unsigned int one = 1;
      const unsigned int zero = 0;
    std::initializer_list<unsigned int>{(dispatch_values.GetHasMissing()
      == switch_values_set ?
      DispatchFromBinType<switch_values_set>(std::move(pos_updater), std::move(dispatch_values),
      std::move(common::BinTypeSizeSequence{})), one : zero)...};
  }

  template <bool missing, unsigned int ... switch_values_set>
  void DispatchFromBinType(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
                           std::integer_sequence<unsigned int, switch_values_set...>) {
      const unsigned int one = 1;
      const unsigned int zero = 0;
      std::initializer_list<unsigned int>{(dispatch_values.GetBinTypeSize() == switch_values_set ?
                    DispatchFromLossGuide<missing,
                    typename common::BinTypeMap<switch_values_set>::type>(std::move(pos_updater),
                    std::move(dispatch_values), std::move(common::BoolSequence{})), one : zero)...};
  }

  template <bool missing, typename BinType, bool ... switch_values_set>
  void DispatchFromLossGuide(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
                             std::integer_sequence<bool, switch_values_set...>) {
      const unsigned int one = 1;
      const unsigned int zero = 0;
                             std::initializer_list<unsigned int>{(dispatch_values.GetLossGuide()
                             == switch_values_set ?
                             DispatchFromHasCategorical<missing, BinType,
                             switch_values_set>(std::move(pos_updater),
                             std::move(dispatch_values),
                             std::move(common::BoolSequence{})), one : zero)...};
  }

  template <bool missing, typename BinType, bool is_loss_guide, bool ... switch_values_set>
  void DispatchFromHasCategorical(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
    std::integer_sequence<bool, switch_values_set...>) {
    const unsigned int one = 1;
    const unsigned int zero = 0;
    std::initializer_list<unsigned int>{(dispatch_values.GetHasCategorical()
    == switch_values_set ?
    pos_updater.template Call<missing, BinType, is_loss_guide,
    switch_values_set>(), one : zero)...};
  }

  /**
   * \brief Turn split values into discrete bin indices.
   */
  void FindSplitConditions(const std::vector<CPUExpandEntry> &nodes,
                           const RegTree &tree, const GHistIndexMatrix &gmat,
                           SplitFtrListT *split_conditions) {
    for (const auto& node : nodes) {
      const int32_t nid = node.nid;
      const bst_uint fid = tree[nid].SplitIndex();
      const bst_float split_pt = tree[nid].SplitCond();
      const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
      const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
      int32_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
        if (split_pt == gmat.cut.Values()[bound]) {
          split_cond = static_cast<int32_t>(bound);
        }
      }
      (*split_conditions)[nid] = split_cond;
    }
  }

  template <typename ... Args>
  void UpdatePositionDispatched(DispatchParameterSet&& dispatch_params, Args&& ... args) {
      UpdatePositionHelper helper(this, std::forward<Args>(args)...);
      DispatchFromHasMissing(std::move(helper), std::move(dispatch_params),
      std::move(common::BoolSequence{}));
  }

  template <bool any_missing, typename BinIdxType,
            bool is_loss_guided, bool any_cat>
  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
    std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree,
    int depth,
    NodeMaskListT* smalest_nodes_mask_ptr,
    const bool loss_guide,
    SplitFtrListT* split_conditions_,
    SplitIndListT* split_ind_, const size_t max_depth,
    NodeIdListT* child_node_ids_,
    bool is_left_small = true,
    bool check_is_left_small = false) {
    common::ColumnMatrix const& column_matrix = gmat.Transpose();
if (column_matrix.GetIndexData() != opt_partition_builder_.data_hash ||
    column_matrix.GetMissing() != opt_partition_builder_.missing_ptr ||
    column_matrix.GetRowId() != opt_partition_builder_.row_ind_ptr) {
  // CHECK(false);
    // common::ColumnMatrix const &column_matrix = gmat.Transpose();
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree,
                                                ctx->Threads(), max_depth,
                                                is_loss_guided);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree,
                                                ctx->Threads(), max_depth,
                                                is_loss_guided);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree,
                                                ctx->Threads(), max_depth,
                                                is_loss_guided);
        break;
      default:
        CHECK(false);  // no default behavior
    }
}

    // 1. Find split condition for each split
    const size_t n_nodes = nodes.size();
    FindSplitConditions(nodes, *p_tree, gmat, split_conditions_);
    // 2.1 Create a blocked space of size SUM(samples in each node)
    const uint32_t* offsets = gmat.index.Offset();
    const uint64_t rows_offset = gmat.row_ptr.size() - 1;
    std::vector<uint32_t> split_nodes(n_nodes, 0);
    // std::cout << "nodes spliting info:" << std::endl;
    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t nid = nodes[i].nid;
        split_nodes[i] = nid;
        const uint64_t fid = (*p_tree)[nid].SplitIndex();
        (*split_ind_)[nid] = fid*((gmat.IsDense() ? rows_offset : 1));
        (*split_conditions_)[nid] = (*split_conditions_)[nid] - gmat.cut.Ptrs()[fid];
    }
    std::vector<uint64_t> split_ind_data_vec;
    std::vector<int32_t> split_conditions_data_vec;
    std::vector<bool> smalest_nodes_mask_vec;
    if (max_depth != 0) {
      split_ind_data_vec.resize((1 << (max_depth + 2)), 0);
      split_conditions_data_vec.resize((1 << (max_depth + 2)), 0);
      smalest_nodes_mask_vec.resize((1 << (max_depth + 2)), 0);
      for (size_t nid = 0; nid < (1 << (max_depth + 2)); ++nid) {
        split_ind_data_vec[nid] = (*split_ind_)[nid];
        split_conditions_data_vec[nid] = (*split_conditions_)[nid];
        smalest_nodes_mask_vec[nid] = (*smalest_nodes_mask_ptr)[nid];
      }
    }
    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    int nthreads = ctx->Threads();
    nthreads = std::max(nthreads, 1);
    const size_t depth_begin = opt_partition_builder_.DepthBegin(*child_node_ids_,
                                                                 loss_guide);
    const size_t depth_size = opt_partition_builder_.DepthSize(gmat, *child_node_ids_,
                                                               loss_guide);

    auto const& index = gmat.index;
    auto const& cut_values = gmat.cut.Values();
    auto const& cut_ptrs = gmat.cut.Ptrs();
    RegTree const tree = *p_tree;
    auto pred = [&](auto ridx, auto bin_id, auto nid, auto split_cond) {
      if (!any_cat) {
        return bin_id <= split_cond;
      }
      bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
      if (any_cat && is_cat) {
        const bst_uint fid = tree[nid].SplitIndex();
        const bool default_left = tree[nid].DefaultLeft();
        // const auto column_ptr = column_matrix.GetColumn<BinIdxType, any_missing>(fid);
        auto node_cats = tree.NodeCats(nid);
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
    if (max_depth != 0) {
    // for (size_t tid = 0; tid < nthreads; ++tid)
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        // std::cout << "partitioner:" << std::endl;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(
          tid, begin, end, numa,
          node_ids_.data(),
          &split_conditions_data_vec,
          &split_ind_data_vec,
          &smalest_nodes_mask_vec,
          column_matrix,
          split_nodes, pred, depth);
        // std::cout << std::endl;
      }
    } else {
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(
          tid, begin, end, numa,
          node_ids_.data(),
          split_conditions_,
          split_ind_,
          smalest_nodes_mask_ptr,
          column_matrix,
          split_nodes, pred, depth);
      }
    }

    if (depth != max_depth || loss_guide) {
      opt_partition_builder_.UpdateRowBuffer(*child_node_ids_,
                                             gmat, n_features, depth,
                                             node_ids_, is_loss_guided);
      opt_partition_builder_.UpdateThreadsWork(*child_node_ids_, gmat,
                                               n_features, depth, is_loss_guided,
                                               is_left_small, check_is_left_small);
    }
  }

  NodeIdListT &GetNodeAssignments() { return node_ids_; }

  void LeafPartition(Context const *ctx, RegTree const &tree,  // common::Span<float const> hess,
                     std::vector<bst_node_t> *p_out_position) const {
    auto& h_pos = *p_out_position;
    const uint16_t* node_ids_data_ptr = node_ids_.data();
    h_pos.resize(node_ids_.size(), std::numeric_limits<bst_node_t>::max());
    xgboost::common::ParallelFor(node_ids_.size(), ctx->Threads(), [&](size_t i) {
      h_pos[i] = node_ids_data_ptr[i];
    });
    // partition_builder_.LeafPartition(ctx, tree, this->Partitions(), p_out_position,
    //                                  [&](size_t idx) -> bool { return hess[idx] - .0f == .0f; });
  }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  auto const &GetOptPartition() const {
    return opt_partition_builder_;
  }

  RowPartitioner() = default;
  explicit RowPartitioner(GenericParameter const *ctx,
                                GHistIndexMatrix const &gmat,
                                const RegTree* p_tree_local,
                                size_t max_depth,
                                bool is_loss_guide) {
    is_loss_guided = is_loss_guide;
    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guided) {
      opt_partition_builder_.ResizeRowsBuffer(gmat.row_ptr.size() - 1);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(ctx->Threads())
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(gmat.row_ptr.size() - 1));
        for (size_t i = ibegin; i < iend; ++i) {
          row_set_collection_vec_p[i] = i;
        }
      }
    }
    common::ColumnMatrix const &column_matrix = gmat.Transpose();
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }

  void Reset(GenericParameter const *ctx,
                     GHistIndexMatrix const &gmat,
                     const RegTree* p_tree_local,
                     size_t max_depth,
                     bool is_loss_guide) {
    common::ColumnMatrix const & column_matrix = gmat.Transpose();
    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guide) {
      opt_partition_builder_.ResizeRowsBuffer(gmat.row_ptr.size() - 1);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(ctx->Threads())
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(gmat.row_ptr.size() - 1));
        for (size_t i = ibegin; i < iend; ++i) {
          row_set_collection_vec_p[i] = i;
        }
      }
    }
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_APPROX_H_
