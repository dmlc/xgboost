/*!
 * Copyright 2017-2024 XGBoost contributors
 */
#ifndef PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_
#define PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/data.h>
#pragma GCC diagnostic pop
#include <xgboost/tree_model.h>

#include <algorithm>
#include <vector>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../../src/common/column_matrix.h"
#pragma GCC diagnostic pop

#include "../data.h"
#include "row_set.h"
#include "../data/gradient_index.h"
#include "../tree/expand_entry.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

// split row indexes (rid_span) to 2 parts (both stored in rid_buf) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
template <bool default_left, typename BinIdxType>
inline ::sycl::event PartitionDenseKernel(
                                 ::sycl::queue* qu,
                                 const GHistIndexMatrix& gmat,
                                 const RowSetCollection::Elem& rid_span,
                                 const size_t fid,
                                 const int32_t split_cond,
                                 xgboost::common::Span<size_t>* rid_buf,
                                 size_t* parts_size,
                                 ::sycl::event event) {
  const size_t row_stride = gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const size_t* rid = rid_span.begin;
  const size_t range_size = rid_span.Size();
  const size_t offset = gmat.cut.Ptrs()[fid];

  size_t* p_rid_buf = rid_buf->data();

  return qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event);
    cgh.parallel_for<>(::sycl::range<1>(range_size), [=](::sycl::item<1> nid) {
      const size_t id = rid[nid.get_id(0)];
      const int32_t value = static_cast<int32_t>(gradient_index[id * row_stride + fid] + offset);
      const bool is_left = value <= split_cond;
      if (is_left) {
        AtomicRef<size_t> n_left(parts_size[0]);
        p_rid_buf[n_left.fetch_add(1)] = id;
      } else {
        AtomicRef<size_t> n_right(parts_size[1]);
        p_rid_buf[range_size - n_right.fetch_add(1) - 1] = id;
      }
    });
  });
}

// split row indexes (rid_span) to 2 parts (both stored in rid_buf) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle sparce columns
template <bool default_left, typename BinIdxType>
inline ::sycl::event PartitionSparseKernel(::sycl::queue* qu,
                                  const GHistIndexMatrix& gmat,
                                  const RowSetCollection::Elem& rid_span,
                                  const size_t fid,
                                  const int32_t split_cond,
                                  xgboost::common::Span<size_t>* rid_buf,
                                  size_t* parts_size,
                                  ::sycl::event event) {
  const size_t row_stride = gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const size_t* rid = rid_span.begin;
  const size_t range_size = rid_span.Size();
  const uint32_t* cut_ptrs = gmat.cut_device.Ptrs().DataConst();

  size_t* p_rid_buf = rid_buf->data();
  return qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event);
    cgh.parallel_for<>(::sycl::range<1>(range_size), [=](::sycl::item<1> nid) {
      const size_t id = rid[nid.get_id(0)];

      const BinIdxType* gr_index_local = gradient_index + row_stride * id;
      const int32_t fid_local = std::lower_bound(gr_index_local,
                                                 gr_index_local + row_stride,
                                                 cut_ptrs[fid]) - gr_index_local;
      const bool is_left = (fid_local >= row_stride ||
                            gr_index_local[fid_local] >= cut_ptrs[fid + 1]) ?
                              default_left :
                              gr_index_local[fid_local] <= split_cond;
      if (is_left) {
        AtomicRef<size_t> n_left(parts_size[0]);
        p_rid_buf[n_left.fetch_add(1)] = id;
      } else {
        AtomicRef<size_t> n_right(parts_size[1]);
        p_rid_buf[range_size - n_right.fetch_add(1) - 1] = id;
      }
    });
  });
}

// The builder is required for samples partition to left and rights children for set of nodes
class PartitionBuilder {
 public:
  template<typename Func>
  void Init(::sycl::queue* qu, size_t n_nodes, Func funcNTaks) {
    qu_ = qu;
    nodes_offsets_.resize(n_nodes+1);
    result_rows_.resize(2 * n_nodes);
    n_nodes_ = n_nodes;


    nodes_offsets_[0] = 0;
    for (size_t i = 1; i < n_nodes+1; ++i) {
      nodes_offsets_[i] = nodes_offsets_[i-1] + funcNTaks(i-1);
    }

    if (data_.Size() < nodes_offsets_[n_nodes]) {
      data_.Resize(qu, nodes_offsets_[n_nodes]);
    }
  }

  size_t GetNLeftElems(int nid) const {
    return result_rows_[2 * nid];
  }

  size_t GetNRightElems(int nid) const {
    return result_rows_[2 * nid + 1];
  }

  // For test purposes only
  void SetNLeftElems(int nid, size_t val) {
    result_rows_[2 * nid] = val;
  }

  // For test purposes only
  void SetNRightElems(int nid, size_t val) {
    result_rows_[2 * nid + 1] = val;
  }

  xgboost::common::Span<size_t> GetData(int nid) {
    return { data_.Data() + nodes_offsets_[nid], nodes_offsets_[nid + 1] - nodes_offsets_[nid] };
  }

  template <typename BinIdxType>
  ::sycl::event Partition(const int32_t split_cond,
                        const GHistIndexMatrix& gmat,
                        const RowSetCollection::Elem& rid_span,
                        const xgboost::RegTree::Node& node,
                        xgboost::common::Span<size_t>* rid_buf,
                        size_t* parts_size,
                        ::sycl::event event) {
    const bst_uint fid = node.SplitIndex();
    const bool default_left = node.DefaultLeft();

    if (gmat.IsDense()) {
      if (default_left) {
        return PartitionDenseKernel<true, BinIdxType>(qu_, gmat, rid_span, fid,
                                                      split_cond, rid_buf, parts_size, event);
      } else {
        return PartitionDenseKernel<false, BinIdxType>(qu_, gmat, rid_span, fid,
                                                      split_cond, rid_buf, parts_size, event);
      }
    } else {
      if (default_left) {
        return PartitionSparseKernel<true, BinIdxType>(qu_, gmat, rid_span, fid,
                                                      split_cond, rid_buf, parts_size, event);
      } else {
        return PartitionSparseKernel<false, BinIdxType>(qu_, gmat, rid_span, fid,
                                                        split_cond, rid_buf, parts_size, event);
      }
    }
  }

  // Entry point for Partition
  void Partition(const GHistIndexMatrix& gmat,
                 const std::vector<tree::ExpandEntry> nodes,
                 const RowSetCollection& row_set_collection,
                 const std::vector<int32_t>& split_conditions,
                 RegTree* p_tree,
                 ::sycl::event* general_event) {
    nodes_events_.resize(n_nodes_);

    parts_size_.ResizeAndFill(qu_, 2 * n_nodes_, 0, general_event);

    for (size_t node_in_set = 0; node_in_set < n_nodes_; node_in_set++) {
      const int32_t nid = nodes[node_in_set].nid;
      ::sycl::event& node_event = nodes_events_[node_in_set];
      const auto& rid_span = row_set_collection[nid];
      if (rid_span.Size() > 0) {
        const RegTree::Node& node = (*p_tree)[nid];
        xgboost::common::Span<size_t> rid_buf = GetData(node_in_set);
        size_t* part_size = parts_size_.Data() + 2 * node_in_set;
        int32_t split_condition = split_conditions[node_in_set];
        switch (gmat.index.GetBinTypeSize()) {
          case common::BinTypeSize::kUint8BinsTypeSize:
            node_event = Partition<uint8_t>(split_condition, gmat, rid_span, node,
                                            &rid_buf, part_size, *general_event);
            break;
          case common::BinTypeSize::kUint16BinsTypeSize:
            node_event = Partition<uint16_t>(split_condition, gmat, rid_span, node,
                                            &rid_buf, part_size, *general_event);
            break;
          case common::BinTypeSize::kUint32BinsTypeSize:
            node_event = Partition<uint32_t>(split_condition, gmat, rid_span, node,
                                            &rid_buf, part_size, *general_event);
            break;
          default:
            CHECK(false);  // no default behavior
        }
      } else {
        node_event = ::sycl::event();
      }
    }

    *general_event = qu_->memcpy(result_rows_.data(),
                                 parts_size_.DataConst(),
                                 sizeof(size_t) * 2 * n_nodes_,
                                 nodes_events_);
  }

  void MergeToArray(size_t nid,
                    size_t* data_result,
                    ::sycl::event* event) {
    size_t n_nodes_total = GetNLeftElems(nid) + GetNRightElems(nid);
    if (n_nodes_total > 0) {
      const size_t* data = data_.Data() + nodes_offsets_[nid];
      qu_->memcpy(data_result, data, sizeof(size_t) * n_nodes_total, *event);
    }
  }

 protected:
  std::vector<size_t> nodes_offsets_;
  std::vector<size_t> result_rows_;
  std::vector<::sycl::event> nodes_events_;
  size_t n_nodes_;

  USMVector<size_t, MemoryType::on_device> parts_size_;
  USMVector<size_t, MemoryType::on_device> data_;

  ::sycl::queue* qu_;
};

}  // namespace common
}  // namespace sycl
}  // namespace xgboost


#endif  // PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_
