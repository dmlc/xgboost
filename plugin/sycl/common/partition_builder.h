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

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

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

  void MergeToArray(size_t nid,
                    size_t* data_result,
                    ::sycl::event event) {
    size_t n_nodes_total = GetNLeftElems(nid) + GetNRightElems(nid);
    if (n_nodes_total > 0) {
      const size_t* data = data_.Data() + nodes_offsets_[nid];
      qu_->memcpy(data_result, data, sizeof(size_t) * n_nodes_total, event);
    }
  }

 protected:
  std::vector<size_t> nodes_offsets_;
  std::vector<size_t> result_rows_;
  size_t n_nodes_;

  USMVector<size_t, MemoryType::on_device> parts_size_;
  USMVector<size_t, MemoryType::on_device> data_;

  ::sycl::queue* qu_;
};

}  // namespace common
}  // namespace sycl
}  // namespace xgboost


#endif  // PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_
