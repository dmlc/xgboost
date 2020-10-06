/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_ROW_SET_ONEAPI_H_
#define XGBOOST_COMMON_ROW_SET_ONEAPI_H_

#include <xgboost/data.h>
#include <algorithm>
#include <vector>
#include <utility>

#include "data_oneapi.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace common {

/*! \brief collection of rowset */
class RowSetCollectionOneAPI {
 public:
  /*! \brief data structure to store an instance set, a subset of
   *  rows (instances) associated with a particular node in a decision
   *  tree. */
  struct Elem {
    const size_t* begin{nullptr};
    const size_t* end{nullptr};
    bst_node_t node_id{-1};
      // id of node associated with this instance set; -1 means uninitialized
    Elem()
         = default;
    Elem(const size_t* begin,
         const size_t* end,
         bst_node_t node_id = -1)
        : begin(begin), end(end), node_id(node_id) {}

    inline size_t Size() const {
      return end - begin;
    }
  };
  /* \brief specifies how to split a rowset into two */
  struct Split {
    std::vector<size_t> left;
    std::vector<size_t> right;
  };

  inline std::vector<Elem>::const_iterator begin() const {  // NOLINT
    return elem_of_each_node_.begin();
  }

  inline std::vector<Elem>::const_iterator end() const {  // NOLINT
    return elem_of_each_node_.end();
  }

  /*! \brief return corresponding element set given the node_id */
  inline const Elem& operator[](unsigned node_id) const {
    const Elem& e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr)
        << "access element that is not in the set";
    return e;
  }

  /*! \brief return corresponding element set given the node_id */
  inline Elem& operator[](unsigned node_id) {
    Elem& e = elem_of_each_node_[node_id];
    return e;
  }

  // clear up things
  inline void Clear() {
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  inline void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0U);

    if (row_indices_.Empty()) {  // edge case: empty instance set
      // assign arbitrary address here, to bypass nullptr check
      // (nullptr usually indicates a nonexistent rowset, but we want to
      //  indicate a valid rowset that happens to have zero length and occupies
      //  the whole instance set)
      // this is okay, as BuildHist will compute (end-begin) as the set size
      const size_t* begin = reinterpret_cast<size_t*>(20);
      const size_t* end = begin;
      elem_of_each_node_.emplace_back(Elem(begin, end, 0));
      return;
    }

    const size_t* begin = row_indices_.Begin();
    const size_t* end = row_indices_.End();
    elem_of_each_node_.emplace_back(Elem(begin, end, 0));
  }

  USMVector<size_t>& Data() { return row_indices_; }

  // split rowset into two
  inline void AddSplit(unsigned node_id,
                       unsigned left_node_id,
                       unsigned right_node_id,
                       size_t n_left,
                       size_t n_right) {
    const Elem e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr);
    size_t* all_begin = row_indices_.Begin();
    size_t* begin = all_begin + (e.begin - all_begin);

    CHECK_EQ(n_left + n_right, e.Size());
    CHECK_LE(begin + n_left, e.end);
    CHECK_EQ(begin + n_left + n_right, e.end);

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(left_node_id + 1, Elem(nullptr, nullptr, -1));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(right_node_id + 1, Elem(nullptr, nullptr, -1));
    }

    elem_of_each_node_[left_node_id] = Elem(begin, begin + n_left, left_node_id);
    elem_of_each_node_[right_node_id] = Elem(begin + n_left, e.end, right_node_id);
    elem_of_each_node_[node_id] = Elem(nullptr, nullptr, -1);
  }

 private:
  // stores the row indexes in the set
  USMVector<size_t> row_indices_;
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};

// The builder is required for samples partition to left and rights children for set of nodes
// Responsible for:
// 1) Effective memory allocation for intermediate results for multi-thread work
// 2) Merging partial results produced by threads into original row set (row_set_collection_)
// BlockSize is template to enable memory alignment easily with C++11 'alignas()' feature
template<size_t BlockSize>
class PartitionBuilderOneAPI {
 public:
  static constexpr size_t maxLocalSums = 256;
  static constexpr size_t subgroupSize = 16;

  template<typename Func>
  void Init(cl::sycl::queue qu, size_t n_nodes, Func funcNTaks) {
    qu_ = qu;
    left_right_nodes_sizes_.resize(n_nodes);
    nodes_offsets_.resize(n_nodes+1);
    result_left_rows_.resize(n_nodes);
    result_right_rows_.resize(n_nodes);

    nodes_offsets_[0] = 0;
    for (size_t i = 1; i < n_nodes+1; ++i) {
      nodes_offsets_[i] = nodes_offsets_[i-1] + funcNTaks(i-1);
    }

    if (left_data_.Size() < nodes_offsets_[n_nodes]) {
      left_data_.Resize(qu_, nodes_offsets_[n_nodes]);
      right_data_.Resize(qu_, nodes_offsets_[n_nodes]);
    }
    prefix_sums_.Resize(qu, maxLocalSums);
  }

  common::Span<size_t> GetLeftData(int nid) {
    return { left_data_.Data() + nodes_offsets_[nid], nodes_offsets_[nid + 1] - nodes_offsets_[nid] };
  }

  common::Span<size_t> GetRightData(int nid) {
    return { right_data_.Data() + nodes_offsets_[nid], nodes_offsets_[nid + 1] - nodes_offsets_[nid] };
  }

  common::Span<size_t> GetPrefixSums() {
    return { prefix_sums_.Data(), prefix_sums_.Size() };
  }

  size_t GetLocalSize(const common::Range1d& range) {
    size_t range_size = range.end() - range.begin();
    size_t local_subgroups = range_size / (maxLocalSums * subgroupSize) + !!(range_size % (maxLocalSums * subgroupSize));
    return subgroupSize * local_subgroups;
  }

  size_t GetSubgroupSize() {
    return subgroupSize;
  }

  void SetNLeftElems(int nid, size_t begin, size_t end, size_t n_left) {
    result_left_rows_[nid] = n_left;
  }

  void SetNRightElems(int nid, size_t begin, size_t end, size_t n_right) {
    result_right_rows_[nid] = n_right;
  }

  size_t GetNLeftElems(int nid) const {
    return left_right_nodes_sizes_[nid].first;
  }

  size_t GetNRightElems(int nid) const {
    return left_right_nodes_sizes_[nid].second;
  }

  void CalculateRowOffsets() {
    for (size_t i = 0; i < nodes_offsets_.size()-1; ++i) {
      left_right_nodes_sizes_[i] = {result_left_rows_[i], result_right_rows_[i]};
    }
  }

  void MergeToArray(int nid, size_t* rows_indexes) {
    size_t* left_result  = rows_indexes;
    size_t* right_result = rows_indexes + result_left_rows_[nid];

    const size_t* left = left_data_.Data() + nodes_offsets_[nid];
    const size_t* right = right_data_.Data() + nodes_offsets_[nid];
    
    qu_.memcpy(left_result, left, sizeof(size_t) * result_left_rows_[nid]).wait();
    qu_.memcpy(right_result, right, sizeof(size_t) * result_right_rows_[nid]).wait();
  }

 protected:
  std::vector<std::pair<size_t, size_t>> left_right_nodes_sizes_;
  std::vector<size_t> nodes_offsets_;
  std::vector<size_t> result_left_rows_;
  std::vector<size_t> result_right_rows_;

  USMVector<size_t> left_data_;
  USMVector<size_t> right_data_;

  USMVector<size_t> prefix_sums_;

  cl::sycl::queue qu_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
