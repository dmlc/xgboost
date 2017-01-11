/*!
 * Copyright 2017 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_ROW_SET_H_
#define XGBOOST_COMMON_ROW_SET_H_

#include <xgboost/data.h>
#include <algorithm>
#include <vector>

namespace xgboost {
namespace common {

/*! \brief collection of rowset */
class RowSetCollection {
 public:
  /*! \brief subset of rows */
  struct Elem {
    const bst_uint* begin;
    const bst_uint* end;
    Elem(void)
        : begin(nullptr), end(nullptr) {}
    Elem(const bst_uint* begin,
         const bst_uint* end)
        : begin(begin), end(end) {}

    inline size_t size() const {
      return end - begin;
    }
  };
  /* \brief specifies how to split a rowset into two */
  struct Split {
    std::vector<bst_uint> left;
    std::vector<bst_uint> right;
  };
  /*! \brief return corresponding element set given the node_id */
  inline const Elem& operator[](unsigned node_id) const {
    const Elem& e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr)
        << "access element that is not in the set";
    return e;
  }
  // clear up things
  inline void Clear() {
    row_indices_.clear();
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  inline void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0);
    const bst_uint* begin = dmlc::BeginPtr(row_indices_);
    const bst_uint* end = dmlc::BeginPtr(row_indices_) + row_indices_.size();
    elem_of_each_node_.emplace_back(Elem(begin, end));
  }
  // split rowset into two
  inline void AddSplit(unsigned node_id,
                       const std::vector<Split>& row_split_tloc,
                       unsigned left_node_id,
                       unsigned right_node_id) {
    const Elem e = elem_of_each_node_[node_id];
    const unsigned nthread = row_split_tloc.size();
    CHECK(e.begin != nullptr);
    bst_uint* all_begin = dmlc::BeginPtr(row_indices_);
    bst_uint* begin = all_begin + (e.begin - all_begin);

    bst_uint* it = begin;
    // TODO(hcho3): parallelize this section
    for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
      std::copy(row_split_tloc[tid].left.begin(), row_split_tloc[tid].left.end(), it);
      it += row_split_tloc[tid].left.size();
    }
    bst_uint* split_pt = it;
    for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
      std::copy(row_split_tloc[tid].right.begin(), row_split_tloc[tid].right.end(), it);
      it += row_split_tloc[tid].right.size();
    }

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(left_node_id + 1, Elem(nullptr, nullptr));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(right_node_id + 1, Elem(nullptr, nullptr));
    }

    elem_of_each_node_[left_node_id] = Elem(begin, split_pt);
    elem_of_each_node_[right_node_id] = Elem(split_pt, e.end);
    elem_of_each_node_[node_id] = Elem(nullptr, nullptr);
  }

  // stores the row indices in the set
  std::vector<bst_uint> row_indices_;

 private:
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
