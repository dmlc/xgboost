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
  /*! \brief data structure to store an instance set, a subset of
   *  rows (instances) associated with a particular node in a decision
   *  tree. */
  struct Elem {
    const size_t* begin{nullptr};
    const size_t* end{nullptr};
    int node_id{-1};
      // id of node associated with this instance set; -1 means uninitialized
    Elem()
         = default;
    Elem(const size_t* begin,
         const size_t* end,
         int node_id)
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
  // clear up things
  inline void Clear() {
    row_indices_.clear();
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  inline void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0U);

    if (row_indices_.empty()) {  // edge case: empty instance set
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

    const size_t* begin = dmlc::BeginPtr(row_indices_);
    const size_t* end = dmlc::BeginPtr(row_indices_) + row_indices_.size();
    elem_of_each_node_.emplace_back(Elem(begin, end, 0));
  }
  // split rowset into two
  inline void AddSplit(unsigned node_id,
                       const std::vector<Split>& row_split_tloc,
                       unsigned left_node_id,
                       unsigned right_node_id) {
    const Elem e = elem_of_each_node_[node_id];
    const auto nthread = static_cast<bst_omp_uint>(row_split_tloc.size());
    CHECK(e.begin != nullptr);
    size_t* all_begin = dmlc::BeginPtr(row_indices_);
    size_t* begin = all_begin + (e.begin - all_begin);

    size_t* it = begin;
    for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
      std::copy(row_split_tloc[tid].left.begin(), row_split_tloc[tid].left.end(), it);
      it += row_split_tloc[tid].left.size();
    }
    size_t* split_pt = it;
    for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
      std::copy(row_split_tloc[tid].right.begin(), row_split_tloc[tid].right.end(), it);
      it += row_split_tloc[tid].right.size();
    }

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(left_node_id + 1, Elem(nullptr, nullptr, -1));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(right_node_id + 1, Elem(nullptr, nullptr, -1));
    }

    elem_of_each_node_[left_node_id] = Elem(begin, split_pt, left_node_id);
    elem_of_each_node_[right_node_id] = Elem(split_pt, e.end, right_node_id);
    elem_of_each_node_[node_id] = Elem(nullptr, nullptr, -1);
  }

  // stores the row indices in the set
  std::vector<size_t> row_indices_;

 private:
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
