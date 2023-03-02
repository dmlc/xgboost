/*!
 * Copyright 2017-2022 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_ROW_SET_H_
#define XGBOOST_COMMON_ROW_SET_H_

#include <xgboost/data.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <memory>

namespace xgboost {
namespace common {
/*! \brief collection of rowset */
class RowSetCollection {
 public:
  RowSetCollection() = default;
  RowSetCollection(RowSetCollection const&) = delete;
  RowSetCollection(RowSetCollection&&) = default;
  RowSetCollection& operator=(RowSetCollection const&) = delete;
  RowSetCollection& operator=(RowSetCollection&&) = default;

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

  std::vector<Elem>::const_iterator begin() const {  // NOLINT
    return elem_of_each_node_.begin();
  }

  std::vector<Elem>::const_iterator end() const {  // NOLINT
    return elem_of_each_node_.end();
  }

  size_t Size() const { return std::distance(begin(), end()); }

  /*! \brief return corresponding element set given the node_id */
  inline const Elem& operator[](unsigned node_id) const {
    const Elem& e = elem_of_each_node_[node_id];
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

    if (row_indices_.empty()) {  // edge case: empty instance set
      constexpr size_t* kBegin = nullptr;
      constexpr size_t* kEnd = nullptr;
      static_assert(kEnd - kBegin == 0);
      elem_of_each_node_.emplace_back(kBegin, kEnd, 0);
      return;
    }

    const size_t* begin = dmlc::BeginPtr(row_indices_);
    const size_t* end = dmlc::BeginPtr(row_indices_) + row_indices_.size();
    elem_of_each_node_.emplace_back(begin, end, 0);
  }

  std::vector<size_t>* Data() { return &row_indices_; }
  std::vector<size_t> const* Data() const { return &row_indices_; }

  // split rowset into two
  inline void AddSplit(unsigned node_id, unsigned left_node_id, unsigned right_node_id,
                       size_t n_left, size_t n_right) {
    const Elem e = elem_of_each_node_[node_id];

    size_t* all_begin{nullptr};
    size_t* begin{nullptr};
    if (e.begin == nullptr) {
      CHECK_EQ(n_left, 0);
      CHECK_EQ(n_right, 0);
    } else {
      all_begin = dmlc::BeginPtr(row_indices_);
      begin = all_begin + (e.begin - all_begin);
    }

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
  std::vector<size_t> row_indices_;
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
