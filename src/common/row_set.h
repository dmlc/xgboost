/**
 * Copyright 2017-2024, XGBoost Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_ROW_SET_H_
#define XGBOOST_COMMON_ROW_SET_H_

#include <cstddef>   // for size_t
#include <iterator>  // for distance
#include <vector>    // for vector

#include "xgboost/base.h"     // for bst_node_t
#include "xgboost/logging.h"  // for CHECK

namespace xgboost::common {
/**
 * @brief Collection of rows for each tree node.
 */
class RowSetCollection {
 public:
  RowSetCollection() = default;
  RowSetCollection(RowSetCollection const&) = delete;
  RowSetCollection(RowSetCollection&&) = default;
  RowSetCollection& operator=(RowSetCollection const&) = delete;
  RowSetCollection& operator=(RowSetCollection&&) = default;

  /**
   * @brief data structure to store an instance set, a subset of rows (instances)
   *        associated with a particular node in a decision tree.
   */
  struct Elem {
    std::size_t const* begin{nullptr};
    std::size_t const* end{nullptr};
    bst_node_t node_id{-1};
    // id of node associated with this instance set; -1 means uninitialized
    Elem() = default;
    Elem(std::size_t const* begin, std::size_t const* end, bst_node_t node_id = -1)
        : begin(begin), end(end), node_id(node_id) {}

    std::size_t Size() const { return end - begin; }
  };

  [[nodiscard]] std::vector<Elem>::const_iterator begin() const {  // NOLINT
    return elem_of_each_node_.cbegin();
  }
  [[nodiscard]] std::vector<Elem>::const_iterator end() const {  // NOLINT
    return elem_of_each_node_.cend();
  }

  [[nodiscard]] std::size_t Size() const { return std::distance(begin(), end()); }

  /** @brief return corresponding element set given the node_id */
  [[nodiscard]] Elem const& operator[](bst_node_t node_id) const {
    Elem const& e = elem_of_each_node_[node_id];
    return e;
  }
  /** @brief return corresponding element set given the node_id */
  [[nodiscard]] Elem& operator[](bst_node_t node_id) {
    Elem& e = elem_of_each_node_[node_id];
    return e;
  }

  // clear up things
  void Clear() {
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  void Init() {
    CHECK(elem_of_each_node_.empty());

    if (row_indices_.empty()) {  // edge case: empty instance set
      constexpr std::size_t* kBegin = nullptr;
      constexpr std::size_t* kEnd = nullptr;
      static_assert(kEnd - kBegin == 0);
      elem_of_each_node_.emplace_back(kBegin, kEnd, 0);
      return;
    }

    const std::size_t* begin = dmlc::BeginPtr(row_indices_);
    const std::size_t* end = dmlc::BeginPtr(row_indices_) + row_indices_.size();
    elem_of_each_node_.emplace_back(begin, end, 0);
  }

  [[nodiscard]] std::vector<std::size_t>* Data() { return &row_indices_; }
  [[nodiscard]] std::vector<std::size_t> const* Data() const { return &row_indices_; }

  // split rowset into two
  void AddSplit(bst_node_t node_id, bst_node_t left_node_id, bst_node_t right_node_id,
                bst_idx_t n_left, bst_idx_t n_right) {
    const Elem e = elem_of_each_node_[node_id];

    std::size_t* all_begin{nullptr};
    std::size_t* begin{nullptr};
    if (e.begin == nullptr) {
      CHECK_EQ(n_left, 0);
      CHECK_EQ(n_right, 0);
    } else {
      all_begin = row_indices_.data();
      begin = all_begin + (e.begin - all_begin);
    }

    CHECK_EQ(n_left + n_right, e.Size());
    CHECK_LE(begin + n_left, e.end);
    CHECK_EQ(begin + n_left + n_right, e.end);

    if (left_node_id >= static_cast<bst_node_t>(elem_of_each_node_.size())) {
      elem_of_each_node_.resize(left_node_id + 1, Elem{nullptr, nullptr, -1});
    }
    if (right_node_id >= static_cast<bst_node_t>(elem_of_each_node_.size())) {
      elem_of_each_node_.resize(right_node_id + 1, Elem{nullptr, nullptr, -1});
    }

    elem_of_each_node_[left_node_id] = Elem{begin, begin + n_left, left_node_id};
    elem_of_each_node_[right_node_id] = Elem{begin + n_left, e.end, right_node_id};
    elem_of_each_node_[node_id] = Elem{nullptr, nullptr, -1};
  }

 private:
  // stores the row indexes in the set
  std::vector<std::size_t> row_indices_;
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_ROW_SET_H_
