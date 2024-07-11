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
   private:
    bst_idx_t* begin_{nullptr};
    bst_idx_t* end_{nullptr};

   public:
    bst_node_t node_id{-1};
    // id of node associated with this instance set; -1 means uninitialized
    Elem() = default;
    Elem(bst_idx_t* begin, bst_idx_t* end, bst_node_t node_id = -1)
        : begin_(begin), end_(end), node_id(node_id) {}

    // Disable copy ctor to avoid casting away the constness via copy.
    Elem(Elem const& that) = delete;
    Elem& operator=(Elem const& that) = delete;
    Elem(Elem&& that) = default;
    Elem& operator=(Elem&& that) = default;

    [[nodiscard]] std::size_t Size() const { return std::distance(begin(), end()); }

    [[nodiscard]] bst_idx_t const* begin() const { return this->begin_; }  // NOLINT
    [[nodiscard]] bst_idx_t const* end() const { return this->end_; }      // NOLINT
    [[nodiscard]] bst_idx_t* begin() { return this->begin_; }              // NOLINT
    [[nodiscard]] bst_idx_t* end() { return this->end_; }                  // NOLINT
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
      constexpr bst_idx_t* kBegin = nullptr;
      constexpr bst_idx_t* kEnd = nullptr;
      static_assert(kEnd - kBegin == 0);
      elem_of_each_node_.emplace_back(kBegin, kEnd, 0);
      return;
    }

    bst_idx_t* begin = row_indices_.data();
    bst_idx_t* end = row_indices_.data() + row_indices_.size();
    elem_of_each_node_.emplace_back(begin, end, 0);
  }

  [[nodiscard]] std::vector<bst_idx_t>* Data() { return &row_indices_; }
  [[nodiscard]] std::vector<bst_idx_t> const* Data() const { return &row_indices_; }

  // split rowset into two
  void AddSplit(bst_node_t node_id, bst_node_t left_node_id, bst_node_t right_node_id,
                bst_idx_t n_left, bst_idx_t n_right) {
    Elem& e = elem_of_each_node_[node_id];

    bst_idx_t* all_begin{nullptr};
    bst_idx_t* begin{nullptr};
    bst_idx_t* end{nullptr};
    if (e.begin() == nullptr) {
      CHECK_EQ(n_left, 0);
      CHECK_EQ(n_right, 0);
    } else {
      all_begin = row_indices_.data();
      begin = all_begin + (e.begin() - all_begin);
      end = elem_of_each_node_[node_id].end();
    }

    CHECK_EQ(n_left + n_right, e.Size());
    CHECK_LE(begin + n_left, e.end());
    CHECK_EQ(begin + n_left + n_right, e.end());

    if (left_node_id >= static_cast<bst_node_t>(elem_of_each_node_.size())) {
      elem_of_each_node_.resize(left_node_id + 1);
    }
    if (right_node_id >= static_cast<bst_node_t>(elem_of_each_node_.size())) {
      elem_of_each_node_.resize(right_node_id + 1);
    }

    elem_of_each_node_[left_node_id] = Elem{begin, begin + n_left, left_node_id};
    elem_of_each_node_[right_node_id] = Elem{begin + n_left, end, right_node_id};
    elem_of_each_node_[node_id] = Elem{nullptr, nullptr, -1};
  }

 private:
  // stores the row indexes in the set
  std::vector<bst_idx_t> row_indices_;
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_ROW_SET_H_
