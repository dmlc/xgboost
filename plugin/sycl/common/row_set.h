/*!
 * Copyright 2017-2023 XGBoost contributors
 */
#ifndef PLUGIN_SYCL_COMMON_ROW_SET_H_
#define PLUGIN_SYCL_COMMON_ROW_SET_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/data.h>
#pragma GCC diagnostic pop
#include <algorithm>
#include <vector>
#include <utility>

#include "../data.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {


/*! \brief Collection of rowsets stored on device in USM memory */
class RowSetCollection {
 public:
  /*! \brief data structure to store an instance set, a subset of
   *  rows (instances) associated with a particular node in a decision
   *  tree. */
  struct Elem {
    const size_t* begin{nullptr};
    const size_t* end{nullptr};
    bst_node_t node_id{-1};  // id of node associated with this instance set; -1 means uninitialized
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

  inline size_t Size() const {
    return elem_of_each_node_.size();
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

    const size_t* begin = row_indices_.Begin();
    const size_t* end = row_indices_.End();
    elem_of_each_node_.emplace_back(Elem(begin, end, 0));
  }

  auto& Data() { return row_indices_; }

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
  USMVector<size_t, MemoryType::on_device> row_indices_;
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};

}  // namespace common
}  // namespace sycl
}  // namespace xgboost


#endif  // PLUGIN_SYCL_COMMON_ROW_SET_H_
