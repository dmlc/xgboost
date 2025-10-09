/*!
 * Copyright by Contributors 2017-2025
 * \file node.h
 */
#ifndef PLUGIN_SYCL_PREDICTOR_NODE_H_
#define PLUGIN_SYCL_PREDICTOR_NODE_H_

#include "../../src/gbm/gbtree_model.h"

namespace xgboost {
namespace sycl {
namespace predictor {

union NodeValue {
  float leaf_weight;
  float fvalue;
};

class Node {
  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

 public:
  Node() = default;

  explicit Node(const RegTree::Node& n) {
    left_child_idx = n.LeftChild();
    right_child_idx = n.RightChild();
    fidx = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx |= (1U << 31);
    }

    if (n.IsLeaf()) {
      val.leaf_weight = n.LeafValue();
    } else {
      val.fvalue = n.SplitCond();
    }
  }

  int LeftChildIdx() const {return left_child_idx; }

  int RightChildIdx() const {return right_child_idx; }

  bool IsLeaf() const { return left_child_idx == -1; }

  int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (fidx >> 31) != 0; }

  int MissingIdx() const {
    if (MissingLeft()) {
      return left_child_idx;
    } else {
      return right_child_idx;
    }
  }

  float GetFvalue() const { return val.fvalue; }

  float GetWeight() const { return val.leaf_weight; }
};

}  // namespace predictor
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_PREDICTOR_NODE_H_
