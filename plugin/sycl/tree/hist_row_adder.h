/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_row_adder.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_
#define PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_

#include <vector>
#include <algorithm>

namespace xgboost {
namespace sycl {
namespace tree {

template <typename GradientSumT>
class HistRowsAdder {
 public:
  virtual void AddHistRows(HistUpdater<GradientSumT>* builder,
                           std::vector<int>* sync_ids, RegTree *p_tree) = 0;
  virtual ~HistRowsAdder() = default;
};

template <typename GradientSumT>
class BatchHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  void AddHistRows(HistUpdater<GradientSumT>* builder,
                   std::vector<int>* sync_ids, RegTree *p_tree) override {
    builder->builder_monitor_.Start("AddHistRows");

    for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
      int nid = entry.nid;
      auto event = builder->hist_.AddHistRow(nid);
    }
    for (auto const& node : builder->nodes_for_subtraction_trick_) {
      auto event = builder->hist_.AddHistRow(node.nid);
    }

    builder->builder_monitor_.Stop("AddHistRows");
  }
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_
