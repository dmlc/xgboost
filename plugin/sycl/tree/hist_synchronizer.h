/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_synchronizer.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_
#define PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_

#include <vector>

#include "../common/hist_util.h"
#include "expand_entry.h"

namespace xgboost {
namespace sycl {
namespace tree {

template <typename GradientSumT>
class HistUpdater;

template <typename GradientSumT>
class HistSynchronizer {
 public:
  virtual void SyncHistograms(HistUpdater<GradientSumT>* builder,
                              const std::vector<int>& sync_ids,
                              RegTree *p_tree) = 0;
  virtual ~HistSynchronizer() = default;
};

template <typename GradientSumT>
class BatchHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  void SyncHistograms(HistUpdater<GradientSumT>* builder,
                      const std::vector<int>& sync_ids,
                      RegTree *p_tree) override {
    builder->builder_monitor_.Start("SyncHistograms");
    const size_t nbins = builder->hist_builder_.GetNumBins();

    hist_sync_events_.resize(builder->nodes_for_explicit_hist_build_.size());
    for (int i = 0; i < builder->nodes_for_explicit_hist_build_.size(); i++) {
      const auto entry = builder->nodes_for_explicit_hist_build_[i];
      auto& this_hist = builder->hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot()) {
        const size_t parent_id = (*p_tree)[entry.nid].Parent();
        auto& parent_hist = builder->hist_[parent_id];
        auto& sibling_hist = builder->hist_[entry.GetSiblingId(p_tree, parent_id)];
        hist_sync_events_[i] = common::SubtractionHist(builder->qu_, &sibling_hist, parent_hist,
                                                       this_hist, nbins, ::sycl::event());
      }
    }
    builder->qu_.wait_and_throw();

    builder->builder_monitor_.Stop("SyncHistograms");
  }

  std::vector<::sycl::event> GetEvents() const {
    return hist_sync_events_;
  }

 private:
  std::vector<::sycl::event> hist_sync_events_;
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_
