/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_updater.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_UPDATER_H_
#define PLUGIN_SYCL_TREE_HIST_UPDATER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/tree_updater.h>
#pragma GCC diagnostic pop

#include <utility>
#include <memory>

#include "../common/partition_builder.h"
#include "split_evaluator.h"

#include "../data.h"

namespace xgboost {
namespace sycl {
namespace tree {

template<typename GradientSumT>
class HistUpdater {
 public:
  explicit HistUpdater(::sycl::queue qu,
                    const xgboost::tree::TrainParam& param,
                    std::unique_ptr<TreeUpdater> pruner,
                    FeatureInteractionConstraintHost int_constraints_,
                    DMatrix const* fmat)
    : qu_(qu), param_(param),
      tree_evaluator_(qu, param, fmat->Info().num_col_),
      pruner_(std::move(pruner)),
      interaction_constraints_{std::move(int_constraints_)},
      p_last_tree_(nullptr), p_last_fmat_(fmat) {
    builder_monitor_.Init("SYCL::Quantile::HistUpdater");
    kernel_monitor_.Init("SYCL::Quantile::HistUpdater");
    const auto sub_group_sizes =
      qu_.get_device().get_info<::sycl::info::device::sub_group_sizes>();
    sub_group_size_ = sub_group_sizes.back();
  }

 protected:
  void InitSampling(const USMVector<GradientPair, MemoryType::on_device> &gpair,
                    USMVector<size_t, MemoryType::on_device>* row_indices);


  void InitData(Context const * ctx,
                const common::GHistIndexMatrix& gmat,
                const USMVector<GradientPair, MemoryType::on_device> &gpair,
                const DMatrix& fmat,
                const RegTree& tree);

  //  --data fields--
  size_t sub_group_size_;

  // the internal row sets
  common::RowSetCollection row_set_collection_;

  const xgboost::tree::TrainParam& param_;
  TreeEvaluator<GradientSumT> tree_evaluator_;
  std::unique_ptr<TreeUpdater> pruner_;
  FeatureInteractionConstraintHost interaction_constraints_;

  // back pointers to tree and data matrix
  const RegTree* p_last_tree_;
  DMatrix const* const p_last_fmat_;

  xgboost::common::Monitor builder_monitor_;
  xgboost::common::Monitor kernel_monitor_;

  uint64_t seed_ = 0;

  ::sycl::queue qu_;
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_UPDATER_H_
