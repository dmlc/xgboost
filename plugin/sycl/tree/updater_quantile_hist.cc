/*!
 * Copyright 2017-2024 by Contributors
 * \file updater_quantile_hist.cc
 */
#include <memory>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/gradient.h"  // for GradientContainer
#include "xgboost/tree_updater.h"
#pragma GCC diagnostic pop

#include "../../src/collective/communicator-inl.h"  // for IsDistributed
#include "../data.h"
#include "updater_quantile_hist.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace sycl {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist_sycl);

DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

std::set<std::string> QuantileHistMaker::Configure(const Args &args) {
  const DeviceOrd device_spec = ctx_->Device();
  qu_ = device_manager.GetQueue(device_spec);

  auto used = UpdateAndGetUsedParameters(&param_, args);
  used.merge(UpdateAndGetUsedParameters(&hist_maker_param_, args));

  bool has_fp64_support = qu_->get_device().has(::sycl::aspect::fp64);
  if (hist_maker_param_.single_precision_histogram || !has_fp64_support) {
    if (!hist_maker_param_.single_precision_histogram) {
      LOG(WARNING) << "Target device doesn't support fp64, using single_precision_histogram=True";
    }
    hist_precision_ = HistPrecision::fp32;
  } else {
    hist_precision_ = HistPrecision::fp64;
  }
  return used;
}

template <typename GradientSumT>
void QuantileHistMaker::SetPimpl(std::unique_ptr<HistUpdater<GradientSumT>> *pimpl, DMatrix *dmat) {
  pimpl->reset(new HistUpdater<GradientSumT>(ctx_, qu_, param_, int_constraint_, dmat));
  if (collective::IsDistributed()) {
    (*pimpl)->SetHistSynchronizer(new DistributedHistSynchronizer<GradientSumT>());
    (*pimpl)->SetHistRowsAdder(new DistributedHistRowsAdder<GradientSumT>());
  } else {
    (*pimpl)->SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
    (*pimpl)->SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  }
}

template <typename GradientSumT>
void QuantileHistMaker::CallUpdate(const std::unique_ptr<HistUpdater<GradientSumT>> &pimpl,
                                   xgboost::tree::TrainParam const *param,
                                   ::xgboost::linalg::Matrix<GradientPair> *gpair, DMatrix *dmat,
                                   xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                                   const std::vector<RegTree *> &trees) {
  CHECK_EQ(out_position.size(), trees.size());
  for (std::size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx) {
    pimpl->Update(param, gmat_, *(gpair->Data()), dmat, &out_position[tree_idx], trees[tree_idx]);
  }
}

void QuantileHistMaker::Update(xgboost::tree::TrainParam const *param, GradientContainer *in_gpair,
                               DMatrix *dmat,
                               xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                               const std::vector<RegTree *> &trees) {
  auto gpair = in_gpair->FullGradOnly();
  gpair->Data()->SetDevice(ctx_->Device());
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    gmat_.Init(qu_, ctx_, dmat, static_cast<uint32_t>(param_.max_bin));
    updater_monitor_.Stop("GmatInitialization");
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  int_constraint_.Configure(param_, dmat->Info().num_col_);
  // build tree
  if (hist_precision_ == HistPrecision::fp32) {
    if (!pimpl_fp32) {
      SetPimpl(&pimpl_fp32, dmat);
    }
    CallUpdate(pimpl_fp32, param, gpair, dmat, out_position, trees);
  } else {
    if (!pimpl_fp64) {
      SetPimpl(&pimpl_fp64, dmat);
    }
    CallUpdate(pimpl_fp64, param, gpair, dmat, out_position, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker_sycl")
    .describe("Grow tree using quantized histogram with SYCL.")
    .set_body([](Context const *ctx, ObjInfo const *task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
