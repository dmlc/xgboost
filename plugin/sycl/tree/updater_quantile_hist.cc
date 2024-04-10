/*!
 * Copyright 2017-2024 by Contributors
 * \file updater_quantile_hist.cc
 */
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/tree_updater.h"
#pragma GCC diagnostic pop

#include "xgboost/logging.h"

#include "updater_quantile_hist.h"
#include "../data.h"

namespace xgboost {
namespace sycl {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist_sycl);

DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  const DeviceOrd device_spec = ctx_->Device();
  qu_ = device_manager.GetQueue(device_spec);

  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

void QuantileHistMaker::Update(xgboost::tree::TrainParam const *param,
                               linalg::Matrix<GradientPair>* gpair,
                               DMatrix *dmat,
                               xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                               const std::vector<RegTree *> &trees) {
  LOG(FATAL) << "Not Implemented yet";
}

bool QuantileHistMaker::UpdatePredictionCache(const DMatrix* data,
                                              linalg::MatrixView<float> out_preds) {
  LOG(FATAL) << "Not Implemented yet";
}

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker_sycl")
.describe("Grow tree using quantized histogram with SYCL.")
.set_body(
    [](Context const* ctx, ObjInfo const * task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
