/*!
 * Copyright 2017-2024 by Contributors
 * \file updater_quantile_hist.h
 */
#ifndef PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
#define PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <xgboost/tree_updater.h>

#include <vector>
#include <memory>

#include "../data/gradient_index.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/partition_builder.h"
#include "split_evaluator.h"
#include "../device_manager.h"
#include "hist_updater.h"
#include "xgboost/data.h"

#include "xgboost/json.h"
#include "../../src/tree/constraints.h"
#include "../../src/common/random.h"

namespace xgboost {
namespace sycl {
namespace tree {

// training parameters specific to this algorithm
struct HistMakerTrainParam
    : public XGBoostParameter<HistMakerTrainParam> {
  bool single_precision_histogram = false;
  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
  }
};

/*! \brief construct a tree using quantized feature values with SYCL backend*/
class QuantileHistMaker: public TreeUpdater {
 public:
  QuantileHistMaker(Context const* ctx, ObjInfo const * task) :
                             TreeUpdater(ctx), task_{task} {
    updater_monitor_.Init("SYCLQuantileHistMaker");
  }
  void Configure(const Args& args) override;

  void Update(xgboost::tree::TrainParam const *param,
              linalg::Matrix<GradientPair>* gpair,
              DMatrix* dmat,
              xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::MatrixView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    FromJson(config.at("sycl_hist_train_param"), &this->hist_maker_param_);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["sycl_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker_sycl";
  }

 protected:
  HistMakerTrainParam hist_maker_param_;
  // training parameter
  xgboost::tree::TrainParam param_;
  // quantized data matrix
  common::GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  // column accessor
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

  xgboost::common::Monitor updater_monitor_;

  template<typename GradientSumT>
  void SetPimpl(std::unique_ptr<HistUpdater<GradientSumT>>*, DMatrix *dmat);

  template<typename GradientSumT>
  void CallUpdate(const std::unique_ptr<HistUpdater<GradientSumT>>& builder,
                  xgboost::tree::TrainParam const *param,
                  linalg::Matrix<GradientPair> *gpair,
                  DMatrix *dmat,
                  xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                  const std::vector<RegTree *> &trees);

  enum class HistPrecision {fp32, fp64};
  HistPrecision hist_precision_;

  std::unique_ptr<HistUpdater<float>> pimpl_fp32;
  std::unique_ptr<HistUpdater<double>> pimpl_fp64;

  FeatureInteractionConstraintHost int_constraint_;

  ::sycl::queue* qu_;
  DeviceManager device_manager;
  ObjInfo const *task_{nullptr};
};


}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
