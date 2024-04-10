/*!
 * Copyright 2017-2024 by Contributors
 * \file updater_quantile_hist.h
 */
#ifndef PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
#define PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <xgboost/tree_updater.h>

#include <vector>

#include "../data/gradient_index.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/partition_builder.h"
#include "split_evaluator.h"
#include "../device_manager.h"

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
    try {
      FromJson(config.at("sycl_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range& e) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING) << "Attempted to load interal configuration for a model file that was generated "
        << "by a previous version of XGBoost. A likely cause for this warning is that the model "
        << "was saved with saveRDS() in R or pickle.dump() in Python. We strongly ADVISE AGAINST "
        << "using saveRDS() or pickle.dump() so that the model remains accessible in current and "
        << "upcoming XGBoost releases. Please use xgb.save() instead to preserve models for the "
        << "long term. For more details and explanation, see "
        << "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html";
      this->hist_maker_param_.UpdateAllowUnknown(Args{});
    }
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

  xgboost::common::Monitor updater_monitor_;

  ::sycl::queue qu_;
  DeviceManager device_manager;
  ObjInfo const *task_{nullptr};
};


}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
