/**
 * Copyright 2022-2023 by XGBoost contributors
 */
#include "init_estimation.h"

#include <memory>                        // unique_ptr

#include "../common/stats.h"             // Mean
#include "../tree/fit_stump.h"           // FitStump
#include "xgboost/base.h"                // GradientPair
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/json.h"                // Json
#include "xgboost/linalg.h"              // Tensor,Vector
#include "xgboost/task.h"                // ObjInfo

namespace xgboost::obj {
void FitIntercept::InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const {
  if (this->Task().task == ObjInfo::kRegression) {
    CheckInitInputs(info);
  }
  // Avoid altering any state in child objective.
  HostDeviceVector<float> dummy_predt(info.labels.Size(), 0.0f, this->ctx_->gpu_id);
  HostDeviceVector<GradientPair> gpair(info.labels.Size(), GradientPair{}, this->ctx_->gpu_id);

  Json config{Object{}};
  this->SaveConfig(&config);

  std::unique_ptr<ObjFunction> new_obj{
      ObjFunction::Create(get<String const>(config["name"]), this->ctx_)};
  new_obj->LoadConfig(config);
  new_obj->GetGradient(dummy_predt, info, 0, &gpair);

  bst_target_t n_targets = this->Targets(info);
  linalg::Vector<float> leaf_weight;
  tree::FitStump(this->ctx_, info, gpair, n_targets, &leaf_weight);
  // workaround, we don't support multi-target due to binary model serialization for
  // base margin.
  common::Mean(this->ctx_, leaf_weight, base_score);
  this->PredTransform(base_score->Data());
}
}  // namespace xgboost::obj
