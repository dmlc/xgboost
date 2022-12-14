/*!
 * Copyright 2015-2022 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <dmlc/registry.h>
#include <xgboost/context.h>  // Context
#include <xgboost/objective.h>

#include <sstream>

#include "../common/stats.h"    // Mean
#include "../tree/fit_stump.h"  // FitStump
#include "validation.h"         // CheckInitInputs
#include "xgboost/data.h"       // MetaInfo
#include "xgboost/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::ObjFunctionReg);
}  // namespace dmlc

namespace xgboost {
// implement factory functions
ObjFunction* ObjFunction::Create(const std::string& name, Context const* ctx) {
  auto *e = ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->Find(name);
  if (e == nullptr) {
    std::stringstream ss;
    for (const auto& entry : ::dmlc::Registry< ::xgboost::ObjFunctionReg>::List()) {
      ss << "Objective candidate: " << entry->name << "\n";
    }
    LOG(FATAL) << "Unknown objective function: `" << name << "`\n"
               << ss.str();
  }
  auto pobj = (e->body)();
  pobj->ctx_ = ctx;
  return pobj;
}

void ObjFunction::InitEstimation(MetaInfo const& info, linalg::Tensor<float, 1>* base_score) const {
  obj::CheckInitInputs(info);
  // Avoid altering any state in child objective.
  HostDeviceVector<float> dummy_predt(info.labels.Size(), 0.0f);
  HostDeviceVector<GradientPair> gpair(info.labels.Size());

  Json config{Object{}};
  this->SaveConfig(&config);

  std::unique_ptr<ObjFunction> new_obj{
      ObjFunction::Create(get<String const>(config["name"]), ctx_)};
  new_obj->LoadConfig(config);
  new_obj->GetGradient(dummy_predt, info, 0, &gpair);
  bst_target_t n_targets = this->Targets(info);
  linalg::Vector<float> leaf_weight;
  tree::FitStump(ctx_, gpair, n_targets, &leaf_weight);

  // workaround, we don't support multi-target due to binary model serialization for
  // base margin.
  common::Mean(ctx_, leaf_weight, base_score);
  auto h_base_margin = base_score->HostView();
  this->PredTransform(base_score->Data());
}
}  // namespace xgboost

namespace xgboost {
namespace obj {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(regression_obj_gpu);
DMLC_REGISTRY_LINK_TAG(hinge_obj_gpu);
DMLC_REGISTRY_LINK_TAG(multiclass_obj_gpu);
DMLC_REGISTRY_LINK_TAG(rank_obj_gpu);
#else
DMLC_REGISTRY_LINK_TAG(regression_obj);
DMLC_REGISTRY_LINK_TAG(hinge_obj);
DMLC_REGISTRY_LINK_TAG(multiclass_obj);
DMLC_REGISTRY_LINK_TAG(rank_obj);
#endif  // XGBOOST_USE_CUDA
}  // namespace obj
}  // namespace xgboost
