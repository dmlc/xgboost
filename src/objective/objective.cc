/*!
 * Copyright 2015-2022 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <dmlc/registry.h>
#include <xgboost/context.h>
#include <xgboost/objective.h>

#include <sstream>

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

void ObjFunction::InitEstimation(MetaInfo const&, linalg::Tensor<float, 1>* base_score) const {
  CHECK(base_score);
  base_score->Reshape(1);
  (*base_score)(0) = DefaultBaseScore();
}
}  // namespace xgboost

namespace xgboost {
namespace obj {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(regression_obj_gpu);
DMLC_REGISTRY_LINK_TAG(quantile_obj_gpu);
DMLC_REGISTRY_LINK_TAG(hinge_obj_gpu);
DMLC_REGISTRY_LINK_TAG(multiclass_obj_gpu);
DMLC_REGISTRY_LINK_TAG(lambdarank_obj);
DMLC_REGISTRY_LINK_TAG(lambdarank_obj_cu);
#else
DMLC_REGISTRY_LINK_TAG(regression_obj);
DMLC_REGISTRY_LINK_TAG(quantile_obj);
DMLC_REGISTRY_LINK_TAG(hinge_obj);
DMLC_REGISTRY_LINK_TAG(multiclass_obj);
DMLC_REGISTRY_LINK_TAG(lambdarank_obj);
#endif  // XGBOOST_USE_CUDA
}  // namespace obj
}  // namespace xgboost
