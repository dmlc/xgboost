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
  std::string obj_name = name;
  if (ctx->IsSycl()) {
    obj_name = GetSyclImplementationName(obj_name);
  }
  auto *e = ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->Find(obj_name);
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

/* If the objective function has sycl-specific implementation,
 * returns the specific implementation name.
 * Otherwise return the orginal name without modifications.
 */
std::string ObjFunction::GetSyclImplementationName(const std::string& name) {
  const std::string sycl_postfix = "_sycl";
  auto *e = ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->Find(name + sycl_postfix);
  if (e != nullptr) {
    // Function has specific sycl implementation
    return name + sycl_postfix;
  } else {
    // Function hasn't specific sycl implementation
    LOG(FATAL) << "`" << name << "` doesn't have sycl implementation yet\n";
    return name;
  }
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
