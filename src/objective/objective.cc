/*!
 * Copyright 2015 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <xgboost/objective.h>
#include <dmlc/registry.h>

#include "../common/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::ObjFunctionReg);
}  // namespace dmlc

namespace xgboost {
// implement factory functions
ObjFunction* ObjFunction::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->Find(name);
  if (e == nullptr) {
    for (const auto& entry : ::dmlc::Registry< ::xgboost::ObjFunctionReg>::List()) {
      LOG(INFO) << "Objective candidate: " << entry->name;
    }
    LOG(FATAL) << "Unknown objective function " << name;
  }
  return (e->body)();
}

}  // namespace xgboost

namespace xgboost {
namespace obj {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(regression_obj);
#ifdef XGBOOST_USE_CUDA
  DMLC_REGISTRY_LINK_TAG(regression_obj_gpu);
#endif
DMLC_REGISTRY_LINK_TAG(multiclass_obj);
DMLC_REGISTRY_LINK_TAG(rank_obj);
DMLC_REGISTRY_LINK_TAG(hinge);
}  // namespace obj
}  // namespace xgboost
