/*!
 * Copyright 2015 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <xgboost/objective.h>
#include <dmlc/registry.h>

#include "../common/dhvec.h"

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

void ObjFunction::GetGradient(dhvec<bst_float>* preds,
                              const MetaInfo& info,
                              int iteration,
                              dhvec<bst_gpair>* out_gpair) {
  GetGradient(preds->data_h(), info, iteration, &out_gpair->data_h());
}

void ObjFunction::PredTransform(dhvec<bst_float> *io_preds) {
  PredTransform(&io_preds->data_h());
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
}  // namespace obj
}  // namespace xgboost
