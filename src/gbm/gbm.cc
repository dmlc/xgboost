/*!
 * Copyright 2015-2020 by Contributors
 * \file gbm.cc
 * \brief Registry of gradient boosters.
 */
#include <dmlc/registry.h>
#include <string>
#include <vector>
#include <memory>

#include "xgboost/gbm.h"
#include "xgboost/learner.h"
#include "xgboost/generic_parameters.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::GradientBoosterReg);
}  // namespace dmlc

namespace xgboost {
GradientBooster* GradientBooster::Create(
    const std::string& name,
    GenericParameter const* generic_param,
    LearnerModelParam const* learner_model_param) {
  auto *e = ::dmlc::Registry< ::xgboost::GradientBoosterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown gbm type " << name;
  }
  auto p_bst =  (e->body)(learner_model_param);
  p_bst->generic_param_ = generic_param;
  return p_bst;
}
}  // namespace xgboost

namespace xgboost {
namespace gbm {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(gblinear);
DMLC_REGISTRY_LINK_TAG(gbtree);
}  // namespace gbm
}  // namespace xgboost
