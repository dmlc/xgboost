/*!
 * Copyright 2015-2022 by XGBoost Contributors
 * \file gbm.cc
 * \brief Registry of gradient boosters.
 */
#include "xgboost/gbm.h"

#include <dmlc/registry.h>

#include <memory>
#include <string>
#include <vector>

#include "xgboost/context.h"
#include "xgboost/learner.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::GradientBoosterReg);
}  // namespace dmlc

namespace xgboost {
GradientBooster* GradientBooster::Create(const std::string& name, Context const* ctx,
                                         LearnerModelParam const* learner_model_param) {
  auto *e = ::dmlc::Registry< ::xgboost::GradientBoosterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown gbm type " << name;
  }
  auto p_bst =  (e->body)(learner_model_param, ctx);
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
