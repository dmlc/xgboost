/*!
 * Copyright 2015 by Contributors
 * \file gbm.cc
 * \brief Registry of gradient boosters.
 */
#include <xgboost/gbm.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::GradientBoosterReg);
}  // namespace dmlc

namespace xgboost {
GradientBooster* GradientBooster::Create(
    const std::string& name,
    const std::vector<std::shared_ptr<DMatrix> >& cache_mats,
    bst_float base_margin) {
  auto *e = ::dmlc::Registry< ::xgboost::GradientBoosterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown gbm type " << name;
  }
  return (e->body)(cache_mats, base_margin);
}

}  // namespace xgboost

namespace xgboost {
namespace gbm {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(gblinear);
DMLC_REGISTRY_LINK_TAG(gbtree);
}  // namespace gbm
}  // namespace xgboost
