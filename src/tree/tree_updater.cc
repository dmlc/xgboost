/**
 * Copyright 2015-2023 by XGBoost Contributors
 * \file tree_updater.cc
 * \brief Registry of tree updaters.
 */
#include "xgboost/tree_updater.h"

#include <dmlc/registry.h>

#include <string>  // for string

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeUpdaterReg);
}  // namespace dmlc

namespace xgboost {
TreeUpdater* TreeUpdater::Create(const std::string& name, Context const* ctx, ObjInfo const* task) {
  auto* e = ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown tree updater " << name;
  }
  auto p_updater = (e->body)(ctx, task);
  return p_updater;
}
}  // namespace xgboost

namespace xgboost::tree {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_colmaker);
DMLC_REGISTRY_LINK_TAG(updater_refresh);
DMLC_REGISTRY_LINK_TAG(updater_prune);
DMLC_REGISTRY_LINK_TAG(updater_quantile_hist);
DMLC_REGISTRY_LINK_TAG(updater_approx);
DMLC_REGISTRY_LINK_TAG(updater_sync);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_hist);
#endif  // XGBOOST_USE_CUDA
}  // namespace xgboost::tree
