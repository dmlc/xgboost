/*!
 * Copyright 2015 by Contributors
 * \file tree_updater.cc
 * \brief Registry of tree updaters.
 */
#include <xgboost/tree_updater.h>
#include <dmlc/registry.h>

#include "../common/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeUpdaterReg);
}  // namespace dmlc

namespace xgboost {

TreeUpdater* TreeUpdater::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown tree updater " << name;
  }
  return (e->body)();
}

}  // namespace xgboost

namespace xgboost {
namespace tree {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_colmaker);
DMLC_REGISTRY_LINK_TAG(updater_skmaker);
DMLC_REGISTRY_LINK_TAG(updater_refresh);
DMLC_REGISTRY_LINK_TAG(updater_prune);
DMLC_REGISTRY_LINK_TAG(updater_fast_hist);
DMLC_REGISTRY_LINK_TAG(updater_histmaker);
DMLC_REGISTRY_LINK_TAG(updater_sync);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu);
DMLC_REGISTRY_LINK_TAG(updater_gpu_hist);
#endif
}  // namespace tree
}  // namespace xgboost
