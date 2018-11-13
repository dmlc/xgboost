/*!
 * Copyright 2018
 */
#include <xgboost/linear_updater.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::LinearUpdaterReg);
}  // namespace dmlc

namespace xgboost {

LinearUpdater* LinearUpdater::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::LinearUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown linear updater " << name;
  }
  return (e->body)();
}

}  // namespace xgboost

namespace xgboost {
namespace linear {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_shotgun);
DMLC_REGISTRY_LINK_TAG(updater_coordinate);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_coordinate);
#endif
}  // namespace linear
}  // namespace xgboost
