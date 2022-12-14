/*!
 * Copyright 2018
 */
#include <xgboost/linear_updater.h>
#include <dmlc/registry.h>
#include "./param.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::LinearUpdaterReg);
}  // namespace dmlc

namespace xgboost {

LinearUpdater* LinearUpdater::Create(const std::string& name, Context const* ctx) {
  auto *e = ::dmlc::Registry< ::xgboost::LinearUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown linear updater " << name;
  }
  auto p_linear = (e->body)();
  p_linear->ctx_ = ctx;
  return p_linear;
}

}  // namespace xgboost

namespace xgboost {
namespace linear {
DMLC_REGISTER_PARAMETER(LinearTrainParam);

// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_shotgun);
DMLC_REGISTRY_LINK_TAG(updater_coordinate);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_coordinate);
#endif  // XGBOOST_USE_CUDA
}  // namespace linear
}  // namespace xgboost
