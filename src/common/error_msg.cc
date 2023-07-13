/**
 * Copyright 2023 by XGBoost contributors
 */
#include "error_msg.h"

#include "../collective/communicator-inl.h"  // for GetRank
#include "xgboost/logging.h"

namespace xgboost::error {
void WarnDeprecatedGPUHist() {
  auto msg =
      "The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` "
      R"(parameter to CUDA instead.

    E.g. tree_method = "hist", device = "CUDA"
)";
  LOG(WARNING) << msg;
}

void WarnManualUpdater() {
  bool static thread_local logged{false};
  if (logged) {
    return;
  }
  LOG(WARNING)
      << "You have manually specified the `updater` parameter. The `tree_method` parameter "
         "will be ignored. Incorrect sequence of updaters will produce undefined "
         "behavior. For common uses, we recommend using `tree_method` parameter instead.";
  logged = true;
}

void WarnDeprecatedGPUId() {
  static thread_local bool logged{false};
  if (logged) {
    return;
  }
  LOG(WARNING) << "`gpu_id` is deprecated in favor of the new `device` parameter: "
               << "device = cpu/cuda/cuda:0";
  logged = true;
}

void WarnEmptyDataset() {
  static thread_local bool logged{false};
  if (logged) {
    return;
  }
  LOG(WARNING) << "Empty dataset at worker: " << collective::GetRank();
  logged = true;
}
}  // namespace xgboost::error
