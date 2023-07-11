/**
 * Copyright 2023 by XGBoost contributors
 */
#include "error_msg.h"

#include "xgboost/logging.h"

namespace xgboost::error {
void WarnDeprecatedGPUHist() {
  bool static thread_local logged{false};
  if (logged) {
    return;
  }
  auto msg =
      "The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` "
      R"(parameter to CUDA instead.

    E.g. tree_method = "hist", device = "CUDA"

)";
  LOG(WARNING) << msg;
  logged = true;
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
}  // namespace xgboost::error
