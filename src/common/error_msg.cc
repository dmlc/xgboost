/**
 * Copyright 2023-2025, XGBoost contributors
 */
#include "error_msg.h"

#include <mutex>         // for call_once, once_flag
#include <sstream>       // for stringstream
#include <system_error>  // for error_code, system_category

#include "../collective/communicator-inl.h"  // for GetRank
#include "xgboost/collective/socket.h"       // for LastError
#include "xgboost/context.h"                 // for Context
#include "xgboost/logging.h"

namespace xgboost::error {
[[nodiscard]] std::string DeprecatedFunc(StringView old, StringView since, StringView replacement) {
  std::stringstream ss;
  ss << "`" << old << "` is deprecated since" << since << ", use `" << replacement << "` instead.";
  return ss.str();
}

[[nodiscard]] std::string InvalidModel(StringView fname) {
  std::stringstream ss;
  ss << "Invalid model format in: `" << fname << "`.";
  return ss.str();
}

[[nodiscard]] std::string OldBinaryModel(StringView fname) {
  std::stringstream ss;
  ss << "Failed to load model: `" << fname << "`. ";
  ss << R"doc(
The binary format has been deprecated in 1.6 and removed in 3.1, use UBJ or JSON
instead. You can port the binary model to UBJ and JSON by re-saving it with XGBoost
3.0. See:

    https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

for more info.
)doc";
  return ss.str();
}

void WarnManualUpdater() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    LOG(WARNING)
        << "You have manually specified the `updater` parameter. The `tree_method` parameter "
           "will be ignored. Incorrect sequence of updaters will produce undefined "
           "behavior. For common uses, we recommend using `tree_method` parameter instead.";
  });
}

void WarnEmptyDataset() {
  static std::once_flag flag;
  std::call_once(flag,
                 [] { LOG(WARNING) << "Empty dataset at worker: " << collective::GetRank(); });
}

void MismatchedDevices(Context const* booster, Context const* data) {
  static std::once_flag flag;
  std::call_once(flag, [&] {
    LOG(WARNING)
        << "Falling back to prediction using DMatrix due to mismatched devices. This might "
           "lead to higher memory usage and slower performance. XGBoost is running on: "
        << booster->DeviceName() << ", while the input data is on: " << data->DeviceName() << ".\n"
        << R"(Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.
)";
  });
}

void CheckOldNccl(std::int32_t major, std::int32_t minor, std::int32_t patch) {
  auto msg = [&] {
    std::stringstream ss;
    ss << "NCCL version too old: " << "(" << major << "." << minor << "." << patch << ")"
       << ". Install NCCL >= 2.23.4 .";
    return ss.str();
  };

  // Minimum required version.
  CHECK_GE(major, 2) << msg();
  CHECK_GE(minor, 21) << msg();

  // With 2.23.4+, we can abort the NCCL communicator after timeout.
  if (minor < 23) {
    LOG(WARNING) << msg();
  }
}

[[nodiscard]] std::error_code SystemError() {
  std::int32_t errsv = system::LastError();
  auto err = std::error_code{errsv, std::system_category()};
  return err;
}

void InvalidIntercept(std::int32_t n_classes, bst_target_t n_targets, std::size_t intercept_len) {
  std::stringstream ss;
  ss << "Invalid `base_score`, it should match the number of outputs for multi-class/target "
     << "models. `base_score` len: " << intercept_len;
  if (n_classes > 1) {
    ss << ", `n_classes`: " << n_classes;
  }
  if (n_targets > 1) {
    ss << ", `n_targets`: " << n_targets;
  }
  LOG(FATAL) << ss.str();
}
}  // namespace xgboost::error
