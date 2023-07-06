/**
 * Copyright 2023 by XGBoost contributors
 *
 * \brief Common error message for various checks.
 */
#ifndef XGBOOST_COMMON_ERROR_MSG_H_
#define XGBOOST_COMMON_ERROR_MSG_H_

#include <cinttypes>  // for uint64_t
#include <limits>     // for numeric_limits

#include "xgboost/base.h"  // for bst_feature_t
#include "xgboost/logging.h"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::error {
constexpr StringView GroupWeight() {
  return "Size of weight must equal to the number of query groups when ranking group is used.";
}

constexpr StringView GroupSize() {
  return "Invalid query group structure. The number of rows obtained from group doesn't equal to ";
}

constexpr StringView LabelScoreSize() {
  return "The size of label doesn't match the size of prediction.";
}

constexpr StringView InfInData() {
  return "Input data contains `inf` or a value too large, while `missing` is not set to `inf`";
}

constexpr StringView NoF128() {
  return "128-bit floating point is not supported on current platform.";
}

constexpr StringView InconsistentMaxBin() {
  return "Inconsistent `max_bin`. `max_bin` should be the same across different QuantileDMatrix, "
         "and consistent with the Booster being trained.";
}

constexpr StringView UnknownDevice() { return "Unknown device type."; }

inline void MaxFeatureSize(std::uint64_t n_features) {
  auto max_n_features = std::numeric_limits<bst_feature_t>::max();
  CHECK_LE(n_features, max_n_features)
      << "Unfortunately, XGBoost does not support data matrices with "
      << std::numeric_limits<bst_feature_t>::max() << " features or greater";
}

constexpr StringView InplacePredictProxy() {
  return "Inplace predict accepts only DMatrixProxy as input.";
}

inline void MaxSampleSize(std::size_t n) {
  LOG(FATAL) << "Sample size too large for the current updater. Maximum number of samples:" << n
             << ". Consider using a different updater or tree_method.";
}

constexpr StringView OldSerialization() {
  return R"doc(If you are loading a serialized model (like pickle in Python, RDS in R) or
configuration generated by an older version of XGBoost, please export the model by calling
`Booster.save_model` from that version first, then load it back in current version. See:

    https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

for more details about differences between saving model and serializing.
)doc";
}

inline void WarnOldSerialization() {
  // Display it once is enough. Otherwise this can be really verbose in distributed
  // environments.
  static thread_local bool logged{false};
  if (logged) {
    return;
  }
  LOG(WARNING) << OldSerialization();
  logged = true;
}

inline void WarnDeprecatedGPUHist() {
  bool static thread_local logged{false};
  if (logged) {
    return;
  }
  auto msg =
      R"(The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA.)";
  LOG(WARNING) << msg;
  logged = true;
}

inline void WarnManualUpdater() {
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
#endif  // XGBOOST_COMMON_ERROR_MSG_H_
