/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_VALIDATION_H_
#define XGBOOST_DATA_VALIDATION_H_
#include <cmath>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/data.h"                // for FeatureType
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/logging.h"

namespace xgboost::data {
struct LabelsCheck {
  XGBOOST_DEVICE bool operator()(float y) {
#if defined(__CUDA_ARCH__)
    return ::isnan(y) || ::isinf(y);
#else
    return std::isnan(y) || std::isinf(y);
#endif
  }
};

struct WeightsCheck {
  XGBOOST_DEVICE bool operator()(float w) { return LabelsCheck{}(w) || w < 0; }  // NOLINT
};

inline void ValidateQueryGroup(std::vector<bst_group_t> const& group_ptr_) {
  bool valid_query_group = true;
  for (size_t i = 1; i < group_ptr_.size(); ++i) {
    valid_query_group = valid_query_group && group_ptr_[i] >= group_ptr_[i - 1];
    if (XGBOOST_EXPECT(!valid_query_group, false)) {
      break;
    }
  }
  CHECK(valid_query_group) << "Invalid group structure.";
}
namespace cuda_impl {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const& lhs,
                       HostDeviceVector<FeatureType> const& rhs);
}

void CheckFeatureTypes(HostDeviceVector<FeatureType> const& lhs,
                       HostDeviceVector<FeatureType> const& rhs);
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_VALIDATION_H_
