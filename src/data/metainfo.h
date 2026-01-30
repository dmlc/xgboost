/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#pragma once

#include <cmath>    // for isnan, isinf
#include <cstdint>  // for int8_t
#include <vector>   // for vector

#include "xgboost/base.h"                // for bst_group_t
#include "xgboost/data.h"                // for FeatureType
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/logging.h"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::data {
enum class MetaField : std::int8_t {
  kLabel = 0,
  kWeight = 1,
  kBaseMargin = 2,
  kLabelLowerBound = 3,
  kLabelUpperBound = 4,
  kFeatureWeights = 5,
  kGroupPtr = 6,
  kQid = 7,  // Converted into group ptr
};

// `group_ptr` is for the output, while input is `group`.
inline MetaField MapMetaField(StringView key, bool is_input) {
  if (key == "label") {
    return MetaField::kLabel;
  } else if (key == "weight") {
    return MetaField::kWeight;
  } else if (key == "base_margin") {
    return MetaField::kBaseMargin;
  } else if (key == "label_lower_bound") {
    return MetaField::kLabelLowerBound;
  } else if (key == "label_upper_bound") {
    return MetaField::kLabelUpperBound;
  } else if (key == "feature_weights") {
    return MetaField::kFeatureWeights;
  } else if (key == "group_ptr" && !is_input) {
    return MetaField::kGroupPtr;
  } else if (key == "group" && is_input) {
    return MetaField::kGroupPtr;
  } else if (key == "qid") {
    return MetaField::kQid;
  } else {
    LOG(FATAL) << "Unknown key:" << key;
  }
  return {};
}

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

// TODO(jiamingy): We have two sets of dtypes in XGBoost, one in `data.h`, another one in array
// interface. We should unify them.
template <typename Fn>
decltype(auto) DispatchDType(DataType dtype, Fn&& fn) {
  switch (dtype) {
    case xgboost::DataType::kFloat32: {
      return fn(float{});
    }
    case xgboost::DataType::kDouble: {
      return fn(double{});
    }
    case xgboost::DataType::kUInt32: {
      return fn(std::uint32_t{});
    }
    case xgboost::DataType::kUInt64: {
      return fn(std::uint64_t{});
    }
    default:
      LOG(FATAL) << "Unknown data type" << static_cast<uint8_t>(dtype);
  }
  LOG(FATAL) << "Unreachable";
  return fn(float{});
}
}  // namespace xgboost::data
