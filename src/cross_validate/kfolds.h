#pragma once
#include <cstddef>  // for size_t

#include "xgboost/data.h"  // for MetaInfo
#include "xgboost/host_device_vector.h"

namespace xgboost::cv {
// k-fold split based on labels.
void KFold(std::size_t k_folds, MetaInfo const& info, std::int32_t k,
           HostDeviceVector<bst_idx_t>* out);
}  // namespace xgboost::cv
