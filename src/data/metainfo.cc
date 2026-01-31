/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#include "metainfo.h"

#include <string>       // for string
#include <type_traits>  // for add_pointer_t

#include "../common/error_msg.h"         // for InconsistentFeatureTypes
#include "xgboost/data.h"                // for FeatureType
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

#if !defined(XGBOOST_USE_CUDA)

#include "../common/common.h"  // for AssertGPUSupport

#endif  // !defined(XGBOOST_USE_CUDA)

namespace xgboost {
std::string TypedArrayRef::ArrayInterfaceStr() const {
  return data::DispatchDType(this->dtype, [this](auto dtype) {
    using DType = decltype(dtype);
    auto ptr = static_cast<std::add_pointer_t<std::add_const_t<DType>>>(this->data);
    if (this->ndim == 1) {
      auto vec = linalg::MakeVec(ptr, this->shape.front());
      return linalg::ArrayInterfaceStr(vec);
    } else {
      auto n = this->Size();
      if (ptr) {
        CHECK_GT(n, 0);
      }
      if (n > 0) {
        CHECK(ptr);
      }
      auto mat = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{ptr, n}, this->shape[0],
                                        this->shape[1]);
      return linalg::ArrayInterfaceStr(mat);
    }
  });
}
}  // namespace xgboost

namespace xgboost::data {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const& lhs,
                       HostDeviceVector<FeatureType> const& rhs) {
  CHECK_EQ(lhs.Size(), rhs.Size()) << error::InconsistentFeatureTypes();
  if (lhs.DeviceCanRead() || rhs.DeviceCanRead()) {
    return cuda_impl::CheckFeatureTypes(lhs, rhs);
  }
  auto const& h_lhs = lhs.ConstHostVector();
  auto const& h_rhs = rhs.ConstHostVector();
  auto ft_is_same = std::equal(h_lhs.cbegin(), h_lhs.cend(), h_rhs.cbegin());
  CHECK(ft_is_same) << error::InconsistentFeatureTypes();
}

#if !defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const&, HostDeviceVector<FeatureType> const&) {
  common::AssertGPUSupport();
}
}  // namespace cuda_impl
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
