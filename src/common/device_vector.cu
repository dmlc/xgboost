/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include "../collective/communicator-inl.h"  // for GetRank
#include "device_helpers.cuh"                // for CurrentDevice
#include "device_vector.cuh"

namespace dh {
namespace detail {
void ThrowOOMError(std::string const &err, size_t bytes) {
  auto device = CurrentDevice();
  auto rank = xgboost::collective::GetRank();
  std::stringstream ss;
  ss << "Memory allocation error on worker " << rank << ": " << err << "\n"
     << "- Free memory: " << dh::AvailableMemory(device) << "\n"
     << "- Requested memory: " << bytes << std::endl;
  LOG(FATAL) << ss.str();
}
}  // namespace detail

#if defined(XGBOOST_USE_RMM)
LoggingResource *GlobalLoggingResource() {
  static auto mr{std::make_unique<LoggingResource>()};
  return mr.get();
}
#endif  // defined(XGBOOST_USE_RMM)
}  // namespace dh
