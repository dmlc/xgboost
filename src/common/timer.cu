/*!
 * Copyright by Contributors 2019
 */
#if defined(XGBOOST_USE_NVTX)
#include <nvToolsExt.h>
#endif  // defined(XGBOOST_USE_NVTX)

#include <string>

#include "xgboost/logging.h"
#include "device_helpers.cuh"
#include "timer.h"

namespace xgboost {
namespace common {

void Monitor::StartCuda(const std::string& name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map[name];
    stats.timer.Start();
#if defined(XGBOOST_USE_NVTX)
    stats.nvtx_id = nvtxRangeStartA(name.c_str());
#endif  // defined(XGBOOST_USE_NVTX)
  }
}

void Monitor::StopCuda(const std::string& name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map[name];
    stats.timer.Stop();
    stats.count++;
#if defined(XGBOOST_USE_NVTX)
    nvtxRangeEnd(stats.nvtx_id);
#endif  // defined(XGBOOST_USE_NVTX)
  }
}
}  // namespace common
}  // namespace xgboost
