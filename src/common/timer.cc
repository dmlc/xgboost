/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include "timer.h"

#include <utility>

#include "../collective/communicator-inl.h"
#include "cuda_rt_utils.h"

#if defined(XGBOOST_USE_NVTX)
#include <nvtx3/nvtx3.hpp>
#endif  // defined(XGBOOST_USE_NVTX)

namespace xgboost::common {
void Monitor::Start(std::string const &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map_[name];
    stats.timer.Start();
#if defined(XGBOOST_USE_NVTX)
    auto range_handle = nvtx3::start_range_in<curt::NvtxDomain>(label_ + "::" + name);
    stats.nvtx_id = range_handle.get_value();
#endif  // defined(XGBOOST_USE_NVTX)
  }
}

void Monitor::Stop(const std::string &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map_[name];
    stats.timer.Stop();
    stats.count++;
#if defined(XGBOOST_USE_NVTX)
    nvtx3::end_range_in<curt::NvtxDomain>(nvtx3::range_handle{stats.nvtx_id});
#endif  // defined(XGBOOST_USE_NVTX)
  }
}

void Monitor::PrintStatistics(StatMap const &statistics) const {
  for (auto &kv : statistics) {
    if (kv.second.first == 0) {
      LOG(WARNING) << "Timer for " << kv.first << " did not get stopped properly.";
      continue;
    }
    LOG(CONSOLE) << kv.first << ": " << static_cast<double>(kv.second.second) / 1e+6 << "s, "
                 << kv.second.first << " calls @ " << kv.second.second << "us" << std::endl;
  }
}

void Monitor::Print() const {
  if (!ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    return;
  }
  auto rank = collective::GetRank();
  StatMap stat_map;
  for (auto const &kv : statistics_map_) {
    stat_map[kv.first] = std::make_pair(
        kv.second.count,
        std::chrono::duration_cast<std::chrono::microseconds>(kv.second.timer.elapsed).count());
  }
  if (stat_map.empty()) {
    return;
  }
  LOG(CONSOLE) << "======== Monitor (" << rank << "): " << label_ << " ========";
  this->PrintStatistics(stat_map);
}
}  // namespace xgboost::common
