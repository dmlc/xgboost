/*!
 * Copyright by Contributors 2019
 */
#include "timer.h"

#include <sstream>
#include <utility>

#include "../collective/communicator-inl.h"

#if defined(XGBOOST_USE_NVTX)
#include <nvToolsExt.h>
#endif  // defined(XGBOOST_USE_NVTX)

namespace xgboost {
namespace common {

void Monitor::Start(std::string const &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map_[name];
    stats.timer.Start();
#if defined(XGBOOST_USE_NVTX)
    std::string nvtx_name = "xgboost::" + label_ + "::" + name;
    stats.nvtx_id = nvtxRangeStartA(nvtx_name.c_str());
#endif  // defined(XGBOOST_USE_NVTX)
  }
}

void Monitor::Stop(const std::string &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map_[name];
    stats.timer.Stop();
    stats.count++;
#if defined(XGBOOST_USE_NVTX)
    nvtxRangeEnd(stats.nvtx_id);
#endif  // defined(XGBOOST_USE_NVTX)
  }
}

void Monitor::PrintStatistics(StatMap const& statistics) const {
  for (auto &kv : statistics) {
    if (kv.second.first == 0) {
      LOG(WARNING) <<
          "Timer for " << kv.first << " did not get stopped properly.";
      continue;
    }
    LOG(CONSOLE) << kv.first << ": " << static_cast<double>(kv.second.second) / 1e+6
                 << "s, " << kv.second.first << " calls @ "
                 << kv.second.second
                 << "us" << std::endl;
  }
}

void Monitor::Print() const {
  if (!ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) { return; }
  auto rank = collective::GetRank();
  StatMap stat_map;
  for (auto const &kv : statistics_map_) {
    stat_map[kv.first] = std::make_pair(
        kv.second.count, std::chrono::duration_cast<std::chrono::microseconds>(
                             kv.second.timer.elapsed)
                             .count());
  }
  LOG(CONSOLE) << "======== Monitor (" << rank << "): " << label_ << " ========";
  this->PrintStatistics(stat_map);
}

}  // namespace common
}  // namespace xgboost
