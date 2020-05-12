/*!
 * Copyright by Contributors 2019
 */
#include <rabit/rabit.h>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>
#include <sstream>
#include "timer.h"
#include "xgboost/json.h"

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
    std::string nvtx_name = label_ + "::" + name;
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

std::vector<Monitor::StatMap> Monitor::CollectFromOtherRanks() const {
  // Since other nodes might have started timers that this one haven't, so
  // we can't simply call all reduce.
  size_t const world_size = rabit::GetWorldSize();
  size_t const rank = rabit::GetRank();

  // It's much easier to work with rabit if we have a string serialization.  So we go with
  // json.
  Json j_statistic { Object() };
  j_statistic["rank"] = Integer(rank);
  j_statistic["statistic"] = Object();

  auto& statistic = j_statistic["statistic"];
  for (auto const& kv : statistics_map_) {
    statistic[kv.first] = Object();
    auto& j_pair = statistic[kv.first];
    j_pair["count"] = Integer(kv.second.count);
    j_pair["elapsed"] = Integer(static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
        kv.second.timer.elapsed).count()));
  }

  std::stringstream ss;
  Json::Dump(j_statistic, &ss);
  std::string const str { ss.str() };

  size_t str_size = str.size();
  rabit::Allreduce<rabit::op::Max>(&str_size, 1);
  std::string buffer;
  buffer.resize(str_size);

  // vector storing stat from all workers
  std::vector<StatMap> world(world_size);

  // Actually only rank 0 is printing.
  for (size_t i = 0; i < world_size; ++i) {
    std::copy(str.cbegin(), str.cend(), buffer.begin());
    rabit::Broadcast(&buffer, i);
    auto j_other = Json::Load(StringView{buffer.c_str(), buffer.size()});
    auto& other = world[i];

    auto const& j_statistic = get<Object>(j_other["statistic"]);

    for (auto const& kv : j_statistic) {
      std::string const& timer_name = kv.first;
      auto const& pair = kv.second;
      other[timer_name] = {get<Integer>(pair["count"]), get<Integer>(pair["elapsed"])};
    }

    // FIXME(trivialfis): How to ask rabit to block here?
  }

  return world;
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

  bool is_distributed = rabit::IsDistributed();

  if (is_distributed) {
    auto world = this->CollectFromOtherRanks();
    // rank zero is in charge of printing
    if (rabit::GetRank() == 0) {
      LOG(CONSOLE) << "======== Monitor: " << label_ << " ========";
      for (size_t i = 0; i < world.size(); ++i) {
        LOG(CONSOLE) << "From rank: " << i << ": " << std::endl;
        auto const& statistic = world[i];
        this->PrintStatistics(statistic);
      }
    }
  } else {
    StatMap stat_map;
    for (auto const& kv : statistics_map_) {
      stat_map[kv.first] = std::make_pair(
          kv.second.count, std::chrono::duration_cast<std::chrono::microseconds>(
              kv.second.timer.elapsed).count());
    }
    LOG(CONSOLE) << "======== Monitor: " << label_ << " ========";
    this->PrintStatistics(stat_map);
  }
}

}  // namespace common
}  // namespace xgboost
