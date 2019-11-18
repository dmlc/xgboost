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
#include "json_experimental.h"
#include "json_reader_experimental.h"
#include "json_writer_experimental.h"

namespace xgboost {
namespace common {

void Monitor::Start(std::string const &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    statistics_map[name].timer.Start();
  }
}

void Monitor::Stop(const std::string &name) {
  if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
    auto &stats = statistics_map[name];
    stats.timer.Stop();
    stats.count++;
  }
}

std::vector<Monitor::StatMap> Monitor::CollectFromOtherRanks() const {
  // Since other nodes might have started timers that this one haven't, so
  // we can't simply call all reduce.
  size_t const world_size = rabit::GetWorldSize();
  size_t const rank = rabit::GetRank();

  // It's much easier to work with rabit if we have a string serialization.  So we go with
  // json.
  experimental::Document j_statistic;
  j_statistic.GetValue().CreateMember("rank") = static_cast<int64_t>(rank);
  auto statistic = j_statistic.CreateMember("statistic");
  statistic.SetObject();

  for (auto const& kv : statistics_map) {
    // statistic[kv.first] = Object();
    auto prop = statistic.CreateMember(kv.first);
    prop.SetObject();
    prop.CreateMember("count").SetInteger(kv.second.count);
    prop.CreateMember("elapsed").SetInteger(
        std::chrono::duration_cast<std::chrono::microseconds>(
            kv.second.timer.elapsed).count());
  }

  std::string str = j_statistic.Dump<experimental::JsonWriter>();

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
    auto j_other =
        experimental::Document::Load<experimental::JsonRecursiveReader>(
            experimental::StringRef{buffer});
    auto& other = world[i];

    auto const& j_statistic = *j_other.GetValue().FindMemberByKey("statistic");

    for (auto it = j_statistic.cbegin(); it != j_statistic.cend(); ++it) {
      auto const& timer_name = it.Key();
      auto pair = *it;
      other[timer_name.Copy()] = {(*pair.FindMemberByKey("count")).GetInt(),
                                  (*pair.FindMemberByKey("elapsed")).GetInt()};
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
    std::cout << kv.first << ": " << static_cast<double>(kv.second.second) / 1e+6
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
      LOG(CONSOLE) << "======== Monitor: " << label << " ========";
      for (size_t i = 0; i < world.size(); ++i) {
        std::cout << "From rank: " << i << ": " << std::endl;
        auto const& statistic = world[i];
        this->PrintStatistics(statistic);
        std::cout << std::endl;
      }
    }
  } else {
    StatMap stat_map;
    for (auto const& kv : statistics_map) {
      stat_map[kv.first] = std::make_pair(
          kv.second.count, std::chrono::duration_cast<std::chrono::microseconds>(
              kv.second.timer.elapsed).count());
    }
    LOG(CONSOLE) << "======== Monitor: " << label << " ========";
    this->PrintStatistics(stat_map);
  }
  std::cout << std::endl;
}

}  // namespace common
}  // namespace xgboost
