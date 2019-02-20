/*!
 * Copyright by Contributors 2017
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#if defined(XGBOOST_INSTRUMENT_CUDA) && defined(__CUDACC__)
#include <nvToolsExt.h>
#endif

namespace xgboost {
namespace common {
struct Timer {
  using ClockT = std::chrono::high_resolution_clock;
  using TimePointT = std::chrono::high_resolution_clock::time_point;
  using DurationT = std::chrono::high_resolution_clock::duration;
  using SecondsT = std::chrono::duration<double>;

  TimePointT start;
  DurationT elapsed;
  Timer() { Reset(); }
  void Reset() {
    elapsed = DurationT::zero();
    Start();
  }
  void Start() { start = ClockT::now(); }
  void Stop() { elapsed += ClockT::now() - start; }
  double ElapsedSeconds() const { return SecondsT(elapsed).count(); }
  void PrintElapsed(std::string label) {
    char buffer[255];
    snprintf(buffer, sizeof(buffer), "%s:\t %fs", label.c_str(),
             SecondsT(elapsed).count());
    LOG(CONSOLE) << buffer;
    Reset();
  }
};

/**
 * \struct  Monitor
 *
 * \brief Timing utility used to measure total method execution time over the
 * lifetime of the containing object.
 */

struct Monitor {
 private:
  struct Statistics {
    Timer timer;
    size_t count{0};
    uint64_t nvtx_id;
  };
  std::string label = "";
  std::map<std::string, Statistics> statistics_map;
  Timer self_timer;

 public:
  Monitor() { self_timer.Start(); }

  ~Monitor() {
    if (!ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) return;

    LOG(CONSOLE) << "======== Monitor: " << label << " ========";
    for (auto &kv : statistics_map) {
      if (kv.second.count == 0) {
        LOG(WARNING) <<
            "Timer for " << kv.first << " did not get stopped properly.";
        continue;
      }
      LOG(CONSOLE) << kv.first << ": " << kv.second.timer.ElapsedSeconds()
                   << "s, " << kv.second.count << " calls @ "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          kv.second.timer.elapsed / kv.second.count)
                          .count()
                   << "us";
    }
    self_timer.Stop();
  }
  void Init(std::string label) { this->label = label; }
  void Start(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      statistics_map[name].timer.Start();
    }
  }
  void Stop(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Stop();
      stats.count++;
    }
  }
  void StartCuda(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Start();
#if defined(XGBOOST_INSTRUMENT_CUDA) && defined(__CUDACC__)
      stats.nvtx_id = nvtxRangeStartA(name.c_str());
#endif
    }
  }
  void StopCuda(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Stop();
      stats.count++;
#if defined(XGBOOST_INSTRUMENT_CUDA) && defined(__CUDACC__)
      nvtxRangeEnd(stats.nvtx_id);
#endif
    }
  }
};
}  // namespace common
}  // namespace xgboost
