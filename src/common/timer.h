/*!
 * Copyright by Contributors 2017
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "common.h"

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
  struct Statistics {
    Timer timer;
    size_t count{0};
  };
  bool debug_verbose = false;
  std::string label = "";
  std::map<std::string, Statistics> statistics_map;
  Timer self_timer;

  Monitor() { self_timer.Start(); }

  ~Monitor() {
    if (!debug_verbose) return;

    LOG(CONSOLE) << "======== Monitor: " << label << " ========";
    for (auto &kv : statistics_map) {
      LOG(CONSOLE) << kv.first << ": " << kv.second.timer.ElapsedSeconds()
                   << "s, " << kv.second.count << " calls @ "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          kv.second.timer.elapsed / kv.second.count)
                          .count()
                   << "us";
    }
    self_timer.Stop();
  }
  void Init(std::string label, bool debug_verbose) {
    this->debug_verbose = debug_verbose;
    this->label = label;
  }
  void Start(const std::string &name) { statistics_map[name].timer.Start(); }
  void Start(const std::string &name, GPUSet devices) {
    if (debug_verbose) {
#ifdef __CUDACC__
      for (auto device : devices) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
      }
#endif
    }
    statistics_map[name].timer.Start();
  }
  void Stop(const std::string &name) {
    statistics_map[name].timer.Stop();
    statistics_map[name].count++;
  }
  void Stop(const std::string &name, GPUSet devices) {
    if (debug_verbose) {
#ifdef __CUDACC__
      for (auto device : devices) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
      }
#endif
    }
    this->Stop(name);
  }
};
}  // namespace common
}  // namespace xgboost
