/*!
 * Copyright by Contributors 2017
 */
#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace xgboost {
namespace common {
struct Timer {
  typedef std::chrono::high_resolution_clock ClockT;
  typedef std::chrono::high_resolution_clock::time_point TimePointT;
  typedef std::chrono::high_resolution_clock::duration DurationT;
  typedef std::chrono::duration<double> SecondsT;

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
    printf("%s:\t %fs\n", label.c_str(), SecondsT(elapsed).count());
    fflush(stdout);
    Reset();
  }
  void PrintElapsed2(std::string label) {
    printf("%s:\t %fs\n", label.c_str(), SecondsT(elapsed).count());
    fflush(stdout);
  }
};

/**
 * \struct  Monitor
 *
 * \brief Timing utility used to measure total method execution time over the
 * lifetime of the containing object.
 */

struct Monitor {
  bool debug_verbose = false;
  std::string label = "";
  std::map<std::string, Timer> timer_map;
  Timer self_timer;

  Monitor() { self_timer.Start(); }

  void Output(void) {
    if (!debug_verbose) return;

    std::cout << "========\n";
    std::cout << "Monitor: " << label << "\n";
    std::cout << "========\n";
    for (auto &kv : timer_map) {
      kv.second.PrintElapsed2(kv.first);
    }
    self_timer.Stop();
    self_timer.PrintElapsed2(label + " Lifetime");
    self_timer.Start();
  }

  ~Monitor() {
    if (!debug_verbose) return;

    std::cout << "========\n";
    std::cout << "Monitor: " << label << "\n";
    std::cout << "========\n";
    for (auto &kv : timer_map) {
      kv.second.PrintElapsed(kv.first);
    }
    self_timer.Stop();
    self_timer.PrintElapsed(label + " Lifetime");
    self_timer.Start();
    fflush(stdout);
  }
  void Init(std::string label, bool debug_verbose) {
    this->debug_verbose = debug_verbose;
    this->label = label;
  }
  void Start(const std::string &name) { timer_map[name].Start(); }
  void Start(const std::string &name, std::vector<int> dList) {
    if (debug_verbose) {
#ifdef __CUDACC__
#include "device_helpers.cuh"
      dh::synchronize_n_devices(dList.size(), dList);
#endif
    }
    timer_map[name].Start();
  }
  void Stop(const std::string &name) { timer_map[name].Stop(); }
  void Stop(const std::string &name, std::vector<int> dList) {
    if (debug_verbose) {
#ifdef __CUDACC__
#include "device_helpers.cuh"
      dh::synchronize_n_devices(dList.size(), dList);
#endif
    }
    timer_map[name].Stop();
  }
};
}  // namespace common
}  // namespace xgboost
