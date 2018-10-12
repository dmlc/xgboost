/*!
 * Copyright 2017-2018 by Contributors
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

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

/*!
 * \brief Singleton class storing device memory usage statistic.
 */
class DeviceMemoryStat {
  struct Usage {
    std::set<std::string> traces_;
    size_t running_;  // running total.
    size_t peak_;
    size_t count_;  // allocation count.

   public:
    Usage() : running_{0}, peak_{0}, count_{0} {}
    Usage(std::set<std::string> traces,
          size_t running, size_t peak, size_t count);

    Usage& operator+=(Usage const& other);
    Usage& operator-=(Usage const& other);

    size_t GetPeak() const { return peak_; }
    size_t GetAllocCount() const { return count_; }
    size_t GetRunningSum() const { return running_; }
    std::set<std::string> GetTraces() const { return traces_; }
  };

  std::mutex mutex_;
  bool profiling_;

  Usage global_usage_;
  std::map<void const*, Usage> usage_map_;

#if !defined(_MSC_VER)
  std::vector<std::string> const units_;
#else
  std::vector<std::string> units_;
#endif

  DeviceMemoryStat() :
      mutex_(), profiling_{false}, global_usage_()  // NOLINT
#if !defined(_MSC_VER)
      , units_{"B", "KB", "MB", "GB"} {}
#else
  // MSVC 2013: list initialization inside member initializer list or
  // non-static data member initializer is not implemented
  {
    units_.push_back("B");
    units_.push_back("KB");
    units_.push_back("MB");
    units_.push_back("GB");
  }
#endif

 public:
  ~DeviceMemoryStat() { PrintSummary(); }

  /*! \brief Get an instance. */
  static DeviceMemoryStat& Ins() {
    static DeviceMemoryStat instance;
    return instance;
  }

  void SetProfiling(bool profiling) { profiling_ = profiling; }

  void Allocate(void* ptr, size_t size);
  void Deallocate(void* ptr, size_t size);
  void Deallocate(void* ptr);
  /*! \brief replace the usage stat of lhs with the one from rhs. */
  void Replace(void const* lhs, void const* rhs);
  void Reset();

  void PrintSummary() const;

  size_t GetPeakUsage() const;
  size_t GetAllocationCount() const;
  Usage const& GetPtrUsage(void const* ptr) const;
};

/**
 * \struct  Monitor
 *
 * \brief Timing utility used to measure total method execution time over the
 * lifetime of the containing object.
 */
class Monitor {
  bool debug_verbose_ = false;
  std::string label_ = "";
  std::map<std::string, Timer> timer_map_;
  Timer self_timer_;

  void PrintElapsed();

 public:
  Monitor() { self_timer_.Start(); }
  ~Monitor() { PrintElapsed(); }

  void Init(std::string label, bool debug_verbose) {
    this->debug_verbose_ = debug_verbose;
    this->label_ = label;
  }
  void Start(const std::string &name) { timer_map_[name].Start(); }
  void Start(const std::string &name, GPUSet devices);
  void Stop(const std::string &name) { timer_map_[name].Stop(); }
  void Stop(const std::string &name, GPUSet devices);
};
}  // namespace common
}  // namespace xgboost
