/*!
 * Copyright by Contributors 2017-2019
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

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

  // from left to right, <name <count, elapsed>>
  using StatMap = std::map<std::string, std::pair<size_t, size_t>>;

  std::string label_ = "";
  std::map<std::string, Statistics> statistics_map_;
  Timer self_timer_;

  /*! \brief Collect time statistics across all workers. */
  std::vector<StatMap> CollectFromOtherRanks() const;
  void PrintStatistics(StatMap const& statistics) const;

 public:
  Monitor() { self_timer_.Start(); }
  /*\brief Print statistics info during destruction.
   *
   * Please note that this may not work, as with distributed frameworks like Dask, the
   * model is pickled to other workers, and the global parameters like `global_verbosity_`
   * are not included in the pickle.
   */
  ~Monitor() {
    this->Print();
    self_timer_.Stop();
  }

  /*! \brief Print all the statistics. */
  void Print() const;

  void Init(std::string label) { this->label_ = label; }
  void Start(const std::string &name);
  void Stop(const std::string &name);
};
}  // namespace common
}  // namespace xgboost
