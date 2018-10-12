/*!
 * Copyright 2018 by Contributors
 */
#include "monitor.h"

#include <dmlc/logging.h>
#include <dmlc/registry.h>

#include <string>
#include <mutex>

namespace xgboost {
namespace common {

DMLC_REGISTRY_FILE_TAG(monitor);

DeviceMemoryStat::Usage::Usage(std::set<std::string> traces,
                               size_t running, size_t peak, size_t count) :
    traces_(std::move(traces)), running_{running}, peak_{peak}, count_{count} {}

DeviceMemoryStat::Usage& DeviceMemoryStat::Usage::operator+=(
    DeviceMemoryStat::Usage const& other) {
  running_ += other.running_;
  if (running_ > peak_) peak_ = running_;
  count_ += other.count_;

  for (auto& t : other.traces_) {
    traces_.insert(t);
  }
  return *this;
}

DeviceMemoryStat::Usage& DeviceMemoryStat::Usage::operator-=(
    DeviceMemoryStat::Usage const& other) {
  running_ -= other.running_;
  return *this;
}

void DeviceMemoryStat::Allocate(void* ptr, size_t size) {
  if (!profiling_) { return; }
  std::lock_guard<std::mutex> lock(mutex_);

  // No trace for global.
  global_usage_ += Usage(std::set<std::string>{""}, size, size, 1);

  if (usage_map_.find(ptr) == usage_map_.end()) {
    usage_map_[ptr] = Usage();
  }
  std::vector<std::string> functions = dmlc::Split(dmlc::StackTrace(), '\n');
  std::set<std::string> new_traces;
  for (auto& func : functions) {
    auto beg = func.find("(", 9);
    if (beg == -1) continue;
    auto end = func.rfind(")");
    CHECK_NE(end, -1);
    beg += 1;  // the "("

    std::string func_name = func.substr(beg, end - beg);
    new_traces.emplace_hint(new_traces.cend(), func_name);
  }
  usage_map_.at(ptr) += Usage(new_traces, size, size, 1);
}

void DeviceMemoryStat::Deallocate(void* ptr, size_t size) {
  if (!profiling_) { return; }
  std::lock_guard<std::mutex> lock(mutex_);

  auto deallocated = Usage(std::set<std::string>{""}, size, size, 1);
  global_usage_ -= deallocated;
  try {
    usage_map_.at(ptr) -= deallocated;
  } catch(std::out_of_range& e) {
    LOG(FATAL) << e.what() << ", ptr: " << ptr;
  }
}

void DeviceMemoryStat::Deallocate(void* ptr) {
  if (!profiling_) { return; }
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    usage_map_.at(ptr) -= usage_map_.at(ptr);
  } catch(std::out_of_range& e) {
    LOG(FATAL) << e.what() << ", ptr: " << ptr;
  }
}

void DeviceMemoryStat::Replace(void const* lhs, void const* rhs) {
  if (!profiling_) { return; }
  std::lock_guard<std::mutex> lock(mutex_);

  bool has_lhs = usage_map_.find(lhs) != usage_map_.cend();
  bool has_rhs = usage_map_.find(rhs) != usage_map_.cend();
  // This way we keep the peak and traces correct.
  if (has_lhs && has_rhs) {
    usage_map_.at(lhs) -= usage_map_.at(lhs);
    usage_map_.at(lhs) += usage_map_.at(rhs);
    usage_map_.erase(rhs);
  } else if (has_lhs) {
    usage_map_.at(lhs) -= usage_map_.at(lhs);
  } else if (has_rhs) {
    usage_map_[lhs] = usage_map_.at(rhs);
    usage_map_.erase(rhs);
  } else {
    usage_map_[lhs] = Usage();
  }
}

void DeviceMemoryStat::Reset() {
  profiling_ = false;
  global_usage_ = Usage();
  usage_map_.clear();
}

void DeviceMemoryStat::PrintSummary() const {
  if (!profiling_) { return; }

  auto CalcSize =
      [this](size_t const size_in) {
        int unit_idx = 0;
        double size_res = static_cast<double>(size_in);
        while (size_res > 1024 && unit_idx < units_.size() - 1) {
          size_res /= 1024;
          unit_idx++;
        }
        return std::to_string(size_res) + " " + units_.at(unit_idx);
      };

  std::ostringstream summary;
  summary << "\nDevice memory usage summary: ===========\n";
  summary << "Peak usage: "
          << CalcSize(global_usage_.GetPeak()) << ".\n";
  summary << "Number of allocations: "
          << std::to_string(global_usage_.GetAllocCount()) << " times.\n";
  summary << "\n";
  // FIXME: Using LOG(INFO) causes seg fault.
  std::cout << summary.str() << std::endl;

  std::ostringstream usage_detail;
  usage_detail << "\nDetailed usage: ========================\n";
  for (auto& usage : usage_map_) {
    usage_detail << "Pointer: " << usage.first << "\n";
    usage_detail << "Peak usage: " << CalcSize(usage.second.GetPeak()) << "\n";
    usage_detail << "Traces:\n";
    for (auto& trace : usage.second.GetTraces()) {
      usage_detail << "  " << trace << "\n";
    }
    usage_detail << "\n";
  }

  std::cout << usage_detail.str() << std::endl << std::endl;
}

size_t DeviceMemoryStat::GetPeakUsage() const {
  return global_usage_.peak_;
}
size_t DeviceMemoryStat::GetAllocationCount() const {
  return global_usage_.count_;
}

DeviceMemoryStat::Usage const& DeviceMemoryStat::GetPtrUsage(
    void const* ptr) const {
  try {
    auto usage = usage_map_.at(ptr);
  } catch(std::out_of_range& e) {
    LOG(FATAL) << e.what() << ", ptr: " << ptr;
  }
  return usage_map_.at(ptr);
}

void Monitor::PrintElapsed() {
  if (!debug_verbose_) return;

  LOG(CONSOLE) << "======== Monitor: " << label_ << " ========";
  for (auto &kv : timer_map_) {
    kv.second.PrintElapsed(kv.first);
  }
  self_timer_.Stop();
  self_timer_.PrintElapsed(label_ + " Lifetime");
}

void Monitor::Start(const std::string &name, GPUSet devices) {
  if (debug_verbose_) {
#ifdef __CUDACC__
#include "device_helpers.cuh"
    dh::SynchronizeNDevices(devices);
#endif
  }
  timer_map_[name].Start();
}

void Monitor::Stop(const std::string &name, GPUSet devices) {
  if (debug_verbose_) {
#ifdef __CUDACC__
#include "device_helpers.cuh"
    dh::SynchronizeNDevices(devices);
#endif
  }
  timer_map_[name].Stop();
}

}  // namespace common
}  // namespace xgboost
