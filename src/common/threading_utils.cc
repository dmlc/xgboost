/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include "threading_utils.h"

#include <algorithm>   // for max
#include <exception>   // for exception
#include <filesystem>  // for path, exists
#include <fstream>     // for ifstream
#include <string>      // for string

#include "common.h"  // for DivRoundUp

namespace xgboost::common {
/**
 * Modified from
 * github.com/psiha/sweater/blob/master/include/boost/sweater/hardware_concurrency.hpp
 *
 * MIT License: Copyright (c) 2016 Domagoj Šarić
 */
std::int32_t GetCGroupV1Count(std::filesystem::path const& quota_path,
                              std::filesystem::path const& peroid_path) {
#if defined(__linux__)
  // https://bugs.openjdk.java.net/browse/JDK-8146115
  // http://hg.openjdk.java.net/jdk/hs/rev/7f22774a5f42
  // RAM limit /sys/fs/cgroup/memory.limit_in_bytes
  // swap limt /sys/fs/cgroup/memory.memsw.limit_in_bytes

  auto read_int = [](char const* const file_path) noexcept {
    std::ifstream fin(file_path);
    if (!fin) {
      return -1;
    }
    std::string value;
    fin >> value;
    try {
      return std::stoi(value);
    } catch (std::exception const&) {
      return -1;
    }
  };
  // complete fair scheduler from Linux
  auto const cfs_quota(read_int(quota_path.c_str()));
  auto const cfs_period(read_int(peroid_path.c_str()));
  if ((cfs_quota > 0) && (cfs_period > 0)) {
    return std::max(cfs_quota / cfs_period, 1);
  }
#endif  //  defined(__linux__)
  return -1;
}

std::int32_t GetCGroupV2Count(std::filesystem::path const& bandwidth_path) noexcept(true) {
  std::int32_t cnt{-1};
#if defined(__linux__)
  namespace fs = std::filesystem;

  std::int32_t a{0}, b{0};

  auto warn = [] { LOG(WARNING) << "Invalid cgroupv2 file."; };
  try {
    std::ifstream fin{bandwidth_path, std::ios::in};
    fin >> a;
    fin >> b;
  } catch (std::exception const&) {
    warn();
    return cnt;
  }
  if (a > 0 && b > 0) {
    cnt = std::max(common::DivRoundUp(a, b), 1);
  }
#endif  //  defined(__linux__)
  return cnt;
}

std::int32_t GetCfsCPUCount() noexcept {
  namespace fs = std::filesystem;
  fs::path const bandwidth_path{"/sys/fs/cgroup/cpu.max"};
  auto has_v2 = fs::exists(bandwidth_path);
  if (has_v2) {
    return GetCGroupV2Count(bandwidth_path);
  }

  fs::path const quota_path{"/sys/fs/cgroup/cpu/cpu.cfs_quota_us"};
  fs::path const peroid_path{"/sys/fs/cgroup/cpu/cpu.cfs_period_us"};
  auto has_v1 = fs::exists(quota_path) && fs::exists(peroid_path);
  if (has_v1) {
    return GetCGroupV1Count(quota_path, peroid_path);
  }

  return -1;
}

std::int32_t OmpGetNumThreads(std::int32_t n_threads) {
  // Don't use parallel if we are in a parallel region.
  if (omp_in_parallel()) {
    return 1;
  }
  // If -1 or 0 is specified by the user, we default to maximum number of threads.
  if (n_threads <= 0) {
    n_threads = std::min(omp_get_num_procs(), omp_get_max_threads());
  }
  // Honor the openmp thread limit, which can be set via environment variable.
  n_threads = std::min(n_threads, OmpGetThreadLimit());
  n_threads = std::max(n_threads, 1);
  return n_threads;
}
}  // namespace xgboost::common
