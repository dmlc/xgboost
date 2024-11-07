/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "threading_utils.h"

#include <algorithm>   // for max, min
#include <exception>   // for exception
#include <filesystem>  // for path, exists
#include <fstream>     // for ifstream
#include <string>      // for string

#include "common.h"  // for DivRoundUp

#if defined(__linux__)
#include <pthread.h>
#include <sys/syscall.h>  // for SYS_getcpu
#include <unistd.h>       // for syscall
#endif

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

  try {
    fs::path const bandwidth_path{"/sys/fs/cgroup/cpu.max"};
    auto has_v2 = fs::exists(bandwidth_path);
    if (has_v2) {
      return GetCGroupV2Count(bandwidth_path);
    }
  } catch (std::exception const&) {
    return -1;
  }

  try {
    fs::path const quota_path{"/sys/fs/cgroup/cpu/cpu.cfs_quota_us"};
    fs::path const peroid_path{"/sys/fs/cgroup/cpu/cpu.cfs_period_us"};
    auto has_v1 = fs::exists(quota_path) && fs::exists(peroid_path);
    if (has_v1) {
      return GetCGroupV1Count(quota_path, peroid_path);
    }
  } catch (std::exception const&) {
    return -1;
  }

  return -1;
}

std::int32_t OmpGetNumThreads(std::int32_t n_threads) noexcept(true) {
  // Don't use parallel if we are in a parallel region.
  if (omp_in_parallel()) {
    return 1;
  }
  // Honor the openmp thread limit, which can be set via environment variable.
  auto max_n_threads = std::min({omp_get_num_procs(), omp_get_max_threads(), OmpGetThreadLimit()});
  // If -1 or 0 is specified by the user, we default to maximum number of threads.
  if (n_threads <= 0) {
    n_threads = max_n_threads;
  }
  n_threads = std::min(n_threads, max_n_threads);
  n_threads = std::max(n_threads, 1);
  return n_threads;
}

[[nodiscard]] bool GetCpuNuma(unsigned int* cpu, unsigned int* numa) {
#ifdef SYS_getcpu
  return syscall(SYS_getcpu, cpu, numa, NULL) == 0;
#else
  return false;
#endif
}

void NameThread(std::thread* t, StringView name) {
#if defined(__linux__)
  auto handle = t->native_handle();
  char old[16];
  auto ret = pthread_getname_np(handle, old, 16);
  if (ret != 0) {
    LOG(DEBUG) << "Failed to get the name from thread";
  }
  auto new_name = std::string{old} + ">" + name.c_str();  // NOLINT
  if (new_name.size() > 15) {
    new_name = new_name.substr(new_name.size() - 15);
  }
  ret = pthread_setname_np(handle, new_name.c_str());
  if (ret != 0) {
    LOG(DEBUG) << "Failed to name thread:" << ret << " :" << new_name;
  }
#else
  (void)name;
  (void)t;
#endif
}
}  // namespace xgboost::common
