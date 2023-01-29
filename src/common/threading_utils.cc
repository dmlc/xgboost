/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include "threading_utils.h"

#include <fstream>
#include <string>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {
int32_t GetCfsCPUCount() noexcept {
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
  auto const cfs_quota(read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"));
  auto const cfs_period(read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
  if ((cfs_quota > 0) && (cfs_period > 0)) {
    return std::max(cfs_quota / cfs_period, 1);
  }
#endif  //  defined(__linux__)
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
}  // namespace common
}  // namespace xgboost
