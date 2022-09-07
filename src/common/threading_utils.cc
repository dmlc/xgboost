/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "threading_utils.h"

#include <fstream>
#include <string>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {
/**
 * \brief Get thread limit from CFS
 *
 * Modified from
 * github.com/psiha/sweater/blob/master/include/boost/sweater/hardware_concurrency.hpp
 *
 * MIT License: Copyright (c) 2016 Domagoj Šarić
 */
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
}  // namespace common
}  // namespace xgboost
