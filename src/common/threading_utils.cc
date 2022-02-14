/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "threading_utils.h"

#include <fcntl.h>
#include <unistd.h>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {
/**
 * \brief Get thread limit from docker container
 *
 * Modified from
 * github.com/psiha/sweater/blob/master/include/boost/sweater/hardware_concurrency.hpp
 *
 * MIT License: Copyright (c) 2016 Domagoj Šarić
 */
int32_t GetCfsCPUCount() noexcept {
  // https://bugs.openjdk.java.net/browse/JDK-8146115
  // http://hg.openjdk.java.net/jdk/hs/rev/7f22774a5f42
  // RAM limit /sys/fs/cgroup/memory.limit_in_bytes
  // swap limt /sys/fs/cgroup/memory.memsw.limit_in_bytes

  auto read_int = [](char const* const file_path) noexcept {
    auto const fd(::open(file_path, O_RDONLY, 0));
    if (fd == -1) return -1;
    char value[64];
    CHECK(::read(fd, value, sizeof(value)) < signed(sizeof(value)));
    return std::atoi(value);
  };
  // complete fair scheduler from Linux
  auto const cfs_quota(read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"));
  auto const cfs_period(read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
  if ((cfs_quota > 0) && (cfs_period > 0)) {
    // Docker allows non-whole core quota assignments - use some sort of
    // heurestical rounding.
    return std::max((cfs_quota + cfs_period / 2) / cfs_period, 1);
  }
  return -1;
}
}  // namespace common
}  // namespace xgboost
