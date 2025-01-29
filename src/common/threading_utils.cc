/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "threading_utils.h"

#include <fstream>
#include <strin#include <mutex>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {

/**
 * \brief Read an integer value from a system file.
 *
 * \param file_path Path to the system file.
 * \return Parsed integer value, or -1 if reading fails.
 */
int ReadIntFromFile(const char* file_path) noexcept {
    std::ifstream fin(file_path);
    if (!fin) {
        return -1;  // Return -1 on failure
    }

    std::string value;
    fin >> value;

    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return -1;  // Return -1 if parsing fails
    }
}

/**
 * \brief Get thread limit from the CFS scheduler.
 *
 * This function caches the result to avoid repeated file system access.
 *
 * \return The computed thread limit or -1 if unavailable.
 */
int32_t GetCfsCPUCount() noexcept {
    static int32_t cached_cfs_cpu_count = -2;
    static std::once_flag cache_flag;

    std::call_once(cache_flag, []() {
#if defined(__linux__)
        const int cfs_quota = ReadIntFromFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        const int cfs_period = ReadIntFromFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us");

        if (cfs_quota > 0 && cfs_period > 0) {
            cached_cfs_cpu_count = std::max(cfs_quota / cfs_period, 1);
            return;
        }
#endif  // defined(__linux__)
        cached_cfs_cpu_count = -1;  // Default to -1 if no valid quota is found
    });

    return cached_cfs_cpu_count;
}

}  // namespace common
}  // namespace xgboost
