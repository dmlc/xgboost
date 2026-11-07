/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "threading_utils.h"

#include <algorithm>    // for equal, find, max, min
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <exception>    // for exception
#include <filesystem>   // for path, exists
#include <fstream>      // for ifstream
#include <optional>     // for optional
#include <string>       // for getline, stoi, string
#include <string_view>  // for string_view
#include <thread>       // for thread
#include <utility>      // for pair
#include <vector>       // for vector

#include "common.h"               // for DivRoundUp, Split
#include "xgboost/logging.h"      // for LOG
#include "xgboost/string_view.h"  // for StringView

#if defined(__linux__)
#include <pthread.h>  // for pthread_getname_np, pthread_setname_np
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

  auto warn = [] {
    LOG(WARNING) << "Invalid cgroupv2 file.";
  };
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

#if defined(__linux__)
namespace {
// Components of a path without the empty ones.
std::vector<std::string> PathParts(std::string const& path) {
  std::vector<std::string> parts;
  for (auto const& part : Split(path, '/')) {
    if (!part.empty()) {
      parts.push_back(part);
    }
  }
  return parts;
}

// mountinfo escapes spaces in paths as \040.
std::string UnescapeMountPath(std::string path) {
  for (auto pos = path.find("\\040"); pos != std::string::npos; pos = path.find("\\040", pos)) {
    path.replace(pos, 4, " ");
  }
  return path;
}

// The cgroup path of this process from /proc/self/cgroup: the "0::" line for cgroup v2, the
// line that lists `controller` for cgroup v1. Empty when the file has no such line, nullopt
// when the file cannot be read.
std::optional<std::string> CGroupMembership(std::filesystem::path const& cgroup_file,
                                            std::string_view controller) {
  std::ifstream fin{cgroup_file};
  if (!fin) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(fin, line)) {
    // hierarchy-ID:controller-list:cgroup-path
    auto first = line.find(':');
    if (first == std::string::npos) {
      continue;
    }
    auto second = line.find(':', first + 1);
    if (second == std::string::npos) {
      continue;
    }
    auto controllers = line.substr(first + 1, second - first - 1);
    auto path = line.substr(second + 1);
    if (controller.empty()) {
      if (controllers.empty()) {
        return path;
      }
      continue;
    }
    for (auto const& name : Split(controllers, ',')) {
      if (name == controller) {
        return path;
      }
    }
  }
  return std::string{};
}

// (mount point, mount root) of the hierarchy from /proc/self/mountinfo. A mount can expose a
// subtree (root other than "/"), and several mounts can match. Of the mounts whose root
// contains `membership`, the one with the shortest root wins: it exposes the most ancestors,
// and an ancestor limit still applies. Then the first match, then `fallback` with root "/".
std::pair<std::string, std::string> CGroupMount(std::filesystem::path const& mountinfo_file,
                                                std::string_view controller,
                                                std::string const& membership,
                                                std::string const& fallback) {
  std::vector<std::pair<std::string, std::string>> candidates;
  std::ifstream fin{mountinfo_file};
  std::string line;
  while (std::getline(fin, line)) {
    // 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
    auto sep = line.find(" - ");
    if (sep == std::string::npos) {
      continue;
    }
    auto mount_fields = Split(line.substr(0, sep), ' ');
    auto fs_fields = Split(line.substr(sep + 3), ' ');
    if (mount_fields.size() < 5 || fs_fields.size() < 3) {
      continue;
    }
    bool matches{false};
    if (controller.empty()) {
      matches = fs_fields[0] == "cgroup2";
    } else if (fs_fields[0] == "cgroup") {
      for (auto const& option : Split(fs_fields[2], ',')) {
        if (option == controller) {
          matches = true;
          break;
        }
      }
    }
    if (matches) {
      candidates.emplace_back(UnescapeMountPath(mount_fields[4]),
                              UnescapeMountPath(mount_fields[3]));
    }
  }

  auto member_parts = PathParts(membership);
  std::pair<std::string, std::string> const* best{nullptr};
  std::size_t best_depth{0};
  for (auto const& candidate : candidates) {
    auto root_parts = PathParts(candidate.second);
    bool contains = root_parts.size() <= member_parts.size() &&
                    std::equal(root_parts.cbegin(), root_parts.cend(), member_parts.cbegin());
    if (contains && (best == nullptr || root_parts.size() < best_depth)) {
      best = &candidate;
      best_depth = root_parts.size();
    }
  }
  if (best != nullptr) {
    return *best;
  }
  if (!candidates.empty()) {
    return candidates.front();
  }
  return {fallback, "/"};
}
}  // anonymous namespace
#endif  // defined(__linux__)

std::vector<std::filesystem::path> CGroupDirs(std::filesystem::path const& proc_self,
                                              std::string_view controller,
                                              std::string const& fallback_mount) {
  std::vector<std::filesystem::path> dirs;
#if defined(__linux__)
  auto maybe_membership = CGroupMembership(proc_self / "cgroup", controller);
  if (maybe_membership.has_value() && maybe_membership->empty()) {
    // The process is in no cgroup of this hierarchy, so no limit of it applies.
    return dirs;
  }
  // An unreadable /proc/self/cgroup keeps the previous behavior: read the mount root.
  auto membership = maybe_membership.value_or(std::string{});
  auto [mount_point, mount_root] =
      CGroupMount(proc_self / "mountinfo", controller, membership, fallback_mount);
  auto components = PathParts(membership);
  auto root_components = PathParts(mount_root);
  if (std::find(components.cbegin(), components.cend(), "..") != components.cend()) {
    // A cgroup outside this cgroup namespace cannot be resolved from here.
    components.clear();
  } else if (root_components.size() <= components.size() &&
             std::equal(root_components.cbegin(), root_components.cend(), components.cbegin())) {
    // /proc/self/cgroup is relative to the hierarchy root, the mount starts at mount_root.
    components.erase(components.begin(),
                     components.begin() + static_cast<std::ptrdiff_t>(root_components.size()));
  }
  for (auto depth = components.size(); depth > 0; --depth) {
    std::filesystem::path dir{mount_point};
    for (std::size_t i = 0; i < depth; ++i) {
      dir /= components[i];
    }
    dirs.push_back(dir);
  }
  dirs.emplace_back(mount_point);
#else
  (void)proc_self;
  (void)controller;
  (void)fallback_mount;
#endif  // defined(__linux__)
  return dirs;
}

std::int32_t GetCfsCPUCount(std::filesystem::path const& proc_self) noexcept {
  namespace fs = std::filesystem;
  std::int32_t cnt{-1};
  auto consider = [&cnt](std::int32_t level) {
    if (level > 0) {
      cnt = cnt > 0 ? std::min(cnt, level) : level;
    }
  };

  try {
    bool has_v2{false};
    for (auto const& dir : CGroupDirs(proc_self, "", "/sys/fs/cgroup")) {
      fs::path const bandwidth_path{dir / "cpu.max"};
      if (fs::exists(bandwidth_path)) {
        has_v2 = true;
        consider(GetCGroupV2Count(bandwidth_path));
      }
    }
    if (has_v2) {
      return cnt;
    }
  } catch (std::exception const&) {
    return -1;
  }

  try {
    for (auto const& dir : CGroupDirs(proc_self, "cpu", "/sys/fs/cgroup/cpu")) {
      fs::path const quota_path{dir / "cpu.cfs_quota_us"};
      fs::path const peroid_path{dir / "cpu.cfs_period_us"};
      if (fs::exists(quota_path) && fs::exists(peroid_path)) {
        consider(GetCGroupV1Count(quota_path, peroid_path));
      }
    }
  } catch (std::exception const&) {
    return -1;
  }

  return cnt;
}

std::int32_t GetCfsCPUCount() noexcept { return GetCfsCPUCount("/proc/self"); }

std::int32_t GetNumThreads(std::int32_t n_threads) noexcept(true) {
  auto max_n_threads = Executor::MaxNumWorkers();
  CHECK_GE(max_n_threads, 1);
  // If -1 or 0 is specified by the user, we default to maximum number of threads.
  if (n_threads <= 0) {
    n_threads = max_n_threads;
  }
  return std::min(n_threads, max_n_threads);
}

void NameThread(std::thread* t, StringView name) {
#if defined(__linux__) && (!defined(__ANDROID__) || __ANDROID_API__ >= 26)
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
