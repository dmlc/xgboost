/**
 * Copyright 2025, XGBoost Contributors
 */
#include "numa_utils.h"

#if defined(__linux__)

#include <linux/mempolicy.h>  // for MPOL_BIND
#include <sys/syscall.h>      // for SYS_get_mempolicy
#include <unistd.h>           // for syscall

#endif  // defined(__linux__)

#include <cctype>      // for isalnum
#include <cstddef>     // for size_t
#include <cstdint>     // for int32_t
#include <filesystem>  // for path
#include <fstream>     // for ifstream
#include <string>      // for string, stoi
#include <vector>      // for vector

#include "common.h"     // for TrimLast, TrimFirst
#include "error_msg.h"  // for SystemError
#include "xgboost/logging.h"

namespace xgboost::common {

namespace {
namespace fs = std::filesystem;
#if defined(__linux__)
// Wrapper for the system call
auto GetMemPolicy(int *mode, unsigned long *nodemask, unsigned long maxnode, void *addr,  // NOLINT
                  unsigned long flags) {                                                  // NOLINT
  return syscall(SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags);
}

auto GetMemPolicy(int *policy, unsigned long *nmask, unsigned long maxnode) {  // NOLINT
  return GetMemPolicy(policy, nmask, maxnode, nullptr, 0);
}
#endif  // defined(__linux__)
}  // namespace

void ReadCpuList(fs::path const &path, std::vector<std::int32_t> *p_cpus) {
  auto &cpus = *p_cpus;
  cpus.clear();

  std::ifstream fin{path};
  std::string buff;
  fin >> buff;
  CHECK(!buff.empty());
  buff = common::TrimFirst(common::TrimLast(buff));

  std::int32_t k = 0;
  CHECK(std::isalnum(buff[k]));
  while (static_cast<std::size_t>(k) < buff.size()) {
    std::int32_t val0 = -1, val1 = -1;
    std::size_t idx = 0;
    CHECK(std::isalnum(buff[k])) << k << " " << buff;
    val0 = std::stoi(buff.data() + k, &idx);
    auto last = k + idx;
    CHECK_LE(last, buff.size());
    k = last + 1;  // new begin
    if (last == buff.size() || buff[last] != '-') {
      cpus.push_back(val0);
      continue;
    }
    CHECK_EQ(buff[last], '-') << last;

    idx = -1;
    CHECK_LT(k, buff.size());
    val1 = std::stoi(buff.data() + k, &idx);
    CHECK_GE(idx, 1);
    // Parse range
    for (auto i = val0; i <= val1; ++i) {
      cpus.push_back(i);
    }
    k += (idx + 1);
  }
}

void GetNumaNodeCpus(std::int32_t node_id, std::vector<std::int32_t> *p_cpus) {
  p_cpus->clear();
#if defined(__linux__)
  std::string nodename = "node" + std::to_string(node_id);
  auto p_cpulist = fs::path{"/sys/devices/system/node"} / fs::path{nodename} / fs::path{"cpulist"};

  if (!fs::exists(p_cpulist)) {
    return;
  }
  ReadCpuList(p_cpulist, p_cpus);
#endif  // defined(__linux__)
}

[[nodiscard]] std::int32_t GetNumaMaxNumNodes() {
#if defined(__linux__)
  auto p_possible = fs::path{"/sys/devices/system/node/possible"};
  std::int32_t max_n_nodes = -1;
  if (fs::exists(p_possible)) {
    std::vector<std::int32_t> cpus;
    ReadCpuList(p_possible, &cpus);
    auto it = std::max_element(cpus.cbegin(), cpus.cend());
    if (it != cpus.cend()) {
      max_n_nodes = *it;
    }
  }

  // Just in case if it keeps getting into error
  constexpr decltype(max_n_nodes) kThresh = 16384;
  // Estimate the size of the CPU set based on the error returned from get mempolicy.
  // Strategy used by hwloc and libnuma.
  while (true) {
    std::vector<std::uint64_t> mask(max_n_nodes, 0);

    std::int32_t mode = -1;
    auto err = GetMemPolicy(&mode, mask.data(), max_n_nodes);
    if (!err || errno != EINVAL) {
      return max_n_nodes;
    }
    max_n_nodes *= 2;

    if (max_n_nodes > kThresh) {
      break;
    }
  }
  return -1;
#else
  return -1;
#endif  // defined(__linux__)
}

[[nodiscard]] bool GetNumaMemBind() {
#if defined(__linux__)
  std::int32_t mode = -1;
  auto max_n_nodes = GetNumaMaxNumNodes();
  if (max_n_nodes <= 0) {
    return false;
  }
  std::vector<std::uint64_t> mask(max_n_nodes / 8);
  CHECK_GE(GetMemPolicy(&mode, mask.data(), max_n_nodes), 0) << error::SystemError().message();
  return mode == MPOL_BIND;
#else
  return false;
#endif  // defined(__linux__)
}

[[nodiscard]] std::int32_t GetNumaNumNodes() {
#if defined(__linux__)
  fs::path p_node{"/sys/devices/system/node"};
  if (!fs::exists(p_node)) {
    return -1;
  }
  std::int32_t n_nodes{0};
  for (auto const &entry : fs::directory_iterator{p_node}) {
    auto name = entry.path().filename().string();
    if (name.find("node") == 0) {  // starts with `node`
      n_nodes += 1;
    }
  }
  if (n_nodes == 0) {
    return -1;
  }
  return n_nodes;
#else
  return -1;
#endif  // defined(__linux__)
}
}  // namespace xgboost::common
