/**
 * Copyright 2025, XGBoost Contributors
 */
#include "numa_topo.h"

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

using MaskT = unsigned long;  // NOLINT
inline constexpr std::size_t kMaskBits = sizeof(MaskT) * 8;

#if defined(__linux__)
// Wrapper for the system call.
//
// https://github.com/torvalds/linux/blob/3f31a806a62e44f7498e2d17719c03f816553f11/mm/mempolicy.c#L1075
auto GetMemPolicy(int *mode, MaskT *nodemask, unsigned long maxnode, void *addr,  // NOLINT
                  unsigned long flags) {                                          // NOLINT
  return syscall(SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags);
}

auto GetMemPolicy(int *policy, MaskT *nodemask, unsigned long maxnode) {  // NOLINT
  return GetMemPolicy(policy, nodemask, maxnode, nullptr, 0);
}
#endif  // defined(__linux__)
}  // namespace

void ReadCpuList(fs::path const &path, std::vector<std::int32_t> *p_cpus) {
  auto &cpus = *p_cpus;
  cpus.clear();

  std::string buff;
  std::ifstream fin{path};
  fin >> buff;
  if (fin.fail()) {
    LOG(WARNING) << "Failed to read: " << path;
    return;
  }

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
      // Single value
      cpus.push_back(val0);
      continue;
    }
    CHECK_EQ(buff[last], '-') << last;

    idx = -1;
    CHECK_LT(k, buff.size());
    val1 = std::stoi(buff.data() + k, &idx);
    CHECK_GE(idx, 1);
    // Range
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
  auto p_cpulist = fs::path{"/sys/devices/system/node"} / nodename / "cpulist";  // NOLINT

  if (!fs::exists(p_cpulist)) {
    return;
  }
  ReadCpuList(p_cpulist, p_cpus);
#endif  // defined(__linux__)
}

[[nodiscard]] std::int32_t GetNumaMaxNumNodes() {
#if defined(__linux__)
  auto p_possible = fs::path{"/sys/devices/system/node/possible"};

  std::int32_t max_n_nodes = kMaskBits;

  if (fs::exists(p_possible)) {
    std::vector<std::int32_t> cpus;
    ReadCpuList(p_possible, &cpus);
    auto it = std::max_element(cpus.cbegin(), cpus.cend());
    // +1 since node/CPU uses 0-based indexing.
    if (it != cpus.cend() && (*it + 1) > max_n_nodes) {
      max_n_nodes = (*it + 1);
    }
  }

  // Just in case if it keeps getting into error
  constexpr decltype(max_n_nodes) kStop = 16384;
  // Estimate the size of the CPU set based on the error returned from get mempolicy.
  // Strategy used by hwloc and libnuma.
  while (true) {
    std::vector<MaskT> mask(max_n_nodes / kMaskBits, 0);

    std::int32_t mode = -1;
    auto err = GetMemPolicy(&mode, mask.data(), max_n_nodes);
    if (!err || errno != EINVAL) {
      return max_n_nodes;  // Got it.
    }
    max_n_nodes *= 2;

    if (max_n_nodes > kStop) {
      break;
    }
  }
#endif  // defined(__linux__)
  return -1;
}

[[nodiscard]] bool GetNumaMemBind() {
#if defined(__linux__)
  std::int32_t mode = -1;
  auto max_n_nodes = GetNumaMaxNumNodes();
  if (max_n_nodes <= 0) {
    return false;  // Sth went wrong, assume there's no membind.
  }
  CHECK_GE(max_n_nodes, kMaskBits);
  std::vector<MaskT> mask(max_n_nodes / kMaskBits);
  auto status = GetMemPolicy(&mode, mask.data(), max_n_nodes);
  if (status < 0) {
    auto msg = error::SystemError().message();
    LOG(WARNING) << msg;
    return false;
  }
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
  try {
    std::int32_t n_nodes{0};
    for (auto const &entry : fs::directory_iterator{p_node}) {
      auto name = entry.path().filename().string();
      if (name.find("node") == 0) {  // starts with `node`
        n_nodes += 1;
      }
    }
    if (n_nodes == 0) {
      // Something went wrong, we should have at lease 1 node.
      LOG(WARNING) << "Failed to list NUMA nodes.";
      return -1;
    }
    return n_nodes;
  } catch (std::exception const &e) {
    LOG(WARNING) << "Failed to list NUMA nodes: " << e.what();
  }
#endif  // defined(__linux__)
  return -1;
}

void GetNumaHasNormalMemoryNodes(std::vector<std::int32_t> *p_nodes) {
#if defined(__linux__)
  fs::path has_nm{"/sys/devices/system/node/has_normal_memory"};
  p_nodes->clear();
  if (!fs::exists(has_nm)) {
    return;
  }
  ReadCpuList(has_nm, p_nodes);
#endif  // defined(__linux__)
}

void GetNumaHasCpuNodes(std::vector<std::int32_t> *p_nodes) {
#if defined(__linux__)
  fs::path has_cpu{"/sys/devices/system/node/has_cpu"};
  p_nodes->clear();
  if (!fs::exists(has_cpu)) {
    return;
  }
  ReadCpuList(has_cpu, p_nodes);
#endif  // defined(__linux__)
}

[[nodiscard]] bool GetCpuNuma(unsigned int* cpu, unsigned int* numa) {
#ifdef SYS_getcpu
  return syscall(SYS_getcpu, cpu, numa, NULL) == 0;
#else
  return false;
#endif
}
}  // namespace xgboost::common
