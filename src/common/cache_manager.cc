/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#include "cache_manager.h"

#include <algorithm>  // for min
#include <cstdint>    // for uint64_t
#include <optional>   // for optional

#if !defined(__x86_64__) && defined(__linux__)
#include <fstream>  // for ifstream
#include <string>   // for string, getline, stoll
#elif !defined(__x86_64__) && defined(__APPLE__)
#include <sys/sysctl.h>  // for sysctlbyname

#include <limits>  // for numeric_limits
#include <string>  // for string, to_string
#include <vector>  // for vector
#endif             // !defined(__x86_64__) && defined(__linux__)

#if defined(__x86_64__)

void RunCpuid(uint32_t eax, uint32_t ecx, uint32_t (&abcd)[4]) {
#if defined(_MSC_VER)
  __cpuidex(reinterpret_cast<int*>(abcd), eax, ecx);
#else
  uint32_t ebx = 0, edx = 0;
  __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
  abcd[0] = eax;
  abcd[1] = ebx;
  abcd[2] = ecx;
  abcd[3] = edx;
#endif
}

#define __extract_bitmask_value(val, mask, shift) (((val) & (mask)) >> shift)

#define _CPUID_GET_TYPE(__eax) __extract_bitmask_value(__eax /*4:0*/, 0x1fU, 0)

#define _CPUID_GET_LEVEL(__eax) __extract_bitmask_value(__eax /*7:5*/, 0xe0U, 5)

#define _CPUID_GET_SETS(__ecx) ((__ecx) + 1)

#define _CPUID_GET_LINE_SIZE(__ebx) (__extract_bitmask_value(__ebx /*11:0*/, 0x7ffU, 0) + 1)

#define _CPUID_GET_PARTITIONS(__ebx) (__extract_bitmask_value(__ebx /*21:11*/, 0x3ff800U, 11) + 1)

#define _CPUID_GET_WAYS(__ebx) (__extract_bitmask_value(__ebx /*31:22*/, 0xffc00000U, 22) + 1)

#define _CPUID_CACHE_INFO_INTEL 0x4U

#define _CPUID_CACHE_INFO_AMD 0x8000001DU

#define _CPUID_VENDOR_ID_AMD 0x68747541

// Run CPUID and collect raw output.
void GetCacheInfo(int cache_num, int* type, int* level, int64_t* sets, int* line_size,
                  int* partitions, int* ways) {
  // Leaf 0x0 returns Vendor ID in EBX, EDX, ECX
  uint32_t vendor_reg[4];
  RunCpuid(0, 0, vendor_reg);
  bool is_amd = (vendor_reg[1] == _CPUID_VENDOR_ID_AMD);

  uint32_t cache_info_leaf = is_amd ? _CPUID_CACHE_INFO_AMD : _CPUID_CACHE_INFO_INTEL;
  static uint32_t abcd[4];
  RunCpuid(cache_info_leaf, cache_num, abcd);

  const uint32_t eax = abcd[0];
  const uint32_t ebx = abcd[1];
  const uint32_t ecx = abcd[2];
  // const uint32_t edx = abcd[3];  // Not used
  *type = _CPUID_GET_TYPE(eax);
  *level = _CPUID_GET_LEVEL(eax);
  *sets = _CPUID_GET_SETS(ecx);
  *line_size = _CPUID_GET_LINE_SIZE(ebx);
  *partitions = _CPUID_GET_PARTITIONS(ebx);
  *ways = _CPUID_GET_WAYS(ebx);
}

constexpr int kCpuidTypeNull = 0;
constexpr int kCpuidTypeData = 1;  // NOLINT
constexpr int kCpuidTypeInst = 2;
constexpr int kCpuidTypeUnif = 3;  // NOLINT

// Interpret the raw CPUID results and extract actual (or unified) cache
// parameters.
template <std::int32_t kMaxCacheSize>
void DetectDataCaches(int64_t* cache_sizes) {
  (void)kCpuidTypeData;
  (void)kCpuidTypeUnif;
  int cache_num = 0, cache_sizes_idx = 0;
  while (cache_sizes_idx < kMaxCacheSize) {
    int type, level, line_size, partitions, ways;
    int64_t sets, size;
    GetCacheInfo(cache_num++, &type, &level, &sets, &line_size, &partitions, &ways);

    if (type == kCpuidTypeNull) break;  // no more caches to read.
    if (type == kCpuidTypeInst) continue;

    size = ways * partitions * line_size * sets;
    cache_sizes[cache_sizes_idx++] = size;
  }
}

#elif defined(__linux__)  // non-x86_64 Linux (e.g. aarch64): read cache sizes from sysfs

namespace {

// Parse a sysfs cache "size" string like "64K", "2048K", "36864K", "2M".
int64_t ParseCacheSize(const std::string& s) {
  if (s.empty()) return -1;  // kUninitCache
  int64_t mult = 1;
  std::string num = s;
  switch (num.back()) {
    case 'K':
    case 'k':
      mult = 1024;
      num.pop_back();
      break;
    case 'M':
    case 'm':
      mult = 1024 * 1024;
      num.pop_back();
      break;
    case 'G':
    case 'g':
      mult = 1024 * 1024 * 1024;
      num.pop_back();
      break;
    default:
      break;
  }
  try {
    return static_cast<int64_t>(std::stoll(num)) * mult;
  } catch (...) {
    return -1;  // kUninitCache
  }
}

// Read data/unified cache sizes from sysfs, storing each at its level:
// L1 -> [0], L2 -> [1], L3 -> [2]. Indexing by level is robust to sysfs ordering.
template <std::int32_t kMaxCacheSize>
void DetectDataCachesSysfs(int64_t* cache_sizes) {
  const std::string base = "/sys/devices/system/cpu/cpu0/cache/index";

  for (int i = 0; i < 16; ++i) {
    const std::string dir = base + std::to_string(i);

    std::ifstream type_f(dir + "/type");
    if (!type_f) break;  // indices are contiguous; no more cache levels exposed

    std::string type;
    std::getline(type_f, type);

    // Keep only data and unified caches (this also skips the instruction cache,
    // as the x86 CPUID path does).
    if (type != "Data" && type != "Unified") continue;

    std::ifstream level_f(dir + "/level");
    int level = 0;
    level_f >> level;

    // cache_sizes has kMaxCacheSize slots addressed by (level - 1); a missing or
    // out-of-range level is skipped, leaving that slot at its kUninitCache default.
    if (level <= 0 || level > kMaxCacheSize) continue;

    std::ifstream size_f(dir + "/size");
    std::string size_s;
    std::getline(size_f, size_s);

    const int64_t sz = ParseCacheSize(size_s);
    if (sz > 0) {
      cache_sizes[level - 1] = sz;
    }
  }
}

}  // namespace

#elif defined(__APPLE__)  // non-x86_64 Apple (Apple Silicon): read cache sizes from sysctl

namespace {

/* Read an integer sysctl by name. Empty when the key is absent, unreadable, or reports a
 * non-positive size.
 */
std::optional<int64_t> ReadSysctlInt(const std::string& name) {
  std::uint64_t value = 0;
  std::size_t len = sizeof(value);
  if (::sysctlbyname(name.c_str(), &value, &len, nullptr, 0) != 0) {
    return std::nullopt;
  }
  // Cache sizes are reported as 64-bit quantities, topology counts such as
  // `hw.nperflevels` as 32-bit ones. Both land in the low bytes of the zero-initialised
  // buffer above, so a single reader handles either width.
  if (len != sizeof(std::uint32_t) && len != sizeof(std::uint64_t)) {
    return std::nullopt;
  }
  if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<int64_t>(value);
}

/* Query the sysctl `perflevel` hierarchy and store L1 -> [0] and L2 -> [1]. Anything that
 * cannot be determined is left at its kUninitCache default so the compiled fallbacks in
 * the header apply.
 *
 * This function is only the platform query; the policy that turns the readings into
 * per-thread sizes lives in ReduceHeterogeneousCaches so it can be tested directly.
 */
template <std::int32_t kMaxCacheSize>
void DetectDataCachesSysctl(int64_t* cache_sizes) {
  static_assert(kMaxCacheSize >= 2, "Need slots for L1 and L2.");

  const std::optional<int64_t> n_levels = ReadSysctlInt("hw.nperflevels");
  if (!n_levels) {
    return;
  }

  std::vector<xgboost::common::detail::PerfLevelCache> levels;
  levels.reserve(static_cast<std::size_t>(*n_levels));
  for (int64_t level = 0; level < *n_levels; ++level) {
    const std::string prefix = "hw.perflevel" + std::to_string(level) + ".";
    levels.push_back({ReadSysctlInt(prefix + "l1dcachesize"), ReadSysctlInt(prefix + "l2cachesize"),
                      ReadSysctlInt(prefix + "cpusperl2")});
  }

  const auto sizes = xgboost::common::detail::ReduceHeterogeneousCaches(
      xgboost::common::Span<xgboost::common::detail::PerfLevelCache const>{levels.data(),
                                                                           levels.size()});
  if (sizes.l1d) {
    cache_sizes[0] = *sizes.l1d;
  }
  if (sizes.l2_per_cpu) {
    cache_sizes[1] = *sizes.l2_per_cpu;
  }
  // The Apple system level cache is not exposed through sysctl; leave L3 unset.
}

}  // namespace

#endif  // defined(__x86_64__)

namespace xgboost::common {
namespace detail {
/* Reduce per-performance-level readings to the sizes one thread can rely on.
 *
 * Heterogeneous CPUs such as Apple Silicon differ from x86 in two ways that matter here:
 *
 *   - Performance and efficiency cores report different L1d and L2 sizes, and XGBoost
 *     spreads its threads over both. The minimum is taken because, as the header notes,
 *     overestimating a cache costs more than underestimating it: a block sized for a
 *     performance core's L1 would spill on an efficiency core.
 *   - L2 is shared by every core in a cluster rather than being private per core, so a
 *     raw cluster size is not what a single thread can rely on. Dividing by the sharing
 *     factor restores the per-thread meaning that the callers in hist_util.cc and
 *     tree/hist/histogram.h assume.
 *
 * A level contributes an L2 estimate only when its sharing factor is also known, since an
 * undivided cluster size would overestimate the per-thread budget several-fold.
 *
 * This is compiled on every platform, not just the ones with heterogeneous cores, so the
 * policy stays under test everywhere.
 */
DataCacheSizes ReduceHeterogeneousCaches(Span<PerfLevelCache const> levels) {
  DataCacheSizes out;
  auto keep_min = [](std::optional<std::int64_t>* acc, std::int64_t value) {
    *acc = acc->has_value() ? std::min(**acc, value) : value;
  };

  for (auto const& level : levels) {
    if (level.l1d && *level.l1d > 0) {
      keep_min(&out.l1d, *level.l1d);
    }
    if (level.l2 && level.cpus_per_l2 && *level.l2 > 0 && *level.cpus_per_l2 > 0) {
      keep_min(&out.l2_per_cpu, *level.l2 / *level.cpus_per_l2);
    }
  }
  // A cluster smaller than its sharing factor would divide to zero, which the callers
  // would then treat as a real size rather than falling back to the default.
  if (out.l2_per_cpu && *out.l2_per_cpu <= 0) {
    out.l2_per_cpu.reset();
  }
  return out;
}
}  // namespace detail

/* Detect CPU cache sizes at runtime: CPUID on x86_64, sysfs on non-x86_64 Linux, sysctl on
 * non-x86_64 Apple, and compiled L1/L2/L3 defaults otherwise (or for any size detection
 * leaves unset).
 */
CacheManager::CacheManager() {
#if defined(__x86_64__)
  DetectDataCaches<kMaxCacheSize>(cache_size_.data());
#elif defined(__linux__)
  DetectDataCachesSysfs<kMaxCacheSize>(cache_size_.data());
#elif defined(__APPLE__)
  DetectDataCachesSysctl<kMaxCacheSize>(cache_size_.data());
#else
  SetDefaultCaches();
#endif  // defined(__x86_64__)
}
}  // namespace xgboost::common
