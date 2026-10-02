/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstdint>   // for int64_t
#include <optional>  // for optional
#include <vector>    // for vector

#include "../../../src/common/cache_manager.h"

#if !defined(__x86_64__) && defined(__APPLE__)
#include <sys/sysctl.h>  // for sysctlbyname

#include <algorithm>  // for min
#include <string>     // for string, to_string
#endif                // !defined(__x86_64__) && defined(__APPLE__)

namespace xgboost::common {
namespace {
std::optional<std::int64_t> Some(std::int64_t v) { return std::optional<std::int64_t>{v}; }
constexpr std::optional<std::int64_t> kNone = std::nullopt;

detail::DataCacheSizes Reduce(std::vector<detail::PerfLevelCache> const& levels) {
  return detail::ReduceHeterogeneousCaches(
      Span<detail::PerfLevelCache const>{levels.data(), levels.size()});
}
}  // namespace

/* The reduction policy is where the judgement calls live, so it is tested directly with
 * synthetic readings. This runs on every platform, including CPU topologies that are not
 * available to test on natively.
 */
TEST(CacheManager, ReduceHeterogeneousCaches) {
  // No readings at all: nothing is claimed, so the callers fall back to defaults.
  {
    auto out = Reduce({});
    ASSERT_FALSE(out.l1d.has_value());
    ASSERT_FALSE(out.l2_per_cpu.has_value());
  }
  // A single homogeneous level, L2 shared by 4 cores.
  {
    auto out = Reduce({{Some(64 * 1024), Some(4 * 1024 * 1024), Some(4)}});
    ASSERT_EQ(out.l1d, 64 * 1024);
    ASSERT_EQ(out.l2_per_cpu, 1024 * 1024);
  }
  // Two levels, as on an Apple M4 Max: 128KB/16MB shared by 6 performance cores and
  // 64KB/4MB shared by 4 efficiency cores. The smaller of each must win, because a
  // thread may land on either.
  {
    auto out = Reduce({{Some(128 * 1024), Some(16 * 1024 * 1024), Some(6)},
                       {Some(64 * 1024), Some(4 * 1024 * 1024), Some(4)}});
    ASSERT_EQ(out.l1d, 64 * 1024);
    ASSERT_EQ(out.l2_per_cpu, 1024 * 1024);
  }
  // Order must not matter.
  {
    auto out = Reduce({{Some(64 * 1024), Some(4 * 1024 * 1024), Some(4)},
                       {Some(128 * 1024), Some(16 * 1024 * 1024), Some(6)}});
    ASSERT_EQ(out.l1d, 64 * 1024);
    ASSERT_EQ(out.l2_per_cpu, 1024 * 1024);
  }
  // A level with an unknown sharing factor contributes no L2 estimate, since an
  // undivided cluster size would overestimate the per-thread budget. Its L1 still counts.
  {
    auto out = Reduce({{Some(32 * 1024), Some(16 * 1024 * 1024), kNone}});
    ASSERT_EQ(out.l1d, 32 * 1024);
    ASSERT_FALSE(out.l2_per_cpu.has_value());
  }
  // Mixed: only the level that reports a sharing factor contributes L2.
  {
    auto out = Reduce({{Some(128 * 1024), Some(16 * 1024 * 1024), kNone},
                       {Some(64 * 1024), Some(4 * 1024 * 1024), Some(4)}});
    ASSERT_EQ(out.l1d, 64 * 1024);
    ASSERT_EQ(out.l2_per_cpu, 1024 * 1024);
  }
  // Missing and non-positive readings are ignored rather than propagated as sizes.
  {
    auto out = Reduce({{kNone, kNone, kNone}, {Some(0), Some(-1), Some(0)}});
    ASSERT_FALSE(out.l1d.has_value());
    ASSERT_FALSE(out.l2_per_cpu.has_value());
  }
  // A sharing factor larger than the cluster would divide to zero; that must not be
  // reported as a real size.
  {
    auto out = Reduce({{Some(64 * 1024), Some(2), Some(64)}});
    ASSERT_EQ(out.l1d, 64 * 1024);
    ASSERT_FALSE(out.l2_per_cpu.has_value());
  }
}

TEST(CacheManager, Sizes) {
  CacheManager cache_manager;

  // The accessors substitute compiled defaults for anything detection left unset, so a
  // size is always usable as a divisor or a budget.
  ASSERT_GT(cache_manager.L1Size(), 0);
  ASSERT_GT(cache_manager.L2Size(), 0);
  ASSERT_GE(cache_manager.L3Size(), 0);

  // The histogram heuristics treat L1 as a subset of the budget they compare against L2,
  // so an inverted hierarchy would silently mis-tune them.
  ASSERT_GE(cache_manager.L2Size(), cache_manager.L1Size());

  // Detection must not depend on when the object is constructed.
  CacheManager other;
  ASSERT_EQ(cache_manager.L1Size(), other.L1Size());
  ASSERT_EQ(cache_manager.L2Size(), other.L2Size());
  ASSERT_EQ(cache_manager.L3Size(), other.L3Size());
}

#if !defined(__x86_64__) && defined(__APPLE__)
namespace {
std::int64_t ReadSysctlInt(std::string const& name) {
  std::uint64_t value = 0;
  std::size_t len = sizeof(value);
  if (::sysctlbyname(name.c_str(), &value, &len, nullptr, 0) != 0) {
    return -1;
  }
  return value > 0 ? static_cast<std::int64_t>(value) : -1;
}
}  // namespace

// Guards against detection silently failing and leaving the compiled defaults in place,
// which is invisible to the checks above because those defaults are also valid sizes.
TEST(CacheManager, AppleSilicon) {
  auto const n_levels = ReadSysctlInt("hw.nperflevels");
  if (n_levels <= 0) {
    GTEST_SKIP() << "sysctl does not report performance levels.";
  }

  std::int64_t expected_l1d = -1;
  std::int64_t expected_l2 = -1;
  for (std::int64_t level = 0; level < n_levels; ++level) {
    auto const prefix = "hw.perflevel" + std::to_string(level) + ".";

    auto const l1d = ReadSysctlInt(prefix + "l1dcachesize");
    if (l1d > 0) {
      expected_l1d = expected_l1d > 0 ? std::min(expected_l1d, l1d) : l1d;
    }

    auto const l2 = ReadSysctlInt(prefix + "l2cachesize");
    auto const cpus_per_l2 = ReadSysctlInt(prefix + "cpusperl2");
    if (l2 > 0 && cpus_per_l2 > 0) {
      auto const per_cpu = l2 / cpus_per_l2;
      expected_l2 = expected_l2 > 0 ? std::min(expected_l2, per_cpu) : per_cpu;
    }
  }

  CacheManager cache_manager;
  if (expected_l1d > 0) {
    ASSERT_EQ(cache_manager.L1Size(), expected_l1d);
  }
  if (expected_l2 > 0) {
    ASSERT_EQ(cache_manager.L2Size(), expected_l2);
  }
}

#endif  // !defined(__x86_64__) && defined(__APPLE__)
}  // namespace xgboost::common
