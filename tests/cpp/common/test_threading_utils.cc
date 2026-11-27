/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <dmlc/omp.h>  // for omp_get_thread_num, omp_get_max_threads, omp_set_num_threads
#include <gtest/gtest.h>

#include <array>       // for array
#include <atomic>      // for atomic
#include <chrono>      // for milliseconds
#include <cstddef>     // for size_t
#include <filesystem>  // for path, create_directories
#include <fstream>     // for ofstream
#include <memory>      // for make_shared, make_unique, shared_ptr
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <thread>      // for sleep_for, thread
#include <vector>      // for vector

#include "../../../src/common/threading_utils.h"  // BlockedSpace2d,ParallelFor2d,ParallelFor
#include "../filesystem.h"                        // for TemporaryDirectory
#include "xgboost/global_config.h"                // for GlobalConfigThreadLocalStore

namespace xgboost::common {
TEST(ParallelFor2d, CreateBlockedSpace2d) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kDim2 = 3;
  constexpr size_t kGrainSize = 1;

  BlockedSpace2d space(kDim1, [&](size_t) { return kDim2; }, kGrainSize);

  ASSERT_EQ(kDim1 * kDim2, space.Size());

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < kDim2; j++) {
      ASSERT_EQ(space.GetFirstDimension(i * kDim2 + j), i);
      ASSERT_EQ(j, space.GetRange(i * kDim2 + j).begin());
      ASSERT_EQ(j + kGrainSize, space.GetRange(i * kDim2 + j).end());
    }
  }
}

TEST(ParallelFor2d, Coverage) {
  std::vector<std::size_t> sizes{0, 1, 5, 8, 19};
  BlockedSpace2d space(sizes.size(), [&](auto i) { return sizes[i]; }, 3);
  for (auto n_threads : {1, 4}) {
    std::vector<std::vector<int>> visits;
    for (auto size : sizes) {
      visits.emplace_back(size, 0);
    }
    auto visit = [&](std::size_t i, Range1d range) {
      for (auto j = range.begin(); j < range.end(); ++j) {
        ++visits[i][j];
      }
    };
    if (n_threads == 1) {
      ParallelFor2d(space, n_threads, visit);
    } else {
      ParallelFor2d(space, n_threads,
                    WithWorker([&, limit = std::make_unique<int>(n_threads)](
                                   std::size_t i, Range1d range, Worker worker) {
                      ASSERT_GE(worker.Id(), 0);
                      ASSERT_LT(worker.Id(), *limit);
                      visit(i, range);
                    }));
    }
    for (auto const& row : visits) {
      for (auto count : row) {
        EXPECT_EQ(count, 1);
      }
    }
  }
}

TEST(ParallelFor, WithWorker) {
  for (auto n_threads : {1, 4}) {
    // Owning the thread limit makes the callback move-only.
    auto make_callback = [n_threads] {
      return WithWorker([limit = std::make_unique<int>(n_threads)](auto, Worker worker) {
#if defined(_OPENMP)
        EXPECT_EQ(worker.Id(), omp_get_thread_num());
#endif
        EXPECT_GE(worker.Id(), 0);
        EXPECT_LT(worker.Id(), *limit);
      });
    };
    ParallelFor(std::size_t{17}, n_threads, make_callback());
    ParallelFor1d<3>(17, n_threads, make_callback());
    ParallelForBlock(17, n_threads, make_callback());
  }
  // Nested serial loops report worker ID 0, not the outer thread ID.
  ParallelFor(4, 4, [](auto) {
    ParallelFor(1, 1, WithWorker([](auto, Worker inner) { EXPECT_EQ(inner.Id(), 0); }));
  });
}

TEST(ParallelFor, Schedules) {
  for (auto sched : {Sched::Auto(), Sched::Static(), Sched::Static(2), Sched::Dyn(), Sched::Dyn(2),
                     Sched::Guided()}) {
    for (int size : {0, 1, 19}) {
      std::array<std::atomic<int>, 19> visits{};
      ParallelFor(size, 4, sched, [&](auto i) { ++visits[i]; });
      for (std::size_t i = 0; i < visits.size(); ++i) {
        EXPECT_EQ(visits[i].load(), i < static_cast<std::size_t>(size) ? 1 : 0);
      }
    }
  }
}

TEST(ParallelFor, ExceptionDrainsTasks) {
  bool finished{false};
  EXPECT_THROW(ParallelFor(2, 2,
                           [&](auto i) {
                             if (i == 0) {
                               throw std::runtime_error{"task failed"};
                             }
                             std::this_thread::sleep_for(std::chrono::milliseconds{10});
                             finished = true;
                           }),
               std::runtime_error);
  EXPECT_TRUE(finished);
}

TEST(GetNumThreads, ConfiguredLimit) {
#if defined(_OPENMP)
  auto previous = omp_get_max_threads();
  omp_set_num_threads(2);
#else
  auto* config = GlobalConfigThreadLocalStore::Get();
  auto previous = config->nthread;
  config->nthread = 2;
#endif
  EXPECT_LE(GetNumThreads(0), 2);
  EXPECT_EQ(GetNumThreads(8), GetNumThreads(0));
  EXPECT_EQ(GetNumThreads(1), 1);
#if defined(_OPENMP)
  ParallelFor(2, 2, [](auto) { EXPECT_EQ(GetNumThreads(0), 1); });
  omp_set_num_threads(previous);
#else
  config->nthread = previous;
#endif
}

TEST(MemStackAllocator, ObjectLifetime) {
  auto value = std::make_shared<int>(7);
  for (std::size_t n : {0, 2, 3}) {
    {
      MemStackAllocator<std::shared_ptr<int>, 2> storage(n, value);
      ASSERT_EQ(value.use_count(), n + 1);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(storage[i], value);
      }
    }
    ASSERT_EQ(value.use_count(), 1);
  }
}

#if defined(__linux__)
namespace {
namespace fs = std::filesystem;

// A fake procfs and fake cgroup mounts under a temporary directory.
class CGroupTree {
  TemporaryDirectory tmp_;

 public:
  CGroupTree() { fs::create_directories(this->Proc()); }

  [[nodiscard]] fs::path Proc() const { return tmp_.Path() / "proc"; }
  [[nodiscard]] fs::path Mount(std::string const& name) const {
    auto mount = tmp_.Path() / "mnt" / name;
    fs::create_directories(mount);
    return mount;
  }

  static void Write(fs::path const& path, std::string const& content) {
    fs::create_directories(path.parent_path());
    std::ofstream fout{path};
    fout << content;
  }
  // One /proc/self/mountinfo line: ID PARENT MAJ:MIN ROOT MOUNTPOINT OPTS - FSTYPE SRC SUPEROPTS
  static std::string MountLine(std::string const& root, std::string const& point,
                               std::string const& fs_type, std::string const& super_options) {
    return "30 20 0:26 " + root + " " + point + " rw,nosuid - " + fs_type + " cgroup " +
           super_options + "\n";
  }
  void SetProc(std::string const& cgroup, std::string const& mountinfo) const {
    Write(this->Proc() / "cgroup", cgroup);
    Write(this->Proc() / "mountinfo", mountinfo);
  }
};
}  // anonymous namespace

TEST(GetCfsCPUCount, V2LimitOnProcessCGroup) {
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/pod/app\n", CGroupTree::MountLine("/", mount.string(), "cgroup2", "rw"));
  CGroupTree::Write(mount / "pod" / "cpu.max", "max 100000\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.max", "200000 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 2);
}

TEST(GetCfsCPUCount, V2ParentLimitWins) {
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/pod/app\n", CGroupTree::MountLine("/", mount.string(), "cgroup2", "rw"));
  // The parent allows 1.5 CPUs with a 50 ms period, the child 4 with the default period.
  CGroupTree::Write(mount / "pod" / "cpu.max", "75000 50000\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.max", "400000 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 2);
}

TEST(GetCfsCPUCount, V2NoLimit) {
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/pod/app\n", CGroupTree::MountLine("/", mount.string(), "cgroup2", "rw"));
  CGroupTree::Write(mount / "pod" / "cpu.max", "max 100000\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.max", "max 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), -1);
}

TEST(GetCfsCPUCount, V2SubtreeMountRootIsStripped) {
  // Only the container's subtree is mounted; its root is below the hierarchy root.
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/docker/abc/pod/app\n",
               CGroupTree::MountLine("/docker/abc", mount.string(), "cgroup2", "rw"));
  CGroupTree::Write(mount / "pod" / "cpu.max", "100000 100000\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.max", "max 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 1);
}

TEST(GetCfsCPUCount, V2SubtreeAliasKeepsAncestors) {
  // A bind mount of the process's own cgroup next to the full hierarchy, in both orders.
  for (bool alias_first : {true, false}) {
    CGroupTree tree;
    auto alias = tree.Mount("alias");
    auto full = tree.Mount("full");
    auto alias_line = CGroupTree::MountLine("/pod/app", alias.string(), "cgroup2", "rw");
    auto full_line = CGroupTree::MountLine("/", full.string(), "cgroup2", "rw");
    tree.SetProc("0::/pod/app\n", alias_first ? alias_line + full_line : full_line + alias_line);
    CGroupTree::Write(alias / "cpu.max", "max 100000\n");
    CGroupTree::Write(full / "pod" / "cpu.max", "300000 100000\n");
    CGroupTree::Write(full / "pod" / "app" / "cpu.max", "max 100000\n");
    ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 3) << "alias_first=" << alias_first;
  }
}

TEST(GetCfsCPUCount, V2EscapedSpaceInMountPoint) {
  CGroupTree tree;
  auto mount = tree.Mount("my cg");
  auto escaped = mount.parent_path().string() + "/my\\040cg";
  tree.SetProc("0::/app\n", CGroupTree::MountLine("/", escaped, "cgroup2", "rw"));
  CGroupTree::Write(mount / "app" / "cpu.max", "200000 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 2);
}

TEST(GetCfsCPUCount, V2OutsideNamespaceReadsRoot) {
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/../other\n", CGroupTree::MountLine("/", mount.string(), "cgroup2", "rw"));
  CGroupTree::Write(mount / "cpu.max", "150000 100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 2);
}

TEST(GetCfsCPUCount, V1CoMountedControllersWithSubtreeRoot) {
  // Docker on a cgroup v1 host: host cgroup namespace, cpu and cpuacct co-mounted.
  CGroupTree tree;
  auto mount = tree.Mount("cpu,cpuacct");
  tree.SetProc("4:cpu,cpuacct:/docker/abc/pod/app\n",
               CGroupTree::MountLine("/docker/abc", mount.string(), "cgroup", "rw,cpu,cpuacct"));
  CGroupTree::Write(mount / "pod" / "cpu.cfs_quota_us", "250000\n");
  CGroupTree::Write(mount / "pod" / "cpu.cfs_period_us", "100000\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.cfs_quota_us", "-1\n");
  CGroupTree::Write(mount / "pod" / "app" / "cpu.cfs_period_us", "100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 2);
}

TEST(GetCfsCPUCount, HybridHostFallsBackToV1) {
  // systemd hybrid layout: cgroup2 mounted without the cpu controller, limits in cgroup v1.
  CGroupTree tree;
  auto unified = tree.Mount("unified");
  auto cpu = tree.Mount("cpu");
  tree.SetProc("4:cpu,cpuacct:/app\n0::/app\n",
               CGroupTree::MountLine("/", unified.string(), "cgroup2", "rw") +
                   CGroupTree::MountLine("/", cpu.string(), "cgroup", "rw,cpu,cpuacct"));
  fs::create_directories(unified / "app");
  CGroupTree::Write(cpu / "app" / "cpu.cfs_quota_us", "300000\n");
  CGroupTree::Write(cpu / "app" / "cpu.cfs_period_us", "100000\n");
  ASSERT_EQ(GetCfsCPUCount(tree.Proc()), 3);
}

TEST(GetCfsCPUCount, NoProcFiles) {
  CGroupTree tree;
  // Only the mount root is read, so the value depends on the host. It must be a valid count.
  auto n = GetCfsCPUCount(tree.Proc() / "missing");
  ASSERT_TRUE(n == -1 || n >= 1);
}

TEST(CGroupDirs, NearestFirstMountLast) {
  CGroupTree tree;
  auto mount = tree.Mount("cg");
  tree.SetProc("0::/a/b\n", CGroupTree::MountLine("/", mount.string(), "cgroup2", "rw"));
  auto dirs = CGroupDirs(tree.Proc(), "", "/unused");
  ASSERT_EQ(dirs.size(), 3);
  ASSERT_EQ(dirs[0], mount / "a" / "b");
  ASSERT_EQ(dirs[1], mount / "a");
  ASSERT_EQ(dirs[2], mount);
}

TEST(CGroupDirs, NoMountinfoUsesFallback) {
  CGroupTree tree;
  CGroupTree::Write(tree.Proc() / "cgroup", "0::/a\n");
  auto dirs = CGroupDirs(tree.Proc(), "", "/fallback");
  ASSERT_EQ(dirs.size(), 2);
  ASSERT_EQ(dirs[0], fs::path("/fallback") / "a");
  ASSERT_EQ(dirs[1], fs::path("/fallback"));
}

TEST(CGroupDirs, NotInHierarchy) {
  // A cgroup v1 process has no "0::" line: the v2 hierarchy does not apply to it.
  CGroupTree tree;
  tree.SetProc("4:cpu,cpuacct:/a\n", "");
  ASSERT_TRUE(CGroupDirs(tree.Proc(), "", "/fallback").empty());
  ASSERT_EQ(CGroupDirs(tree.Proc(), "cpu", "/fallback").size(), 2);
  // No readable /proc/self/cgroup at all: the previous behavior, the fallback mount.
  ASSERT_EQ(CGroupDirs(tree.Proc() / "missing", "", "/fallback").size(), 1);
}
#endif  // defined(__linux__)
}  // namespace xgboost::common
