/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <filesystem>  // for path
#include <fstream>     // for ofstream
#include <vector>      // for vector

#include "../../../src/common/numa_topo.h"
#include "../filesystem.h"  // for TemporaryDirectory

namespace xgboost::common {
namespace {
namespace fs = std::filesystem;
}

TEST(Numa, CpuListParser) {
  common::TemporaryDirectory tmpdir;
  auto path = tmpdir.Path() / "cpulist";
  std::vector<std::int32_t> cpus;

  auto write = [&](auto const& cpulist) {
    std::ofstream fout{path};
    fout << cpulist;
  };

  {
    std::string cpulist = R"(1
)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus[0], 1);
    ASSERT_EQ(cpus.size(), 1);
  }
  {
    std::string cpulist = R"(2)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus.size(), 1);
    ASSERT_EQ(cpus[0], 2);
  }
  {
    std::string cpulist = R"(2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus.size(), 2);
    ASSERT_EQ(cpus[0], 2);
    ASSERT_EQ(cpus[1], 3);
  }

  auto check_4cpu_case = [&] {
    ASSERT_EQ(cpus.size(), 4);
    for (std::size_t i = 0; i < cpus.size(); ++i) {
      ASSERT_EQ(cpus[i], static_cast<std::int32_t>(i));
    }
  };
  {
    std::string cpulist = R"(0-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1-2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1,2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1,2-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-1,2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-1,2-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    auto path = tmpdir.Path() / "foo";
    testing::internal::CaptureStderr();
    ReadCpuList(path, &cpus);
    std::string output = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(cpus.empty());
    ASSERT_NE(output.find("foo"), std::string::npos);
  }
}

TEST(Numa, GetCpus) {
  std::vector<std::int32_t> cpus;
  if (GetNumaNumNodes() > 0) {
    GetNumaNodeCpus(0, &cpus);
    ASSERT_FALSE(cpus.empty());
  } else {
    GTEST_SKIP();
  }
}

TEST(Numa, GetMaxNumNodes) {
  auto n_nodes = GetNumaMaxNumNodes();
#if defined(__linux__)
  ASSERT_GE(n_nodes, 0);
#else
  ASSERT_EQ(n_nodes, -1);
#endif  // defined(__linux__)
}

TEST(Numa, GetMemBind) {
  // You can run this test with:
  // numactl --membind=0 ./testxgboost --gtest_filter="Numa.GetMemBind"
  // or
  // hwloc-bind --strict --membind node:0 ./testxgboost --gtest_filter="Numa.GetMemBind"
  // The strict flag is required.
  [[maybe_unused]] auto bind = GetNumaMemBind();
}

TEST(Numa, GetNumNodes) {
  auto n_nodes = GetNumaNumNodes();
#if defined(__linux__)
  ASSERT_GE(n_nodes, 1);
#else
  ASSERT_EQ(n_nodes, -1);
#endif  // defined(__linux__)
}

TEST(Numa, GetHasCpuNodes) {
  std::vector<std::int32_t> nodes;
  GetNumaHasCpuNodes(&nodes);
#if defined(__linux__)
  ASSERT_GE(nodes.size(), 1);
#else
  ASSERT_EQ(nodes.size(), 0);
#endif  // defined(__linux__)
}

TEST(Numa, GetHasNormalMemoryNodes) {
  std::vector<std::int32_t> nodes;
  GetNumaHasNormalMemoryNodes(&nodes);
#if defined(__linux__)
  ASSERT_GE(nodes.size(), 1);
#else
  ASSERT_EQ(nodes.size(), 0);
#endif  // defined(__linux__)
}
}  // namespace xgboost::common
