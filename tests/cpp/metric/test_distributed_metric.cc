/**
 * Copyright 2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for DeviceOrd
#include <xgboost/data.h>     // for DMatrix

#include <algorithm>   // for min
#include <cstdint>     // for int32_t
#include <functional>  // for function
#include <string>      // for string
#include <thread>      // for thread

#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "test_auc.h"
#include "test_elementwise_metric.h"
#include "test_multiclass_metric.h"
#include "test_rank_metric.h"
#include "test_survival_metric.h"

namespace xgboost::metric {
namespace {
using Verifier = std::function<void(DeviceOrd)>;
struct Param {
  bool is_dist;      // is distributed
  Verifier v;        // test function
  std::string name;  // metric name
  DeviceOrd device;  // device to run
};

class TestDistributedMetric : public ::testing::TestWithParam<Param> {
 protected:
  template <typename Fn>
  void Run(bool is_dist, Fn fn, DeviceOrd device) {
    if (!is_dist) {
      fn(device);
      return;
    }

    std::int32_t n_workers{0};
    if (device.IsCUDA()) {
      n_workers = curt::AllVisibleGPUs();
    } else {
      n_workers = std::min(static_cast<std::int32_t>(std::thread::hardware_concurrency()), 3);
    }
    auto fn1 = [&]() {
      auto r = collective::GetRank();
      if (device.IsCPU()) {
        fn(DeviceOrd::CPU());
      } else {
        fn(DeviceOrd::CUDA(r));
      }
    };
    collective::TestDistributedGlobal(n_workers, fn1);
  }
};
}  // anonymous namespace

TEST_P(TestDistributedMetric, BinaryAUCRowSplit) {
  auto p = GetParam();
  this->Run(p.is_dist, p.v, p.device);
}

constexpr bool UseNCCL() {
#if defined(XGBOOST_USE_NCCL)
  return true;
#else
  return false;
#endif  // defined(XGBOOST_USE_NCCL)
}

constexpr bool UseCUDA() {
#if defined(XGBOOST_USE_CUDA)
  return true;
#else
  return false;
#endif  // defined(XGBOOST_USE_CUDA)
}

auto MakeParamsForTest() {
  std::vector<Param> cases;

  auto push = [&](std::string name, auto fn) {
    for (auto d : {DeviceOrd::CPU(), DeviceOrd::CUDA(0)}) {
      if (!UseCUDA() && d.IsCUDA()) {
        // skip CUDA tests
        continue;
      }

      auto p = Param{true, fn, name, d};
      // Distributed CUDA tests require NCCL, but local CUDA tests do not.
      if (d.IsCPU() || UseNCCL()) {
        cases.push_back(p);
      }
      // Add a local test.
      p.is_dist = false;
      cases.push_back(p);
    }
  };

#define REFLECT_NAME(name) push(#name, Verify##name)
  // AUC
  REFLECT_NAME(BinaryAUC);
  REFLECT_NAME(MultiClassAUC);
  REFLECT_NAME(MultiLabelAUC);
  REFLECT_NAME(RankingAUC);
  REFLECT_NAME(PRAUC);
  REFLECT_NAME(MultiClassPRAUC);
  REFLECT_NAME(MultiLabelPRAUC);
  REFLECT_NAME(RankingPRAUC);
  // Elementwise
  REFLECT_NAME(RMSE);
  REFLECT_NAME(RMSLE);
  REFLECT_NAME(MAE);
  REFLECT_NAME(MAPE);
  REFLECT_NAME(MPHE);
  REFLECT_NAME(LogLoss);
  REFLECT_NAME(Error);
  REFLECT_NAME(PoissonNegLogLik);
  REFLECT_NAME(MultiRMSE);
  REFLECT_NAME(Quantile);
  REFLECT_NAME(Expectile);
  // Multi-Class
  REFLECT_NAME(MultiClassError);
  REFLECT_NAME(MultiClassLogLoss);
  // Ranking
  REFLECT_NAME(Precision);
  REFLECT_NAME(NDCG);
  REFLECT_NAME(MAP);
  REFLECT_NAME(NDCGExpGain);
  // AFT
  using namespace xgboost::common;  // NOLINT
  REFLECT_NAME(AFTNegLogLik);
  REFLECT_NAME(IntervalRegressionAccuracy);

#undef REFLECT_NAME

  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    DistributedMetric, TestDistributedMetric, ::testing::ValuesIn(MakeParamsForTest()),
    [](const ::testing::TestParamInfo<TestDistributedMetric::ParamType>& info) {
      std::string result;
      if (info.param.is_dist) {
        result += "Dist_";
      }
      result += info.param.device.IsCPU() ? "CPU" : "MGPU";
      result += "_";
      result += info.param.name;
      return result;
    });

TEST(Metric, ExpectileLoadConfig) {
  auto ctx = MakeCUDACtx(GPUIDX);
  std::unique_ptr<xgboost::Metric> metric{xgboost::Metric::Create("expectile", &ctx)};
  metric->Configure({{"expectile_alpha", "0.8"}});
  Json config{Object{}};
  metric->SaveConfig(&config);

  std::unique_ptr<xgboost::Metric> loaded{xgboost::Metric::Create("expectile", &ctx)};
  loaded->LoadConfig(config);

  xgboost::HostDeviceVector<float> preds;
  preds.HostVector() = {0.1f, 0.9f};
  auto result = GetMetricEval(loaded.get(), preds, {0.0f, 1.0f}, {}, {});
  // alpha=0.8, diffs {0.1, -0.1} => losses {0.2*0.01, 0.8*0.01} -> mean 0.005.
  EXPECT_NEAR(result, 0.005f, 1e-6f);
}

TEST(AUC, MultiLabelEmptyWorker) {
  collective::TestDistributedGlobal(2, [] {
    VerifyMultiLabelAUCEmptyWorker("auc", DeviceOrd::CPU());
    VerifyMultiLabelAUCEmptyWorker("aucpr", DeviceOrd::CPU());
  });
  if (UseCUDA() && UseNCCL() && curt::AllVisibleGPUs() >= 2) {
    collective::TestDistributedGlobal(2, [] {
      auto device = DeviceOrd::CUDA(collective::GetRank());
      VerifyMultiLabelAUCEmptyWorker("auc", device);
      VerifyMultiLabelAUCEmptyWorker("aucpr", device);
    });
  }
}
}  // namespace xgboost::metric
