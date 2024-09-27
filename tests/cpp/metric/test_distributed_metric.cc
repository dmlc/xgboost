/**
 * Copyright 2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for DeviceOrd
#include <xgboost/data.h>     // for DataSplitMode

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

#if defined(XGBOOST_USE_FEDERATED)

#include "../plugin/federated/test_worker.h"  // for TestFederatedGlobal

#endif  // defined(XGBOOST_USE_FEDERATED)

namespace xgboost::metric {
namespace {
using Verifier = std::function<void(DataSplitMode, DeviceOrd)>;
struct Param {
  bool is_dist;         // is distributed
  bool is_fed;          // is federated learning
  DataSplitMode split;  // how to split data
  Verifier v;           // test function
  std::string name;     // metric name
  DeviceOrd device;     // device to run
};

class TestDistributedMetric : public ::testing::TestWithParam<Param> {
 protected:
  template <typename Fn>
  void Run(bool is_dist, bool is_fed, DataSplitMode split_mode, Fn fn, DeviceOrd device) {
    if (!is_dist) {
      fn(split_mode, device);
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
        fn(split_mode, DeviceOrd::CPU());
      } else {
        fn(split_mode, DeviceOrd::CUDA(r));
      }
    };
    if (is_fed) {
#if defined(XGBOOST_USE_FEDERATED)
      collective::TestFederatedGlobal(n_workers, fn1);
#endif  // defined(XGBOOST_USE_FEDERATED)
    } else {
      collective::TestDistributedGlobal(n_workers, fn1);
    }
  }
};
}  // anonymous namespace

TEST_P(TestDistributedMetric, BinaryAUCRowSplit) {
  auto p = GetParam();
  this->Run(p.is_dist, p.is_fed, p.split, p.v, p.device);
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

constexpr bool UseFederated() {
#if defined(XGBOOST_USE_FEDERATED)
  return true;
#else
  return false;
#endif
}

auto MakeParamsForTest() {
  std::vector<Param> cases;

  auto push = [&](std::string name, auto fn) {
    for (bool is_federated : {false, true}) {
      for (DataSplitMode m : {DataSplitMode::kCol, DataSplitMode::kRow}) {
        for (auto d : {DeviceOrd::CPU(), DeviceOrd::CUDA(0)}) {
          if (!is_federated && !UseNCCL() && d.IsCUDA()) {
            // Federated doesn't use nccl.
            continue;
          }
          if (!UseCUDA() && d.IsCUDA()) {
            // skip CUDA tests
            continue;
          }
          if (!UseFederated() && is_federated) {
            // skip GRPC tests
            continue;
          }

          auto p = Param{true, is_federated, m, fn, name, d};
          cases.push_back(p);
          if (!is_federated) {
            // Add a local test.
            p.is_dist = false;
            cases.push_back(p);
          }
        }
      }
    }
  };

#define REFLECT_NAME(name) push(#name, Verify##name)
  // AUC
  REFLECT_NAME(BinaryAUC);
  REFLECT_NAME(MultiClassAUC);
  REFLECT_NAME(RankingAUC);
  REFLECT_NAME(PRAUC);
  REFLECT_NAME(MultiClassPRAUC);
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
      if (info.param.is_fed) {
        result += "Federated_";
      }
      if (info.param.split == DataSplitMode::kRow) {
        result += "RowSplit";
      } else {
        result += "ColSplit";
      }
      result += "_";
      result += info.param.device.IsCPU() ? "CPU" : "CUDA";
      result += "_";
      result += info.param.name;
      return result;
    });
}  // namespace xgboost::metric
