/**
 * Copyright 2023-2024, XGBoost contributors
 *
 * Some other tests for federated learning are in the main test suite (test_learner.cc),
 * gaurded by the `XGBOOST_USE_FEDERATED`.
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/objective.h>

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/linalg_op.h"  // for begin, end
#include "../helpers.h"
#include "../objective_helpers.h"  // for MakeObjNamesForTest, ObjTestNameGenerator
#include "federated/test_worker.h"

namespace xgboost {
namespace {
auto MakeModel(std::string tree_method, std::string device, std::string objective,
               std::shared_ptr<DMatrix> dmat) {
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParam("tree_method", tree_method);
  learner->SetParam("device", device);
  learner->SetParam("objective", objective);
  if (objective.find("quantile") != std::string::npos) {
    learner->SetParam("quantile_alpha", "0.5");
  }
  if (objective.find("multi") != std::string::npos) {
    learner->SetParam("num_class", "3");
  }
  learner->UpdateOneIter(0, dmat);
  Json config{Object{}};
  learner->SaveConfig(&config);

  Json model{Object{}};
  learner->SaveModel(&model);
  return model;
}

void VerifyObjective(std::size_t rows, std::size_t cols, float expected_base_score,
                     Json const &expected_model, std::string const &tree_method, std::string device,
                     std::string const &objective) {
  auto rank = collective::GetRank();
  std::shared_ptr<DMatrix> dmat{RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(rank == 0)};

  if (rank == 0) {
    MakeLabelForObjTest(dmat, objective);
  }
  std::shared_ptr<DMatrix> sliced{dmat->SliceCol(collective::GetWorldSize(), rank)};

  auto model = MakeModel(tree_method, device, objective, sliced);
  auto base_score = GetBaseScore(model);

  std::unique_ptr<Learner> expected{Learner::Create({})};
  expected->LoadModel(expected_model);

  std::unique_ptr<Learner> got{Learner::Create({})};
  got->LoadModel(model);

  if (rank == 0) {
    ASSERT_EQ(base_score, expected_base_score) << " rank " << rank;
    HostDeviceVector<float> expected_predt;
    expected->Predict(dmat, false, &expected_predt, 0, 0);
    HostDeviceVector<float> got_predt;
    expected->Predict(dmat, false, &got_predt, 0, 0);
    ASSERT_EQ(expected_predt.HostVector(), got_predt.HostVector());
  }
}
}  // namespace

class VerticalFederatedLearnerTest : public ::testing::TestWithParam<std::string> {
  static int constexpr kWorldSize{3};

 protected:
  void Run(std::string tree_method, std::string device, std::string objective) {
    // Following objectives are not yet supported.
    if (objective.find("multi:") != std::string::npos) {
      GTEST_SKIP();
      return;
    }
    if (objective.find("quantile") != std::string::npos) {
      GTEST_SKIP();
      return;
    }
    if (objective.find("absoluteerror") != std::string::npos) {
      GTEST_SKIP();
    }

    static auto constexpr kRows{16};
    static auto constexpr kCols{16};

    std::shared_ptr<DMatrix> dmat{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};
    MakeLabelForObjTest(dmat, objective);

    auto &h_upper = dmat->Info().labels_upper_bound_.HostVector();
    auto &h_lower = dmat->Info().labels_lower_bound_.HostVector();
    h_lower.resize(kRows);
    h_upper.resize(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      h_lower[i] = 1;
      h_upper[i] = 10;
    }
    if (objective.find("rank:") != std::string::npos) {
      auto h_label = dmat->Info().labels.HostView();
      std::size_t k = 0;
      for (auto &v : h_label) {
        v = k % 2 == 0;
        ++k;
      }
    }

    auto model = MakeModel(tree_method, device, objective, dmat);
    auto score = GetBaseScore(model);

    collective::TestFederatedGlobal(kWorldSize, [&]() {
      VerifyObjective(kRows, kCols, score, model, tree_method, device, objective);
    });
  }
};

TEST_P(VerticalFederatedLearnerTest, Approx) {
  std::string objective = GetParam();
  this->Run("approx", DeviceSym::CPU(), objective);
}

TEST_P(VerticalFederatedLearnerTest, Hist) {
  std::string objective = GetParam();
  this->Run("hist", DeviceSym::CPU(), objective);
}

#if defined(XGBOOST_USE_CUDA)
TEST_P(VerticalFederatedLearnerTest, GPUApprox) {
  std::string objective = GetParam();
  // Not yet supported by the plugin system
  GTEST_SKIP();
  this->Run("approx", DeviceSym::CUDA(), objective);
}

TEST_P(VerticalFederatedLearnerTest, GPUHist) {
  std::string objective = GetParam();
  // Not yet supported by the plugin system
  GTEST_SKIP();
  this->Run("hist", DeviceSym::CUDA(), objective);
}
#endif  // defined(XGBOOST_USE_CUDA)

INSTANTIATE_TEST_SUITE_P(
    FederatedLearnerObjective, VerticalFederatedLearnerTest,
    ::testing::ValuesIn(MakeObjNamesForTest()),
    [](const ::testing::TestParamInfo<VerticalFederatedLearnerTest::ParamType> &info) {
      return ObjTestNameGenerator(info);
    });
}  // namespace xgboost
