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
  ASSERT_EQ(base_score, expected_base_score) << " rank " << rank;
  ASSERT_EQ(model, expected_model) << " rank " << rank;
}
}  // namespace

class VerticalFederatedLearnerTest
    : public ::testing::TestWithParam<std::tuple<std::string, bool>> {
  static int constexpr kWorldSize{3};

 protected:
  void Run(std::string tree_method, std::string device, std::string objective, bool is_encrypted) {
    // Following objectives are not yet supported.
    if (is_encrypted) {
      std::vector<std::string> unsupported{"multi:", "quantile", "absoluteerror"};
      auto skip = std::any_of(unsupported.cbegin(), unsupported.cend(), [&](auto const &name) {
        return objective.find(name) != std::string::npos;
      });
      if (skip) {
        GTEST_SKIP_("Not supported by the plugin.");
      }
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

  auto GetTestParam() {
    std::string objective = get<0>(GetParam());
    auto is_encrypted = get<1>(GetParam());
    return std::make_tuple(objective, is_encrypted);
  }
};

namespace {
auto MakeTestParams() {
  auto objs = MakeObjNamesForTest();
  std::vector<std::tuple<std::string, bool>> values;
  for (auto const &v : objs) {
    values.emplace_back(v, true);
    values.emplace_back(v, false);
  }
  return values;
}
}  // namespace

TEST_P(VerticalFederatedLearnerTest, Approx) {
  auto [objective, is_encrypted] = this->GetTestParam();
  if (is_encrypted) {
    GTEST_SKIP();
  }
  this->Run("approx", DeviceSym::CPU(), objective, is_encrypted);
}

TEST_P(VerticalFederatedLearnerTest, Hist) {
  auto [objective, is_encrypted] = this->GetTestParam();
  this->Run("hist", DeviceSym::CPU(), objective, is_encrypted);
}

#if defined(XGBOOST_USE_CUDA)
TEST_P(VerticalFederatedLearnerTest, GPUApprox) {
  auto [objective, is_encrypted] = this->GetTestParam();
  if (is_encrypted) {
    GTEST_SKIP();
  }
  this->Run("approx", DeviceSym::CUDA(), objective, is_encrypted);
}

TEST_P(VerticalFederatedLearnerTest, GPUHist) {
  auto [objective, is_encrypted] = this->GetTestParam();
  if (is_encrypted) {
    GTEST_SKIP();
  }
  this->Run("hist", DeviceSym::CUDA(), objective, is_encrypted);
}
#endif  // defined(XGBOOST_USE_CUDA)

INSTANTIATE_TEST_SUITE_P(
    FederatedLearnerObjective, VerticalFederatedLearnerTest, ::testing::ValuesIn(MakeTestParams()),
    [](const ::testing::TestParamInfo<VerticalFederatedLearnerTest::ParamType> &info) {
      auto name = ObjTestNameGenerator(std::get<0>(info.param));
      if (std::get<1>(info.param)) {
        name += "_enc";
      }
      return name;
    });
}  // namespace xgboost
