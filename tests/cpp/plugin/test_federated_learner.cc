/*!
 * Copyright 2023 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/objective.h>

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/linalg_op.h"
#include "../helpers.h"
#include "../objective_helpers.h"  // for MakeObjNamesForTest, ObjTestNameGenerator
#include "helpers.h"

namespace xgboost {
namespace {
auto MakeModel(std::string tree_method, std::string objective, std::shared_ptr<DMatrix> dmat) {
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParam("tree_method", tree_method);
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

void VerifyObjective(size_t rows, size_t cols, float expected_base_score, Json expected_model,
                     std::string tree_method, std::string objective) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::shared_ptr<DMatrix> dmat{RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(rank == 0)};

  if (rank == 0) {
    auto &h_upper = dmat->Info().labels_upper_bound_.HostVector();
    auto &h_lower = dmat->Info().labels_lower_bound_.HostVector();
    h_lower.resize(rows);
    h_upper.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
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
  }
  std::shared_ptr<DMatrix> sliced{dmat->SliceCol(world_size, rank)};

  auto model = MakeModel(tree_method, objective, sliced);
  auto base_score = GetBaseScore(model);
  ASSERT_EQ(base_score, expected_base_score);
  ASSERT_EQ(model, expected_model);
}
}  // namespace

class FederatedLearnerTest : public ::testing::TestWithParam<std::string> {
  std::unique_ptr<ServerForTest> server_;
  static int constexpr kWorldSize{3};

 protected:
  void SetUp() override { server_ = std::make_unique<ServerForTest>(kWorldSize); }
  void TearDown() override { server_.reset(nullptr); }

  void Run(std::string tree_method, std::string objective) {
    static auto constexpr kRows{16};
    static auto constexpr kCols{16};

    std::shared_ptr<DMatrix> dmat{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};

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

    auto model = MakeModel(tree_method, objective, dmat);
    auto score = GetBaseScore(model);

    RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyObjective, kRows, kCols,
                                 score, model, tree_method, objective);
  }
};

TEST_P(FederatedLearnerTest, Approx) {
  std::string objective = GetParam();
  this->Run("approx", objective);
}

TEST_P(FederatedLearnerTest, Hist) {
  std::string objective = GetParam();
  this->Run("hist", objective);
}

INSTANTIATE_TEST_SUITE_P(FederatedLearnerObjective, FederatedLearnerTest,
                         ::testing::ValuesIn(MakeObjNamesForTest()),
                         [](const ::testing::TestParamInfo<FederatedLearnerTest::ParamType> &info) {
                           return ObjTestNameGenerator(info);
                         });
}  // namespace xgboost
