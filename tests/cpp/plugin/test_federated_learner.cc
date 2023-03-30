/*!
 * Copyright 2023 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/objective.h>

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"
#include "../helpers.h"
#include "helpers.h"

namespace xgboost {

void VerifyBaseScoreAndModel(size_t rows, size_t cols, std::string const& objective,
                             float expected_base_score, Json const& expected_model) {
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
  }

  std::shared_ptr<DMatrix> sliced{dmat->SliceCol(world_size, rank)};
  std::unique_ptr<Learner> learner{Learner::Create({sliced})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("objective", objective);
  if (objective.find("quantile") != std::string::npos) {
    learner->SetParam("quantile_alpha", "0.5");
  }
  if (objective.find("multi") != std::string::npos) {
    learner->SetParam("num_class", "3");
  }
  learner->UpdateOneIter(0, sliced);

  Json config{Object{}};
  learner->SaveConfig(&config);
  auto base_score = GetBaseScore(config);
  ASSERT_EQ(base_score, expected_base_score);

  Json model{Object{}};
  learner->SaveModel(&model);
  ASSERT_EQ(model, expected_model);
}

class FederatedLearnerTest : public BaseFederatedTest {
 protected:
  static auto constexpr kRows{16};
  static auto constexpr kCols{16};

  void TestObjective(std::string const& objective) {
    std::shared_ptr<DMatrix> dmat{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};

    auto &h_upper = dmat->Info().labels_upper_bound_.HostVector();
    auto &h_lower = dmat->Info().labels_lower_bound_.HostVector();
    h_lower.resize(kRows);
    h_upper.resize(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      h_lower[i] = 1;
      h_upper[i] = 10;
    }

    std::unique_ptr<Learner> learner{Learner::Create({dmat})};
    learner->SetParam("tree_method", "approx");
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
    auto base_score = GetBaseScore(config);

    Json model{Object{}};
    learner->SaveModel(&model);

    RunWithFederatedCommunicator(kWorldSize, server_address_, &VerifyBaseScoreAndModel, kRows,
                                 kCols, objective, base_score, model);
  }
};

TEST_F(FederatedLearnerTest, RegSquaredError) { TestObjective("reg:squarederror"); }

TEST_F(FederatedLearnerTest, RegSquaredLogError) { TestObjective("reg:squaredlogerror"); }

TEST_F(FederatedLearnerTest, RegLogistic) { TestObjective("reg:logistic"); }

TEST_F(FederatedLearnerTest, RegPseudoHuberError) { TestObjective("reg:pseudohubererror"); }

TEST_F(FederatedLearnerTest, RegAsoluteError) { TestObjective("reg:absoluteerror"); }

TEST_F(FederatedLearnerTest, RegQuantileError) { TestObjective("reg:quantileerror"); }

TEST_F(FederatedLearnerTest, BinaryLogistic) { TestObjective("binary:logistic"); }

TEST_F(FederatedLearnerTest, BinaryLogitRaw) { TestObjective("binary:logitraw"); }

TEST_F(FederatedLearnerTest, BinaryHinge) { TestObjective("binary:hinge"); }

TEST_F(FederatedLearnerTest, CountPoisson) { TestObjective("count:poisson"); }

TEST_F(FederatedLearnerTest, SurvivalCox) { TestObjective("survival:cox"); }

TEST_F(FederatedLearnerTest, SurvivalAft) { TestObjective("survival:aft"); }

TEST_F(FederatedLearnerTest, MultiSoftmax) { TestObjective("multi:softmax"); }

TEST_F(FederatedLearnerTest, MultiSoftprob) { TestObjective("multi:softprob"); }

TEST_F(FederatedLearnerTest, RankPairwise) { TestObjective("rank:pairwise"); }

TEST_F(FederatedLearnerTest, RankNdcg) { TestObjective("rank:ndcg"); }

TEST_F(FederatedLearnerTest, RankMap) { TestObjective("rank:map"); }

TEST_F(FederatedLearnerTest, RegGamma) { TestObjective("reg:gamma"); }

TEST_F(FederatedLearnerTest, RegTweedie) { TestObjective("reg:tweedie"); }
}  // namespace xgboost
