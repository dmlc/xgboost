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

class FederatedLearnerTest : public BaseFederatedTest {
 protected:
  static auto constexpr kRows{16};
  static auto constexpr kCols{16};
};

void VerifyBaseScore(size_t rows, size_t cols, float expected_base_score) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(rank == 0)};
  std::shared_ptr<DMatrix> sliced{Xy_->SliceCol(world_size, rank)};
  std::unique_ptr<Learner> learner{Learner::Create({sliced})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("objective", "binary:logistic");
  learner->UpdateOneIter(0, sliced);
  Json config{Object{}};
  learner->SaveConfig(&config);
  auto base_score = GetBaseScore(config);
  ASSERT_EQ(base_score, expected_base_score);
}

void VerifyModel(size_t rows, size_t cols, Json const& expected_model) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(rank == 0)};
  std::shared_ptr<DMatrix> sliced{Xy_->SliceCol(world_size, rank)};
  std::unique_ptr<Learner> learner{Learner::Create({sliced})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("objective", "binary:logistic");
  learner->UpdateOneIter(0, sliced);
  Json model{Object{}};
  learner->SaveModel(&model);
  ASSERT_EQ(model, expected_model);
}

TEST_F(FederatedLearnerTest, BaseScore) {
  std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};
  std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("objective", "binary:logistic");
  learner->UpdateOneIter(0, Xy_);
  Json config{Object{}};
  learner->SaveConfig(&config);
  auto base_score = GetBaseScore(config);
  ASSERT_NE(base_score, ObjFunction::DefaultBaseScore());

  RunWithFederatedCommunicator(kWorldSize, server_address_, &VerifyBaseScore, kRows, kCols,
                               base_score);
}

TEST_F(FederatedLearnerTest, Model) {
  std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};
  std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("objective", "binary:logistic");
  learner->UpdateOneIter(0, Xy_);
  Json model{Object{}};
  learner->SaveModel(&model);

  RunWithFederatedCommunicator(kWorldSize, server_address_, &VerifyModel, kRows, kCols,
                               std::cref(model));
}
}  // namespace xgboost
