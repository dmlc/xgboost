/*!
 * Copyright 2023 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/objective.h>

#include <thread>

#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/communicator-inl.h"
#include "../helpers.h"
#include "helpers.h"

namespace xgboost {

class FederatedLearnerTest : public BaseFederatedTest {
 public:
  void VerifyBaseScore(int rank, float expected_base_score) {
    InitCommunicator(rank);

    std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(rank == 0)};
    std::shared_ptr<DMatrix> sliced{Xy_->SliceCol(kWorldSize, rank)};
    std::unique_ptr<Learner> learner{Learner::Create({sliced})};
    learner->SetParam("objective", "binary:logistic");
    learner->UpdateOneIter(0, sliced);
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, expected_base_score);

    xgboost::collective::Finalize();
  }

 protected:
  static auto constexpr kRows{16};
  static auto constexpr kCols{16};
};

TEST_F(FederatedLearnerTest, BaseScore) {
  std::shared_ptr<DMatrix> Xy_{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};
  std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
  learner->SetParam("objective", "binary:logistic");
  learner->UpdateOneIter(0, Xy_);
  Json config{Object{}};
  learner->SaveConfig(&config);
  auto base_score = GetBaseScore(config);
  ASSERT_NE(base_score, ObjFunction::DefaultBaseScore());

  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedLearnerTest_BaseScore_Test::VerifyBaseScore, this, rank,
                         base_score);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
}  // namespace xgboost
