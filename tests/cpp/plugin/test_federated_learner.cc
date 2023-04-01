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

void VerifyObjectives(size_t rows, size_t cols, std::vector<float> const &expected_base_scores,
                      std::vector<Json> const &expected_models) {
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

  auto i = 0;
  for (auto const *entry : ::dmlc::Registry<::xgboost::ObjFunctionReg>::List()) {
    std::unique_ptr<Learner> learner{Learner::Create({sliced})};
    learner->SetParam("tree_method", "approx");
    learner->SetParam("objective", entry->name);
    if (entry->name.find("quantile") != std::string::npos) {
      learner->SetParam("quantile_alpha", "0.5");
    }
    if (entry->name.find("multi") != std::string::npos) {
      learner->SetParam("num_class", "3");
    }
    learner->UpdateOneIter(0, sliced);

    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, expected_base_scores[i]);

    Json model{Object{}};
    learner->SaveModel(&model);
    ASSERT_EQ(model, expected_models[i]);

    i++;
  }
}

class FederatedLearnerTest : public BaseFederatedTest {
 protected:
  static auto constexpr kRows{16};
  static auto constexpr kCols{16};
};

TEST_F(FederatedLearnerTest, Objectives) {
  std::shared_ptr<DMatrix> dmat{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};

  auto &h_upper = dmat->Info().labels_upper_bound_.HostVector();
  auto &h_lower = dmat->Info().labels_lower_bound_.HostVector();
  h_lower.resize(kRows);
  h_upper.resize(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    h_lower[i] = 1;
    h_upper[i] = 10;
  }

  std::vector<float> base_scores;
  std::vector<Json> models;
  for (auto const *entry : ::dmlc::Registry<::xgboost::ObjFunctionReg>::List()) {
    std::unique_ptr<Learner> learner{Learner::Create({dmat})};
    learner->SetParam("tree_method", "approx");
    learner->SetParam("objective", entry->name);
    if (entry->name.find("quantile") != std::string::npos) {
      learner->SetParam("quantile_alpha", "0.5");
    }
    if (entry->name.find("multi") != std::string::npos) {
      learner->SetParam("num_class", "3");
    }
    learner->UpdateOneIter(0, dmat);
    Json config{Object{}};
    learner->SaveConfig(&config);
    base_scores.emplace_back(GetBaseScore(config));

    Json model{Object{}};
    learner->SaveModel(&model);
    models.emplace_back(model);
  }

  RunWithFederatedCommunicator(kWorldSize, server_address_, &VerifyObjectives, kRows, kCols,
                               base_scores, models);
}
}  // namespace xgboost
