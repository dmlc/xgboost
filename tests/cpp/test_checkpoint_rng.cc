/**
 * Copyright 2026 XGBoost Contributors
 * \file test_checkpoint_rng.cc
 * \brief Test RNG state preservation during model save/load (issue #11982)
 */
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "../../src/common/random.h"
#include "xgboost/learner.h"
#include "xgboost/json.h"
#include "helpers.h"

namespace xgboost {

class TestCheckpointRNG : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset RNG to a known state
    common::GlobalRandom().seed(42);
  }
};

TEST_F(TestCheckpointRNG, RNGStateSerialization) {
  // Test that RNG state can be serialized and deserialized correctly
  auto& rng = common::GlobalRandom();
  
  // Generate some random numbers to evolve the RNG state
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < 100; ++i) {
    dist(rng);
  }
  
  // Serialize RNG state
  std::ostringstream oss;
  oss << rng;
  std::string rng_state = oss.str();
  
  // Generate a number before deserialization
  float before_reload = dist(rng);
  
  // Reset RNG to different state
  rng.seed(999);
  
  // Deserialize RNG state
  std::istringstream iss(rng_state);
  iss >> rng;
  
  // Generate a number after deserialization - should match before_reload
  float after_reload = dist(rng);
  
  EXPECT_EQ(before_reload, after_reload) 
    << "RNG state serialization/deserialization failed";
}

TEST_F(TestCheckpointRNG, ModelSaveLoadPreservesRNG) {
  // Create a simple model and verify RNG state is preserved
  size_t constexpr kRows = 10, kCols = 10;
  auto p_dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  
  std::unique_ptr<Learner> learner{Learner::Create({})};
  learner->SetParam("verbosity", "0");
  learner->SetParam("objective", "reg:squarederror");
  learner->SetParam("subsample", "0.5");  // Enable subsampling to use RNG
  learner->SetParam("seed", "42");
  learner->Configure();
  
  // Train for a few rounds to evolve RNG state
  for (int i = 0; i < 5; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  
  // Generate a number to check RNG state
  auto& rng = common::GlobalRandom();
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float before_save = dist(rng);
  
  // Save model
  Json model{Object()};
  learner->SaveModel(&model);
  
  // Reset RNG to different state
  rng.seed(999);
  
  // Load model - this should restore RNG state
  std::unique_ptr<Learner> learner2{Learner::Create({})};
  learner2->LoadModel(model);
  
  // Generate number - should match the one after the original RNG state
  float after_load = dist(common::GlobalRandom());
  
  EXPECT_EQ(before_save, after_load)
    << "Model save/load did not preserve RNG state";
}

TEST_F(TestCheckpointRNG, CheckpointConsistency) {
  // Test the full checkpoint scenario from issue #11982
  size_t constexpr kRows = 1000, kCols = 50;
  auto p_dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  
  // Scenario 1: Train full model
  std::unique_ptr<Learner> learner_full{Learner::Create({})};
  learner_full->SetParam("verbosity", "0");
  learner_full->SetParam("objective", "reg:squarederror");
  learner_full->SetParam("subsample", "0.2");  // Low subsample like in the issue
  learner_full->SetParam("seed", "1994");
  learner_full->Configure();
  
  for (int i = 0; i < 50; ++i) {
    learner_full->UpdateOneIter(i, p_dmat);
  }
  
  HostDeviceVector<float> pred_full;
  learner_full->Predict(p_dmat, false, &pred_full, 0, 0);
  
  // Scenario 2: Train with checkpoint at round 25
  std::unique_ptr<Learner> learner_part1{Learner::Create({})};
  learner_part1->SetParam("verbosity", "0");
  learner_part1->SetParam("objective", "reg:squarederror");
  learner_part1->SetParam("subsample", "0.2");
  learner_part1->SetParam("seed", "1994");
  learner_part1->Configure();
  
  for (int i = 0; i < 25; ++i) {
    learner_part1->UpdateOneIter(i, p_dmat);
  }
  
  // Save checkpoint
  Json checkpoint{Object()};
  learner_part1->SaveModel(&checkpoint);
  
  // Load checkpoint and continue training
  std::unique_ptr<Learner> learner_part2{Learner::Create({})};
  learner_part2->LoadModel(checkpoint);
  learner_part2->Configure();
  
  for (int i = 25; i < 50; ++i) {
    learner_part2->UpdateOneIter(i, p_dmat);
  }
  
  HostDeviceVector<float> pred_checkpoint;
  learner_part2->Predict(p_dmat, false, &pred_checkpoint, 0, 0);
  
  // Predictions should be identical
  ASSERT_EQ(pred_full.Size(), pred_checkpoint.Size());
  
  auto const& h_pred_full = pred_full.ConstHostVector();
  auto const& h_pred_checkpoint = pred_checkpoint.ConstHostVector();
  
  for (size_t i = 0; i < pred_full.Size(); ++i) {
    EXPECT_FLOAT_EQ(h_pred_full[i], h_pred_checkpoint[i])
      << "Prediction mismatch at index " << i 
      << " - checkpoint did not preserve RNG state";
  }
}

}  // namespace xgboost