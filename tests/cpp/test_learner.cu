/**
 * Copyright 2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>        // for DeviceSym
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore
#include <xgboost/learner.h>

#include <cstdint>  // for int32_t
#include <memory>   // for unique_ptr

#include "../../src/common/device_vector.cuh"  // for GlobalMemoryLogger
#include "helpers.h"                           // for RandomDataGenerator

namespace xgboost {
TEST(Learner, Reset) {
  dh::GlobalMemoryLogger().Clear();

  auto verbosity = GlobalConfigThreadLocalStore::Get()->verbosity;
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto p_fmat = RandomDataGenerator{1024, 32, 0.0}.GenerateDMatrix(true);
  std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
  learner->SetParam("device", DeviceSym::CUDA());
  learner->Configure();
  for (std::int32_t i = 0; i < 2; ++i) {
    learner->UpdateOneIter(i, p_fmat);
  }

  auto cur = dh::GlobalMemoryLogger().CurrentlyAllocatedBytes();
  p_fmat.reset();
  auto after_p_fmat_reset = dh::GlobalMemoryLogger().CurrentlyAllocatedBytes();
  ASSERT_LT(after_p_fmat_reset, cur);
  learner->Reset();
  auto after_learner_reset = dh::GlobalMemoryLogger().CurrentlyAllocatedBytes();
  ASSERT_LT(after_learner_reset, after_p_fmat_reset);
  ASSERT_LE(after_learner_reset, 64);
  ConsoleLogger::Configure({{"verbosity", std::to_string(verbosity)}});
}
}  // namespace xgboost
