/**
 * Copyright 2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>        // for DeviceSym
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore
#include <xgboost/learner.h>

#include <cstdint>  // for int32_t
#include <memory>   // for unique_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../../src/common/device_vector.cuh"  // for GlobalMemoryLogger
#include "helpers.h"                           // for RandomDataGenerator

namespace xgboost {
TEST(LearnerModelState, ChangeCUDADevice) {
  if (curt::AllVisibleGPUs() < 2) {
    GTEST_SKIP() << "At least 2 GPUs are required.";
  }

  auto ctx = MakeCUDACtx(0);
  std::vector<float> h_base_score{0.5f};
  linalg::Vector<float> base_score{
      h_base_score.cbegin(), h_base_score.cend(), {h_base_score.size()}, ctx.Device()};
  LearnerModelState state{&ctx,
                          1,
                          0,
                          1,
                          true,
                          h_base_score,
                          std::move(base_score),
                          ObjInfo{ObjInfo::kRegression},
                          MultiStrategy::kOneOutputPerTree};

  ctx = MakeCUDACtx(1);
  EXPECT_NO_THROW(state.ConfigureDevice(&ctx));
  EXPECT_EQ(state.BaseScore(DeviceOrd::CPU())(0), h_base_score[0]);
}

TEST(Learner, Reset) {
  dh::GlobalMemoryLogger().Clear();

  auto verbosity = GlobalConfigThreadLocalStore::Get()->verbosity;
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto p_fmat = RandomDataGenerator{1024, 32, 0.0}.GenerateDMatrix(true);
  std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
  learner->Configure({{"device", DeviceSym::CUDA()}});
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
