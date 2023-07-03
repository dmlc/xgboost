/**
 * Copyright 2023, XGBoost contributors
 */
#include <xgboost/context.h>      // for Context
#include <xgboost/learner.h>      // for Learner
#include <xgboost/string_view.h>  // for StringView

#include <limits>  // for numeric_limits
#include <memory>  // for shared_ptr
#include <string>  // for string

#include "../../../src/data/adapter.h"           // for ArrayAdapter
#include "../../../src/data/device_adapter.cuh"  // for CupyAdapter
#include "../../../src/data/proxy_dmatrix.h"     // for DMatrixProxy
#include "../helpers.h"                          // for RandomDataGenerator

namespace xgboost {
void TestInplaceFallback(Context const* ctx) {
  // prepare data
  bst_row_t n_samples{1024};
  bst_feature_t n_features{32};
  HostDeviceVector<float> X_storage;
  // use a different device than the learner
  std::int32_t data_ordinal = ctx->IsCPU() ? 0 : -1;
  auto X = RandomDataGenerator{n_samples, n_features, 0.0}
               .Device(data_ordinal)
               .GenerateArrayInterface(&X_storage);
  HostDeviceVector<float> y_storage;
  auto y = RandomDataGenerator{n_samples, 1u, 0.0}.GenerateArrayInterface(&y_storage);

  std::shared_ptr<DMatrix> Xy;
  if (data_ordinal == Context::kCpuId) {
    auto X_adapter = data::ArrayAdapter{StringView{X}};
    Xy.reset(DMatrix::Create(&X_adapter, std::numeric_limits<float>::quiet_NaN(), ctx->Threads()));
  } else {
    auto X_adapter = data::CupyAdapter{StringView{X}};
    Xy.reset(DMatrix::Create(&X_adapter, std::numeric_limits<float>::quiet_NaN(), ctx->Threads()));
  }

  Xy->SetInfo("label", y);

  // learner is configured to the device specified by ctx
  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  ConfigLearnerByCtx(ctx, learner.get());
  for (std::int32_t i = 0; i < 3; ++i) {
    learner->UpdateOneIter(i, Xy);
  }

  std::shared_ptr<DMatrix> p_m{new data::DMatrixProxy};
  auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
  if (data_ordinal == Context::kCpuId) {
    proxy->SetArrayData(StringView{X});
  } else {
    proxy->SetCUDAArray(X.c_str());
  }

  HostDeviceVector<float>* out_predt{nullptr};
  ConsoleLogger::Configure(Args{{"verbosity", "1"}});
  // test whether the warning is raised
  ::testing::internal::CaptureStderr();
  learner->InplacePredict(p_m, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &out_predt, 0, 0);
  auto output = testing::internal::GetCapturedStderr();
  ASSERT_NE(output.find("Falling back"), std::string::npos);

  // test when the contexts match
  Context new_ctx = *proxy->Ctx();
  ASSERT_NE(new_ctx.gpu_id, ctx->gpu_id);

  ConfigLearnerByCtx(&new_ctx, learner.get());
  HostDeviceVector<float>* out_predt_1{nullptr};
  // no warning is raised
  ::testing::internal::CaptureStderr();
  learner->InplacePredict(p_m, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &out_predt_1, 0, 0);
  output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(output.empty());

  ASSERT_EQ(out_predt->ConstHostVector(), out_predt_1->ConstHostVector());
}

TEST(GBTree, InplacePredictFallback) {
  auto ctx = MakeCUDACtx(0);
  TestInplaceFallback(&ctx);
}
}  // namespace xgboost
