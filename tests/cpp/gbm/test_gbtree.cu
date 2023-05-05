#include <xgboost/context.h>      // for Context
#include <xgboost/learner.h>      // for Learner
#include <xgboost/string_view.h>  // for StringView

#include <limits>  // for numeric_limits
#include <memory>  // for shared_ptr
#include <string>  // for string

#include "../../../src/data/adapter.h"        // for ArrayAdapter
#include "../../../src/data/proxy_dmatrix.h"  // for DMatrixProxy
#include "../helpers.h"                       // for RandomDataGenerator

namespace xgboost {
void TestInplaceFallback(std::string tree_method) {
  bst_row_t n_samples{1024};
  bst_feature_t n_features{32};
  HostDeviceVector<float> X_storage;
  auto X = RandomDataGenerator{n_samples, n_features, 0.0}.GenerateArrayInterface(&X_storage);
  HostDeviceVector<float> y_storage;
  auto y = RandomDataGenerator{n_samples, 1u, 0.0}.GenerateArrayInterface(&y_storage);

  auto X_adapter = data::ArrayAdapter{StringView{X}};

  Context ctx;
  std::shared_ptr<DMatrix> Xy{
      DMatrix::Create(&X_adapter, std::numeric_limits<float>::quiet_NaN(), ctx.Threads())};
  Xy->SetInfo("label", y);

  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  learner->SetParam("tree_method", tree_method);
  for (std::int32_t i = 0; i < 3; ++i) {
    learner->UpdateOneIter(i, Xy);
  }

  std::shared_ptr<DMatrix> p_m{new data::DMatrixProxy};
  auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
  proxy->SetArrayData(StringView{X});

  HostDeviceVector<float>* out_predt{nullptr};

  ::testing::internal::CaptureStderr();
  learner->InplacePredict(p_m, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &out_predt, 0, 0);
  auto output = testing::internal::GetCapturedStderr();
  ASSERT_NE(output.find("Falling back"), std::string::npos);

  learner->SetParam("tree_method", "hist");
  learner->SetParam("gpu_id", "-1");
  learner->Configure();
  HostDeviceVector<float>* out_predt_1{nullptr};

  ::testing::internal::CaptureStderr();
  learner->InplacePredict(p_m, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &out_predt_1, 0, 0);
  output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(output.empty());

  ASSERT_EQ(out_predt->ConstHostVector(), out_predt_1->ConstHostVector());
}

TEST(GBTree, InplacePredictFallback) { TestInplaceFallback("gpu_hist"); }
}  // namespace xgboost
