/**
 * Copyright 2022-2023 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/data/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../helpers.h"
#include "xgboost/context.h"

namespace xgboost {
namespace {
class DMatrixForTest : public data::SimpleDMatrix {
  size_t n_regen_{0};

 public:
  using SimpleDMatrix::SimpleDMatrix;
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const* ctx,
                                              const BatchParam& param) override {
    auto backup = this->gradient_index_;
    auto iter = SimpleDMatrix::GetGradientIndex(ctx, param);
    n_regen_ += (backup != this->gradient_index_);
    return iter;
  }

  BatchSet<EllpackPage> GetEllpackBatches(Context const* ctx, const BatchParam& param) override {
    auto backup = this->ellpack_page_;
    auto iter = SimpleDMatrix::GetEllpackBatches(ctx, param);
    n_regen_ += (backup != this->ellpack_page_);
    return iter;
  }

  auto NumRegen() const { return n_regen_; }

  void Reset() {
    this->gradient_index_.reset();
    this->ellpack_page_.reset();
    n_regen_ = 0;
  }
};

/**
 * \brief Test for whether the gradient index is correctly regenerated.
 */
class RegenTest : public ::testing::Test {
 protected:
  std::shared_ptr<DMatrix> p_fmat_;

  void SetUp() override {
    size_t constexpr kRows = 256, kCols = 10;
    HostDeviceVector<float> storage;
    auto dense = RandomDataGenerator{kRows, kCols, 0.5}.GenerateArrayInterface(&storage);
    auto adapter = data::ArrayAdapter(StringView{dense});
    p_fmat_ = std::shared_ptr<DMatrix>(
        new DMatrixForTest{&adapter, std::numeric_limits<float>::quiet_NaN(), AllThreadsForTest()});

    p_fmat_->Info().labels.Reshape(256, 1);
    auto labels = p_fmat_->Info().labels.Data();
    RandomDataGenerator{kRows, 1, 0}.GenerateDense(labels);
  }

  auto constexpr Iter() const { return 4; }

  template <typename Page>
  size_t TestTreeMethod(Context const* ctx, std::string tree_method, std::string obj,
                        bool reset = true) const {
    auto learner = std::unique_ptr<Learner>{Learner::Create({p_fmat_})};
    learner->SetParam("device", ctx->DeviceName());
    learner->SetParam("tree_method", tree_method);
    learner->SetParam("objective", obj);
    learner->Configure();

    for (auto i = 0; i < Iter(); ++i) {
      learner->UpdateOneIter(i, p_fmat_);
    }

    auto for_test = dynamic_cast<DMatrixForTest*>(p_fmat_.get());
    CHECK(for_test);
    auto backup = for_test->NumRegen();
    for_test->GetBatches<Page>(p_fmat_->Ctx(), BatchParam{});
    CHECK_EQ(for_test->NumRegen(), backup);

    if (reset) {
      for_test->Reset();
    }
    return backup;
  }
};
}  // anonymous namespace

TEST_F(RegenTest, Approx) {
  Context ctx;
  auto n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "approx", "reg:squarederror");
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "approx", "reg:logistic");
  ASSERT_EQ(n, this->Iter());
}

TEST_F(RegenTest, Hist) {
  Context ctx;
  auto n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "hist", "reg:squarederror");
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "hist", "reg:logistic");
  ASSERT_EQ(n, 1);
}

TEST_F(RegenTest, Mixed) {
  Context ctx;
  auto n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "hist", "reg:squarederror", false);
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "approx", "reg:logistic", true);
  ASSERT_EQ(n, this->Iter() + 1);

  n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "approx", "reg:logistic", false);
  ASSERT_EQ(n, this->Iter());
  n = this->TestTreeMethod<GHistIndexMatrix>(&ctx, "hist", "reg:squarederror", true);
  ASSERT_EQ(n, this->Iter() + 1);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(RegenTest, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  auto n = this->TestTreeMethod<EllpackPage>(&ctx, "approx", "reg:squarederror", true);
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<EllpackPage>(&ctx, "approx", "reg:logistic", false);
  ASSERT_EQ(n, this->Iter());

  n = this->TestTreeMethod<EllpackPage>(&ctx, "approx", "reg:logistic", true);
  ASSERT_EQ(n, this->Iter() * 2);
}

TEST_F(RegenTest, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  auto n = this->TestTreeMethod<EllpackPage>(&ctx, "hist", "reg:squarederror", true);
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<EllpackPage>(&ctx, "hist", "reg:logistic", false);
  ASSERT_EQ(n, 1);

  {
    Context ctx;
    n = this->TestTreeMethod<EllpackPage>(&ctx, "hist", "reg:logistic");
    ASSERT_EQ(n, 2);
  }
}

TEST_F(RegenTest, GpuMixed) {
  auto ctx = MakeCUDACtx(0);
  auto n = this->TestTreeMethod<EllpackPage>(&ctx, "hist", "reg:squarederror", false);
  ASSERT_EQ(n, 1);
  n = this->TestTreeMethod<EllpackPage>(&ctx, "approx", "reg:logistic", true);
  ASSERT_EQ(n, this->Iter() + 1);

  n = this->TestTreeMethod<EllpackPage>(&ctx, "approx", "reg:logistic", false);
  ASSERT_EQ(n, this->Iter());
  n = this->TestTreeMethod<EllpackPage>(&ctx, "hist", "reg:squarederror", true);
  ASSERT_EQ(n, this->Iter() + 1);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
