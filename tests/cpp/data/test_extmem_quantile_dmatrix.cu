/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <vector>  // for vector

#include "../../../src/data/ellpack_page.cuh"  // for EllpackPageImpl
#include "../helpers.h"                        // for RandomDataGenerator
#include "test_extmem_quantile_dmatrix.h"      // for TestExtMemQdmBasic

namespace xgboost::data {
class ExtMemQuantileDMatrixGpu : public ::testing::TestWithParam<float> {
 public:
  void Run(float sparsity) {
    auto equal = [](Context const* ctx, EllpackPage const& orig, EllpackPage const& sparse) {
      auto const& orig_cuts = orig.Cuts();
      auto const& sparse_cuts = sparse.Cuts();
      ASSERT_EQ(orig_cuts.Values(), sparse_cuts.Values());
      ASSERT_EQ(orig_cuts.MinValues(), sparse_cuts.MinValues());
      ASSERT_EQ(orig_cuts.Ptrs(), sparse_cuts.Ptrs());

      std::vector<common::CompressedByteT> h_orig, h_sparse;
      auto orig_acc = orig.Impl()->GetHostAccessor(ctx, &h_orig, {});
      auto sparse_acc = sparse.Impl()->GetHostAccessor(ctx, &h_sparse, {});
      ASSERT_EQ(h_orig.size(), h_sparse.size());

      auto equal = std::equal(h_orig.cbegin(), h_orig.cend(), h_sparse.cbegin());
      ASSERT_TRUE(equal);
    };

    auto ctx = MakeCUDACtx(0);
    TestExtMemQdmBasic<EllpackPage>(&ctx, true, sparsity, equal);
    TestExtMemQdmBasic<EllpackPage>(&ctx, false, sparsity, equal);
  }
};

TEST_P(ExtMemQuantileDMatrixGpu, Basic) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, ExtMemQuantileDMatrixGpu, ::testing::ValuesIn([] {
                           std::vector<float> sparsities{0.0f, 0.2f, 0.4f, 0.8f};
                           return sparsities;
                         }()));
}  // namespace xgboost::data
