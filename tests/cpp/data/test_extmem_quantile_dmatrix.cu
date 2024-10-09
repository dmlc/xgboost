/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <tuple>   // for tuple
#include <vector>  // for vector

#include "../../../src/data/ellpack_page.cuh"  // for EllpackPageImpl
#include "../helpers.h"                        // for RandomDataGenerator
#include "test_extmem_quantile_dmatrix.h"      // for TestExtMemQdmBasic

namespace xgboost::data {
class ExtMemQuantileDMatrixGpu : public ::testing::TestWithParam<std::tuple<float, bool>> {
 public:
  void Run(float sparsity, bool on_host) {
    auto equal = [](Context const* ctx, EllpackPage const& orig, EllpackPage const& sparse) {
      auto const& orig_cuts = orig.Cuts();
      auto const& sparse_cuts = sparse.Cuts();
      ASSERT_EQ(orig_cuts.Values(), sparse_cuts.Values());
      ASSERT_EQ(orig_cuts.MinValues(), sparse_cuts.MinValues());
      ASSERT_EQ(orig_cuts.Ptrs(), sparse_cuts.Ptrs());

      std::vector<common::CompressedByteT> h_orig, h_sparse;
      [[maybe_unused]] auto orig_acc = orig.Impl()->GetHostAccessor(ctx, &h_orig, {});
      [[maybe_unused]] auto sparse_acc = sparse.Impl()->GetHostAccessor(ctx, &h_sparse, {});
      ASSERT_EQ(h_orig.size(), h_sparse.size());

      auto equal = std::equal(h_orig.cbegin(), h_orig.cend(), h_sparse.cbegin());
      ASSERT_TRUE(equal);
    };
    auto no_missing = [](EllpackPage const& page) {
      return page.Impl()->IsDense();
    };

    auto ctx = MakeCUDACtx(0);
    TestExtMemQdmBasic<EllpackPage>(&ctx, on_host, sparsity, equal, no_missing);
  }
};

TEST_P(ExtMemQuantileDMatrixGpu, Basic) {
  auto [sparsity, on_host] = this->GetParam();
  this->Run(sparsity, on_host);
}

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, ExtMemQuantileDMatrixGpu,
                         ::testing::Combine(::testing::Values(0.0f, 0.2f, 0.4f, 0.8f),
                                            ::testing::Bool()));
}  // namespace xgboost::data
