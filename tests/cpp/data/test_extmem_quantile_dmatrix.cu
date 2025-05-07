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
auto AssertEllpackEq(Context const* ctx, EllpackPageImpl const* lhs, EllpackPageImpl const* rhs) {
  ASSERT_EQ(lhs->n_rows, rhs->n_rows);
  ASSERT_EQ(lhs->info.row_stride, rhs->info.row_stride);
  ASSERT_EQ(lhs->info.n_symbols, rhs->info.n_symbols);
  ASSERT_EQ(lhs->gidx_buffer.size(), rhs->gidx_buffer.size());

  ASSERT_EQ(lhs->Cuts().Values(), rhs->Cuts().Values());
  ASSERT_EQ(lhs->Cuts().MinValues(), rhs->Cuts().MinValues());
  ASSERT_EQ(lhs->Cuts().Ptrs(), rhs->Cuts().Ptrs());

  std::vector<common::CompressedByteT> h_buf, d_buf;
  auto h_acc = rhs->GetHostAccessor(ctx, &h_buf);
  auto d_acc = rhs->GetHostAccessor(ctx, &d_buf);
  for (std::size_t i = 0; i < h_acc.n_rows * h_acc.row_stride; ++i) {
    ASSERT_EQ(h_acc.gidx_iter[i], d_acc.gidx_iter[i]);
  }
}

class ExtMemQuantileDMatrixGpu : public ::testing::TestWithParam<std::tuple<float, bool>> {
 public:
  void Run(float sparsity, bool on_host) {
    auto equal = [](Context const* ctx, EllpackPage const& orig, EllpackPage const& sparse) {
      AssertEllpackEq(ctx, orig.Impl(), sparse.Impl());
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

class EllpackHostCacheTest : public ::testing::TestWithParam<std::tuple<double, bool>> {
 public:
  static constexpr bst_idx_t NumSamples() { return 8192; }
  static constexpr bst_idx_t NumFeatures() { return 4; }
  static constexpr bst_bin_t NumBins() { return 256; }
  // Assumes dense
  static constexpr bst_idx_t NumBytes() { return NumFeatures() * NumSamples(); }

  void Run(float sparsity, bool is_concat) {
    auto ctx = MakeCUDACtx(0);
    auto param = BatchParam{NumBins(), tree::TrainParam::DftSparseThreshold()};
    auto n_batches = 4;
    auto p_fmat = RandomDataGenerator{NumSamples(), NumFeatures(), sparsity}
                      .Device(ctx.Device())
                      .GenerateDMatrix();
    bst_idx_t min_page_cache_bytes = 0;
    if (is_concat) {
      min_page_cache_bytes =
          p_fmat->GetBatches<EllpackPage>(&ctx, param).begin().Page()->Impl()->MemCostBytes() / 3;
    }

    auto p_ext_fmat = RandomDataGenerator{NumSamples(), NumFeatures(), sparsity}
                          .Batches(n_batches)
                          .Bins(param.max_bin)
                          .Device(ctx.Device())
                          .OnHost(true)
                          .MinPageCacheBytes(min_page_cache_bytes)
                          .GenerateExtMemQuantileDMatrix("temp", true);
    if (!is_concat) {
      ASSERT_EQ(p_ext_fmat->NumBatches(), n_batches);
    } else {
      ASSERT_EQ(p_ext_fmat->NumBatches(), n_batches / 2);
    }
    ASSERT_EQ(p_fmat->Info().num_row_, p_ext_fmat->Info().num_row_);
    for (auto const& page_s : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
      auto impl_s = page_s.Impl();
      auto cuts_s = impl_s->CutsShared();
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, cuts_s, sparsity == 0.0,
                                                        impl_s->info.row_stride, impl_s->n_rows);
      new_impl->CopyInfo(impl_s);
      bst_idx_t offset = 0;
      for (auto const& page_m : p_ext_fmat->GetBatches<EllpackPage>(&ctx, param)) {
        auto impl_m = page_m.Impl();
        offset += new_impl->Copy(&ctx, impl_m, offset);
      }
      AssertEllpackEq(&ctx, impl_s, new_impl.get());
    }
  }
};

TEST_P(EllpackHostCacheTest, Basic) {
  auto ctx = MakeCUDACtx(0);
  auto [sparsity, min_page_cache_bytes] = this->GetParam();
  this->Run(sparsity, min_page_cache_bytes);
}

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, EllpackHostCacheTest,
                         ::testing::Combine(::testing::Values(0.0f, 0.2f, 0.4f, 0.8f),
                                            ::testing::Bool()));
}  // namespace xgboost::data
