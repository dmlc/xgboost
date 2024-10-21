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

class EllpackDeviceCacheTest : public ::testing::TestWithParam<float> {
 public:
  void Run() {
    auto sparsity = this->GetParam();
    auto ctx = MakeCUDACtx(0);
    bst_idx_t n_samples = 2048, n_features = 16;
    bst_bin_t n_bins = 32;
    auto p = BatchParam{n_bins, tree::TrainParam::DftSparseThreshold()};
    auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                      .Batches(4)
                      .Device(ctx.Device())
                      .Bins(p.max_bin)
                      .OnHost(true)
                      .MinPageCacheBytes(0)
                      .GenerateExtMemQuantileDMatrix("temp", true);

    auto p_fmat_valid_d = RandomDataGenerator{n_samples, n_features, sparsity}
                              .Batches(4)
                              .Device(ctx.Device())
                              .Bins(p.max_bin)
                              .OnHost(true)
                              .Ref(p_fmat)
                              .MinPageCacheBytes(0)
                              .MaxNumDevicePages(4)
                              .GenerateExtMemQuantileDMatrix("temp", true);
    ASSERT_EQ(p_fmat_valid_d->NumBatches(), 4);
    auto p_fmat_valid_h = RandomDataGenerator{n_samples, n_features, sparsity}
                              .Batches(4)
                              .Device(ctx.Device())
                              .Bins(p.max_bin)
                              .OnHost(true)
                              .Ref(p_fmat)
                              .MinPageCacheBytes(0)
                              .MaxNumDevicePages(0)
                              .GenerateExtMemQuantileDMatrix("temp", true);
    ASSERT_EQ(p_fmat_valid_h->NumBatches(), 4);

    auto d_it = p_fmat_valid_d->GetBatches<EllpackPage>(&ctx, p).begin();
    std::vector<std::shared_ptr<EllpackPage const>> d_pages;
    auto h_it = p_fmat_valid_h->GetBatches<EllpackPage>(&ctx, p).begin();
    std::vector<std::shared_ptr<EllpackPage const>> h_pages;
    for (; !d_it.AtEnd(); ++d_it) {
      d_pages.push_back(d_it.Page());
    }
    for (; !h_it.AtEnd(); ++h_it) {
      h_pages.push_back(h_it.Page());
    }
    ASSERT_EQ(h_pages.size(), d_pages.size());
    for (std::size_t i = 0; i < h_pages.size(); ++i) {
      if (sparsity != 0.0) {
        ASSERT_LT(d_pages[i]->Impl()->info.row_stride, p_fmat_valid_d->Info().num_col_);
      } else {
        ASSERT_EQ(d_pages[i]->Impl()->info.row_stride, p_fmat_valid_d->Info().num_col_);
      }
      AssertEllpackEq(&ctx, h_pages[i]->Impl(), d_pages[i]->Impl());
    }
  }
};

TEST_P(EllpackDeviceCacheTest, Basic) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, EllpackDeviceCacheTest,
                         ::testing::Values(0.0f, 0.8f));
}  // namespace xgboost::data
