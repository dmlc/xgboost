/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <tuple>   // for tuple
#include <vector>  // for vector

#include "../../../src/data/batch_utils.h"     // for AutoHostRatio
#include "../../../src/data/ellpack_page.cuh"  // for EllpackPageImpl
#include "../helpers.h"                        // for RandomDataGenerator, GMockThrow
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
  auto h_acc = rhs->GetHostEllpack(ctx, &h_buf);
  auto d_acc = rhs->GetHostEllpack(ctx, &d_buf);
  std::visit(
      [&](auto&& h_acc, auto&& d_acc) {
        for (std::size_t i = 0; i < h_acc.n_rows * h_acc.row_stride; ++i) {
          ASSERT_EQ(h_acc.gidx_iter[i], d_acc.gidx_iter[i]);
        }
      },
      h_acc, d_acc);
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

class EllpackHostCacheTest : public ::testing::TestWithParam<std::tuple<double, bool, float>> {
 public:
  static constexpr bst_idx_t NumSamples() { return 8192; }
  static constexpr bst_idx_t NumFeatures() { return 4; }
  static constexpr bst_bin_t NumBins() { return 256; }
  // Assumes dense
  static constexpr bst_idx_t NumBytes() { return NumFeatures() * NumSamples(); }

  void Run(float sparsity, bool is_concat, float cache_host_ratio) {
    auto ctx = MakeCUDACtx(0);
    auto param = BatchParam{NumBins(), tree::TrainParam::DftSparseThreshold()};
    auto n_batches = 4;
    auto p_fmat = RandomDataGenerator{NumSamples(), NumFeatures(), sparsity}
                      .Device(ctx.Device())
                      .GenerateDMatrix();
    bst_idx_t min_page_cache_bytes = ::xgboost::cuda_impl::MatchingPageBytes();
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
                          .CacheHostRatio(cache_host_ratio)
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
  auto [sparsity, is_concat, cache_host_ratio] = this->GetParam();
  this->Run(sparsity, is_concat, cache_host_ratio);
}

INSTANTIATE_TEST_SUITE_P(
    ExtMemQuantileDMatrix, EllpackHostCacheTest,
    ::testing::Combine(::testing::Values(0.0f, 0.2f, 0.4f, 0.8f), ::testing::Bool(),
                       ::testing::Values(0.0f, 0.5f, 1.0f, ::xgboost::cuda_impl::AutoHostRatio())));

TEST(EllpackHostCacheTest, Accessor) {
  auto ctx = MakeCUDACtx(0);
  auto param = BatchParam{32, tree::TrainParam::DftSparseThreshold()};
  param.prefetch_copy = false;
  std::size_t n_bytes = 0;
  {
    auto p_ext_fmat = RandomDataGenerator{128, 16, 0.0}
                          .Batches(4)
                          .Bins(param.max_bin)
                          .Device(ctx.Device())
                          .OnHost(true)
                          .MinPageCacheBytes(1024 * 1024 * 1024)
                          .CacheHostRatio(0.0)
                          .GenerateExtMemQuantileDMatrix("temp", true);
    ASSERT_EQ(p_ext_fmat->NumBatches(), 1);

    for (auto const& page : p_ext_fmat->GetBatches<EllpackPage>(&ctx, param)) {
      auto acc = page.Impl()->GetDeviceEllpack(&ctx, {});
      // Fully on device
      auto dacc = std::get_if<EllpackDeviceAccessor>(&acc);
      ASSERT_TRUE(dacc);
      n_bytes = page.Impl()->MemCostBytes();
    }
  }
  if (!curt::SupportsPageableMem()) {
    GTEST_SKIP_("Requires HMM or ATS.");
  }
  {
    std::size_t n_pages = 2;  // split for 2 pages
    auto p_ext_fmat = RandomDataGenerator{128, 16, 0.0}
                          .Batches(4)
                          .Bins(param.max_bin)
                          .Device(ctx.Device())
                          .OnHost(true)
                          .MinPageCacheBytes(n_bytes / n_pages)
                          .CacheHostRatio(0.5)
                          .GenerateExtMemQuantileDMatrix("temp", true);
    ASSERT_EQ(p_ext_fmat->NumBatches(), n_pages);
    for (auto const& page : p_ext_fmat->GetBatches<EllpackPage>(&ctx, param)) {
      auto acc = page.Impl()->GetDeviceEllpack(&ctx, {});
      // Host + device
      auto dacc = std::get_if<DoubleEllpackAccessor>(&acc);
      ASSERT_TRUE(dacc);
    }
  }
}

class EllpackDecompTest : public ::testing::TestWithParam<float> {
 public:
  void Run(float hw_decomp_ratio) {
    auto ctx = MakeCUDACtx(0);
    auto param = BatchParam{128, tree::TrainParam::DftSparseThreshold()};
    std::size_t n_samples = 8192, n_features = 512;
    float sparsity = 0.6;
    auto full_p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                           .Batches(4)
                           .Bins(param.max_bin)
                           .Device(ctx.Device())
                           .HwDecompRatio(0.0)
                           .OnHost(true)
                           .MinPageCacheBytes(n_samples * n_features / 4)
                           .CacheHostRatio(0.8)
                           .GenerateExtMemQuantileDMatrix("temp", false);

    auto comp_p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                           .Batches(4)
                           .Bins(param.max_bin)
                           .Device(ctx.Device())
                           .HwDecompRatio(hw_decomp_ratio)
                           .OnHost(true)
                           .MinPageCacheBytes(n_samples * n_features / 4)
                           .CacheHostRatio(0.8)
                           .GenerateExtMemQuantileDMatrix("temp", false);

    auto get_pages = [&](std::shared_ptr<DMatrix> p_fmat) {
      std::vector<std::shared_ptr<EllpackPage const>> pages;
      auto it = p_fmat->GetBatches<EllpackPage>(&ctx, param).begin();
      while (!it.AtEnd()) {
        auto page = it.Page();
        EXPECT_FALSE(page->Impl()->IsDenseCompressed());
        pages.emplace_back(std::move(page));
        ++it;
      }
      return pages;
    };

    std::vector<std::shared_ptr<EllpackPage const>> full_pages = get_pages(full_p_fmat);
    std::vector<std::shared_ptr<EllpackPage const>> comp_pages = get_pages(comp_p_fmat);

    ASSERT_EQ(full_pages.size(), comp_pages.size());
    for (std::size_t i = 0, n = full_pages.size(); i < n; ++i) {
      auto impl_f = full_pages[i]->Impl();
      auto impl_c = comp_pages[i]->Impl();
      ASSERT_EQ(impl_f->gidx_buffer.size(), impl_c->gidx_buffer.size());
      ASSERT_EQ(impl_f->d_gidx_buffer.size(), impl_c->d_gidx_buffer.size());
      ASSERT_EQ(impl_f->NumNonMissing(&ctx, {}), impl_c->NumNonMissing(&ctx, {}));

      std::vector<common::CompressedByteT> buf_f;
      [[maybe_unused]] auto acc_f = impl_f->GetHostEllpack(&ctx, &buf_f);

      std::vector<common::CompressedByteT> buf_c;
      [[maybe_unused]] auto acc_c = impl_c->GetHostEllpack(&ctx, &buf_c);

      ASSERT_EQ(buf_f.size(), buf_c.size());
      for (std::size_t i = 0, m = buf_f.size(); i < m; ++i) {
        ASSERT_EQ(buf_f[i], buf_c[i]) << i;
      }
    }
  }
};

TEST_P(EllpackDecompTest, Basic) {
  auto ctx = MakeCUDACtx(0);
  auto hw_decomp_ratio = this->GetParam();
  this->Run(hw_decomp_ratio);
}

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, EllpackDecompTest,
                         ::testing::Values(1.0f, 0.1f, 0.5f, 0.0f));
}  // namespace xgboost::data
