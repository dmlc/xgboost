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
      auto cuts = impl_s->CutsShared();
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, cuts, sparsity == 0.0,
                                                        impl_s->info.row_stride, impl_s->n_rows);
      new_impl->CopyInfo(impl_s);
      bst_idx_t offset = 0;
      for (auto const& page_m : p_ext_fmat->GetBatches<EllpackPage>(&ctx, param)) {
        auto impl_m = page_m.Impl();
        auto cuts_m = page_m.Impl()->CutsShared();
        ASSERT_EQ(cuts->min_vals_.ConstHostVector(), cuts_m->min_vals_.ConstHostVector());
        ASSERT_EQ(cuts->cut_values_.ConstHostVector(), cuts_m->cut_values_.ConstHostVector());
        ASSERT_EQ(cuts->cut_ptrs_.ConstHostVector(), cuts_m->cut_ptrs_.ConstHostVector());
        offset += new_impl->Copy(&ctx, impl_m, offset);
      }
      std::vector<common::CompressedByteT> buffer_s;
      auto acc_s = impl_s->GetHostAccessor(&ctx, &buffer_s, {});
      std::vector<common::CompressedByteT> buffer_m;
      auto acc_m = new_impl->GetHostAccessor(&ctx, &buffer_m, {});
      ASSERT_EQ(acc_m.row_stride * acc_m.n_rows, acc_s.row_stride * acc_s.n_rows);
      for (std::size_t i = 0; i < acc_m.row_stride * acc_m.n_rows; ++i) {
        ASSERT_EQ(acc_s.gidx_iter[i], acc_m.gidx_iter[i]);
      }
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
