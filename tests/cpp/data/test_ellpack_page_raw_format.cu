/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../../../src/data/ellpack_page.cuh"           // for EllpackPage, GetRowStride
#include "../../../src/data/ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "../../../src/data/ellpack_page_source.h"      // for EllpackFormatStreamPolicy
#include "../../../src/tree/param.h"                    // for TrainParam
#include "../filesystem.h"                              // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost::data {
namespace {
[[nodiscard]] EllpackCacheInfo CInfoForTest(Context const *ctx, DMatrix *Xy, bst_idx_t row_stride,
                                            BatchParam param,
                                            std::shared_ptr<common::HistogramCuts const> cuts) {
  EllpackCacheInfo cinfo{param, false, 1, std::numeric_limits<float>::quiet_NaN()};
  ExternalDataInfo ext_info;
  ext_info.n_batches = 1;
  ext_info.row_stride = row_stride;
  ext_info.base_rowids.push_back(Xy->Info().num_row_);

  CalcCacheMapping(ctx, Xy->IsDense(), cuts, 0, ext_info, &cinfo);
  CHECK_EQ(ext_info.n_batches, cinfo.cache_mapping.size());
  if (cinfo.NumBatchesCc() == 1) {
    EXPECT_TRUE(cinfo.prefer_device);
    cinfo.prefer_device = false;  // We test the host cache.
  }
  return cinfo;
}

class TestEllpackPageRawFormat : public ::testing::TestWithParam<bool> {
 public:
  template <typename FormatStreamPolicy>
  void Run(FormatStreamPolicy *p_policy, bool prefetch_copy) {
    auto &policy = *p_policy;
    auto ctx = MakeCUDACtx(0);
    auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
    param.prefetch_copy = prefetch_copy;

    auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
    dmlc::TemporaryDirectory tmpdir;
    std::string path = tmpdir.path + "/ellpack.page";

    std::shared_ptr<common::HistogramCuts const> cuts;
    for (auto const &page : m->GetBatches<EllpackPage>(&ctx, param)) {
      cuts = page.Impl()->CutsShared();
    }

    ASSERT_EQ(cuts->cut_values_.Device(), ctx.Device());
    ASSERT_TRUE(cuts->cut_values_.DeviceCanRead());

    auto row_stride = GetRowStride(m.get());
    EllpackCacheInfo cinfo = CInfoForTest(&ctx, m.get(), row_stride, param, cuts);
    policy.SetCuts(cuts, ctx.Device(), cinfo);

    std::unique_ptr<EllpackPageRawFormat> format{policy.CreatePageFormat(param)};

    std::size_t n_bytes{0};
    {
      auto fo = policy.CreateWriter(StringView{path}, 0);
      for (auto const &ellpack : m->GetBatches<EllpackPage>(&ctx, param)) {
        n_bytes += format->Write(ellpack, fo.get());
      }
    }

    EllpackPage page;
    auto fi = policy.CreateReader(StringView{path}, static_cast<bst_idx_t>(0), n_bytes);
    ASSERT_TRUE(format->Read(&page, fi.get()));

    for (auto const &ellpack : m->GetBatches<EllpackPage>(&ctx, param)) {
      auto loaded = page.Impl();
      auto orig = ellpack.Impl();
      ASSERT_EQ(loaded->Cuts().Ptrs(), orig->Cuts().Ptrs());
      ASSERT_EQ(loaded->Cuts().MinValues(), orig->Cuts().MinValues());
      ASSERT_EQ(loaded->Cuts().Values(), orig->Cuts().Values());
      ASSERT_EQ(loaded->base_rowid, orig->base_rowid);
      ASSERT_EQ(loaded->info.row_stride, orig->info.row_stride);
      std::vector<common::CompressedByteT> h_loaded, h_orig;
      [[maybe_unused]] auto h_loaded_acc = loaded->GetHostAccessor(&ctx, &h_loaded);
      [[maybe_unused]] auto h_orig_acc = orig->GetHostAccessor(&ctx, &h_orig);
      ASSERT_EQ(h_loaded, h_orig);
    }
  }
};
}  // anonymous namespace

TEST_P(TestEllpackPageRawFormat, DiskIO) {
  EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy> policy{false};
  this->Run(&policy, this->GetParam());
}

TEST_P(TestEllpackPageRawFormat, DiskIOHmm) {
  if (curt::SupportsPageableMem()) {
    EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy> policy{true};
    this->Run(&policy, this->GetParam());
  } else {
    GTEST_SKIP_("HMM is not supported.");
  }
}

TEST_P(TestEllpackPageRawFormat, HostIO) {
  {
    EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy> policy;
    this->Run(&policy, this->GetParam());
  }
  {
    auto ctx = MakeCUDACtx(0);
    auto param = BatchParam{32, tree::TrainParam::DftSparseThreshold()};
    param.n_prefetch_batches = 1;
    param.prefetch_copy = this->GetParam();

    EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy> policy;
    std::unique_ptr<EllpackPageRawFormat> format{};
    Cache cache{false, "name", "ellpack", true};
    for (std::size_t i = 0; i < 3; ++i) {
      auto p_fmat = RandomDataGenerator{100, 14, 0.5}.Seed(i).GenerateDMatrix();
      for (auto const &page : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
        if (!format) {
          // Prepare the mapping info.
          EllpackCacheInfo cinfo{param, false, 1, std::numeric_limits<float>::quiet_NaN()};
          for (std::size_t i = 0; i < 3; ++i) {
            cinfo.cache_mapping.push_back(i);
            cinfo.buffer_bytes.push_back(page.Impl()->MemCostBytes());
            cinfo.buffer_rows.push_back(page.Impl()->n_rows);
          }
          policy.SetCuts(page.Impl()->CutsShared(), ctx.Device(), std::move(cinfo));
          format = policy.CreatePageFormat(param);
        }
        auto writer = policy.CreateWriter({}, i);
        auto n_bytes = format->Write(page, writer.get());
        ASSERT_EQ(n_bytes, page.Impl()->MemCostBytes());
        cache.Push(n_bytes);
      }
    }
    cache.Commit();

    for (std::size_t i = 0; i < 3; ++i) {
      auto reader = policy.CreateReader({}, cache.offset[i], cache.Bytes(i));
      EllpackPage page;
      ASSERT_TRUE(format->Read(&page, reader.get()));
      ASSERT_EQ(page.Impl()->MemCostBytes(), cache.Bytes(i));
      auto p_fmat = RandomDataGenerator{100, 14, 0.5}.Seed(i).GenerateDMatrix();
      for (auto const &orig : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
        std::vector<common::CompressedByteT> h_orig;
        auto h_acc_orig = orig.Impl()->GetHostAccessor(&ctx, &h_orig, {});
        std::vector<common::CompressedByteT> h_page;
        auto h_acc = page.Impl()->GetHostAccessor(&ctx, &h_page, {});
        ASSERT_EQ(h_orig, h_page);
        ASSERT_EQ(h_acc_orig.NumFeatures(), h_acc.NumFeatures());
        ASSERT_EQ(h_acc_orig.row_stride, h_acc.row_stride);
        ASSERT_EQ(h_acc_orig.n_rows, h_acc.n_rows);
        ASSERT_EQ(h_acc_orig.base_rowid, h_acc.base_rowid);
        ASSERT_EQ(h_acc_orig.IsDenseCompressed(), h_acc.IsDenseCompressed());
        ASSERT_EQ(h_acc_orig.NullValue(), h_acc.NullValue());
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(EllpackPageRawFormat, TestEllpackPageRawFormat, ::testing::Bool());

TEST(EllpackPageRawFormat, DevicePageConcat) {
  auto ctx = MakeCUDACtx(0);
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  bst_idx_t n_features = 16, n_samples = 128;

  auto test = [&](std::int32_t max_num_device_pages, std::int64_t min_cache_page_bytes) {
    EllpackCacheInfo cinfo{param, true, max_num_device_pages,
                           std::numeric_limits<float>::quiet_NaN()};
    ExternalDataInfo ext_info;

    ext_info.n_batches = 8;
    ext_info.row_stride = n_features;
    for (bst_idx_t i = 0; i < ext_info.n_batches; ++i) {
      ext_info.base_rowids.push_back(n_samples);
    }
    std::partial_sum(ext_info.base_rowids.cbegin(), ext_info.base_rowids.cend(),
                     ext_info.base_rowids.begin());
    ext_info.accumulated_rows = n_samples * ext_info.n_batches;
    ext_info.nnz = ext_info.accumulated_rows * n_features;

    auto p_fmat = RandomDataGenerator{n_samples, n_features, 0}.Seed(0).GenerateDMatrix();
    EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy> policy;

    for (auto const &page : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
      auto cuts = page.Impl()->CutsShared();
      CalcCacheMapping(&ctx, true, cuts, min_cache_page_bytes, ext_info, &cinfo);
      [&] {
        ASSERT_EQ(cinfo.buffer_rows.size(), 4ul);
      }();
      policy.SetCuts(page.Impl()->CutsShared(), ctx.Device(), std::move(cinfo));
    }

    auto format = policy.CreatePageFormat(param);

    // write multipe pages
    for (bst_idx_t i = 0; i < ext_info.n_batches; ++i) {
      for (auto const &page : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
        auto writer = policy.CreateWriter({}, i);
        [[maybe_unused]] auto n_bytes = format->Write(page, writer.get());
      }
    }
    // check correct concatenation.
    auto mem_cache = policy.Share();
    return mem_cache;
  };

  {
    auto mem_cache = test(1, n_features * n_samples);
    ASSERT_EQ(mem_cache->on_device.size(), 4);
    ASSERT_TRUE(mem_cache->on_device[0]);
    ASSERT_EQ(mem_cache->NumDevicePages(), 1);
  }
  {
    auto mem_cache = test(2, n_features * n_samples);
    ASSERT_EQ(mem_cache->on_device.size(), 4);
    ASSERT_TRUE(mem_cache->on_device[0]);
    ASSERT_TRUE(mem_cache->on_device[1]);
    ASSERT_EQ(mem_cache->NumDevicePages(), 2);
  }
}
}  // namespace xgboost::data
