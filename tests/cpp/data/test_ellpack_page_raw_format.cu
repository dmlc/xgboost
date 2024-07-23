/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../../../src/data/ellpack_page.cuh"           // for EllpackPage
#include "../../../src/data/ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "../../../src/data/ellpack_page_source.h"      // for EllpackFormatStreamPolicy
#include "../../../src/tree/param.h"                    // for TrainParam
#include "../filesystem.h"                              // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost::data {
namespace {
template <typename FormatStreamPolicy>
void TestEllpackPageRawFormat(FormatStreamPolicy *p_policy) {
  auto &policy = *p_policy;
  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};

  auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/ellpack.page";

  std::shared_ptr<common::HistogramCuts const> cuts;
  for (auto const &page : m->GetBatches<EllpackPage>(&ctx, param)) {
    cuts = page.Impl()->CutsShared();
  }

  ASSERT_EQ(cuts->cut_values_.Device(), ctx.Device());
  ASSERT_TRUE(cuts->cut_values_.DeviceCanRead());
  policy.SetCuts(cuts, ctx.Device());

  std::unique_ptr<EllpackPageRawFormat> format{policy.CreatePageFormat()};

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
    ASSERT_EQ(loaded->row_stride, orig->row_stride);
    std::vector<common::CompressedByteT> h_loaded, h_orig;
    [[maybe_unused]] auto h_loaded_acc = loaded->GetHostAccessor(&ctx, &h_loaded);
    [[maybe_unused]] auto h_orig_acc = orig->GetHostAccessor(&ctx, &h_orig);
    ASSERT_EQ(h_loaded, h_orig);
  }
}
}  // anonymous namespace

TEST(EllpackPageRawFormat, DiskIO) {
  EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy> policy{false};
  TestEllpackPageRawFormat(&policy);
}

TEST(EllpackPageRawFormat, DiskIOHmm) {
  if (common::SupportsPageableMem()) {
    EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy> policy{true};
    TestEllpackPageRawFormat(&policy);
  } else {
    GTEST_SKIP_("HMM is not supported.");
  }
}

TEST(EllpackPageRawFormat, HostIO) {
  EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy> policy;
  TestEllpackPageRawFormat(&policy);
}
}  // namespace xgboost::data
