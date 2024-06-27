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
void TestEllpackPageRawFormat() {
  FormatStreamPolicy policy;

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
    ASSERT_EQ(loaded->gidx_buffer.HostVector(), orig->gidx_buffer.HostVector());
  }
}
}  // anonymous namespace

TEST(EllpackPageRawFormat, DiskIO) {
  TestEllpackPageRawFormat<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>();
}

TEST(EllpackPageRawFormat, HostIO) {
  TestEllpackPageRawFormat<EllpackFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>();
}
}  // namespace xgboost::data
