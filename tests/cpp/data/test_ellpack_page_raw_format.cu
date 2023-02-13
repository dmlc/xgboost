/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/sparse_page_source.h"
#include "../../../src/tree/param.h"  // TrainParam
#include "../filesystem.h"            // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost {
namespace data {
TEST(EllpackPageRawFormat, IO) {
  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};

  std::unique_ptr<SparsePageFormat<EllpackPage>> format{CreatePageFormat<EllpackPage>("raw")};

  auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/ellpack.page";

  {
    std::unique_ptr<dmlc::Stream> fo{dmlc::Stream::Create(path.c_str(), "w")};
    for (auto const &ellpack : m->GetBatches<EllpackPage>(&ctx, param)) {
      format->Write(ellpack, fo.get());
    }
  }

  EllpackPage page;
  std::unique_ptr<dmlc::SeekStream> fi{dmlc::SeekStream::CreateForRead(path.c_str())};
  format->Read(&page, fi.get());

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
}  // namespace data
}  // namespace xgboost
