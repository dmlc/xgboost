/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../../../src/data/gradient_index.h"
#include "../../../src/data/sparse_page_source.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
TEST(GHistIndexPageRawFormat, IO) {
  Context ctx;

  std::unique_ptr<SparsePageFormat<GHistIndexMatrix>> format{
      CreatePageFormat<GHistIndexMatrix>("raw")};
  auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/ghistindex.page";
  auto batch = BatchParam{256, 0.5};

  {
    std::unique_ptr<dmlc::Stream> fo{dmlc::Stream::Create(path.c_str(), "w")};
    for (auto const &index : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      format->Write(index, fo.get());
    }
  }

  GHistIndexMatrix page;
  std::unique_ptr<dmlc::SeekStream> fi{dmlc::SeekStream::CreateForRead(path.c_str())};
  format->Read(&page, fi.get());

  for (auto const &gidx : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
    auto const &loaded = gidx;
    ASSERT_EQ(loaded.cut.Ptrs(), page.cut.Ptrs());
    ASSERT_EQ(loaded.cut.MinValues(), page.cut.MinValues());
    ASSERT_EQ(loaded.cut.Values(), page.cut.Values());
    ASSERT_EQ(loaded.base_rowid, page.base_rowid);
    ASSERT_EQ(loaded.IsDense(), page.IsDense());
    ASSERT_TRUE(std::equal(loaded.index.begin(), loaded.index.end(), page.index.begin()));
    ASSERT_TRUE(std::equal(loaded.index.Offset(), loaded.index.Offset() + loaded.index.OffsetSize(),
                           page.index.Offset()));

    ASSERT_EQ(loaded.Transpose().GetTypeSize(), loaded.Transpose().GetTypeSize());
  }
}
}  // namespace data
}  // namespace xgboost
