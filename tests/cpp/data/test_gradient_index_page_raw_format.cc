/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr

#include "../../../src/common/column_matrix.h"  // for common::ColumnMatrix
#include "../../../src/common/io.h"             // for MmapResource, AlignedResourceReadStream...
#include "../../../src/data/gradient_index.h"   // for GHistIndexMatrix
#include "../../../src/data/gradient_index_format.h"  // for GHistIndexRawFormat
#include "../helpers.h"                               // for RandomDataGenerator

namespace xgboost::data {
TEST(GHistIndexPageRawFormat, IO) {
  Context ctx;

  auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/ghistindex.page";
  auto batch = BatchParam{256, 0.5};

  common::HistogramCuts cuts;
  for (auto const &index : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
    cuts = index.Cuts();
    break;
  }
  auto format = std::make_unique<GHistIndexRawFormat>(std::move(cuts));

  std::size_t bytes{0};
  {
    auto fo = std::make_unique<common::AlignedFileWriteStream>(StringView{path}, "wb");
    for (auto const &index : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      bytes += format->Write(index, fo.get());
    }
  }

  GHistIndexMatrix page;

  std::unique_ptr<common::AlignedResourceReadStream> fi{
      std::make_unique<common::PrivateMmapConstStream>(path, 0, bytes)};
  ASSERT_TRUE(format->Read(&page, fi.get()));

  for (auto const &gidx : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
    auto const &loaded = gidx;
    ASSERT_EQ(loaded.cut.Ptrs(), page.cut.Ptrs());
    ASSERT_EQ(loaded.cut.MinValues(), page.cut.MinValues());
    ASSERT_EQ(loaded.cut.Values(), page.cut.Values());
    ASSERT_EQ(loaded.base_rowid, page.base_rowid);
    ASSERT_EQ(loaded.row_ptr.size(), page.row_ptr.size());
    ASSERT_TRUE(std::equal(loaded.row_ptr.cbegin(), loaded.row_ptr.cend(), page.row_ptr.cbegin()));
    ASSERT_EQ(loaded.IsDense(), page.IsDense());
    ASSERT_TRUE(std::equal(loaded.index.begin(), loaded.index.end(), page.index.begin()));
    ASSERT_TRUE(std::equal(loaded.index.Offset(), loaded.index.Offset() + loaded.index.OffsetSize(),
                           page.index.Offset()));

    ASSERT_EQ(loaded.Transpose().GetTypeSize(), loaded.Transpose().GetTypeSize());
  }
}
}  // namespace xgboost::data
