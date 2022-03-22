/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/data.h>

#include "../../../src/data/sparse_page_source.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
template <typename S> void TestSparsePageRawFormat() {
  std::unique_ptr<SparsePageFormat<S>> format{CreatePageFormat<S>("raw")};

  auto m = RandomDataGenerator{100, 14, 0.5}.GenerateDMatrix();
  ASSERT_TRUE(m->SingleColBlock());
  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/sparse.page";
  S orig;
  {
    // block code to flush the stream
    std::unique_ptr<dmlc::Stream> fo{dmlc::Stream::Create(path.c_str(), "w")};
    for (auto const &page : m->GetBatches<S>()) {
      orig.Push(page);
      format->Write(page, fo.get());
    }
  }

  S page;
  std::unique_ptr<dmlc::SeekStream> fi{dmlc::SeekStream::CreateForRead(path.c_str())};
  format->Read(&page, fi.get());
  for (size_t i = 0; i < orig.data.Size(); ++i) {
    ASSERT_EQ(page.data.HostVector()[i].fvalue,
              orig.data.HostVector()[i].fvalue);
    ASSERT_EQ(page.data.HostVector()[i].index, orig.data.HostVector()[i].index);
  }
  for (size_t i = 0; i < orig.offset.Size(); ++i) {
    ASSERT_EQ(page.offset.HostVector()[i], orig.offset.HostVector()[i]);
  }
  ASSERT_EQ(page.base_rowid, orig.base_rowid);
}

TEST(SparsePageRawFormat, SparsePage) {
  TestSparsePageRawFormat<SparsePage>();
}

TEST(SparsePageRawFormat, CSCPage) {
  TestSparsePageRawFormat<CSCPage>();
}

TEST(SparsePageRawFormat, SortedCSCPage) {
  TestSparsePageRawFormat<SortedCSCPage>();
}
}  // namespace data
}  // namespace xgboost
