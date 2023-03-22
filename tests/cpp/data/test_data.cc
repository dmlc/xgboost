/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <vector>

#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "xgboost/data.h"

namespace xgboost {
TEST(SparsePage, PushCSC) {
  std::vector<bst_row_t> offset {0};
  std::vector<Entry> data;
  SparsePage batch;
  batch.offset.HostVector() = offset;
  batch.data.HostVector() = data;

  offset = {0, 1, 4};
  for (size_t i = 0; i < offset.back(); ++i) {
    data.emplace_back(i, 0.1f);
  }

  SparsePage other;
  other.offset.HostVector() = offset;
  other.data.HostVector() = data;

  batch.PushCSC(other);

  ASSERT_EQ(batch.offset.HostVector().size(), offset.size());
  ASSERT_EQ(batch.data.HostVector().size(), data.size());
  for (size_t i = 0; i < offset.size(); ++i) {
    ASSERT_EQ(batch.offset.HostVector()[i], offset[i]);
  }
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(batch.data.HostVector()[i].index, data[i].index);
  }

  batch.PushCSC(other);
  ASSERT_EQ(batch.offset.HostVector().size(), offset.size());
  ASSERT_EQ(batch.data.Size(), data.size() * 2);

  for (size_t i = 0; i < offset.size(); ++i) {
    ASSERT_EQ(batch.offset.HostVector()[i], offset[i] * 2);
  }

  auto page = batch.GetView();
  auto inst = page[0];
  ASSERT_EQ(inst.size(), 2ul);
  for (auto entry : inst) {
    ASSERT_EQ(entry.index, 0u);
  }

  inst = page[1];
  ASSERT_EQ(inst.size(), 6ul);
  std::vector<size_t> indices_sol {1, 2, 3};
  for (size_t i = 0; i < inst.size(); ++i) {
    ASSERT_EQ(inst[i].index, indices_sol[i % 3]);
  }
}

TEST(SparsePage, PushCSCAfterTranspose) {
  size_t constexpr kPageSize = 1024, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(kEntries);
  const int ncols = dmat->Info().num_col_;
  SparsePage page; // Consolidated sparse page
  for (const auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    // Transpose each batch and push
    SparsePage tmp = batch.GetTranspose(ncols, AllThreadsForTest());
    page.PushCSC(tmp);
  }

  // Make sure that the final sparse page has the right number of entries
  ASSERT_EQ(kEntries, page.data.Size());

  page.SortRows(AllThreadsForTest());
  auto v = page.GetView();
  for (size_t i = 0; i < v.Size(); ++i) {
    auto column = v[i];
    for (size_t j = 1; j < column.size(); ++j) {
      ASSERT_GE(column[j].fvalue, column[j-1].fvalue);
    }
  }
}

TEST(SparsePage, SortIndices) {
  auto p_fmat = RandomDataGenerator{100, 10, 0.6}.GenerateDMatrix();
  auto n_threads = AllThreadsForTest();
  SparsePage copy;
  for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
    ASSERT_TRUE(page.IsIndicesSorted(n_threads));
    copy.Push(page);
  }
  ASSERT_TRUE(copy.IsIndicesSorted(n_threads));

  for (size_t ridx = 0; ridx < copy.Size(); ++ridx) {
    auto beg = copy.offset.HostVector()[ridx];
    auto end = copy.offset.HostVector()[ridx + 1];
    auto& h_data = copy.data.HostVector();
    if (end - beg >= 2) {
      std::swap(h_data[beg], h_data[end - 1]);
    }
  }
  ASSERT_FALSE(copy.IsIndicesSorted(n_threads));

  copy.SortIndices(n_threads);
  ASSERT_TRUE(copy.IsIndicesSorted(n_threads));
}

TEST(DMatrix, Uri) {
  auto constexpr kRows {16};
  auto constexpr kCols {8};

  dmlc::TemporaryDirectory tmpdir;
  auto const path = tmpdir.path + "/small.csv";
  CreateTestCSV(path, kRows, kCols);

  std::unique_ptr<DMatrix> dmat;
  // FIXME(trivialfis): Enable the following test by restricting csv parser in dmlc-core.
  // EXPECT_THROW(dmat.reset(DMatrix::Load(path, false, true)), dmlc::Error);

  std::string uri = path + "?format=csv";
  dmat.reset(DMatrix::Load(uri, false));

  ASSERT_EQ(dmat->Info().num_col_, kCols);
  ASSERT_EQ(dmat->Info().num_row_, kRows);
}
}  // namespace xgboost
