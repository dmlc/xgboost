#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <fstream>
#include <memory>
#include <vector>

#include "xgboost/data.h"
#include "../helpers.h"

namespace xgboost {
TEST(SparsePage, PushCSC) {
  std::vector<bst_row_t> offset {0};
  std::vector<Entry> data;
  SparsePage batch;
  batch.offset.HostVector() = offset;
  batch.data.HostVector() = data;

  offset = {0, 1, 4};
  for (size_t i = 0; i < offset.back(); ++i) {
    data.emplace_back(Entry(i, 0.1f));
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
    SparsePage tmp = batch.GetTranspose(ncols);
    page.PushCSC(tmp);
  }

  // Make sure that the final sparse page has the right number of entries
  ASSERT_EQ(kEntries, page.data.Size());

  page.SortRows();
  auto v = page.GetView();
  for (size_t i = 0; i < v.Size(); ++i) {
    auto column = v[i];
    for (size_t j = 1; j < column.size(); ++j) {
      ASSERT_GE(column[j].fvalue, column[j-1].fvalue);
    }
  }
}

TEST(DMatrix, Uri) {
  size_t constexpr kRows {16};
  size_t constexpr kCols {8};
  std::vector<float> data (kRows * kCols);

  for (size_t i = 0; i < kRows * kCols; ++i) {
    data[i] = i;
  }

  dmlc::TemporaryDirectory tmpdir;
  std::string path = tmpdir.path + "/small.csv";

  std::ofstream fout(path);
  size_t i = 0;
  for (size_t r = 0; r < kRows; ++r) {
    for (size_t c = 0; c < kCols; ++c) {
      fout << data[i];
      i++;
      if (c != kCols - 1) {
        fout << ",";
      }
    }
    fout << "\n";
  }
  fout.flush();
  fout.close();

  std::unique_ptr<DMatrix> dmat;
  // FIXME(trivialfis): Enable the following test by restricting csv parser in dmlc-core.
  // EXPECT_THROW(dmat.reset(DMatrix::Load(path, false, true)), dmlc::Error);

  std::string uri = path + "?format=csv";
  dmat.reset(DMatrix::Load(uri, false, true));

  ASSERT_EQ(dmat->Info().num_col_, kCols);
  ASSERT_EQ(dmat->Info().num_row_, kRows);
}
}  // namespace xgboost
