#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../helpers.h"


namespace xgboost {
namespace common {

TEST(DenseColumn, Test) {
  auto dmat = CreateDMatrix(100, 10, 0.0);
  GHistIndexMatrix gmat;
  gmat.Init((*dmat).get(), 256);
  ColumnMatrix column_matrix;
  column_matrix.Init(gmat, 0.2);

  for (auto i = 0ull; i < (*dmat)->Info().num_row_; i++) {
    for (auto j = 0ull; j < (*dmat)->Info().num_col_; j++) {
        auto col = column_matrix.GetColumn(j);
        ASSERT_EQ(gmat.index[i * (*dmat)->Info().num_col_ + j],
                  col.GetGlobalBinIdx(i));
    }
  }
  delete dmat;
}

TEST(SparseColumn, Test) {
  auto dmat = CreateDMatrix(100, 1, 0.85);
  GHistIndexMatrix gmat;
  gmat.Init((*dmat).get(), 256);
  ColumnMatrix column_matrix;
  column_matrix.Init(gmat, 0.5);
  auto col = column_matrix.GetColumn(0);
  ASSERT_EQ(col.Size(), gmat.index.size());
  for (auto i = 0ull; i < col.Size(); i++) {
    ASSERT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]],
              col.GetGlobalBinIdx(i));
  }
  delete dmat;
}

TEST(DenseColumnWithMissing, Test) {
  auto dmat = CreateDMatrix(100, 1, 0.5);
  GHistIndexMatrix gmat;
  gmat.Init((*dmat).get(), 256);
  ColumnMatrix column_matrix;
  column_matrix.Init(gmat, 0.2);
  auto col = column_matrix.GetColumn(0);
  for (auto i = 0ull; i < col.Size(); i++) {
    if (col.IsMissing(i)) continue;
    EXPECT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]],
              col.GetGlobalBinIdx(i));
  }
  delete dmat;
}

void TestGHistIndexMatrixCreation(size_t nthreads) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  /* This should create multiple sparse pages */
  std::unique_ptr<DMatrix> dmat{ CreateSparsePageDMatrix(1024, 1024, filename) };
  omp_set_num_threads(nthreads);
  GHistIndexMatrix gmat;
  gmat.Init(dmat.get(), 256);
}

TEST(HistIndexCreationWithExternalMemory, Test) {
  // Vary the number of threads to make sure that the last batch
  // is distributed properly to the available number of threads
  // in the thread pool
  TestGHistIndexMatrixCreation(20);
  TestGHistIndexMatrixCreation(30);
  TestGHistIndexMatrixCreation(40);
}
}  // namespace common
}  // namespace xgboost
