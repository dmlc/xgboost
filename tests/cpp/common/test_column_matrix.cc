#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../helpers.h"


namespace xgboost {
namespace common {

TEST(DenseColumn, Test) {
  uint64_t max_num_bins[] = {static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (size_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 10, 0.0).GenerateDMatrix();
    GHistIndexMatrix gmat;
    gmat.Init(dmat.get(), max_num_bin);
    ColumnMatrix column_matrix;
    column_matrix.Init(gmat, 0.2);

    for (auto i = 0ull; i < dmat->Info().num_row_; i++) {
      for (auto j = 0ull; j < dmat->Info().num_col_; j++) {
          switch (column_matrix.GetTypeSize()) {
            case kUint8BinsTypeSize: {
                auto col = column_matrix.GetColumn<uint8_t>(j);
                ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j],
                          (*col.get()).GetGlobalBinIdx(i));
              }
              break;
            case kUint16BinsTypeSize: {
                auto col = column_matrix.GetColumn<uint16_t>(j);
                ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j],
                          (*col.get()).GetGlobalBinIdx(i));
              }
              break;
            case kUint32BinsTypeSize: {
                auto col = column_matrix.GetColumn<uint32_t>(j);
                ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j],
                          (*col.get()).GetGlobalBinIdx(i));
              }
              break;
        }
      }
    }
  }
}

template<typename BinIdxType>
inline void CheckSparseColumn(const Column<BinIdxType>& col_input, const GHistIndexMatrix& gmat) {
  const SparseColumn<BinIdxType>& col = static_cast<const SparseColumn<BinIdxType>& >(col_input);
  ASSERT_EQ(col.Size(), gmat.index.Size());
  for (auto i = 0ull; i < col.Size(); i++) {
    ASSERT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(SparseColumn, Test) {
  uint64_t max_num_bins[] = {static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (size_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.85).GenerateDMatrix();
    GHistIndexMatrix gmat;
    gmat.Init(dmat.get(), max_num_bin);
    ColumnMatrix column_matrix;
    column_matrix.Init(gmat, 0.5);
    switch (column_matrix.GetTypeSize()) {
      case kUint8BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint8_t>(0);
          CheckSparseColumn(*col.get(), gmat);
        }
        break;
      case kUint16BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint16_t>(0);
          CheckSparseColumn(*col.get(), gmat);
        }
        break;
      case kUint32BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint32_t>(0);
          CheckSparseColumn(*col.get(), gmat);
        }
        break;
    }
  }
}

template<typename BinIdxType>
inline void CheckColumWithMissingValue(const Column<BinIdxType>& col_input,
                                       const GHistIndexMatrix& gmat) {
  const DenseColumn<BinIdxType>& col = static_cast<const DenseColumn<BinIdxType>& >(col_input);
  for (auto i = 0ull; i < col.Size(); i++) {
    if (col.IsMissing(i)) continue;
    EXPECT_EQ(gmat.index[gmat.row_ptr[i]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(DenseColumnWithMissing, Test) {
  uint64_t max_num_bins[] = { static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                              static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                              static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2 };
  for (size_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.5).GenerateDMatrix();
    GHistIndexMatrix gmat;
    gmat.Init(dmat.get(), max_num_bin);
    ColumnMatrix column_matrix;
    column_matrix.Init(gmat, 0.2);
    switch (column_matrix.GetTypeSize()) {
      case kUint8BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint8_t>(0);
          CheckColumWithMissingValue(*col.get(), gmat);
        }
        break;
      case kUint16BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint16_t>(0);
          CheckColumWithMissingValue(*col.get(), gmat);
        }
        break;
      case kUint32BinsTypeSize: {
          auto col = column_matrix.GetColumn<uint32_t>(0);
          CheckColumWithMissingValue(*col.get(), gmat);
        }
        break;
    }
  }
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
