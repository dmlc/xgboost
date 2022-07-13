/*!
 * Copyright 2018-2022 by XGBoost Contributors
 */
#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../helpers.h"


namespace xgboost {
namespace common {

TEST(DenseColumn, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 10, 0.0).GenerateDMatrix();
    auto sparse_thresh = 0.2;
    GHistIndexMatrix gmat{dmat.get(), max_num_bin, sparse_thresh, false,
                          common::OmpGetNumThreads(0)};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, sparse_thresh, common::OmpGetNumThreads(0));
    }

    const auto& column_list = column_matrix.GetColumnList();
    for (auto i = 0ull; i < dmat->Info().num_row_; i++) {
      for (auto j = 0ull; j < dmat->Info().num_col_; j++) {
        const auto& col = column_list[j];
        ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j],
                  (*col.get()).GetGlobalBinIdx(i));
      }
    }
  }
}

inline void CheckSparseColumn(const Column& col_input, const GHistIndexMatrix& gmat) {
  const SparseColumn& col = static_cast<const SparseColumn& >(col_input);
  ASSERT_EQ(col.Size(), gmat.index.Size());
  for (auto i = 0ull; i < col.Size(); i++) {
    ASSERT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(SparseColumn, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.85).GenerateDMatrix();
    GHistIndexMatrix gmat{dmat.get(), max_num_bin, 0.5f, false, common::OmpGetNumThreads(0)};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, 1.0, common::OmpGetNumThreads(0));
    }
    const auto& column_list = column_matrix.GetColumnList();
    const auto& col = column_list[0];
    CheckSparseColumn(*col.get(), gmat);
  }
}

inline void CheckColumWithMissingValue(const Column& col_input,
                                       const GHistIndexMatrix& gmat) {
  const DenseColumn& col = static_cast<const DenseColumn& >(col_input);
  for (auto i = 0ull; i < col.Size(); i++) {
    if (col.IsMissing(i)) continue;
    EXPECT_EQ(gmat.index[gmat.row_ptr[i]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(DenseColumnWithMissing, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.5).GenerateDMatrix();
    GHistIndexMatrix gmat(dmat.get(), max_num_bin, 0.2, false, common::OmpGetNumThreads(0));
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, 0.2, common::OmpGetNumThreads(0));
    }
    const auto& column_list = column_matrix.GetColumnList();
    const auto& col = column_list[0];
    CheckColumWithMissingValue(*col.get(), gmat);
  }
}
}  // namespace common
}  // namespace xgboost