/**
 * Copyright 2018-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../helpers.h"


namespace xgboost {
namespace common {

TEST(DenseColumn, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  BinTypeSize last{kUint8BinsTypeSize};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 10, 0.0).GenerateDMatrix();
    auto sparse_thresh = 0.2;
    GHistIndexMatrix gmat{&ctx, dmat.get(), max_num_bin, sparse_thresh, false};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, sparse_thresh, AllThreadsForTest());
    }
    ASSERT_GE(column_matrix.GetTypeSize(), last);
    ASSERT_LE(column_matrix.GetTypeSize(), kUint32BinsTypeSize);
    last = column_matrix.GetTypeSize();
    ASSERT_FALSE(column_matrix.AnyMissing());
    for (auto i = 0ull; i < dmat->Info().num_row_; i++) {
      for (auto j = 0ull; j < dmat->Info().num_col_; j++) {
        DispatchBinType(column_matrix.GetTypeSize(), [&](auto dtype) {
          using T = decltype(dtype);
          auto col = column_matrix.DenseColumn<T, false>(j);
          ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j], col.GetGlobalBinIdx(i));
        });
      }
    }
  }
}

template <typename BinIdxType>
void CheckSparseColumn(SparseColumnIter<BinIdxType>* p_col, const GHistIndexMatrix& gmat) {
  auto& col = *p_col;

  size_t n_samples = gmat.row_ptr.size() - 1;
  ASSERT_EQ(col.Size(), gmat.index.Size());
  for (auto i = 0ull; i < col.Size(); i++) {
    ASSERT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]], col.GetGlobalBinIdx(i));
  }

  for (auto i = 0ull; i < n_samples; i++) {
    if (col[i] == Column<BinIdxType>::kMissingId) {
      auto beg = gmat.row_ptr[i];
      auto end = gmat.row_ptr[i + 1];
      ASSERT_EQ(end - beg, 0);
    }
  }
}

TEST(SparseColumn, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.85).GenerateDMatrix();
    GHistIndexMatrix gmat{&ctx, dmat.get(), max_num_bin, 0.5f, false};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, 1.0, AllThreadsForTest());
    }
    common::DispatchBinType(column_matrix.GetTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto col = column_matrix.SparseColumn<T>(0, 0);
      CheckSparseColumn(&col, gmat);
    });
  }
}

template <typename BinIdxType>
void CheckColumWithMissingValue(const DenseColumnIter<BinIdxType, true>& col,
                                const GHistIndexMatrix& gmat) {
  for (auto i = 0ull; i < col.Size(); i++) {
    if (col.IsMissing(i)) continue;
    EXPECT_EQ(gmat.index[gmat.row_ptr[i]], col.GetGlobalBinIdx(i));
  }
}

TEST(DenseColumnWithMissing, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.5).GenerateDMatrix();
    GHistIndexMatrix gmat(&ctx, dmat.get(), max_num_bin, 0.2, false);
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, 0.2, AllThreadsForTest());
    }
    ASSERT_TRUE(column_matrix.AnyMissing());
    DispatchBinType(column_matrix.GetTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto col = column_matrix.DenseColumn<T, true>(0);
      CheckColumWithMissingValue(col, gmat);
    });
  }
}
}  // namespace common
}  // namespace xgboost
