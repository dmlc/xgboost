/**
 * Copyright 2018-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>     // for bst_bin_t
#include <xgboost/context.h>  // for Context
#include <xgboost/data.h>     // for BatchIterator, BatchSet, DMatrix, Met...

#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t, uint16_t, uint8_t
#include <limits>       // for numeric_limits
#include <memory>       // for shared_ptr, __shared_ptr_access, allo...
#include <type_traits>  // for remove_reference_t

#include "../../../src/common/column_matrix.h"      // for ColumnMatrix, Column, DenseColumnIter
#include "../../../src/common/hist_util.h"          // for DispatchBinType, BinTypeSize, Index
#include "../../../src/common/ref_resource_view.h"  // for RefResourceView
#include "../../../src/data/gradient_index.h"       // for GHistIndexMatrix
#include "../../../src/data/iterative_dmatrix.h"    // for IterativeDMatrix
#include "../../../src/tree/param.h"                // for TrainParam
#include "../helpers.h"                             // for RandomDataGenerator, NumpyArrayIterFo...

namespace xgboost::common {
TEST(ColumnMatrix, Basic) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  Context ctx;
  BinTypeSize last{kUint8BinsTypeSize};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 10, 0.0).GenerateDMatrix();
    auto sparse_thresh = 0.2;
    GHistIndexMatrix gmat{&ctx, dmat.get(), max_num_bin, sparse_thresh, false};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, sparse_thresh, ctx.Threads());
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

TEST(ColumnMatrix, SparseColumn) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  Context ctx;
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.85).GenerateDMatrix();
    GHistIndexMatrix gmat{&ctx, dmat.get(), max_num_bin, 0.5f, false};
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, 1.0, ctx.Threads());
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
    if (col.IsMissing(i)) {
      continue;
    }
    EXPECT_EQ(gmat.index[gmat.row_ptr[i]], col.GetGlobalBinIdx(i));
  }
}

TEST(ColumnMatrix, DenseColumnWithMissing) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  Context ctx;
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.5).GenerateDMatrix();
    GHistIndexMatrix gmat(&ctx, dmat.get(), max_num_bin, 0.2, false);
    ColumnMatrix column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.InitFromSparse(page, gmat, 0.2, ctx.Threads());
    }
    ASSERT_TRUE(column_matrix.AnyMissing());
    DispatchBinType(column_matrix.GetTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto col = column_matrix.DenseColumn<T, true>(0);
      CheckColumWithMissingValue(col, gmat);
    });
  }
}

TEST(ColumnMatrix, GrowMissing) {
  float sparsity = 0.5;
  NumpyArrayIterForTest iter(sparsity);
  auto n_threads = 0;
  bst_bin_t n_bins = 16;
  BatchParam batch{n_bins, tree::TrainParam::DftSparseThreshold()};
  Context ctx;
  auto m = std::make_shared<data::IterativeDMatrix>(
      &iter, iter.Proxy(), nullptr, Reset, Next, std::numeric_limits<float>::quiet_NaN(), n_threads,
      n_bins, std::numeric_limits<std::int64_t>::max());
  for (auto const& page : m->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
    auto const& column_matrix = page.Transpose();
    auto const& missing = column_matrix.Missing();
    auto n = NumpyArrayIterForTest::Rows() * NumpyArrayIterForTest::Cols();
    auto expected = std::remove_reference_t<decltype(missing)>::BitFieldT::ComputeStorageSize(n);
    auto got = missing.storage.size();
    ASSERT_EQ(expected, got);
    DispatchBinType(column_matrix.GetTypeSize(), [&](auto dtype) {
      using T = decltype(dtype);
      auto col = column_matrix.DenseColumn<T, true>(0);
      CheckColumWithMissingValue(col, page);
    });
  }
}
}  // namespace xgboost::common
