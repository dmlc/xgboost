/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/gradient.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../src/common/exact_multinomial/packed_stats.h"

namespace xgboost {
TEST(ExactHessianTransport, DefaultState) {
  ExactHessian hessian;
  EXPECT_TRUE(hessian.Empty());
  EXPECT_EQ(hessian.n_free, 0);
  EXPECT_EQ(hessian.NumRows(), 0);
  EXPECT_EQ(hessian.RowSize(), 0);
  // Default construction must not reserve storage for rows that were never requested.
  EXPECT_EQ(hessian.data.Size(), 0);

  GradientContainer container;
  EXPECT_FALSE(container.HasExactHessian());
  EXPECT_TRUE(container.exact_hessian.Empty());
  // The existing sidecar is unaffected by the new one.
  EXPECT_FALSE(container.HasValueGrad());
}

/** The transport and the packed views must not hold separate copies of the layout math. */
TEST(ExactHessianTransport, LayoutSingleSourceOfTruth) {
  for (std::size_t n_free = 0; n_free < 12; ++n_free) {
    ASSERT_EQ(common::PackedHessianSize(n_free), ExactHessian::PackedSize(n_free))
        << "n_free=" << n_free;
    ASSERT_EQ(ExactHessian::PackedSize(n_free), n_free * (n_free + 1) / 2) << "n_free=" << n_free;
  }
  // (K-1)*K/2 for K classes.
  static_assert(ExactHessian::PackedSize(6) == 21, "K=7 packs into 21 scalars");
}

TEST(ExactHessianTransport, Shape) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    auto n_free = common::NumFreeClasses(n_classes);
    bst_idx_t constexpr kRows = 6;

    ExactHessian hessian;
    hessian.Reshape(kRows, static_cast<bst_target_t>(n_free));

    EXPECT_FALSE(hessian.Empty());
    EXPECT_EQ(hessian.n_free, n_free);
    EXPECT_EQ(hessian.NumRows(), kRows);
    EXPECT_EQ(hessian.RowSize(), common::PackedHessianSize(n_free));
    // (K-1) * K / 2 scalars per row.
    EXPECT_EQ(hessian.RowSize(), static_cast<std::size_t>(n_free) * n_classes / 2);
    EXPECT_EQ(hessian.data.Size(), kRows * hessian.RowSize());

    GradientContainer container;
    container.exact_hessian.Reshape(kRows, static_cast<bst_target_t>(n_free));
    EXPECT_TRUE(container.HasExactHessian());
  }
}

/**
 * Every row must be a slice of one allocation: this is what keeps a per-row Hessian off the
 * heap and lets a future histogram walk rows without chasing pointers.
 */
TEST(ExactHessianTransport, ContiguousStorage) {
  bst_target_t constexpr kNumFree = 4;
  bst_idx_t constexpr kRows = 5;
  auto row_size = common::PackedHessianSize(kNumFree);

  ExactHessian hessian;
  hessian.Reshape(kRows, kNumFree);

  auto values = hessian.HostValues();
  ASSERT_EQ(values.size(), kRows * row_size);
  for (bst_idx_t row = 0; row < kRows; ++row) {
    auto span = hessian.HostRow(row);
    ASSERT_EQ(span.size(), row_size);
    // Adjacent rows, one buffer, no per-row allocation.
    ASSERT_EQ(span.data(), values.data() + row * row_size) << "row " << row;
  }

  ExactHessian const& const_hessian = hessian;
  ASSERT_EQ(const_hessian.HostRow(2).data(), hessian.HostRow(2).data());
  ASSERT_EQ(const_hessian.HostValues().size(), values.size());
}

/**
 * The transport rows and the packed statistics used for accumulation must agree on the
 * layout, so that a later histogram can add a row into a bin without translating indices.
 */
TEST(ExactHessianTransport, SharesPackedLayout) {
  bst_target_t constexpr kNumFree = 3;
  bst_idx_t constexpr kRows = 2;
  ExactHessian transport;
  transport.Reshape(kRows, kNumFree);

  auto row = transport.HostRow(1);
  auto view = common::PackedHessianAtRow(row, kNumFree, 0);
  view.Set(0, 0, 1.0f);
  view.Set(1, 0, 2.0f);
  view.Set(1, 1, 3.0f);
  view.Set(2, 0, 4.0f);
  view.Set(2, 1, 5.0f);
  view.Set(2, 2, 6.0f);

  // Written in packed lower triangle order.
  EXPECT_EQ(row[0], 1.0f);
  EXPECT_EQ(row[1], 2.0f);
  EXPECT_EQ(row[2], 3.0f);
  EXPECT_EQ(row[3], 4.0f);
  EXPECT_EQ(row[4], 5.0f);
  EXPECT_EQ(row[5], 6.0f);

  // Reading through the mirrored index reaches the same scalar.
  EXPECT_EQ(view.Get(0, 1), 2.0f);
  EXPECT_EQ(view.Get(0, 2), 4.0f);
  EXPECT_EQ(view.Get(1, 2), 5.0f);

  // Row 0 was not touched by writes aimed at row 1.
  for (auto v : transport.HostRow(0)) {
    EXPECT_EQ(v, 0.0f);
  }

  // A bin statistic accumulates the transported row with the same indexing.
  std::vector<double> bin(common::PackedStatsStride(kNumFree), 0.0);
  auto stats = common::PackedMultinomialStats<double>{
      common::Span<double>{bin.data(), bin.size()}, kNumFree};
  for (std::size_t i = 0; i < kNumFree; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      stats.AddHessian(i, j, view.Get(i, j));
    }
  }
  EXPECT_EQ(stats.GetHessian(2, 1), 5.0);
  EXPECT_EQ(stats.GetHessian(1, 2), 5.0);
  // The packed triangles are byte compatible element by element.
  auto bin_triangle = stats.Hessian();
  ASSERT_EQ(bin_triangle.size(), row.size());
  for (std::size_t k = 0; k < row.size(); ++k) {
    EXPECT_EQ(bin_triangle[k], static_cast<double>(row[k])) << "entry " << k;
  }
}

/**
 * Lifecycle invariant: the sidecar must never outlive the gradient it describes.
 *
 * GradientContainer persists across boosting rounds. If a round used the exact producer and
 * the next round used the scalar path, a surviving sidecar would advertise statistics for a
 * gradient that had already been overwritten -- and with a different row count if the data
 * changed. Whoever starts a new gradient computation clears it.
 */
TEST(ExactHessianTransport, ClearInvalidatesSidecar) {
  GradientContainer container;
  ASSERT_FALSE(container.HasExactHessian());

  // Round N: the exact path populates the sidecar.
  container.exact_hessian.Reshape(8, 3);
  ASSERT_TRUE(container.HasExactHessian());
  ASSERT_EQ(container.exact_hessian.NumRows(), 8);
  ASSERT_EQ(container.exact_hessian.n_free, 3);

  // Round N+1 begins: whatever the objective does next, the stale sidecar is gone.
  container.ClearExactHessian();
  EXPECT_FALSE(container.HasExactHessian());
  EXPECT_TRUE(container.exact_hessian.Empty());
  EXPECT_EQ(container.exact_hessian.NumRows(), 0);
  EXPECT_EQ(container.exact_hessian.RowSize(), 0);
  EXPECT_EQ(container.exact_hessian.n_free, 0);

  // Clearing twice is harmless, and a cleared container can be refilled with a different
  // shape -- the row-count mismatch that would otherwise corrupt a later consumer.
  container.ClearExactHessian();
  EXPECT_FALSE(container.HasExactHessian());
  container.exact_hessian.Reshape(5, 6);
  EXPECT_TRUE(container.HasExactHessian());
  EXPECT_EQ(container.exact_hessian.NumRows(), 5);
  EXPECT_EQ(container.exact_hessian.RowSize(), common::PackedHessianSize(6));
}

/**
 * The documented transport cost: sizeof(float) * (K-1) * K / 2 per row.
 */
TEST(ExactHessianTransport, MemoryFootprint) {
  bst_target_t constexpr kNumClasses = 7;
  auto n_free = common::NumFreeClasses(kNumClasses);
  ExactHessian hessian;
  hessian.Reshape(1024, static_cast<bst_target_t>(n_free));

  // 21 floats, 84 bytes per row for K = 7.
  EXPECT_EQ(hessian.RowSize(), 21);
  EXPECT_EQ(hessian.RowSize() * sizeof(float), 84);
  EXPECT_EQ(hessian.data.Size() * sizeof(float), 1024 * 84);
}
}  // namespace xgboost
