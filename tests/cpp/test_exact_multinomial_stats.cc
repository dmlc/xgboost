/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for sort
#include <cstddef>    // for size_t
#include <vector>     // for vector

#include "../../src/common/exact_multinomial/packed_stats.h"

namespace xgboost::common {
namespace {
// Back a view with its own storage. The header is deliberately non-owning, the tests own
// the buffer the same way a histogram would.
class StatsStorage {
 public:
  explicit StatsStorage(std::size_t n_free)
      : n_free_{n_free}, buffer_(PackedStatsStride(n_free), 0.0) {}

  [[nodiscard]] PackedMultinomialStats<double> View() {
    return {Span<double>{buffer_.data(), buffer_.size()}, n_free_};
  }

 private:
  std::size_t n_free_;
  std::vector<double> buffer_;
};
}  // anonymous namespace

TEST(ExactMultinomialStats, PackedSize) {
  ASSERT_EQ(NumFreeClasses(3), 2);
  // (K - 1) * K / 2 entries for the lower triangle.
  ASSERT_EQ(PackedHessianSize(NumFreeClasses(3)), 3);
  ASSERT_EQ(PackedHessianSize(NumFreeClasses(4)), 6);
  ASSERT_EQ(PackedHessianSize(NumFreeClasses(10)), 45);
  // Gradient followed by the triangle.
  ASSERT_EQ(PackedStatsStride(2), 5);
  ASSERT_EQ(PackedStatsStride(3), 9);
}

TEST(ExactMultinomialStats, PackedIndex) {
  ASSERT_EQ(PackedHessianIndex(0, 0), 0);
  ASSERT_EQ(PackedHessianIndex(1, 0), 1);
  ASSERT_EQ(PackedHessianIndex(1, 1), 2);
  ASSERT_EQ(PackedHessianIndex(2, 0), 3);
  ASSERT_EQ(PackedHessianIndex(2, 1), 4);
  ASSERT_EQ(PackedHessianIndex(2, 2), 5);
  ASSERT_EQ(PackedHessianIndex(3, 0), 6);

  // The lower triangle must map onto [0, size) exactly once.
  for (std::size_t n_free = 1; n_free < 8; ++n_free) {
    std::vector<std::size_t> seen;
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto idx = PackedHessianIndex(i, j);
        ASSERT_LT(idx, PackedHessianSize(n_free));
        seen.push_back(idx);
      }
    }
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(seen.size(), PackedHessianSize(n_free));
    for (std::size_t k = 0; k < seen.size(); ++k) {
      ASSERT_EQ(seen[k], k);
    }
  }
}

TEST(ExactMultinomialStats, Symmetry) {
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      ASSERT_EQ(PackedHessianIndex(i, j), PackedHessianIndex(j, i));
    }
  }

  StatsStorage storage{4};
  auto stats = storage.View();
  stats.SetHessian(3, 1, 2.5);
  ASSERT_EQ(stats.GetHessian(3, 1), 2.5);
  ASSERT_EQ(stats.GetHessian(1, 3), 2.5);
  // Writing through the mirrored index overwrites the same scalar.
  stats.SetHessian(1, 3, -4.0);
  ASSERT_EQ(stats.GetHessian(3, 1), -4.0);
  stats.AddHessian(1, 3, 1.0);
  ASSERT_EQ(stats.GetHessian(3, 1), -3.0);
}

TEST(ExactMultinomialStats, Add) {
  StatsStorage lhs_storage{3};
  StatsStorage rhs_storage{3};
  auto lhs = lhs_storage.View();
  auto rhs = rhs_storage.View();

  for (std::size_t i = 0; i < 3; ++i) {
    lhs.SetGradient(i, static_cast<double>(i));
    rhs.SetGradient(i, 10.0 * static_cast<double>(i));
    for (std::size_t j = 0; j <= i; ++j) {
      lhs.SetHessian(i, j, static_cast<double>(i + j));
      rhs.SetHessian(i, j, 100.0 * static_cast<double>(i + j));
    }
  }

  lhs.Add(rhs);
  for (std::size_t i = 0; i < 3; ++i) {
    ASSERT_DOUBLE_EQ(lhs.GetGradient(i), 11.0 * static_cast<double>(i));
    for (std::size_t j = 0; j <= i; ++j) {
      ASSERT_DOUBLE_EQ(lhs.GetHessian(i, j), 101.0 * static_cast<double>(i + j));
      ASSERT_DOUBLE_EQ(lhs.GetHessian(j, i), lhs.GetHessian(i, j));
    }
  }
}

TEST(ExactMultinomialStats, Subtract) {
  StatsStorage parent_storage{3};
  StatsStorage left_storage{3};
  auto parent = parent_storage.View();
  auto left = left_storage.View();

  for (std::size_t i = 0; i < 3; ++i) {
    parent.SetGradient(i, 7.0 + static_cast<double>(i));
    left.SetGradient(i, 2.0 + static_cast<double>(i));
    for (std::size_t j = 0; j <= i; ++j) {
      parent.SetHessian(i, j, 5.0 * static_cast<double>(i + 1) + static_cast<double>(j));
      left.SetHessian(i, j, static_cast<double>(i + 1));
    }
  }

  // The sibling subtraction trick histograms rely on.
  StatsStorage right_storage{3};
  auto right = right_storage.View();
  right.Add(parent);
  right.Subtract(left);

  for (std::size_t i = 0; i < 3; ++i) {
    ASSERT_DOUBLE_EQ(right.GetGradient(i), 5.0);
    for (std::size_t j = 0; j <= i; ++j) {
      auto expected = 5.0 * static_cast<double>(i + 1) + static_cast<double>(j) -
                      static_cast<double>(i + 1);
      ASSERT_DOUBLE_EQ(right.GetHessian(i, j), expected);
    }
  }

  // Adding a sibling back must restore the parent exactly.
  right.Add(left);
  for (std::size_t i = 0; i < 3; ++i) {
    ASSERT_DOUBLE_EQ(right.GetGradient(i), parent.GetGradient(i));
    for (std::size_t j = 0; j <= i; ++j) {
      ASSERT_DOUBLE_EQ(right.GetHessian(i, j), parent.GetHessian(i, j));
    }
  }
}

TEST(ExactMultinomialStats, Reset) {
  StatsStorage storage{4};
  auto stats = storage.View();
  for (std::size_t i = 0; i < 4; ++i) {
    stats.SetGradient(i, 3.0);
    for (std::size_t j = 0; j <= i; ++j) {
      stats.SetHessian(i, j, 9.0);
    }
  }

  stats.Zero();
  for (auto v : stats.Data()) {
    ASSERT_EQ(v, 0.0);
  }
  ASSERT_EQ(stats.Gradient().size(), 4);
  ASSERT_EQ(stats.Hessian().size(), PackedHessianSize(4));
}

TEST(ExactMultinomialStats, KnownMultinomialHessian) {
  // Three classes with class 2 as the reference class.
  std::vector<double> p{0.1, 0.3, 0.6};
  auto n_free = NumFreeClasses(p.size());
  ASSERT_EQ(n_free, 2);

  StatsStorage storage{n_free};
  auto stats = storage.View();
  // H = diag(p) - p p^T restricted to the free classes.
  for (std::size_t i = 0; i < n_free; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      stats.SetHessian(i, j, (i == j ? p[i] : 0.0) - p[i] * p[j]);
    }
  }

  ASSERT_NEAR(stats.GetHessian(0, 0), 0.09, 1e-15);
  ASSERT_NEAR(stats.GetHessian(0, 1), -0.03, 1e-15);
  ASSERT_NEAR(stats.GetHessian(1, 0), -0.03, 1e-15);
  ASSERT_NEAR(stats.GetHessian(1, 1), 0.21, 1e-15);

  // The packed buffer holds the gradient first, then the lower triangle row by row.
  auto tri = stats.Hessian();
  ASSERT_EQ(tri.size(), 3);
  ASSERT_NEAR(tri[0], 0.09, 1e-15);
  ASSERT_NEAR(tri[1], -0.03, 1e-15);
  ASSERT_NEAR(tri[2], 0.21, 1e-15);
}

TEST(ExactMultinomialStats, HistogramLayout) {
  // A histogram carves every bin out of one buffer, so accumulation never allocates.
  std::size_t constexpr kNumFree = 3;
  std::size_t constexpr kNumBins = 5;
  auto stride = PackedStatsStride(kNumFree);
  std::vector<double> buffer(stride * kNumBins, 0.0);
  auto span = Span<double>{buffer.data(), buffer.size()};

  for (std::size_t bin = 0; bin < kNumBins; ++bin) {
    auto stats = PackedStatsAtBin(span, kNumFree, bin);
    stats.SetGradient(0, static_cast<double>(bin));
    stats.SetHessian(2, 1, static_cast<double>(bin) + 0.5);
  }

  for (std::size_t bin = 0; bin < kNumBins; ++bin) {
    auto stats = PackedStatsAtBin(Span<double const>{buffer.data(), buffer.size()}, kNumFree, bin);
    ASSERT_EQ(stats.GetGradient(0), static_cast<double>(bin));
    ASSERT_EQ(stats.GetHessian(1, 2), static_cast<double>(bin) + 0.5);
    // Each bin views its own slice of the shared allocation.
    ASSERT_EQ(stats.Data().data(), buffer.data() + bin * stride);
  }
}
}  // namespace xgboost::common
