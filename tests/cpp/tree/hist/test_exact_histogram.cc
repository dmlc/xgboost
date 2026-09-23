/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>

#include <chrono>   // for steady_clock
#include <cmath>    // for fabs
#include <cstddef>  // for size_t
#include <cstdio>   // for printf
#include <numeric>  // for accumulate, iota
#include <utility>  // for pair
#include <vector>   // for vector

#include "../../../../src/common/exact_multinomial/packed_stats.h"
#include "../../../../src/data/gradient_index.h"
#include "../../../../src/tree/hist/exact_histogram.h"
#include "../../helpers.h"

namespace xgboost::tree {
namespace {
/** @brief Per-row gradients and exact Hessian for a deterministic toy problem. */
struct ExactRows {
  linalg::Matrix<GradientPair> gpair;
  ExactHessian hessian;
};

/**
 * @brief Build per-row statistics from explicit probabilities and weights.
 *
 * The Hessian is written with the same formula the objective uses,
 * `H_ij = w * p_i * (delta_ij - p_j)`, so these tests exercise the histogram layer against
 * statistics of the right shape without depending on the objective.
 */
ExactRows MakeRows(std::vector<std::vector<double>> const& probabilities,
                   std::vector<float> const& weights, std::vector<std::size_t> const& labels) {
  auto n_samples = probabilities.size();
  auto n_classes = probabilities.front().size();
  auto n_free = static_cast<bst_target_t>(n_classes - 1);

  ExactRows out;
  out.gpair.Reshape(n_samples, n_classes);
  out.hessian.Reshape(n_samples, n_free);
  auto gpair = out.gpair.HostView();

  for (std::size_t r = 0; r < n_samples; ++r) {
    auto const& p = probabilities[r];
    auto w = weights[r];
    for (std::size_t k = 0; k < n_classes; ++k) {
      auto grad = static_cast<float>(p[k]) - (labels[r] == k ? 1.0f : 0.0f);
      gpair(r, k) = GradientPair{grad * w, 1.0f};
    }
    auto row = out.hessian.HostRow(r);
    auto view = common::PackedHessianAtRow(row, n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.Set(i, j, static_cast<float>(w * p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
      }
    }
  }
  return out;
}

std::vector<std::vector<double>> UniformProbabilities(std::size_t n_samples,
                                                      std::size_t n_classes) {
  std::vector<std::vector<double>> out;
  for (std::size_t r = 0; r < n_samples; ++r) {
    std::vector<double> p(n_classes);
    double total = 0.0;
    for (std::size_t k = 0; k < n_classes; ++k) {
      // Deterministic, spread, and strictly positive.
      p[k] = 1.0 + static_cast<double>((r * 7 + k * 3) % 11);
      total += p[k];
    }
    for (auto& v : p) {
      v /= total;
    }
    out.push_back(p);
  }
  return out;
}
}  // anonymous namespace

TEST(ExactHistogram, RecordShape) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    auto n_free = static_cast<bst_target_t>(n_classes - 1);
    // d gradients + d(d+1)/2 Hessian entries.
    auto expected = static_cast<std::size_t>(n_free) + n_free * (n_free + 1) / 2;
    EXPECT_EQ(ExactHistRecordSize(n_free), expected) << "K=" << n_classes;
  }
  EXPECT_EQ(ExactHistRecordSize(1), 2);   // K=2: 1 gradient + 1 Hessian
  EXPECT_EQ(ExactHistRecordSize(2), 5);   // K=3
  EXPECT_EQ(ExactHistRecordSize(6), 27);  // K=7
}

TEST(ExactHistogram, MemoryFormula) {
  // K=7: 27 doubles = 216 bytes per bin.
  // One bin plus the node-total record: two records of 27 doubles.
  EXPECT_EQ(ExactHistMemoryBytes(1, 1, 6), 2 * 216);
  EXPECT_EQ(ExactHistMemoryBytes(256, 4, 6), 216ul * 257 * 4);

  // Against the existing multi-target baseline of 2K doubles per bin.
  for (bst_target_t n_classes : {2u, 3u, 4u, 7u, 10u, 20u}) {
    auto n_free = static_cast<bst_target_t>(n_classes - 1);
    double exact = static_cast<double>(ExactHistRecordSize(n_free));
    double multi_target = 2.0 * n_classes;
    double ratio = exact / multi_target;
    // Closed form: (K-1)(K+2) / (4K).
    double closed = static_cast<double>(n_classes - 1) * (n_classes + 2) / (4.0 * n_classes);
    EXPECT_NEAR(ratio, closed, 1e-12) << "K=" << n_classes;
  }
  // Exact mode is cheaper than the existing multi-target histogram for small K.
  EXPECT_LT(ExactHistRecordSize(2), 2u * 3u);   // K=3
  EXPECT_GT(ExactHistRecordSize(6), 2u * 7u);   // K=7
}

TEST(ExactHistogram, CacheAllocation) {
  bst_bin_t constexpr kBins = 12;
  bst_target_t constexpr kNumFree = 3;
  ExactHistCollection hist;
  hist.Reset(kBins, kNumFree, 4);

  EXPECT_EQ(hist.RecordSize(), ExactHistRecordSize(kNumFree));
  EXPECT_EQ(hist.TotalBins(), kBins);
  EXPECT_EQ(hist.NumFree(), kNumFree);
  EXPECT_EQ(hist.Size(), 0);
  EXPECT_EQ(hist.MemoryBytes(), 0);

  std::vector<bst_node_t> build{0, 1};
  std::vector<bst_node_t> sub{2};
  EXPECT_TRUE(hist.CanHost(common::Span<bst_node_t const>{build},
                           common::Span<bst_node_t const>{sub}));
  hist.AllocateHistograms(common::Span<bst_node_t const>{build},
                          common::Span<bst_node_t const>{sub});

  // One record per bin plus the node-total record.
  auto node_stride = (static_cast<std::size_t>(kBins) + 1) * ExactHistRecordSize(kNumFree);
  EXPECT_EQ(hist.Size(), 3 * node_stride);
  EXPECT_EQ(hist.MemoryBytes(), 3 * node_stride * sizeof(double));
  EXPECT_EQ(hist.MemoryBytes(), ExactHistMemoryBytes(kBins, 3, kNumFree));

  for (bst_node_t nidx : {0, 1, 2}) {
    EXPECT_TRUE(hist.HistogramExists(nidx));
    EXPECT_EQ(hist[nidx].size(), node_stride);
  }
  EXPECT_FALSE(hist.HistogramExists(3));

  // Nodes are laid out contiguously in allocation order, which is what lets a later
  // reduction treat the whole cache as one flat buffer.
  EXPECT_EQ(hist[1].data(), hist[0].data() + node_stride);
  EXPECT_EQ(hist[2].data(), hist[1].data() + node_stride);

  // The node-count budget is enforced exactly as the scalar cache does.
  std::vector<bst_node_t> too_many{3, 4};
  EXPECT_FALSE(hist.CanHost(common::Span<bst_node_t const>{too_many},
                            common::Span<bst_node_t const>{}));
  hist.Clear(true);
  EXPECT_TRUE(hist.HasExceeded());
  EXPECT_EQ(hist.Size(), 0);
}

TEST(ExactHistogram, RecordAtIsPackedView) {
  bst_bin_t constexpr kBins = 5;
  bst_target_t constexpr kNumFree = 2;
  ExactHistCollection hist;
  hist.Reset(kBins, kNumFree, 8);
  hist.AllocateHistograms(std::vector<bst_node_t>{0});
  ZeroExactHist(hist[0]);

  auto record = hist.RecordAt(0, 3);
  record.SetGradient(0, 1.5);
  record.SetGradient(1, -2.5);
  record.SetHessian(0, 0, 0.09);
  record.SetHessian(1, 0, -0.03);
  record.SetHessian(1, 1, 0.21);

  // The bin's record is a slice of the node buffer at the expected stride.
  auto node = hist[0];
  auto stride = ExactHistRecordSize(kNumFree);
  EXPECT_EQ(record.Data().data(), node.data() + 3 * stride);
  EXPECT_EQ(node[3 * stride + 0], 1.5);
  EXPECT_EQ(node[3 * stride + 1], -2.5);
  EXPECT_EQ(node[3 * stride + 2], 0.09);
  EXPECT_EQ(node[3 * stride + 3], -0.03);
  EXPECT_EQ(node[3 * stride + 4], 0.21);

  // Symmetry through the shared index convention.
  EXPECT_EQ(record.GetHessian(0, 1), -0.03);
  // Other bins untouched.
  EXPECT_EQ(node[0], 0.0);
}

/**
 * trace(H_full) = 2 * sum(packed triangle), and equals 1 - sum_k p_k^2 per unit weight.
 * This is the quantity exact mode uses for min_child_weight, so it must be exact and must
 * not depend on the choice of reference class.
 */
TEST(ExactHistogram, TotalCurvatureIsReferenceInvariant) {
  std::vector<double> p{0.1, 0.2, 0.3, 0.4};
  auto n_classes = p.size();
  double expected = 1.0;
  for (auto v : p) {
    expected -= v * v;
  }
  ASSERT_NEAR(expected, 0.70, 1e-12);

  // Drop each class in turn as the reference; the free block changes, the total does not.
  for (std::size_t reference = 0; reference < n_classes; ++reference) {
    std::vector<double> free_p;
    for (std::size_t k = 0; k < n_classes; ++k) {
      if (k != reference) {
        free_p.push_back(p[k]);
      }
    }
    auto n_free = static_cast<bst_target_t>(free_p.size());
    std::vector<double> buffer(common::PackedStatsStride(n_free), 0.0);
    auto stats = common::PackedMultinomialStats<double>{
        common::Span<double>{buffer.data(), buffer.size()}, n_free};
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        stats.SetHessian(i, j, free_p[i] * ((i == j ? 1.0 : 0.0) - free_p[j]));
      }
    }
    EXPECT_NEAR(stats.TotalCurvature(), expected, 1e-12) << "reference class " << reference;

    // The free-block trace alone is NOT invariant, which is why it is not used.
    double free_trace = 0.0;
    for (std::size_t i = 0; i < n_free; ++i) {
      free_trace += stats.GetHessian(i, i);
    }
    EXPECT_NEAR(free_trace, expected - p[reference] * (1.0 - p[reference]), 1e-12);
  }
}

TEST(ExactHistogram, TotalCurvatureAccumulates) {
  bst_target_t constexpr kNumFree = 2;
  std::vector<double> buffer(common::PackedStatsStride(kNumFree), 0.0);
  auto stats = common::PackedMultinomialStats<double>{
      common::Span<double>{buffer.data(), buffer.size()}, kNumFree};

  // Two rows with different distributions and weights.
  std::vector<std::vector<double>> rows{{0.1, 0.3, 0.6}, {0.5, 0.25, 0.25}};
  std::vector<double> weights{1.0, 2.5};
  double expected = 0.0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto const& p = rows[r];
    double curvature = 1.0;
    for (auto v : p) {
      curvature -= v * v;
    }
    expected += weights[r] * curvature;
    for (std::size_t i = 0; i < kNumFree; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        stats.AddHessian(i, j, weights[r] * p[i] * ((i == j ? 1.0 : 0.0) - p[j]));
      }
    }
  }
  EXPECT_NEAR(stats.TotalCurvature(), expected, 1e-12);
}

/** Accumulation over a real binned matrix, checked against a direct reference sum. */
TEST(ExactHistogram, BuildAgainstReference) {
  for (std::size_t n_classes : {2ul, 3ul, 7ul}) {
    Context ctx;
    std::size_t constexpr kRows = 64, kCols = 8;
    bst_bin_t constexpr kMaxBins = 4;
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0.0).Seed(11).GenerateDMatrix();
    auto const& gmat =
        *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

    auto n_free = static_cast<bst_target_t>(n_classes - 1);
    auto probabilities = UniformProbabilities(kRows, n_classes);
    std::vector<float> weights(kRows);
    std::vector<std::size_t> labels(kRows);
    for (std::size_t r = 0; r < kRows; ++r) {
      weights[r] = 0.5f + static_cast<float>(r % 4);
      labels[r] = r % n_classes;
    }
    auto rows = MakeRows(probabilities, weights, labels);

    std::vector<bst_idx_t> row_indices(kRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);

    ExactHistCollection hist;
    hist.Reset(gmat.cut.TotalBins(), n_free, 8);
    hist.AllocateHistograms(std::vector<bst_node_t>{0});
    ZeroExactHist(hist[0]);

    ExactHistThreadBuffer buffer;
    BuildExactHist(&ctx, hist[0], n_free, gmat, common::Span<bst_idx_t const>{row_indices},
                   rows.gpair.HostView(), rows.hessian, &buffer);

    // Independent reference: accumulate serially with no thread buffers.
    auto record_size = ExactHistRecordSize(n_free);
    // Bins plus the node-total record.
    std::vector<double> reference(
        (static_cast<std::size_t>(gmat.cut.TotalBins()) + 1) * record_size, 0.0);
    auto gpair = rows.gpair.HostView();
    for (std::size_t r = 0; r < kRows; ++r) {
      auto h_row = rows.hessian.HostRow(r);
      for (bst_feature_t f = 0; f < gmat.Features(); ++f) {
        auto bin = gmat.GetGindex(r, f);
        if (bin < 0) {
          continue;
        }
        auto* rec = reference.data() + static_cast<std::size_t>(bin) * record_size;
        for (bst_target_t i = 0; i < n_free; ++i) {
          rec[i] += static_cast<double>(gpair(r, i).GetGrad());
        }
        for (std::size_t k = 0; k < h_row.size(); ++k) {
          rec[n_free + k] += static_cast<double>(h_row[k]);
        }
      }
      // The node total accumulates once per row, not once per feature.
      auto* total = reference.data() +
                    static_cast<std::size_t>(gmat.cut.TotalBins()) * record_size;
      for (bst_target_t i = 0; i < n_free; ++i) {
        total[i] += static_cast<double>(gpair(r, i).GetGrad());
      }
      for (std::size_t k = 0; k < h_row.size(); ++k) {
        total[n_free + k] += static_cast<double>(h_row[k]);
      }
    }

    auto built = hist[0];
    ASSERT_EQ(built.size(), reference.size()) << "K=" << n_classes;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      EXPECT_NEAR(built[i], reference[i], 1e-9) << "K=" << n_classes << " entry " << i;
    }

    // Every row lands somewhere: the summed gradient over all bins equals the summed
    // gradient over rows times the number of features each row contributes to.
    double total_g0 = 0.0;
    for (bst_bin_t b = 0; b < gmat.cut.TotalBins(); ++b) {
      total_g0 += built[static_cast<std::size_t>(b) * record_size];
    }
    double row_g0 = 0.0;
    for (std::size_t r = 0; r < kRows; ++r) {
      row_g0 += static_cast<double>(gpair(r, 0).GetGrad());
    }
    EXPECT_NEAR(total_g0, row_g0 * gmat.Features(), 1e-6) << "K=" << n_classes;
  }
}

/**
 * The node total must count each row ONCE, no matter how many features it has or how many of
 * them the row actually carries.
 *
 * This is the invariant the split evaluator depends on: it derives the complementary child as
 * `node_total - accumulated`. Summing histogram bins instead gets it wrong in both
 * directions -- on dense data every row lands in one bin per feature, inflating the total by
 * `n_features`; on sparse data a row lands in a bin only for the features it carries, so the
 * inflation is uneven across rows and cannot be divided out. Neither failure is visible with
 * a single dense feature, so this sweeps several of each.
 */
TEST(ExactHistogram, NodeTotalCountsEachRowOnce) {
  for (auto [n_features, sparsity] : std::vector<std::pair<std::size_t, float>>{
           {1ul, 0.0f}, {3ul, 0.0f}, {8ul, 0.0f}, {3ul, 0.5f}, {8ul, 0.7f}}) {
    Context ctx;
    std::size_t constexpr kRows = 128;
    bst_target_t constexpr kNumClasses = 4;
    auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
    auto record_size = ExactHistRecordSize(n_free);

    auto p_fmat = RandomDataGenerator(kRows, n_features, sparsity).Seed(21).GenerateDMatrix();
    auto const& gmat =
        *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{8, 0.5}).begin());
    ASSERT_EQ(gmat.Features(), n_features);
    ASSERT_EQ(gmat.IsDense(), sparsity == 0.0f);

    // How many features each row actually carries, read from the matrix.
    std::vector<std::size_t> nnz(kRows, 0);
    for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
      auto batch = page.GetView();
      for (std::size_t r = 0; r < batch.Size(); ++r) {
        nnz[page.base_rowid + r] = batch[r].size();
      }
    }

    auto probabilities = UniformProbabilities(kRows, kNumClasses);
    std::vector<float> weights(kRows);
    std::vector<std::size_t> labels(kRows);
    for (std::size_t r = 0; r < kRows; ++r) {
      weights[r] = 0.5f + static_cast<float>(r % 3);
      labels[r] = r % kNumClasses;
    }
    auto rows = MakeRows(probabilities, weights, labels);

    std::vector<bst_idx_t> row_indices(kRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);

    ExactHistCollection hist;
    hist.Reset(gmat.cut.TotalBins(), n_free, 8);
    hist.AllocateHistograms(std::vector<bst_node_t>{0});
    ZeroExactHist(hist[0]);
    ExactHistThreadBuffer buffer;
    BuildExactHist(&ctx, hist[0], n_free, gmat, common::Span<bst_idx_t const>{row_indices},
                   rows.gpair.HostView(), rows.hessian, &buffer);

    // The independent truth: sum every row's statistics exactly once.
    std::vector<double> expected(record_size, 0.0);
    auto gpair = rows.gpair.HostView();
    for (std::size_t r = 0; r < kRows; ++r) {
      for (bst_target_t i = 0; i < n_free; ++i) {
        expected[i] += static_cast<double>(gpair(r, i).GetGrad());
      }
      auto row = rows.hessian.HostRow(r);
      for (std::size_t k = 0; k < row.size(); ++k) {
        expected[n_free + k] += static_cast<double>(row[k]);
      }
    }

    auto const* total = hist[0].data() +
                        static_cast<std::size_t>(gmat.cut.TotalBins()) * record_size;
    for (std::size_t i = 0; i < record_size; ++i) {
      EXPECT_NEAR(total[i], expected[i], 1e-6)
          << "n_features=" << n_features << " sparsity=" << sparsity << " entry " << i;
    }

    // The bin sum instead weights each row by how many features it carries, which is the
    // quantity that used to be mistaken for the node total.
    double bin_sum_g0 = 0.0;
    for (bst_bin_t b = 0; b < gmat.cut.TotalBins(); ++b) {
      bin_sum_g0 += hist[0][static_cast<std::size_t>(b) * record_size];
    }
    double weighted_g0 = 0.0;
    for (std::size_t r = 0; r < kRows; ++r) {
      weighted_g0 += static_cast<double>(nnz[r]) * static_cast<double>(gpair(r, 0).GetGrad());
    }
    EXPECT_NEAR(bin_sum_g0, weighted_g0, 1e-5)
        << "n_features=" << n_features << " sparsity=" << sparsity;
    if (n_features > 1) {
      EXPECT_GT(std::fabs(bin_sum_g0 - expected[0]), 1e-3)
          << "n_features=" << n_features << " sparsity=" << sparsity
          << ": the bin sum coincides with the node total, so this test can no longer "
             "distinguish them";
    }
  }
}

/**
 * The sibling trick must reproduce direct accumulation exactly, including the negative
 * off-diagonal Hessian terms that distinguish the dense Hessian from a diagonal one.
 */
TEST(ExactHistogram, SubtractionMatchesDirectAccumulation) {
  Context ctx;
  std::size_t constexpr kRows = 48, kCols = 6, kNumClasses = 4;
  bst_bin_t constexpr kMaxBins = 4;
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);

  auto p_fmat = RandomDataGenerator(kRows, kCols, 0.0).Seed(7).GenerateDMatrix();
  auto const& gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  auto probabilities = UniformProbabilities(kRows, kNumClasses);
  std::vector<float> weights(kRows, 1.0f);
  std::vector<std::size_t> labels(kRows);
  for (std::size_t r = 0; r < kRows; ++r) {
    weights[r] = 0.25f + static_cast<float>(r % 3);
    labels[r] = r % kNumClasses;
  }
  auto rows = MakeRows(probabilities, weights, labels);

  // Split the rows into two disjoint children.
  std::vector<bst_idx_t> all(kRows), left, right;
  std::iota(all.begin(), all.end(), 0);
  for (auto r : all) {
    (r % 3 == 0 ? left : right).push_back(r);
  }
  ASSERT_FALSE(left.empty());
  ASSERT_FALSE(right.empty());

  ExactHistCollection hist;
  hist.Reset(gmat.cut.TotalBins(), n_free, 8);
  hist.AllocateHistograms(std::vector<bst_node_t>{0, 1, 2, 3});
  ExactHistThreadBuffer buffer;

  auto build = [&](bst_node_t nidx, std::vector<bst_idx_t> const& r) {
    ZeroExactHist(hist[nidx]);
    BuildExactHist(&ctx, hist[nidx], n_free, gmat, common::Span<bst_idx_t const>{r},
                   rows.gpair.HostView(), rows.hessian, &buffer);
  };
  build(0, all);    // parent
  build(1, left);   // left child, directly
  build(2, right);  // right child, directly

  // Right child by subtraction instead.
  SubtractExactHist(hist[3], common::Span<double const>{hist[0]},
                    common::Span<double const>{hist[1]});

  auto direct = hist[2];
  auto subtracted = hist[3];
  ASSERT_EQ(direct.size(), subtracted.size());
  for (std::size_t i = 0; i < direct.size(); ++i) {
    EXPECT_NEAR(subtracted[i], direct[i], 1e-9) << "entry " << i;
  }

  // The test is only meaningful if genuine negative off-diagonal entries are present.
  std::size_t negative_off_diagonal = 0;
  for (bst_bin_t b = 0; b < gmat.cut.TotalBins(); ++b) {
    auto record = hist.RecordAt(0, b);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (record.GetHessian(i, j) < -1e-12) {
          ++negative_off_diagonal;
        }
      }
    }
  }
  EXPECT_GT(negative_off_diagonal, 0u) << "no negative off-diagonal Hessian entries were built";

  // Parent must equal left + right exactly in the accumulated statistics.
  for (std::size_t i = 0; i < direct.size(); ++i) {
    EXPECT_NEAR(hist[0][i], hist[1][i] + hist[2][i], 1e-9) << "entry " << i;
  }
}

/** Thread-local blocks are disjoint and reduce to the serial answer. */
TEST(ExactHistogram, ThreadLocalReduction) {
  bst_bin_t constexpr kBins = 7;
  bst_target_t constexpr kNumFree = 3;
  auto record_size = ExactHistRecordSize(kNumFree);
  std::int32_t constexpr kThreads = 4;

  ExactHistThreadBuffer buffer;
  buffer.Reset(kThreads, static_cast<std::size_t>(kBins), record_size);
  EXPECT_EQ(buffer.Stride(), static_cast<std::size_t>(kBins) * record_size);
  EXPECT_EQ(buffer.NumThreads(), kThreads);

  // Disjointness: consecutive thread blocks are adjacent and non-overlapping.
  for (std::int32_t t = 1; t < kThreads; ++t) {
    EXPECT_EQ(buffer.ThreadSpan(t).data(), buffer.ThreadSpan(t - 1).data() + buffer.Stride());
  }

  // Each thread contributes a distinct pattern; the reduction must sum them all.
  double expected_total = 0.0;
  for (std::int32_t t = 0; t < kThreads; ++t) {
    auto span = buffer.ThreadSpan(t);
    for (std::size_t i = 0; i < span.size(); ++i) {
      auto v = static_cast<double>(t + 1) * 0.5 - static_cast<double>(i % 3);
      span[i] = v;
      expected_total += v;
    }
  }

  std::vector<double> out(buffer.Stride(), 0.0);
  buffer.ReduceTo(common::Span<double>{out.data(), out.size()});
  EXPECT_NEAR(std::accumulate(out.begin(), out.end(), 0.0), expected_total, 1e-9);

  for (std::size_t i = 0; i < out.size(); ++i) {
    double expected = 0.0;
    for (std::int32_t t = 0; t < kThreads; ++t) {
      expected += static_cast<double>(t + 1) * 0.5 - static_cast<double>(i % 3);
    }
    EXPECT_NEAR(out[i], expected, 1e-9) << "entry " << i;
  }

  // ReduceTo accumulates into the destination rather than overwriting it, so a pre-loaded
  // destination must not silently lose its contents.
  std::vector<double> preloaded(buffer.Stride(), 1.0);
  buffer.ReduceTo(common::Span<double>{preloaded.data(), preloaded.size()});
  for (std::size_t i = 0; i < preloaded.size(); ++i) {
    EXPECT_NEAR(preloaded[i], out[i] + 1.0, 1e-9);
  }
}

/**
 * Structural performance guard.
 *
 * A correct build visits each row once and writes one contiguous record per touched bin, so
 * the cost should track the record size: K=3 stores 5 doubles per bin and K=7 stores 27, a
 * ratio of 5.4x. An implementation that re-traversed the rows once per Hessian entry would
 * instead scale with the square of the record size, roughly 29x. The bound below is loose
 * enough not to be timing-flaky while still catching that class of mistake.
 */
TEST(ExactHistogram, BuildCostTracksRecordSize) {
  Context ctx;
  std::size_t constexpr kRows = 4096, kCols = 16;
  bst_bin_t constexpr kMaxBins = 16;
  auto p_fmat = RandomDataGenerator(kRows, kCols, 0.0).Seed(5).GenerateDMatrix();
  auto const& gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  std::vector<bst_idx_t> row_indices(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  auto measure = [&](std::size_t n_classes) {
    auto n_free = static_cast<bst_target_t>(n_classes - 1);
    auto probabilities = UniformProbabilities(kRows, n_classes);
    std::vector<float> weights(kRows, 1.0f);
    std::vector<std::size_t> labels(kRows, 0);
    auto rows = MakeRows(probabilities, weights, labels);

    ExactHistCollection hist;
    hist.Reset(gmat.cut.TotalBins(), n_free, 8);
    hist.AllocateHistograms(std::vector<bst_node_t>{0});
    ExactHistThreadBuffer buffer;

    // One warm-up pass so the comparison is not dominated by first-touch page faults.
    ZeroExactHist(hist[0]);
    BuildExactHist(&ctx, hist[0], n_free, gmat, common::Span<bst_idx_t const>{row_indices},
                   rows.gpair.HostView(), rows.hessian, &buffer);

    auto start = std::chrono::steady_clock::now();
    int constexpr kRepeats = 5;
    for (int i = 0; i < kRepeats; ++i) {
      ZeroExactHist(hist[0]);
      BuildExactHist(&ctx, hist[0], n_free, gmat, common::Span<bst_idx_t const>{row_indices},
                     rows.gpair.HostView(), rows.hessian, &buffer);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count() / kRepeats;
  };

  auto ms_k3 = measure(3);
  auto ms_k7 = measure(7);
  std::printf("  exact histogram build: K=3 %.3f ms, K=7 %.3f ms, ratio %.2fx\n", ms_k3, ms_k7,
              ms_k7 / ms_k3);

  ASSERT_GT(ms_k3, 0.0);
  // Record size ratio is 27/5 = 5.4x; a per-entry re-traversal would be ~29x.
  EXPECT_LT(ms_k7, ms_k3 * 12.0)
      << "K=7 build cost grew far beyond the record-size ratio, which suggests the row loop "
         "is being repeated per Hessian entry";
}

/** Zeroing and the empty state. */
TEST(ExactHistogram, ZeroInitialization) {
  bst_bin_t constexpr kBins = 4;
  bst_target_t constexpr kNumFree = 2;
  ExactHistCollection hist;
  hist.Reset(kBins, kNumFree, 4);
  hist.AllocateHistograms(std::vector<bst_node_t>{0});

  auto node = hist[0];
  for (std::size_t i = 0; i < node.size(); ++i) {
    node[i] = 3.5;
  }
  ZeroExactHist(node);
  for (auto v : node) {
    EXPECT_EQ(v, 0.0);
  }
  EXPECT_EQ(hist.RecordAt(0, 2).TotalCurvature(), 0.0);
}
}  // namespace xgboost::tree
