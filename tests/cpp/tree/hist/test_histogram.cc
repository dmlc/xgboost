/**
 * Copyright 2018-2023 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context

#include <cstddef>  // for size_t
#include <limits>

#include "../../../../src/common/row_set.h"
#include "../../../../src/tree/hist/expand_entry.h"
#include "../../../../src/tree/hist/histogram.h"
#include "../../../../src/tree/hist/param.h"
#include "../../categorical_helpers.h"
#include "../../helpers.h"

namespace xgboost::tree {
namespace {
void InitRowPartitionForTest(common::RowSetCollection *row_set, bst_row_t n_samples,
                             std::size_t base_rowid = 0) {
  auto &row_indices = *row_set->Data();
  row_indices.resize(n_samples);
  std::iota(row_indices.begin(), row_indices.end(), base_rowid);
  row_set->Init();
}
}  // anonymous namespace

TEST(CPUHistogram, HistCollection) {
  Context ctx;
  bst_feature_t n_features = 12;
  bst_bin_t n_bins = 13;
  bst_bin_t n_total_bins = n_features * n_bins;
  HistMakerTrainParam hist_param;
  HistogramStorage hist{n_total_bins, &hist_param};
  hist.AllocateHistograms({0});
  ASSERT_TRUE(hist.HistogramExist(0));
  ASSERT_EQ(hist.NodeCapacity(), 1);

  auto node_hist = hist.GetHist(0);
  ASSERT_EQ(node_hist.size(), n_total_bins);

  auto tloc_hist = hist.GetHist(0);
  ASSERT_EQ(tloc_hist.size(), n_total_bins);
  for (auto v : tloc_hist) {
    ASSERT_EQ(v, GradientPairPrecise{});
  }

  hist.MarkFree(0);
  ASSERT_FALSE(hist.HistogramExist(0));
  ASSERT_EQ(hist.NodeCapacity(), 0);

  hist.AllocateHistograms({0});
  ASSERT_TRUE(hist.HistogramExist(0));
  ASSERT_EQ(hist.NodeCapacity(), 1);

  /**
   * Reuse
   */
  {
    HistMakerTrainParam hist_param;
    HistogramStorage collection{n_total_bins, &hist_param};
    collection.AllocateHistograms({0});
    collection.AllocateHistograms({1, 2});
    collection.MarkFree(0);
    ASSERT_EQ(collection.NodeCapacity(), 2);
    collection.AllocateHistograms({3, 4, 5, 6});
    ASSERT_EQ(collection.NodeCapacity(), 6);
    for (std::int32_t tidx = 0; tidx < ctx.Threads(); ++tidx) {
      for (bst_node_t nidx = 3; nidx < 7; ++nidx) {
        auto node_hist = collection.GetHist(nidx);
        for (auto v : node_hist) {
          ASSERT_EQ(v, GradientPairPrecise{});
        }
      }
    }
  }
}

void TestBuildHistogram(bool is_distributed, bool force_read_by_column, bool is_col_split) {
  std::size_t constexpr kNRows = 8, kNCols = 16;
  std::int32_t constexpr kMaxBins = 4;
  Context ctx;

  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  if (is_col_split) {
    p_fmat = std::shared_ptr<DMatrix>{
        p_fmat->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  }
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());
  uint32_t total_bins = gmat.cut.Ptrs().back();

  static double constexpr kEps = 1e-6;
  std::vector<GradientPair> gpair = {
      {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
      {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f}};

  bst_node_t nidx = RegTree::kRoot;
  HistogramBuilder<CPUExpandEntry> histogram;
  HistMakerTrainParam hist_param;
  histogram.Reset(&ctx, total_bins, {kMaxBins, 0.5}, is_distributed, is_col_split, &hist_param);

  RegTree tree;

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<std::size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kNRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); },
      DefaultHistSpaceGran()};
  std::vector<bst_node_t> nodes{nidx};
  histogram.BuildRootHist<true>(&ctx, 0, space, gmat,
                                linalg::MakeTensorView(&ctx, gpair, gpair.size()),
                                row_set_collection, nodes, force_read_by_column);
  histogram.ReduceHist(nodes);

  // Check if number of histogram bins is correct
  ASSERT_EQ(histogram.Histogram()[nidx].size(), gmat.cut.Ptrs().back());
  std::vector<GradientPairPrecise> histogram_expected(histogram.Histogram()[nidx].size());

  // Compute the correct histogram (histogram_expected)
  CHECK_EQ(gpair.size(), kNRows);
  for (std::size_t rid = 0; rid < kNRows; ++rid) {
    const std::size_t ibegin = gmat.row_ptr[rid];
    const std::size_t iend = gmat.row_ptr[rid + 1];
    for (std::size_t i = ibegin; i < iend; ++i) {
      const std::size_t bin_id = gmat.index[i];
      histogram_expected[bin_id] += GradientPairPrecise(gpair[rid]);
    }
  }

  // Now validate the computed histogram returned by BuildHist
  for (std::size_t i = 0; i < histogram.Histogram()[nidx].size(); ++i) {
    GradientPairPrecise sol = histogram_expected[i];
    ASSERT_NEAR(sol.GetGrad(), histogram.Histogram()[nidx][i].GetGrad(), kEps);
    ASSERT_NEAR(sol.GetHess(), histogram.Histogram()[nidx][i].GetHess(), kEps);
  }
}

TEST(CPUHistogram, BuildHist) {
  TestBuildHistogram(false, false, false);
  TestBuildHistogram(true, false, false);
  TestBuildHistogram(true, true, false);
  TestBuildHistogram(false, true, false);
}

TEST(CPUHistogram, BuildHistColSplit) {
  auto constexpr kWorkers = 4;
  RunWithInMemoryCommunicator(kWorkers, TestBuildHistogram, true, true, true);
  RunWithInMemoryCommunicator(kWorkers, TestBuildHistogram, true, false, true);
}

namespace {
template <typename GradientSumT>
void ValidateCategoricalHistogram(size_t n_categories,
                                  common::Span<GradientSumT> onehot,
                                  common::Span<GradientSumT> cat) {
  auto cat_sum = std::accumulate(cat.cbegin(), cat.cend(), GradientPairPrecise{});
  for (size_t c = 0; c < n_categories; ++c) {
    auto zero = onehot[c * 2];
    auto one = onehot[c * 2 + 1];

    auto chosen = cat[c];
    auto not_chosen = cat_sum - chosen;

    ASSERT_LE(RelError(zero.GetGrad(), not_chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(zero.GetHess(), not_chosen.GetHess()), kRtEps);

    ASSERT_LE(RelError(one.GetGrad(), chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(one.GetHess(), chosen.GetHess()), kRtEps);
  }
}

void TestHistogramCategorical(size_t n_categories, bool force_read_by_column) {
  size_t constexpr kRows = 340;
  bst_bin_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, n_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  Context ctx;

  BatchParam batch_param{0, static_cast<int32_t>(kBins)};

  RegTree tree;
  std::vector<bst_node_t> nodes_for_explicit_hist_build{RegTree::kRoot};

  auto gpair = GenerateRandomGradients(kRows, 0, 2);

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  common::BlockedSpace2d space{
      nodes_for_explicit_hist_build.size(),
      [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); },
      DefaultHistSpaceGran()};

  /**
   * Generate hist with cat data.
   */
  HistogramBuilder<CPUExpandEntry> cat_hist;
  HistMakerTrainParam hist_param;
  for (auto const &gidx : cat_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto n_total_bins = gidx.cut.TotalBins();
    cat_hist.Reset(&ctx, n_total_bins, {kBins, 0.5}, false, false, &hist_param);
    cat_hist.BuildRootHist<false>(
        &ctx, 0, space, gidx, linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
        row_set_collection, nodes_for_explicit_hist_build, force_read_by_column);
    cat_hist.ReduceHist(nodes_for_explicit_hist_build);
  }

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, n_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, n_categories);
  HistogramBuilder<CPUExpandEntry> onehot_hist;
  for (auto const &gidx : encode_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto n_total_bins = gidx.cut.TotalBins();
    onehot_hist.Reset(&ctx, n_total_bins, {kBins, 0.5}, false, false, &hist_param);
    onehot_hist.BuildRootHist<false>(
        &ctx, 0, space, gidx, linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
        row_set_collection, nodes_for_explicit_hist_build, force_read_by_column);
    onehot_hist.ReduceHist(nodes_for_explicit_hist_build);
  }

  auto cat = cat_hist.Histogram()[0];
  auto onehot = onehot_hist.Histogram()[0];
  ValidateCategoricalHistogram(n_categories, onehot, cat);
}
}  // anonymous namespace

TEST(CPUHistogram, Categorical) {
  for (size_t n_categories = 2; n_categories < 8; ++n_categories) {
    TestHistogramCategorical(n_categories, false);
  }
  for (size_t n_categories = 2; n_categories < 8; ++n_categories) {
    TestHistogramCategorical(n_categories, true);
  }
}

namespace {
void TestHistogramExternalMemory(Context const *ctx, BatchParam batch_param, bool is_approx,
                                 bool force_read_by_column) {
  std::size_t n_batches = 3;
  auto m = CreateSparsePageDMatrix(512, 13, n_batches, "cache");
  std::vector<float> hess(m->Info().num_row_, 1.0);
  if (is_approx) {
    batch_param.hess = hess;
  }

  std::size_t partition_size{0};
  bst_bin_t n_total_bins{0};
  bst_row_t n_samples{0};

  auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);

  RegTree tree;
  std::vector<bst_node_t> nodes{RegTree::kRoot};

  common::GHistRow multi_page;
  HistogramBuilder<CPUExpandEntry> multi_build;
  HistMakerTrainParam hist_param;
  {
    /**
     * Multiple pages
     */
    std::vector<common::RowSetCollection> row_sets;
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      CHECK_LT(page.base_rowid, m->Info().num_row_);
      auto n_rows_in_node = page.Size();
      partition_size = std::max(partition_size, n_rows_in_node);
      if (n_total_bins != 0) {
        ASSERT_EQ(n_total_bins, page.cut.TotalBins());
      }
      n_total_bins = page.cut.TotalBins();
      n_samples += n_rows_in_node;

      row_sets.emplace_back();
      InitRowPartitionForTest(&row_sets.back(), n_rows_in_node, page.base_rowid);
    }
    ASSERT_EQ(n_samples, m->Info().num_row_);

    multi_build.Reset(ctx, n_total_bins, batch_param, false, false, &hist_param);
    common::BlockedSpace2d space{1, [&](std::size_t) { return partition_size; },
                                 DefaultHistSpaceGran()};
    std::size_t page_idx{0};
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      multi_build.BuildRootHist<false>(
          ctx, page_idx, space, page,
          linalg::MakeTensorView(ctx, gpair.ConstHostSpan(), gpair.Size()), row_sets.at(page_idx),
          nodes, force_read_by_column);
      ++page_idx;
    }
    multi_build.ReduceHist(nodes);
    ASSERT_EQ(page_idx, 3);
    multi_page = multi_build.Histogram()[0];
  }

  HistogramBuilder<CPUExpandEntry> single_build;
  common::GHistRow single_page;
  {
    /**
     * Single page
     */
    common::RowSetCollection row_set_collection;
    InitRowPartitionForTest(&row_set_collection, n_samples);

    SparsePage concat;
    std::vector<float> hess(m->Info().num_row_, 1.0f);
    for (auto const &page : m->GetBatches<SparsePage>()) {
      concat.Push(page);
    }
    ASSERT_EQ(concat.Size(), m->Info().num_row_);

    auto cut = common::SketchOnDMatrix(ctx, m.get(), batch_param.max_bin, false, hess);
    GHistIndexMatrix gmat{
        concat,        {}, cut, batch_param.max_bin, true, std::numeric_limits<double>::quiet_NaN(),
        ctx->Threads()};
    ASSERT_TRUE(gmat.IsDense());
    common::BlockedSpace2d space{
        1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); },
        DefaultHistSpaceGran()};
    ASSERT_EQ(gmat.Size(), n_samples);

    single_build.Reset(ctx, n_total_bins, batch_param, false, false, &hist_param);
    single_build.BuildRootHist<false>(
        ctx, /*page_idx=*/0, space, gmat,
        linalg::MakeTensorView(ctx, gpair.ConstHostSpan(), gpair.Size()), row_set_collection, nodes,
        force_read_by_column);
    single_build.ReduceHist(nodes);
    single_page = single_build.Histogram()[0];
  }

  for (size_t i = 0; i < single_page.size(); ++i) {
    ASSERT_NEAR(single_page[i].GetGrad(), multi_page[i].GetGrad(), kRtEps);
    ASSERT_NEAR(single_page[i].GetHess(), multi_page[i].GetHess(), kRtEps);
  }
}
}  // anonymous namespace

TEST(CPUHistogram, ExternalMemory) {
  bst_bin_t constexpr kBins = 256;
  Context ctx;

  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, false);
  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, true);

  float sparse_thresh{0.5};
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
  sparse_thresh = std::numeric_limits<float>::quiet_NaN();
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
}
}  // namespace xgboost::tree
