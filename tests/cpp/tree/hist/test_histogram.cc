/**
 * Copyright 2018-2023 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context

#include <limits>

#include "../../../../src/common/categorical.h"
#include "../../../../src/common/row_set.h"
#include "../../../../src/tree/hist/expand_entry.h"
#include "../../../../src/tree/hist/histogram.h"
#include "../../categorical_helpers.h"
#include "../../helpers.h"

namespace xgboost {
namespace tree {
namespace {
void InitRowPartitionForTest(common::RowSetCollection *row_set, size_t n_samples, size_t base_rowid = 0) {
  auto &row_indices = *row_set->Data();
  row_indices.resize(n_samples);
  std::iota(row_indices.begin(), row_indices.end(), base_rowid);
  row_set->Init();
}
}  // anonymous namespace

void TestAddHistRows(bool is_distributed) {
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  RegTree tree;

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build_.emplace_back(3, tree.GetDepth(3));
  nodes_for_explicit_hist_build_.emplace_back(4, tree.GetDepth(4));
  nodes_for_subtraction_trick_.emplace_back(5, tree.GetDepth(5));
  nodes_for_subtraction_trick_.emplace_back(6, tree.GetDepth(6));

  HistogramBuilder<CPUExpandEntry> histogram_builder;
  histogram_builder.Reset(gmat.cut.TotalBins(), {kMaxBins, 0.5}, omp_get_max_threads(), 1,
                          is_distributed, false);
  histogram_builder.AddHistRows(&starting_index, &sync_count,
                                nodes_for_explicit_hist_build_,
                                nodes_for_subtraction_trick_, &tree);

  ASSERT_EQ(sync_count, 2);
  ASSERT_EQ(starting_index, 3);

  for (const CPUExpandEntry &node : nodes_for_explicit_hist_build_) {
    ASSERT_EQ(histogram_builder.Histogram().RowExists(node.nid), true);
  }
  for (const CPUExpandEntry &node : nodes_for_subtraction_trick_) {
    ASSERT_EQ(histogram_builder.Histogram().RowExists(node.nid), true);
  }
}


TEST(CPUHistogram, AddRows) {
  TestAddHistRows(true);
  TestAddHistRows(false);
}

void TestSyncHist(bool is_distributed) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);

  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;
  RegTree tree;

  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  HistogramBuilder<CPUExpandEntry> histogram;
  uint32_t total_bins = gmat.cut.Ptrs().back();
  histogram.Reset(total_bins, {kMaxBins, 0.5}, omp_get_max_threads(), 1, is_distributed, false);

  common::RowSetCollection row_set_collection_;
  {
    row_set_collection_.Clear();
    std::vector<size_t> &row_indices = *row_set_collection_.Data();
    row_indices.resize(kNRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
    row_set_collection_.Init();
  }

  // level 0
  nodes_for_explicit_hist_build_.emplace_back(0, tree.GetDepth(0));
  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();

  // level 1
  nodes_for_explicit_hist_build_.emplace_back(tree[0].LeftChild(), tree.GetDepth(1));
  nodes_for_subtraction_trick_.emplace_back(tree[0].RightChild(), tree.GetDepth(2));

  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  // level 2
  nodes_for_explicit_hist_build_.emplace_back(3, tree.GetDepth(3));
  nodes_for_subtraction_trick_.emplace_back(4, tree.GetDepth(4));
  nodes_for_explicit_hist_build_.emplace_back(5, tree.GetDepth(5));
  nodes_for_subtraction_trick_.emplace_back(6, tree.GetDepth(6));

  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  ASSERT_EQ(n_nodes, 2ul);
  row_set_collection_.AddSplit(0, tree[0].LeftChild(), tree[0].RightChild(), 4,
                               4);
  row_set_collection_.AddSplit(1, tree[1].LeftChild(), tree[1].RightChild(), 2,
                               2);
  row_set_collection_.AddSplit(2, tree[2].LeftChild(), tree[2].RightChild(), 2,
                               2);

  common::BlockedSpace2d space(
      n_nodes,
      [&](size_t node) {
        const int32_t nid = nodes_for_explicit_hist_build_[node].nid;
        return row_set_collection_[nid].Size();
      },
      256);

  std::vector<common::GHistRow> target_hists(n_nodes);
  for (size_t i = 0; i < nodes_for_explicit_hist_build_.size(); ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    target_hists[i] = histogram.Histogram()[nid];
  }

  // set values to specific nodes hist
  std::vector<size_t> n_ids = {1, 2};
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    double *p_hist = reinterpret_cast<double *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = 2 * bin_id;
    }
  }
  n_ids[0] = 3;
  n_ids[1] = 5;
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    double *p_hist = reinterpret_cast<double *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = bin_id;
    }
  }

  histogram.Buffer().Reset(1, n_nodes, space, target_hists);
  // sync hist
  if (is_distributed) {
    histogram.SyncHistogramDistributed(&tree, nodes_for_explicit_hist_build_,
                                       nodes_for_subtraction_trick_,
                                       starting_index, sync_count);
  } else {
    histogram.SyncHistogramLocal(&tree, nodes_for_explicit_hist_build_,
                                 nodes_for_subtraction_trick_);
  }

  using GHistRowT = common::GHistRow;
  auto check_hist = [](const GHistRowT parent, const GHistRowT left, const GHistRowT right,
                       size_t begin, size_t end) {
    const double *p_parent = reinterpret_cast<const double *>(parent.data());
    const double *p_left = reinterpret_cast<const double *>(left.data());
    const double *p_right = reinterpret_cast<const double *>(right.data());
    for (size_t i = 2 * begin; i < 2 * end; ++i) {
      ASSERT_EQ(p_parent[i], p_left[i] + p_right[i]);
    }
  };
  size_t node_id = 0;
  for (const CPUExpandEntry &node : nodes_for_explicit_hist_build_) {
    auto this_hist = histogram.Histogram()[node.nid];
    const size_t parent_id = tree[node.nid].Parent();
    const size_t subtraction_node_id =
        nodes_for_subtraction_trick_[node_id].nid;
    auto parent_hist = histogram.Histogram()[parent_id];
    auto sibling_hist = histogram.Histogram()[subtraction_node_id];

    check_hist(parent_hist, this_hist, sibling_hist, 0, total_bins);
    ++node_id;
  }
  node_id = 0;
  for (const CPUExpandEntry &node : nodes_for_subtraction_trick_) {
    auto this_hist = histogram.Histogram()[node.nid];
    const size_t parent_id = tree[node.nid].Parent();
    const size_t subtraction_node_id =
        nodes_for_explicit_hist_build_[node_id].nid;
    auto parent_hist = histogram.Histogram()[parent_id];
    auto sibling_hist = histogram.Histogram()[subtraction_node_id];

    check_hist(parent_hist, this_hist, sibling_hist, 0, total_bins);
    ++node_id;
  }
}

TEST(CPUHistogram, SyncHist) {
  TestSyncHist(true);
  TestSyncHist(false);
}

void TestBuildHistogram(bool is_distributed, bool force_read_by_column, bool is_col_split) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
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

  bst_node_t nid = 0;
  HistogramBuilder<CPUExpandEntry> histogram;
  histogram.Reset(total_bins, {kMaxBins, 0.5}, omp_get_max_threads(), 1, is_distributed,
                  is_col_split);

  RegTree tree;

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kNRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(0)};
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build;
  nodes_for_explicit_hist_build.push_back(node);
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(&ctx, {kMaxBins, 0.5})) {
    histogram.BuildHist(0, gidx, &tree, row_set_collection, nodes_for_explicit_hist_build, {},
                        gpair, force_read_by_column);
  }

  // Check if number of histogram bins is correct
  ASSERT_EQ(histogram.Histogram()[nid].size(), gmat.cut.Ptrs().back());
  std::vector<GradientPairPrecise> histogram_expected(histogram.Histogram()[nid].size());

  // Compute the correct histogram (histogram_expected)
  CHECK_EQ(gpair.size(), kNRows);
  for (size_t rid = 0; rid < kNRows; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      const size_t bin_id = gmat.index[i];
      histogram_expected[bin_id] += GradientPairPrecise(gpair[rid]);
    }
  }

  // Now validate the computed histogram returned by BuildHist
  for (size_t i = 0; i < histogram.Histogram()[nid].size(); ++i) {
    GradientPairPrecise sol = histogram_expected[i];
    ASSERT_NEAR(sol.GetGrad(), histogram.Histogram()[nid][i].GetGrad(), kEps);
    ASSERT_NEAR(sol.GetHess(), histogram.Histogram()[nid][i].GetHess(), kEps);
  }
}

TEST(CPUHistogram, BuildHist) {
  TestBuildHistogram(true, false, false);
  TestBuildHistogram(false, false, false);
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
  int32_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, n_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);

  BatchParam batch_param{0, static_cast<int32_t>(kBins)};

  RegTree tree;
  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(0)};
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build;
  nodes_for_explicit_hist_build.push_back(node);

  auto gpair = GenerateRandomGradients(kRows, 0, 2);

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  /**
   * Generate hist with cat data.
   */
  HistogramBuilder<CPUExpandEntry> cat_hist;
  for (auto const &gidx : cat_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    cat_hist.Reset(total_bins, {kBins, 0.5}, omp_get_max_threads(), 1, false, false);
    cat_hist.BuildHist(0, gidx, &tree, row_set_collection, nodes_for_explicit_hist_build, {},
                       gpair.HostVector(), force_read_by_column);
  }

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, n_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, n_categories);
  HistogramBuilder<CPUExpandEntry> onehot_hist;
  for (auto const &gidx : encode_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    onehot_hist.Reset(total_bins, {kBins, 0.5}, omp_get_max_threads(), 1, false, false);
    onehot_hist.BuildHist(0, gidx, &tree, row_set_collection, nodes_for_explicit_hist_build, {},
                          gpair.HostVector(), force_read_by_column);
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
  size_t constexpr kEntries = 1 << 16;
  auto m = CreateSparsePageDMatrix(kEntries, "cache");

  std::vector<float> hess(m->Info().num_row_, 1.0);
  if (is_approx) {
    batch_param.hess = hess;
  }

  std::vector<size_t> partition_size(1, 0);
  size_t total_bins{0};
  size_t n_samples{0};

  auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);
  auto const &h_gpair = gpair.HostVector();

  RegTree tree;
  std::vector<CPUExpandEntry> nodes;
  nodes.emplace_back(0, tree.GetDepth(0));

  common::GHistRow multi_page;
  HistogramBuilder<CPUExpandEntry> multi_build;
  {
    /**
     * Multi page
     */
    std::vector<common::RowSetCollection> rows_set;
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      CHECK_LT(page.base_rowid, m->Info().num_row_);
      auto n_rows_in_node = page.Size();
      partition_size[0] = std::max(partition_size[0], n_rows_in_node);
      total_bins = page.cut.TotalBins();
      n_samples += n_rows_in_node;

      rows_set.emplace_back();
      InitRowPartitionForTest(&rows_set.back(), n_rows_in_node, page.base_rowid);
    }
    ASSERT_EQ(n_samples, m->Info().num_row_);

    common::BlockedSpace2d space{
        1, [&](size_t nidx_in_set) { return partition_size.at(nidx_in_set); },
        256};

    multi_build.Reset(total_bins, batch_param, ctx->Threads(), rows_set.size(), false, false);

    size_t page_idx{0};
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      multi_build.BuildHist(page_idx, space, page, &tree, rows_set.at(page_idx), nodes, {}, h_gpair,
                            force_read_by_column);
      ++page_idx;
    }
    ASSERT_EQ(page_idx, 2);
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

    single_build.Reset(total_bins, batch_param, ctx->Threads(), 1, false, false);
    SparsePage concat;
    std::vector<float> hess(m->Info().num_row_, 1.0f);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      concat.Push(page);
    }

    auto cut = common::SketchOnDMatrix(ctx, m.get(), batch_param.max_bin, false, hess);
    GHistIndexMatrix gmat(concat, {}, cut, batch_param.max_bin, false,
                          std::numeric_limits<double>::quiet_NaN(), ctx->Threads());
    single_build.BuildHist(0, gmat, &tree, row_set_collection, nodes, {}, h_gpair, force_read_by_column);
    single_page = single_build.Histogram()[0];
  }

  for (size_t i = 0; i < single_page.size(); ++i) {
    ASSERT_NEAR(single_page[i].GetGrad(), multi_page[i].GetGrad(), kRtEps);
    ASSERT_NEAR(single_page[i].GetHess(), multi_page[i].GetHess(), kRtEps);
  }
}
}  // anonymous namespace

TEST(CPUHistogram, ExternalMemory) {
  int32_t constexpr kBins = 256;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);

  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, false);
  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, true);

  float sparse_thresh{0.5};
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
  sparse_thresh = std::numeric_limits<float>::quiet_NaN();
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
}
}  // namespace tree
}  // namespace xgboost
