/**
 * Copyright 2018-2023 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                // for bst_node_t, bst_bin_t, Gradient...
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for BatchIterator, BatchSet, DMatrix
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/linalg.h>              // for MakeTensorView
#include <xgboost/logging.h>             // for Error, LogCheck_EQ, LogCheck_LT
#include <xgboost/span.h>                // for Span, operator!=
#include <xgboost/tree_model.h>          // for RegTree

#include <algorithm>   // for max
#include <cstddef>     // for size_t
#include <cstdint>     // for int32_t, uint32_t
#include <iterator>    // for back_inserter
#include <limits>      // for numeric_limits
#include <memory>      // for shared_ptr, allocator, unique_ptr
#include <numeric>     // for iota, accumulate
#include <vector>      // for vector

#include "../../../../src/collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../../../../src/common/hist_util.h"             // for GHistRow, HistogramCuts, Sketch...
#include "../../../../src/common/ref_resource_view.h"     // for RefResourceView
#include "../../../../src/common/row_set.h"               // for RowSetCollection
#include "../../../../src/common/threading_utils.h"       // for BlockedSpace2d
#include "../../../../src/data/gradient_index.h"          // for GHistIndexMatrix
#include "../../../../src/tree/common_row_partitioner.h"  // for CommonRowPartitioner
#include "../../../../src/tree/hist/expand_entry.h"       // for CPUExpandEntry
#include "../../../../src/tree/hist/hist_cache.h"         // for BoundedHistCollection
#include "../../../../src/tree/hist/histogram.h"          // for HistogramBuilder
#include "../../../../src/tree/hist/param.h"              // for HistMakerTrainParam
#include "../../categorical_helpers.h"                    // for OneHotEncodeFeature
#include "../../collective/test_worker.h"                 // for TestDistributedGlobal
#include "../../helpers.h"                                // for RandomDataGenerator, GenerateRa...

namespace xgboost::tree {
namespace {
void InitRowPartitionForTest(common::RowSetCollection *row_set, size_t n_samples, size_t base_rowid = 0) {
  auto &row_indices = *row_set->Data();
  row_indices.resize(n_samples);
  std::iota(row_indices.begin(), row_indices.end(), base_rowid);
  row_set->Init();
}
}  // anonymous namespace

void TestAddHistRows(bool is_distributed) {
  Context ctx;
  std::vector<bst_node_t> nodes_to_build;
  std::vector<bst_node_t> nodes_to_sub;

  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  RegTree tree;

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_to_build.emplace_back(3);
  nodes_to_build.emplace_back(4);
  nodes_to_sub.emplace_back(5);
  nodes_to_sub.emplace_back(6);

  HistMakerTrainParam hist_param;
  HistogramBuilder histogram_builder;
  histogram_builder.Reset(&ctx, gmat.cut.TotalBins(), {kMaxBins, 0.5}, is_distributed, false,
                          &hist_param);
  histogram_builder.AddHistRows(&tree, &nodes_to_build, &nodes_to_sub, false);

  for (bst_node_t const &nidx : nodes_to_build) {
    ASSERT_TRUE(histogram_builder.Histogram().HistogramExists(nidx));
  }
  for (bst_node_t const &nidx : nodes_to_sub) {
    ASSERT_TRUE(histogram_builder.Histogram().HistogramExists(nidx));
  }
}


TEST(CPUHistogram, AddRows) {
  TestAddHistRows(true);
  TestAddHistRows(false);
}

void TestSyncHist(bool is_distributed) {
  std::size_t constexpr kNRows = 8, kNCols = 16;
  bst_bin_t constexpr kMaxBins = 4;
  Context ctx;

  std::vector<bst_bin_t> nodes_for_explicit_hist_build;
  std::vector<bst_bin_t> nodes_for_subtraction_trick;
  RegTree tree;

  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  HistogramBuilder histogram;
  uint32_t total_bins = gmat.cut.Ptrs().back();
  HistMakerTrainParam hist_param;
  histogram.Reset(&ctx, total_bins, {kMaxBins, 0.5}, is_distributed, false, &hist_param);

  common::RowSetCollection row_set_collection;
  {
    row_set_collection.Clear();
    std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
    row_indices.resize(kNRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
    row_set_collection.Init();
  }

  // level 0
  nodes_for_explicit_hist_build.emplace_back(0);
  histogram.AddHistRows(&tree, &nodes_for_explicit_hist_build, &nodes_for_subtraction_trick, false);

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build.clear();
  nodes_for_subtraction_trick.clear();

  // level 1
  nodes_for_explicit_hist_build.emplace_back(tree[0].LeftChild());
  nodes_for_subtraction_trick.emplace_back(tree[0].RightChild());

  histogram.AddHistRows(&tree, &nodes_for_explicit_hist_build, &nodes_for_subtraction_trick, false);

  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  nodes_for_explicit_hist_build.clear();
  nodes_for_subtraction_trick.clear();
  // level 2
  nodes_for_explicit_hist_build.emplace_back(3);
  nodes_for_subtraction_trick.emplace_back(4);
  nodes_for_explicit_hist_build.emplace_back(5);
  nodes_for_subtraction_trick.emplace_back(6);

  histogram.AddHistRows(&tree, &nodes_for_explicit_hist_build, &nodes_for_subtraction_trick, false);

  const size_t n_nodes = nodes_for_explicit_hist_build.size();
  ASSERT_EQ(n_nodes, 2ul);
  row_set_collection.AddSplit(0, tree[0].LeftChild(), tree[0].RightChild(), 4, 4);
  row_set_collection.AddSplit(1, tree[1].LeftChild(), tree[1].RightChild(), 2, 2);
  row_set_collection.AddSplit(2, tree[2].LeftChild(), tree[2].RightChild(), 2, 2);

  common::BlockedSpace2d space(
      n_nodes,
      [&](std::size_t nidx_in_set) {
        bst_node_t nidx = nodes_for_explicit_hist_build[nidx_in_set];
        return row_set_collection[nidx].Size();
      },
      256);

  std::vector<common::GHistRow> target_hists(n_nodes);
  for (size_t i = 0; i < nodes_for_explicit_hist_build.size(); ++i) {
    bst_node_t nidx = nodes_for_explicit_hist_build[i];
    target_hists[i] = histogram.Histogram()[nidx];
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
  histogram.SyncHistogram(&ctx, &tree, nodes_for_explicit_hist_build, nodes_for_subtraction_trick);

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
  for (auto const &nidx : nodes_for_explicit_hist_build) {
    auto this_hist = histogram.Histogram()[nidx];
    const size_t parent_id = tree[nidx].Parent();
    const size_t subtraction_node_id = nodes_for_subtraction_trick[node_id];
    auto parent_hist = histogram.Histogram()[parent_id];
    auto sibling_hist = histogram.Histogram()[subtraction_node_id];

    check_hist(parent_hist, this_hist, sibling_hist, 0, total_bins);
    ++node_id;
  }
  node_id = 0;
  for (auto const &nidx : nodes_for_subtraction_trick) {
    auto this_hist = histogram.Histogram()[nidx];
    const size_t parent_id = tree[nidx].Parent();
    const size_t subtraction_node_id = nodes_for_explicit_hist_build[node_id];
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
  Context ctx;
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
  HistogramBuilder histogram;
  HistMakerTrainParam hist_param;
  histogram.Reset(&ctx, total_bins, {kMaxBins, 0.5}, is_distributed, is_col_split, &hist_param);

  RegTree tree;

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kNRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(0)};
  std::vector<bst_node_t> nodes_to_build{node.nid};
  std::vector<bst_node_t> dummy_sub;

  histogram.AddHistRows(&tree, &nodes_to_build, &dummy_sub, false);
  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); }, 256};
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(&ctx, {kMaxBins, 0.5})) {
    histogram.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                        linalg::MakeTensorView(&ctx, gpair, gpair.size()), force_read_by_column);
  }
  histogram.SyncHistogram(&ctx, &tree, nodes_to_build, {});

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
  collective::TestDistributedGlobal(kWorkers, [] { TestBuildHistogram(true, true, true); });
  collective::TestDistributedGlobal(kWorkers, [] { TestBuildHistogram(true, false, true); });
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

  BatchParam batch_param{0, kBins};

  RegTree tree;
  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(RegTree::kRoot)};
  std::vector<bst_node_t> nodes_to_build;
  nodes_to_build.push_back(node.nid);

  auto gpair = GenerateRandomGradients(kRows, 0, 2);

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();
  HistMakerTrainParam hist_param;
  std::vector<bst_node_t> dummy_sub;

  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); }, 256};

  /**
   * Generate hist with cat data.
   */
  HistogramBuilder cat_hist;
  for (auto const &gidx : cat_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    cat_hist.Reset(&ctx, total_bins, {kBins, 0.5}, false, false, &hist_param);
    cat_hist.AddHistRows(&tree, &nodes_to_build, &dummy_sub, false);
    cat_hist.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                       linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
                       force_read_by_column);
  }
  cat_hist.SyncHistogram(&ctx, &tree, nodes_to_build, {});

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, n_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, n_categories);
  HistogramBuilder onehot_hist;
  for (auto const &gidx : encode_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    onehot_hist.Reset(&ctx, total_bins, {kBins, 0.5}, false, false, &hist_param);
    onehot_hist.AddHistRows(&tree, &nodes_to_build, &dummy_sub, false);
    onehot_hist.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                          linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
                          force_read_by_column);
  }
  onehot_hist.SyncHistogram(&ctx, &tree, nodes_to_build, {});

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

  std::vector<bst_idx_t> partition_size(1, 0);
  bst_bin_t total_bins{0};
  bst_idx_t n_samples{0};

  auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);
  auto const &h_gpair = gpair.HostVector();

  RegTree tree;
  std::vector<bst_node_t> nodes{RegTree::kRoot};
  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return partition_size.at(nidx_in_set); }, 256};

  common::GHistRow multi_page;
  HistogramBuilder multi_build;
  HistMakerTrainParam hist_param;
  std::vector<bst_node_t> dummy_sub;
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

    multi_build.Reset(ctx, total_bins, batch_param, false, false, &hist_param);
    multi_build.AddHistRows(&tree, &nodes, &dummy_sub, false);
    std::size_t page_idx{0};
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      multi_build.BuildHist(page_idx, space, page, rows_set[page_idx], nodes,
                            linalg::MakeTensorView(ctx, h_gpair, h_gpair.size()),
                            force_read_by_column);
      ++page_idx;
    }
    multi_build.SyncHistogram(ctx, &tree, nodes, {});

    multi_page = multi_build.Histogram()[RegTree::kRoot];
  }

  HistogramBuilder single_build;
  common::GHistRow single_page;
  {
    /**
     * Single page
     */
    common::RowSetCollection row_set_collection;
    InitRowPartitionForTest(&row_set_collection, n_samples);

    single_build.Reset(ctx, total_bins, batch_param, false, false, &hist_param);
    SparsePage concat;
    std::vector<float> hess(m->Info().num_row_, 1.0f);
    for (auto const &page : m->GetBatches<SparsePage>()) {
      concat.Push(page);
    }

    auto cut = common::SketchOnDMatrix(ctx, m.get(), batch_param.max_bin, false, hess);
    GHistIndexMatrix gmat(concat, {}, cut, batch_param.max_bin, false,
                          std::numeric_limits<double>::quiet_NaN(), ctx->Threads());

    single_build.AddHistRows(&tree, &nodes, &dummy_sub, false);
    single_build.BuildHist(0, space, gmat, row_set_collection, nodes,
                           linalg::MakeTensorView(ctx, h_gpair, h_gpair.size()),
                           force_read_by_column);
    single_build.SyncHistogram(ctx, &tree, nodes, {});

    single_page = single_build.Histogram()[RegTree::kRoot];
  }

  for (size_t i = 0; i < single_page.size(); ++i) {
    ASSERT_NEAR(single_page[i].GetGrad(), multi_page[i].GetGrad(), kRtEps);
    ASSERT_NEAR(single_page[i].GetHess(), multi_page[i].GetHess(), kRtEps);
  }
}
}  // anonymous namespace

TEST(CPUHistogram, ExternalMemory) {
  int32_t constexpr kBins = 256;
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

namespace {
class OverflowTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 public:
  std::vector<GradientPairPrecise> TestOverflow(bool limit, bool is_distributed,
                                                bool is_col_split) {
    bst_bin_t constexpr kBins = 256;
    Context ctx;
    HistMakerTrainParam hist_param;
    if (limit) {
      hist_param.Init(Args{{"max_cached_hist_node", "1"}});
    }

    std::shared_ptr<DMatrix> Xy =
        is_col_split ? RandomDataGenerator{8192, 16, 0.5}.GenerateDMatrix(true)
                     : RandomDataGenerator{8192, 16, 0.5}.Bins(kBins).GenerateQuantileDMatrix(true);
    if (is_col_split) {
      Xy =
          std::shared_ptr<DMatrix>{Xy->SliceCol(collective::GetWorldSize(), collective::GetRank())};
    }

    double sparse_thresh{TrainParam::DftSparseThreshold()};
    auto batch = BatchParam{kBins, sparse_thresh};
    bst_bin_t n_total_bins{0};
    float split_cond{0};
    for (auto const &page : Xy->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      n_total_bins = page.cut.TotalBins();
      // use a cut point in the second column for split
      split_cond = page.cut.Values()[kBins + kBins / 2];
    }

    RegTree tree;
    MultiHistogramBuilder hist_builder;
    CHECK_EQ(Xy->Info().IsColumnSplit(), is_col_split);

    hist_builder.Reset(&ctx, n_total_bins, tree.NumTargets(), batch, is_distributed,
                       Xy->Info().IsColumnSplit(), &hist_param);

    std::vector<CommonRowPartitioner> partitioners;
    partitioners.emplace_back(&ctx, Xy->Info().num_row_, /*base_rowid=*/0,
                              Xy->Info().IsColumnSplit());

    auto gpair = GenerateRandomGradients(Xy->Info().num_row_, 0.0, 1.0);

    CPUExpandEntry best;
    hist_builder.BuildRootHist(Xy.get(), &tree, partitioners,
                               linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size(), 1),
                               best, batch);

    best.split.Update(1.0f, 1, split_cond, false, false, GradStats{1.0, 1.0}, GradStats{1.0, 1.0});
    tree.ExpandNode(best.nid, best.split.SplitIndex(), best.split.split_value, false,
                    /*base_weight=*/2.0f,
                    /*left_leaf_weight=*/1.0f, /*right_leaf_weight=*/1.0f, best.GetLossChange(),
                    /*sum_hess=*/2.0f, best.split.left_sum.GetHess(),
                    best.split.right_sum.GetHess());

    std::vector<CPUExpandEntry> valid_candidates{best};
    for (auto const &page : Xy->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      partitioners.front().UpdatePosition(&ctx, page, valid_candidates, &tree);
    }
    CHECK_NE(partitioners.front()[tree.LeftChild(best.nid)].Size(), 0);
    CHECK_NE(partitioners.front()[tree.RightChild(best.nid)].Size(), 0);

    hist_builder.BuildHistLeftRight(
        &ctx, Xy.get(), &tree, partitioners, valid_candidates,
        linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size(), 1), batch);

    if (limit) {
      CHECK(!hist_builder.Histogram(0).HistogramExists(best.nid));
    } else {
      CHECK(hist_builder.Histogram(0).HistogramExists(best.nid));
    }

    std::vector<GradientPairPrecise> result;
    auto hist = hist_builder.Histogram(0)[tree.LeftChild(best.nid)];
    std::copy(hist.cbegin(), hist.cend(), std::back_inserter(result));
    hist = hist_builder.Histogram(0)[tree.RightChild(best.nid)];
    std::copy(hist.cbegin(), hist.cend(), std::back_inserter(result));

    return result;
  }

  void RunTest() {
    auto param = GetParam();
    auto res0 = this->TestOverflow(false, std::get<0>(param), std::get<1>(param));
    auto res1 = this->TestOverflow(true, std::get<0>(param), std::get<1>(param));
    ASSERT_EQ(res0, res1);
  }
};

auto MakeParamsForTest() {
  std::vector<std::tuple<bool, bool>> configs;
  for (auto i : {true, false}) {
    for (auto j : {true, false}) {
      configs.emplace_back(i, j);
    }
  }
  return configs;
}
}  // anonymous namespace

TEST_P(OverflowTest, Overflow) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(CPUHistogram, OverflowTest, ::testing::ValuesIn(MakeParamsForTest()));
}  // namespace xgboost::tree
