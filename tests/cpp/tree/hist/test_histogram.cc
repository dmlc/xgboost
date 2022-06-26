/*!
 * Copyright 2018-2022 by Contributors
 */
#include <gtest/gtest.h>

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
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(BatchParam{kMaxBins, 0.5}).begin());

  RegTree tree;

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build_.emplace_back(3, tree.GetDepth(3), 0.0f);
  nodes_for_explicit_hist_build_.emplace_back(4, tree.GetDepth(4), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(5, tree.GetDepth(5), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(6, tree.GetDepth(6), 0.0f);

  HistogramBuilder<CPUExpandEntry> histogram_builder;
  histogram_builder.Reset(gmat.cut.TotalBins(), kMaxBins,
                          omp_get_max_threads(), 8, is_distributed);
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

  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;
  RegTree tree;

  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat
                           ->GetBatches<GHistIndexMatrix>(
                               BatchParam{GenericParameter::kCpuId, kMaxBins})
                           .begin());

  HistogramBuilder<CPUExpandEntry> histogram;
  uint32_t total_bins = gmat.cut.Ptrs().back();
  histogram.Reset(total_bins, kMaxBins, 1, 1, 3, is_distributed);

  // level 0
  nodes_for_explicit_hist_build_.emplace_back(0, tree.GetDepth(0), 0.0f);
  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();

  // level 1
  nodes_for_explicit_hist_build_.emplace_back(tree[0].LeftChild(),
                                              tree.GetDepth(1), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(tree[0].RightChild(),
                                            tree.GetDepth(2), 0.0f);

  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  // level 2
  nodes_for_explicit_hist_build_.emplace_back(3, tree.GetDepth(3), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(4, tree.GetDepth(4), 0.0f);
  nodes_for_explicit_hist_build_.emplace_back(5, tree.GetDepth(5), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(6, tree.GetDepth(6), 0.0f);

  histogram.AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build_,
                        nodes_for_subtraction_trick_, &tree);

  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  ASSERT_EQ(n_nodes, 2ul);

  common::OptPartitionBuilder opt_partition_builder;
  opt_partition_builder.template Init<uint8_t>(gmat, gmat.Transpose(), &tree,
                                          1, 3, false);
  // initiat work
  for (auto & threads_id_for_node : opt_partition_builder.threads_id_for_nodes) {
    threads_id_for_node.second.resize(1, 0);
  }
  opt_partition_builder.node_id_for_threads.resize(1);
  opt_partition_builder.node_id_for_threads[0].resize(2);
  opt_partition_builder.node_id_for_threads[0][0] = 3;
  opt_partition_builder.node_id_for_threads[0][1] = 5;
  histogram.Buffer().AllocateHistForLocalThread(opt_partition_builder.node_id_for_threads[0], 0);

  // set values to specific nodes hist
  std::vector<size_t> n_ids = {1, 2};
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    double *p_hist = reinterpret_cast<double *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = 2 * bin_id;
    }
  }
  for (size_t i = 0; i < nodes_for_explicit_hist_build_.size(); ++i) {
    double *p_hist = reinterpret_cast<double *>(
      (*(histogram.Buffer().GetHistBuffer()))[0][i].data());
    ASSERT_EQ(2 * total_bins, (*(histogram.Buffer().GetHistBuffer()))[0][i].size());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = bin_id;
    }
  }
  // sync hist
  if (is_distributed) {
    histogram.SyncHistogramDistributed(&tree, nodes_for_explicit_hist_build_,
                                       nodes_for_subtraction_trick_,
                                       starting_index, sync_count, &opt_partition_builder);
  } else {
    histogram.SyncHistogramLocal(&tree, nodes_for_explicit_hist_build_,
// <<<<<<< HEAD
                                 nodes_for_subtraction_trick_, starting_index,
                                 sync_count, &opt_partition_builder);
// =======
//                                  nodes_for_subtraction_trick_);
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
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

void TestBuildHistogram(bool is_distributed) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat
                           ->GetBatches<GHistIndexMatrix>(
                               BatchParam{GenericParameter::kCpuId, kMaxBins})
                           .begin());
  uint32_t total_bins = gmat.cut.Ptrs().back();

  static double constexpr kEps = 1e-6;
  std::vector<GradientPair> gpair = {
      {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
      {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f}};

  bst_node_t nid = 0;
  HistogramBuilder<CPUExpandEntry> histogram;
  histogram.Reset(total_bins, kMaxBins,
                  omp_get_max_threads(), 1, 8, is_distributed);

  RegTree tree;

  CPUExpandEntry node(RegTree::kRoot, tree.GetDepth(0), 0.0f);
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  nodes_for_explicit_hist_build_.push_back(node);
  common::OptPartitionBuilder opt_partition_builder;

  opt_partition_builder.template Init<uint8_t>(gmat, gmat.Transpose(), &tree,
    omp_get_max_threads(), 3, false);
  std::vector<uint16_t> node_ids(kNRows, 0);
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(
           {GenericParameter::kCpuId, kMaxBins})) {
    histogram.template BuildHist<uint8_t, true>(0, gidx, &tree,
                      nodes_for_explicit_hist_build_, {}, gpair,
                      &opt_partition_builder, &node_ids);
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
  TestBuildHistogram(true);
  TestBuildHistogram(false);
}

namespace {
void TestHistogramCategorical(size_t n_categories) {
  size_t constexpr kRows = 340;
  int32_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, n_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  BatchParam batch_param{0, static_cast<int32_t>(kBins)};

  RegTree tree;
  CPUExpandEntry node(RegTree::kRoot, tree.GetDepth(0), 0.0f);
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
  // auto const& spage = *(cat_m->GetBatches<SparsePage>().begin());
  for (auto const &gidx : cat_m->GetBatches<GHistIndexMatrix>({kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();

    common::OptPartitionBuilder opt_partition_builder;
    auto n_rows_in_node = gidx.Size();
    opt_partition_builder.template Init<uint8_t>(gidx, gidx.Transpose(), &tree,
    omp_get_max_threads(), 8, false);
    std::vector<uint16_t> node_ids(n_rows_in_node, 0);

    cat_hist.Reset(total_bins, kBins,
                      omp_get_max_threads(), 1, 8, false);
    cat_hist.template BuildHist<uint8_t, true>(0, gidx, &tree,
                          nodes_for_explicit_hist_build, {},
                          gpair.HostVector(), &opt_partition_builder, &node_ids);
  }

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, n_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, n_categories);
  HistogramBuilder<CPUExpandEntry> onehot_hist;
  for (auto const &gidx : encode_m->GetBatches<GHistIndexMatrix>({kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();

    common::OptPartitionBuilder opt_partition_builder;
    auto n_rows_in_node = gidx.Size();
    opt_partition_builder.template Init<uint8_t>(gidx, gidx.Transpose(), &tree,
    omp_get_max_threads(), 8, false);
    std::vector<uint16_t> node_ids(n_rows_in_node, 0);

    onehot_hist.Reset(total_bins, kBins,
                      omp_get_max_threads(), 1, 8, false);
    onehot_hist.template BuildHist<uint8_t, true>(0, gidx, &tree,
                          nodes_for_explicit_hist_build, {},
                          gpair.HostVector(), &opt_partition_builder, &node_ids);
  }

  auto cat = cat_hist.Histogram()[0];
  auto onehot = onehot_hist.Histogram()[0];
  ValidateCategoricalHistogram(n_categories, onehot, cat);
}
}  // anonymous namespace

TEST(CPUHistogram, Categorical) {
  for (size_t n_categories = 2; n_categories < 8; ++n_categories) {
    TestHistogramCategorical(n_categories);
  }
}
namespace {
void TestHistogramExternalMemory(BatchParam batch_param, bool is_approx) {
  size_t constexpr kEntries = 1 << 16;
  auto m = CreateSparsePageDMatrix(kEntries, "cache");

  std::vector<float> hess(m->Info().num_row_, 1.0);
  if (is_approx) {
    batch_param.hess = hess;
  }

  size_t total_bins{0};

  for (auto const &page : m->GetBatches<GHistIndexMatrix>(batch_param)) {
    total_bins = page.cut.TotalBins();
  }

  common::GHistRow multi_page;
  HistogramBuilder<CPUExpandEntry> multi_build;
  {
    auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);
    auto const &h_gpair = gpair.HostVector();
    RegTree tree;
    std::vector<CPUExpandEntry> nodes;
    nodes.emplace_back(0, tree.GetDepth(0), 0.0f);
    /**
     * Multi page
     */
    int32_t constexpr kBins = 256;
    std::vector<float> hess(m->Info().num_row_, 1.0);

    multi_build.Reset(total_bins, kBins,
                      omp_get_max_threads(), 2, 8, false);

    size_t page_idx{0};
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(batch_param)) {
      common::OptPartitionBuilder opt_partition_builder;
      auto n_rows_in_node = page.Size();
      opt_partition_builder.template Init<uint8_t>(page, page.Transpose(), &tree,
        omp_get_max_threads(), 8, false);
      std::vector<uint16_t> node_ids(n_rows_in_node, 0);
      multi_build.template BuildHist<uint8_t, true>(page_idx, page, &tree, nodes, {},
                            h_gpair, &opt_partition_builder, &node_ids);
      ++page_idx;
    }
    ASSERT_EQ(page_idx, 2);
    multi_page = multi_build.Histogram()[0];
  }

  HistogramBuilder<CPUExpandEntry> single_build;
  common::GHistRow single_page;
  {
    auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);
    auto const &h_gpair = gpair.HostVector();
    RegTree tree;
    std::vector<CPUExpandEntry> nodes;
    nodes.emplace_back(0, tree.GetDepth(0), 0.0f);
    /**
     * Single page
     */
// <<<<<<< HEAD
    int32_t constexpr kBins = 256;
    single_build.Reset(total_bins, kBins,
                       omp_get_max_threads(), 1, 8, false);
    // single_build.Reset(total_bins, batch_param, common::OmpGetNumThreads(0), 1, false);
    SparsePage concat;
    std::vector<float> hess(m->Info().num_row_, 1.0f);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      concat.Push(page);
    }

    auto cut = common::SketchOnDMatrix(m.get(), batch_param.max_bin, common::OmpGetNumThreads(0),
                                       false, hess);
    GHistIndexMatrix gmat;
    gmat.Init(concat, {}, cut, batch_param.max_bin, false, std::numeric_limits<double>::quiet_NaN(),
              common::OmpGetNumThreads(0));

    // GHistIndexMatrix gmat;
    // std::vector<float> hess(m->Info().num_row_, 1.0f);
    // gmat.Init(m.get(), batch_param.max_bin, std::numeric_limits<double>::quiet_NaN(), false,
    //           common::OmpGetNumThreads(0), hess);
    size_t n_batches{0};
      common::OptPartitionBuilder opt_partition_builder;
      opt_partition_builder.template Init<uint8_t>(gmat, gmat.Transpose(), &tree,
        omp_get_max_threads(), 8, false);
      std::vector<uint16_t> node_ids(kEntries, 0);
      single_build.template BuildHist<uint8_t, true>(0, gmat, &tree, nodes, {},
                             h_gpair, &opt_partition_builder, &node_ids);
      n_batches ++;
    ASSERT_EQ(n_batches, 1);
// =======
//     common::RowSetCollection row_set_collection;
//     InitRowPartitionForTest(&row_set_collection, n_samples);

//     single_build.Reset(total_bins, batch_param, common::OmpGetNumThreads(0), 1, false);
//     SparsePage concat;
//     std::vector<float> hess(m->Info().num_row_, 1.0f);
//     for (auto const& page : m->GetBatches<SparsePage>()) {
//       concat.Push(page);
//     }

//     auto cut = common::SketchOnDMatrix(m.get(), batch_param.max_bin, common::OmpGetNumThreads(0),
//                                        false, hess);
//     GHistIndexMatrix gmat;
//     gmat.Init(concat, {}, cut, batch_param.max_bin, false, std::numeric_limits<double>::quiet_NaN(),
//               common::OmpGetNumThreads(0));
//     single_build.BuildHist(0, gmat, &tree, row_set_collection, nodes, {}, h_gpair);
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
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
  TestHistogramExternalMemory(BatchParam{kBins, common::Span<float>{}, false}, true);

  float sparse_thresh{0.5};
  TestHistogramExternalMemory({kBins, sparse_thresh}, false);
  sparse_thresh = std::numeric_limits<float>::quiet_NaN();
  TestHistogramExternalMemory({kBins, sparse_thresh}, false);

}
}  // namespace tree
}  // namespace xgboost
