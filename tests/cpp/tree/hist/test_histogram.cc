/*!
 * Copyright 2018-2021 by Contributors
 */
#include <gtest/gtest.h>
#include "../../helpers.h"
#include "../../../../src/tree/hist/histogram.h"
#include "../../../../src/tree/updater_quantile_hist.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT>
void TestAddHistRows(bool is_distributed) {
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat
                           ->GetBatches<GHistIndexMatrix>(
                               BatchParam{GenericParameter::kCpuId, kMaxBins})
                           .begin());

  RegTree tree;

  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  nodes_for_explicit_hist_build_.emplace_back(3, tree.GetDepth(3), 0.0f);
  nodes_for_explicit_hist_build_.emplace_back(4, tree.GetDepth(4), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(5, tree.GetDepth(5), 0.0f);
  nodes_for_subtraction_trick_.emplace_back(6, tree.GetDepth(6), 0.0f);

  HistogramBuilder<GradientSumT, CPUExpandEntry> histogram_builder;
  histogram_builder.Reset(gmat.cut.TotalBins(), kMaxBins, omp_get_max_threads(),
                          is_distributed);
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
  TestAddHistRows<float>(true);
  TestAddHistRows<double>(true);

  TestAddHistRows<float>(false);
  TestAddHistRows<double>(false);
}

template <typename GradientSumT>
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

  HistogramBuilder<GradientSumT, CPUExpandEntry> histogram;
  uint32_t total_bins = gmat.cut.Ptrs().back();
  histogram.Reset(total_bins, kMaxBins, omp_get_max_threads(), is_distributed);

  RowSetCollection row_set_collection_;
  {
    row_set_collection_.Clear();
    std::vector<size_t> &row_indices = *row_set_collection_.Data();
    row_indices.resize(kNRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
    row_set_collection_.Init();
  }

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

  std::vector<common::GHistRow<GradientSumT>> target_hists(n_nodes);
  for (size_t i = 0; i < nodes_for_explicit_hist_build_.size(); ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    target_hists[i] = histogram.Histogram()[nid];
  }

  // set values to specific nodes hist
  std::vector<size_t> n_ids = {1, 2};
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    GradientSumT *p_hist = reinterpret_cast<GradientSumT *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = 2 * bin_id;
    }
  }
  n_ids[0] = 3;
  n_ids[1] = 5;
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    GradientSumT *p_hist = reinterpret_cast<GradientSumT *>(this_hist.data());
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
                                 nodes_for_subtraction_trick_, starting_index,
                                 sync_count);
  }

  using GHistRowT = common::GHistRow<GradientSumT>;
  auto check_hist = [](const GHistRowT parent, const GHistRowT left,
                       const GHistRowT right, size_t begin, size_t end) {
    const GradientSumT *p_parent =
        reinterpret_cast<const GradientSumT *>(parent.data());
    const GradientSumT *p_left =
        reinterpret_cast<const GradientSumT *>(left.data());
    const GradientSumT *p_right =
        reinterpret_cast<const GradientSumT *>(right.data());
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
  TestSyncHist<float>(true);
  TestSyncHist<double>(true);

  TestSyncHist<float>(false);
  TestSyncHist<double>(false);
}

template <typename GradientSumT>
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
  HistogramBuilder<GradientSumT, CPUExpandEntry> histogram;
  histogram.Reset(total_bins, kMaxBins, omp_get_max_threads(), is_distributed);

  RegTree tree;

  RowSetCollection row_set_collection_;
  row_set_collection_.Clear();
  std::vector<size_t> &row_indices = *row_set_collection_.Data();
  row_indices.resize(kNRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection_.Init();

  CPUExpandEntry node(CPUExpandEntry::kRootNid, tree.GetDepth(0), 0.0f);
  std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;
  nodes_for_explicit_hist_build_.push_back(node);
  histogram.BuildHist(p_fmat.get(), &tree, row_set_collection_,
                      nodes_for_explicit_hist_build_, {}, gpair);

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
  TestBuildHistogram<float>(true);
  TestBuildHistogram<double>(true);

  TestBuildHistogram<float>(false);
  TestBuildHistogram<double>(false);
}
}  // namespace tree
}  // namespace xgboost
