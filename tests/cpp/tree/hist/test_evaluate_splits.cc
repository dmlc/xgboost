#include <gtest/gtest.h>
#include <xgboost/base.h>
#include "../../../../src/tree/hist/evaluate_splits.h"
#include "../../../../src/tree/updater_quantile_hist.h"
#include "../../../../src/common/hist_util.h"
#include "../../helpers.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT> void TestEvaluateSplits() {
  int static constexpr kRows = 8, kCols = 16;
  auto orig = omp_get_max_threads();
  int32_t n_threads = std::min(omp_get_max_threads(), 4);
  omp_set_num_threads(n_threads);
  auto sampler = std::make_shared<common::ColumnSampler>();

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});

  auto dmat = RandomDataGenerator(kRows, kCols, 0).Seed(3).GenerateDMatrix();

  auto evaluator = HistEvaluator<GradientSumT, CPUExpandEntry>{
      param, dmat->Info(), n_threads, sampler, ObjInfo{ObjInfo::kRegression}};
  common::HistCollection<GradientSumT> hist;
  std::vector<GradientPair> row_gpairs = {
      {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f},  {2.27f, 0.28f},
      {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f}};

  size_t constexpr kMaxBins = 4;
  // dense, no missing values

  GHistIndexMatrix gmat(dmat.get(), kMaxBins);
  common::RowSetCollection row_set_collection;
  std::vector<size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  auto hist_builder = GHistBuilder<GradientSumT>(gmat.cut.Ptrs().back());
  hist.Init(gmat.cut.Ptrs().back());
  hist.AddHistRow(0);
  hist.AllocateAllData();
  hist_builder.template BuildHist<false>(row_gpairs, row_set_collection[0],
                                         gmat, hist[0]);

  // Compute total gradient for all data points
  GradientPairPrecise total_gpair;
  for (const auto &e : row_gpairs) {
    total_gpair += GradientPairPrecise(e);
  }

  RegTree tree;
  std::vector<CPUExpandEntry> entries(1);
  entries.front().nid = 0;
  entries.front().depth = 0;

  evaluator.InitRoot(GradStats{total_gpair});
  evaluator.EvaluateSplits(hist, gmat.cut, {}, tree, &entries);

  auto best_loss_chg =
      evaluator.Evaluator().CalcSplitGain(
          param, 0, entries.front().split.SplitIndex(),
          entries.front().split.left_sum, entries.front().split.right_sum) -
      evaluator.Stats().front().root_gain;
  ASSERT_EQ(entries.front().split.loss_chg, best_loss_chg);
  ASSERT_GT(entries.front().split.loss_chg, 16.2f);

  // Assert that's the best split
  for (size_t i = 1; i < gmat.cut.Ptrs().size(); ++i) {
    GradStats left, right;
    for (size_t j = gmat.cut.Ptrs()[i-1]; j < gmat.cut.Ptrs()[i]; ++j) {
      auto loss_chg =
          evaluator.Evaluator().CalcSplitGain(param, 0, i - 1, left, right) -
          evaluator.Stats().front().root_gain;
      ASSERT_GE(best_loss_chg, loss_chg);
      left.Add(hist[0][j].GetGrad(), hist[0][j].GetHess());
      right.SetSubstract(GradStats{total_gpair}, left);
    }
  }

  omp_set_num_threads(orig);
}

TEST(HistEvaluator, Evaluate) {
  TestEvaluateSplits<float>();
  TestEvaluateSplits<double>();
}

TEST(HistEvaluator, Apply) {
  RegTree tree;
  int static constexpr kNRows = 8, kNCols = 16;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{}});
  auto dmat = RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
  auto sampler = std::make_shared<common::ColumnSampler>();
  auto evaluator_ = HistEvaluator<float, CPUExpandEntry>{param, dmat->Info(), 4, sampler,
                                                         ObjInfo{ObjInfo::kRegression}};

  CPUExpandEntry entry{0, 0, 10.0f};
  entry.split.left_sum = GradStats{0.4, 0.6f};
  entry.split.right_sum = GradStats{0.5, 0.7f};

  evaluator_.ApplyTreeSplit(entry, &tree);
  ASSERT_EQ(tree.NumExtraNodes(), 2);
  ASSERT_EQ(tree.Stat(tree[0].LeftChild()).sum_hess, 0.6f);
  ASSERT_EQ(tree.Stat(tree[0].RightChild()).sum_hess, 0.7f);
}

TEST(HistEvaluator, CategoricalPartition) {
  int static constexpr kRows = 128, kCols = 1;
  using GradientSumT = double;
  std::vector<FeatureType> ft(kCols, FeatureType::kCategorical);

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});

  size_t n_cats{8};

  auto dmat =
      RandomDataGenerator(kRows, kCols, 0).Seed(3).Type(ft).MaxCategory(n_cats).GenerateDMatrix();

  int32_t n_threads = 16;
  auto sampler = std::make_shared<common::ColumnSampler>();
  auto evaluator = HistEvaluator<GradientSumT, CPUExpandEntry>{
      param, dmat->Info(), n_threads, sampler, ObjInfo{ObjInfo::kRegression}};

  for (auto const &gmat : dmat->GetBatches<GHistIndexMatrix>({GenericParameter::kCpuId, 32})) {
    common::HistCollection<GradientSumT> hist;

    std::vector<CPUExpandEntry> entries(1);
    entries.front().nid = 0;
    entries.front().depth = 0;

    hist.Init(gmat.cut.TotalBins());
    hist.AddHistRow(0);
    hist.AllocateAllData();
    auto node_hist = hist[0];
    ASSERT_EQ(node_hist.size(), n_cats);
    ASSERT_EQ(node_hist.size(), gmat.cut.Ptrs().back());

    GradientPairPrecise total_gpair;
    for (size_t i = 0; i < node_hist.size(); ++i) {
      node_hist[i] = {static_cast<double>(node_hist.size() - i), 1.0};
      total_gpair += node_hist[i];
    }
    SimpleLCG lcg;
    std::shuffle(node_hist.begin(), node_hist.end(), lcg);

    RegTree tree;
    evaluator.InitRoot(GradStats{total_gpair});
    evaluator.EvaluateSplits(hist, gmat.cut, ft, tree, &entries);
    ASSERT_TRUE(entries.front().split.is_cat);

    auto run_eval = [&](auto fn) {
      for (size_t i = 1; i < gmat.cut.Ptrs().size(); ++i) {
        GradStats left, right;
        for (size_t j = gmat.cut.Ptrs()[i - 1]; j < gmat.cut.Ptrs()[i]; ++j) {
          auto loss_chg = evaluator.Evaluator().CalcSplitGain(param, 0, i - 1, left, right) -
                          evaluator.Stats().front().root_gain;
          fn(loss_chg);
          left.Add(node_hist[j].GetGrad(), node_hist[j].GetHess());
          right.SetSubstract(GradStats{total_gpair}, left);
        }
      }
    };
    // Assert that's the best split
    auto best_loss_chg = entries.front().split.loss_chg;
    run_eval([&](auto loss_chg) {
      // Approximated test that gain returned by optimal partition is greater than
      // numerical split.
      ASSERT_GT(best_loss_chg, loss_chg);
    });
    // node_hist is captured in lambda.
    std::sort(node_hist.begin(), node_hist.end(), [&](auto l, auto r) {
      return evaluator.Evaluator().CalcWeightCat(param, l) <
             evaluator.Evaluator().CalcWeightCat(param, r);
    });

    double reimpl = 0;
    run_eval([&](auto loss_chg) { reimpl = std::max(loss_chg, reimpl); });
    CHECK_EQ(reimpl, best_loss_chg);
  }
}

namespace {
auto CompareOneHotAndPartition(bool onehot) {
  int static constexpr kRows = 128, kCols = 1;
  using GradientSumT = double;
  std::vector<FeatureType> ft(kCols, FeatureType::kCategorical);

  TrainParam param;
  if (onehot) {
    // force use one-hot
    param.UpdateAllowUnknown(
        Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "100"}});
  } else {
    param.UpdateAllowUnknown(
        Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "1"}});
  }

  size_t n_cats{2};

  auto dmat =
      RandomDataGenerator(kRows, kCols, 0).Seed(3).Type(ft).MaxCategory(n_cats).GenerateDMatrix();

  int32_t n_threads = 16;
  auto sampler = std::make_shared<common::ColumnSampler>();
  auto evaluator = HistEvaluator<GradientSumT, CPUExpandEntry>{
      param, dmat->Info(), n_threads, sampler, ObjInfo{ObjInfo::kRegression}};
  std::vector<CPUExpandEntry> entries(1);

  for (auto const &gmat : dmat->GetBatches<GHistIndexMatrix>({GenericParameter::kCpuId, 32})) {
    common::HistCollection<GradientSumT> hist;

    entries.front().nid = 0;
    entries.front().depth = 0;

    hist.Init(gmat.cut.TotalBins());
    hist.AddHistRow(0);
    hist.AllocateAllData();
    auto node_hist = hist[0];

    CHECK_EQ(node_hist.size(), n_cats);
    CHECK_EQ(node_hist.size(), gmat.cut.Ptrs().back());

    GradientPairPrecise total_gpair;
    for (size_t i = 0; i < node_hist.size(); ++i) {
      node_hist[i] = {static_cast<double>(node_hist.size() - i), 1.0};
      total_gpair += node_hist[i];
    }
    RegTree tree;
    evaluator.InitRoot(GradStats{total_gpair});
    evaluator.EvaluateSplits(hist, gmat.cut, ft, tree, &entries);
  }
  return entries.front();
}
}  // anonymous namespace

TEST(HistEvaluator, Categorical) {
  auto with_onehot = CompareOneHotAndPartition(true);
  auto with_part = CompareOneHotAndPartition(false);

  ASSERT_EQ(with_onehot.split.loss_chg, with_part.split.loss_chg);
}
}  // namespace tree
}  // namespace xgboost
