/*!
 * Copyright 2018 by Contributors
 */
#include "../helpers.h"
#include "../../../src/tree/param.h"
#include "../../../src/tree/updater_quantile_hist.h"
#include "../../../src/common/host_device_vector.h"

#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>

#include <vector>
#include <string>

namespace xgboost {
namespace tree {

class QuantileHistMock : public QuantileHistMaker {
  static double constexpr kEps = 1e-6;

  struct BuilderMock : public QuantileHistMaker::Builder {
    using RealImpl = QuantileHistMaker::Builder;

    BuilderMock(const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     std::unique_ptr<SplitEvaluator> spliteval)
        : RealImpl(param, std::move(pruner), std::move(spliteval)) {}

   public:
    void TestInitData(const GHistIndexMatrix& gmat,
                  const std::vector<GradientPair>& gpair,
                  const DMatrix& fmat,
                  const RegTree& tree) {
      RealImpl::InitData(gmat, gpair, fmat, tree);
      ASSERT_EQ(data_layout_, kSparseData);
    }

    void TestBuildHist(int nid,
                       const GHistIndexMatrix& gmat,
                       const DMatrix& fmat,
                       const RegTree& tree) {
      std::vector<GradientPair> gpair =
          { {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f} };
      RealImpl::InitData(gmat, gpair, fmat, tree);
      GHistIndexBlockMatrix quantile_index_block;
      hist_.AddHistRow(nid);
      BuildHist(gpair, row_set_collection_[nid],
                gmat, quantile_index_block, hist_[nid]);
      std::vector<GradientPairPrecise> solution {
        {0.27f, 0.29f}, {0.27f, 0.29f}, {0.47f, 0.49f},
        {0.27f, 0.29f}, {0.57f, 0.59f}, {0.26f, 0.27f},
        {0.37f, 0.39f}, {0.23f, 0.24f}, {0.37f, 0.39f},
        {0.27f, 0.28f}, {0.27f, 0.29f}, {0.37f, 0.39f},
        {0.26f, 0.27f}, {0.23f, 0.24f}, {0.57f, 0.59f},
        {0.47f, 0.49f}, {0.47f, 0.49f}, {0.37f, 0.39f},
        {0.26f, 0.27f}, {0.23f, 0.24f}, {0.27f, 0.28f},
        {0.57f, 0.59f}, {0.23f, 0.24f}, {0.47f, 0.49f}};

      for (size_t i = 0; i < hist_[nid].size(); ++i) {
        GradientPairPrecise sol = solution[i];
        ASSERT_NEAR(sol.GetGrad(), hist_[nid][i].GetGrad(), kEps);
        ASSERT_NEAR(sol.GetHess(), hist_[nid][i].GetHess(), kEps);
      }
    }

    void TestEvaluateSplit(const GHistIndexBlockMatrix& quantile_index_block,
                           const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr max_bins = 4;
      auto dmat = CreateDMatrix(n_rows, n_cols, 0, 3);  // dense

      common::GHistIndexMatrix gmat;
      gmat.Init((*dmat).get(), max_bins);

      RealImpl::InitData(gmat, row_gpairs, *(*dmat), tree);
      hist_.AddHistRow(0);

      BuildHist(row_gpairs, row_set_collection_[0],
                gmat, quantile_index_block, hist_[0]);

      RealImpl::InitNewNode(0, gmat, row_gpairs, *(*dmat), tree);
      // Manipulate the root_gain so that I don't have to invent an actual
      // split.  Yes, I'm cheating.
      snode_[0].root_gain = 0.8;
      RealImpl::EvaluateSplit(0, gmat, hist_, *(*dmat), tree);

      ASSERT_NEAR(snode_.at(0).best.loss_chg, 0.7128048, kEps);
      ASSERT_EQ(snode_.at(0).best.SplitIndex(), 10);
      ASSERT_NEAR(snode_.at(0).best.split_value, 0.182258, kEps);

      delete dmat;
    }
  };

  int static constexpr n_rows = 8, n_cols = 16;
  std::shared_ptr<xgboost::DMatrix> *dmat;
  const std::vector<std::pair<std::string, std::string> > cfg;
  std::shared_ptr<BuilderMock> builder_;

 public:
  explicit QuantileHistMock(
      const std::vector<std::pair<std::string, std::string> >& args) :
      cfg{args} {
    QuantileHistMaker::Init(args);
    builder_.reset(
        new BuilderMock(
            param_,
            std::move(pruner_),
            std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone())));
    dmat = CreateDMatrix(n_rows, n_cols, 0.8, 3);
  }
  ~QuantileHistMock() { delete dmat; }

  static size_t GetNumColumns() { return n_cols; }

  void TestInitData() {
    size_t constexpr max_bins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init((*dmat).get(), max_bins);

    RegTree tree = RegTree();
    tree.param.InitAllowUnknown(cfg);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };

    builder_->TestInitData(gmat, gpair, *(*dmat), tree);
  }

  void TestBuildHist() {
    RegTree tree = RegTree();
    tree.param.InitAllowUnknown(cfg);

    size_t constexpr max_bins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init((*dmat).get(), max_bins);

    builder_->TestBuildHist(0, gmat, *(*dmat).get(), tree);
  }

  void TestEvaluateSplit() {
    RegTree tree = RegTree();
    tree.param.InitAllowUnknown(cfg);

    builder_->TestEvaluateSplit(gmatb_, tree);
  }
};

TEST(Updater, QuantileHist_InitData) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg);
  maker.TestInitData();
}

TEST(Updater, QuantileHist_BuildHist) {
  // Don't enable feature grouping
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"enable_feature_grouping", std::to_string(0)}};
  QuantileHistMock maker(cfg);
  maker.TestBuildHist();
}

TEST(Updater, QuantileHist_EvalSplits) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"split_evaluator", "elastic_net"}};
  QuantileHistMock maker(cfg);
  maker.TestEvaluateSplit();
}

}  // namespace tree
}  // namespace xgboost
