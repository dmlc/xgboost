/*!
 * Copyright 2018-2019 by Contributors
 */
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <string>

#include "../helpers.h"
#include "../../../src/tree/param.h"
#include "../../../src/tree/updater_quantile_hist.h"
#include "../../../src/tree/split_evaluator.h"

namespace xgboost {
namespace tree {

class QuantileHistMock : public QuantileHistMaker {
  static double constexpr kEps = 1e-6;

  struct BuilderMock : public QuantileHistMaker::Builder {
    using RealImpl = QuantileHistMaker::Builder;

    BuilderMock(const TrainParam& param,
                std::unique_ptr<TreeUpdater> pruner,
                std::unique_ptr<SplitEvaluator> spliteval,
                FeatureInteractionConstraintHost int_constraint)
        : RealImpl(param, std::move(pruner), std::move(spliteval), std::move(int_constraint)) {}

   public:
    void TestInitData(const GHistIndexMatrix& gmat,
                      const std::vector<GradientPair>& gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      RealImpl::InitData(gmat, gpair, *p_fmat, tree);
      ASSERT_EQ(data_layout_, kSparseData);

      /* The creation of HistCutMatrix and GHistIndexMatrix are not technically
       * part of QuantileHist updater logic, but we include it here because
       * QuantileHist updater object currently stores GHistIndexMatrix
       * internally. According to https://github.com/dmlc/xgboost/pull/3803,
       * we should eventually move GHistIndexMatrix out of the QuantileHist
       * updater. */

      const size_t num_row = p_fmat->Info().num_row_;
      const size_t num_col = p_fmat->Info().num_col_;
      /* Validate HistCutMatrix */
      ASSERT_EQ(gmat.cut.Ptrs().size(), num_col + 1);
      for (size_t fid = 0; fid < num_col; ++fid) {
        const size_t ibegin = gmat.cut.Ptrs()[fid];
        const size_t iend = gmat.cut.Ptrs()[fid + 1];
        // Ordered,  but empty feature is allowed.
        ASSERT_LE(ibegin, iend);
        for (size_t i = ibegin; i < iend - 1; ++i) {
          // Quantile points must be sorted in ascending order
          // No duplicates allowed
          ASSERT_LT(gmat.cut.Values()[i], gmat.cut.Values()[i + 1])
              << "ibegin: " << ibegin << ", "
              << "iend: " << iend;
        }
      }

      /* Validate GHistIndexMatrix */
      ASSERT_EQ(gmat.row_ptr.size(), num_row + 1);
      ASSERT_LT(*std::max_element(gmat.index.begin(), gmat.index.end()),
                gmat.cut.Ptrs().back());
      for (const auto& batch : p_fmat->GetBatches<xgboost::SparsePage>()) {
        for (size_t i = 0; i < batch.Size(); ++i) {
          const size_t rid = batch.base_rowid + i;
          ASSERT_LT(rid, num_row);
          const size_t gmat_row_offset = gmat.row_ptr[rid];
          ASSERT_LT(gmat_row_offset, gmat.index.size());
          SparsePage::Inst inst = batch[i];
          ASSERT_EQ(gmat.row_ptr[rid] + inst.size(), gmat.row_ptr[rid + 1]);
          for (size_t j = 0; j < inst.size(); ++j) {
            // Each entry of GHistIndexMatrix represents a bin ID
            const size_t bin_id = gmat.index[gmat_row_offset + j];
            const size_t fid = inst[j].index;
            // The bin ID must correspond to correct feature
            ASSERT_GE(bin_id, gmat.cut.Ptrs()[fid]);
            ASSERT_LT(bin_id, gmat.cut.Ptrs()[fid + 1]);
            // The bin ID must correspond to a region between two
            // suitable quantile points
            ASSERT_LT(inst[j].fvalue, gmat.cut.Values()[bin_id]);
            if (bin_id > gmat.cut.Ptrs()[fid]) {
              ASSERT_GE(inst[j].fvalue, gmat.cut.Values()[bin_id - 1]);
            } else {
              ASSERT_GE(inst[j].fvalue, gmat.cut.MinValues()[fid]);
            }
          }
        }
      }
    }

    void TestBuildHist(int nid,
                       const GHistIndexMatrix& gmat,
                       const DMatrix& fmat,
                       const RegTree& tree) {
      const std::vector<GradientPair> gpair =
          { {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f} };
      RealImpl::InitData(gmat, gpair, fmat, tree);
      GHistIndexBlockMatrix dummy;
      hist_.AddHistRow(nid);
      BuildHist(gpair, row_set_collection_[nid],
                gmat, dummy, hist_[nid], false);

      // Check if number of histogram bins is correct
      ASSERT_EQ(hist_[nid].size(), gmat.cut.Ptrs().back());
      std::vector<GradientPairPrecise> histogram_expected(hist_[nid].size());

      // Compute the correct histogram (histogram_expected)
      const size_t num_row = fmat.Info().num_row_;
      CHECK_EQ(gpair.size(), num_row);
      for (size_t rid = 0; rid < num_row; ++rid) {
        const size_t ibegin = gmat.row_ptr[rid];
        const size_t iend = gmat.row_ptr[rid + 1];
        for (size_t i = ibegin; i < iend; ++i) {
          const size_t bin_id = gmat.index[i];
          histogram_expected[bin_id] += GradientPairPrecise(gpair[rid]);
        }
      }

      // Now validate the computed histogram returned by BuildHist
      for (size_t i = 0; i < hist_[nid].size(); ++i) {
        GradientPairPrecise sol = histogram_expected[i];
        ASSERT_NEAR(sol.GetGrad(), hist_[nid][i].GetGrad(), kEps);
        ASSERT_NEAR(sol.GetHess(), hist_[nid][i].GetHess(), kEps);
      }
    }

    void TestEvaluateSplit(const GHistIndexBlockMatrix& quantile_index_block,
                           const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;
      auto dmat = CreateDMatrix(kNRows, kNCols, 0, 3);
      // dense, no missing values

      common::GHistIndexMatrix gmat;
      gmat.Init((*dmat).get(), kMaxBins);

      RealImpl::InitData(gmat, row_gpairs, *(*dmat), tree);
      hist_.AddHistRow(0);

      BuildHist(row_gpairs, row_set_collection_[0],
                gmat, quantile_index_block, hist_[0], false);

      RealImpl::InitNewNode(0, gmat, row_gpairs, *(*dmat), tree);

      /* Compute correct split (best_split) using the computed histogram */
      const size_t num_row = dmat->get()->Info().num_row_;
      const size_t num_feature = dmat->get()->Info().num_col_;
      CHECK_EQ(num_row, row_gpairs.size());
      // Compute total gradient for all data points
      GradientPairPrecise total_gpair;
      for (const auto& e : row_gpairs) {
        total_gpair += GradientPairPrecise(e);
      }
      // Initialize split evaluator
      std::unique_ptr<SplitEvaluator> evaluator(SplitEvaluator::Create("elastic_net"));
      evaluator->Init(&param_);

      // Now enumerate all feature*threshold combination to get best split
      // To simplify logic, we make some assumptions:
      // 1) no missing values in data
      // 2) no regularization, i.e. set min_child_weight, reg_lambda, reg_alpha,
      //    and max_delta_step to 0.
      bst_float best_split_gain = 0.0f;
      size_t best_split_threshold = std::numeric_limits<size_t>::max();
      size_t best_split_feature = std::numeric_limits<size_t>::max();
      // Enumerate all features
      for (size_t fid = 0; fid < num_feature; ++fid) {
        const size_t bin_id_min = gmat.cut.Ptrs()[fid];
        const size_t bin_id_max = gmat.cut.Ptrs()[fid + 1];
        // Enumerate all bin ID in [bin_id_min, bin_id_max), i.e. every possible
        // choice of thresholds for feature fid
        for (size_t split_thresh = bin_id_min;
             split_thresh < bin_id_max; ++split_thresh) {
          // left_sum, right_sum: Gradient sums for data points whose feature
          //                      value is left/right side of the split threshold
          GradientPairPrecise left_sum, right_sum;
          for (size_t rid = 0; rid < num_row; ++rid) {
            for (size_t offset = gmat.row_ptr[rid];
                 offset < gmat.row_ptr[rid + 1]; ++offset) {
              const size_t bin_id = gmat.index[offset];
              if (bin_id >= bin_id_min && bin_id < bin_id_max) {
                if (bin_id <= split_thresh) {
                  left_sum += GradientPairPrecise(row_gpairs[rid]);
                } else {
                  right_sum += GradientPairPrecise(row_gpairs[rid]);
                }
              }
            }
          }
          // Now compute gain (change in loss)
          const auto split_gain
            = evaluator->ComputeSplitScore(0, fid, GradStats(left_sum),
                                           GradStats(right_sum));
          if (split_gain > best_split_gain) {
            best_split_gain = split_gain;
            best_split_feature = fid;
            best_split_threshold = split_thresh;
          }
        }
      }

      /* Now compare against result given by EvaluateSplit() */
      RealImpl::EvaluateSplit(0, gmat, hist_, *(*dmat), tree);
      ASSERT_EQ(snode_[0].best.SplitIndex(), best_split_feature);
      ASSERT_EQ(snode_[0].best.split_value, gmat.cut.Values()[best_split_threshold]);

      delete dmat;
    }

    void TestEvaluateSplitParallel(const GHistIndexBlockMatrix &quantile_index_block,
                                   const RegTree &tree) {
      omp_set_num_threads(2);
      TestEvaluateSplit(quantile_index_block, tree);
      omp_set_num_threads(1);
    }

  };

  int static constexpr kNRows = 8, kNCols = 16;
  std::shared_ptr<xgboost::DMatrix> *dmat_;
  const std::vector<std::pair<std::string, std::string> > cfg_;
  std::shared_ptr<BuilderMock> builder_;

 public:
  explicit QuantileHistMock(
      const std::vector<std::pair<std::string, std::string> >& args) :
      cfg_{args} {
    QuantileHistMaker::Configure(args);
    spliteval_->Init(&param_);
    builder_.reset(
        new BuilderMock(
            param_,
            std::move(pruner_),
            std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
            int_constraint_));
    dmat_ = CreateDMatrix(kNRows, kNCols, 0.8, 3);
  }
  ~QuantileHistMock() override { delete dmat_; }

  static size_t GetNumColumns() { return kNCols; }

  void TestInitData() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init((*dmat_).get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };

    builder_->TestInitData(gmat, gpair, dmat_->get(), tree);
  }

  void TestBuildHist() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init((*dmat_).get(), kMaxBins);

    builder_->TestBuildHist(0, gmat, *(*dmat_).get(), tree);
  }

  void TestEvaluateSplit() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

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
       {"split_evaluator", "elastic_net"},
       {"reg_lambda", "0"}, {"reg_alpha", "0"}, {"max_delta_step", "0"},
       {"min_child_weight", "0"}};
  QuantileHistMock maker(cfg);
  maker.TestEvaluateSplit();
}

}  // namespace tree
}  // namespace xgboost
