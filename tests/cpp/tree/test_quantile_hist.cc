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
#include "xgboost/data.h"

namespace xgboost {
namespace tree {

class QuantileHistMock : public QuantileHistMaker {
  static double constexpr kEps = 1e-6;

  template <typename GradientSumT>
  struct BuilderMock : public QuantileHistMaker::Builder<GradientSumT> {
    using RealImpl = QuantileHistMaker::Builder<GradientSumT>;
    using ExpandEntryT = typename RealImpl::ExpandEntry;
    using GHistRowT = typename RealImpl::GHistRowT;

    BuilderMock(const TrainParam& param,
                std::unique_ptr<TreeUpdater> pruner,
                std::unique_ptr<SplitEvaluator> spliteval,
                FeatureInteractionConstraintHost int_constraint,
                DMatrix const* fmat)
        : RealImpl(param, std::move(pruner), std::move(spliteval),
          std::move(int_constraint), fmat) {}

   public:
    void TestInitData(const GHistIndexMatrix& gmat,
                      const std::vector<GradientPair>& gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      RealImpl::InitData(gmat, gpair, *p_fmat, tree);
      ASSERT_EQ(this->data_layout_, RealImpl::kSparseData);

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
          ASSERT_LT(gmat_row_offset, gmat.index.Size());
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

    void TestInitDataSampling(const GHistIndexMatrix& gmat,
                      const std::vector<GradientPair>& gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      const size_t nthreads = omp_get_num_threads();
      // save state of global rng engine
      auto initial_rnd = common::GlobalRandom();
      RealImpl::InitData(gmat, gpair, *p_fmat, tree);
      std::vector<size_t> row_indices_initial = *(this->row_set_collection_.Data());

      for (size_t i_nthreads = 1; i_nthreads < 4; ++i_nthreads) {
        omp_set_num_threads(i_nthreads);
        // return initial state of global rng engine
        common::GlobalRandom() = initial_rnd;
        RealImpl::InitData(gmat, gpair, *p_fmat, tree);
        std::vector<size_t>& row_indices = *(this->row_set_collection_.Data());
        ASSERT_EQ(row_indices_initial.size(), row_indices.size());
        for (size_t i = 0; i < row_indices_initial.size(); ++i) {
          ASSERT_EQ(row_indices_initial[i], row_indices[i]);
        }
      }
      omp_set_num_threads(nthreads);
    }

    void TestAddHistRows(const GHistIndexMatrix& gmat,
                         const std::vector<GradientPair>& gpair,
                         DMatrix* p_fmat,
                         RegTree* tree) {
      RealImpl::InitData(gmat, gpair, *p_fmat, *tree);

      int starting_index = std::numeric_limits<int>::max();
      int sync_count = 0;
      this->nodes_for_explicit_hist_build_.clear();
      this->nodes_for_subtraction_trick_.clear();

      tree->ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
      tree->ExpandNode((*tree)[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
      tree->ExpandNode((*tree)[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
      this->nodes_for_explicit_hist_build_.emplace_back(3, 4, tree->GetDepth(3), 0.0f, 0);
      this->nodes_for_explicit_hist_build_.emplace_back(4, 3, tree->GetDepth(4), 0.0f, 0);
      this->nodes_for_subtraction_trick_.emplace_back(5, 6, tree->GetDepth(5), 0.0f, 0);
      this->nodes_for_subtraction_trick_.emplace_back(6, 5, tree->GetDepth(6), 0.0f, 0);

      this->hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, tree);
      ASSERT_EQ(sync_count, 2);
      ASSERT_EQ(starting_index, 3);

      for (const ExpandEntryT& node : this->nodes_for_explicit_hist_build_) {
        ASSERT_EQ(this->hist_.RowExists(node.nid), true);
      }
      for (const ExpandEntryT& node : this->nodes_for_subtraction_trick_) {
        ASSERT_EQ(this->hist_.RowExists(node.nid), true);
      }
    }


    void TestSyncHistograms(const GHistIndexMatrix& gmat,
                            const std::vector<GradientPair>& gpair,
                            DMatrix* p_fmat,
                            RegTree* tree) {
      // init
      RealImpl::InitData(gmat, gpair, *p_fmat, *tree);

      int starting_index = std::numeric_limits<int>::max();
      int sync_count = 0;
      this->nodes_for_explicit_hist_build_.clear();
      this->nodes_for_subtraction_trick_.clear();
      // level 0
      this->nodes_for_explicit_hist_build_.emplace_back(0, -1, tree->GetDepth(0), 0.0f, 0);
      this->hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, tree);
      tree->ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

      this->nodes_for_explicit_hist_build_.clear();
      this->nodes_for_subtraction_trick_.clear();
      // level 1
      this->nodes_for_explicit_hist_build_.emplace_back((*tree)[0].LeftChild(),
                                                (*tree)[0].RightChild(),
                                                tree->GetDepth(1), 0.0f, 0);
      this->nodes_for_subtraction_trick_.emplace_back((*tree)[0].RightChild(),
                                              (*tree)[0].LeftChild(),
                                              tree->GetDepth(2), 0.0f, 0);
      this->hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, tree);
      tree->ExpandNode((*tree)[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
      tree->ExpandNode((*tree)[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

      this->nodes_for_explicit_hist_build_.clear();
      this->nodes_for_subtraction_trick_.clear();
      // level 2
      this->nodes_for_explicit_hist_build_.emplace_back(3, 4, tree->GetDepth(3), 0.0f, 0);
      this->nodes_for_subtraction_trick_.emplace_back(4, 3, tree->GetDepth(4), 0.0f, 0);
      this->nodes_for_explicit_hist_build_.emplace_back(5, 6, tree->GetDepth(5), 0.0f, 0);
      this->nodes_for_subtraction_trick_.emplace_back(6, 5, tree->GetDepth(6), 0.0f, 0);
      this->hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, tree);

      const size_t n_nodes = this->nodes_for_explicit_hist_build_.size();
      ASSERT_EQ(n_nodes, 2);
      this->row_set_collection_.AddSplit(0, (*tree)[0].LeftChild(),
          (*tree)[0].RightChild(), 4, 4);
      this->row_set_collection_.AddSplit(1, (*tree)[1].LeftChild(),
          (*tree)[1].RightChild(), 2, 2);
      this->row_set_collection_.AddSplit(2, (*tree)[2].LeftChild(),
          (*tree)[2].RightChild(), 2, 2);

      common::BlockedSpace2d space(n_nodes, [&](size_t node) {
        const int32_t nid = this->nodes_for_explicit_hist_build_[node].nid;
        return this->row_set_collection_[nid].Size();
      }, 256);

      std::vector<GHistRowT> target_hists(n_nodes);
      for (size_t i = 0; i < this->nodes_for_explicit_hist_build_.size(); ++i) {
        const int32_t nid = this->nodes_for_explicit_hist_build_[i].nid;
        target_hists[i] = this->hist_[nid];
      }

      const size_t nbins = this->hist_builder_.GetNumBins();
      // set values to specific nodes hist
      std::vector<size_t> n_ids = {1, 2};
      for (size_t i : n_ids) {
        auto this_hist = this->hist_[i];
        GradientSumT* p_hist = reinterpret_cast<GradientSumT*>(this_hist.data());
        for (size_t bin_id = 0; bin_id < 2*nbins; ++bin_id) {
          p_hist[bin_id] = 2*bin_id;
        }
      }
      n_ids[0] = 3;
      n_ids[1] = 5;
      for (size_t i : n_ids) {
        auto this_hist = this->hist_[i];
        GradientSumT* p_hist = reinterpret_cast<GradientSumT*>(this_hist.data());
        for (size_t bin_id = 0; bin_id < 2*nbins; ++bin_id) {
          p_hist[bin_id] = bin_id;
        }
      }

      this->hist_buffer_.Reset(1, n_nodes, space, target_hists);
      // sync hist
      this->hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, tree);

      auto check_hist = [] (const GHistRowT parent, const GHistRowT left,
                            const GHistRowT right, size_t begin, size_t end) {
        const GradientSumT* p_parent = reinterpret_cast<const GradientSumT*>(parent.data());
        const GradientSumT* p_left = reinterpret_cast<const GradientSumT*>(left.data());
        const GradientSumT* p_right = reinterpret_cast<const GradientSumT*>(right.data());
        for (size_t i = 2 * begin; i < 2 * end; ++i) {
          ASSERT_EQ(p_parent[i], p_left[i] + p_right[i]);
        }
      };
      for (const ExpandEntryT& node : this->nodes_for_explicit_hist_build_) {
        auto this_hist = this->hist_[node.nid];
        const size_t parent_id = (*tree)[node.nid].Parent();
        auto parent_hist =  this->hist_[parent_id];
        auto sibling_hist = this->hist_[node.sibling_nid];

        check_hist(parent_hist, this_hist, sibling_hist, 0, nbins);
      }
      for (const ExpandEntryT& node : this->nodes_for_subtraction_trick_) {
        auto this_hist = this->hist_[node.nid];
        const size_t parent_id = (*tree)[node.nid].Parent();
        auto parent_hist =  this->hist_[parent_id];
        auto sibling_hist = this->hist_[node.sibling_nid];

        check_hist(parent_hist, this_hist, sibling_hist, 0, nbins);
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
      this->hist_.AddHistRow(nid);
      this->BuildHist(gpair, this->row_set_collection_[nid],
                gmat, dummy, this->hist_[nid]);

      // Check if number of histogram bins is correct
      ASSERT_EQ(this->hist_[nid].size(), gmat.cut.Ptrs().back());
      std::vector<GradientPairPrecise> histogram_expected(this->hist_[nid].size());

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
      for (size_t i = 0; i < this->hist_[nid].size(); ++i) {
        GradientPairPrecise sol = histogram_expected[i];
        ASSERT_NEAR(sol.GetGrad(), this->hist_[nid][i].GetGrad(), kEps);
        ASSERT_NEAR(sol.GetHess(), this->hist_[nid][i].GetHess(), kEps);
      }
    }

    void TestEvaluateSplit(const GHistIndexBlockMatrix& quantile_index_block,
                           const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;
      auto dmat = RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
      // dense, no missing values

      common::GHistIndexMatrix gmat;
      gmat.Init(dmat.get(), kMaxBins);

      RealImpl::InitData(gmat, row_gpairs, *dmat, tree);
      this->hist_.AddHistRow(0);

      this->BuildHist(row_gpairs, this->row_set_collection_[0],
                      gmat, quantile_index_block, this->hist_[0]);

      RealImpl::InitNewNode(0, gmat, row_gpairs, *dmat, tree);

      /* Compute correct split (best_split) using the computed histogram */
      const size_t num_row = dmat->Info().num_row_;
      const size_t num_feature = dmat->Info().num_col_;
      CHECK_EQ(num_row, row_gpairs.size());
      // Compute total gradient for all data points
      GradientPairPrecise total_gpair;
      for (const auto& e : row_gpairs) {
        total_gpair += GradientPairPrecise(e);
      }
      // Initialize split evaluator
      std::unique_ptr<SplitEvaluator> evaluator(SplitEvaluator::Create("elastic_net"));
      evaluator->Init(&this->param_);

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
      typename RealImpl::ExpandEntry node(RealImpl::ExpandEntry::kRootNid,
                                          RealImpl::ExpandEntry::kEmptyNid,
                                          tree.GetDepth(0),
                                          this->snode_[0].best.loss_chg, 0);
      RealImpl::EvaluateSplits({node}, gmat, this->hist_, tree);
      ASSERT_EQ(this->snode_[0].best.SplitIndex(), best_split_feature);
      ASSERT_EQ(this->snode_[0].best.split_value, gmat.cut.Values()[best_split_threshold]);
    }

    void TestEvaluateSplitParallel(const GHistIndexBlockMatrix &quantile_index_block,
                                   const RegTree &tree) {
      omp_set_num_threads(2);
      TestEvaluateSplit(quantile_index_block, tree);
      omp_set_num_threads(1);
    }

    void TestApplySplit(const GHistIndexBlockMatrix& quantile_index_block,
                        const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;

      // try out different sparsity to get different number of missing values
      for (double sparsity : {0.0, 0.1, 0.2}) {
        // kNRows samples with kNCols features
        auto dmat = RandomDataGenerator(kNRows, kNCols, sparsity).Seed(3).GenerateDMatrix();

        common::GHistIndexMatrix gmat;
        gmat.Init(dmat.get(), kMaxBins);
        ColumnMatrix cm;

        // treat everything as dense, as this is what we intend to test here
        cm.Init(gmat, 0.0);
        RealImpl::InitData(gmat, row_gpairs, *dmat, tree);
        this->hist_.AddHistRow(0);

        RealImpl::InitNewNode(0, gmat, row_gpairs, *dmat, tree);

        const size_t num_row = dmat->Info().num_row_;
        // split by feature 0
        const size_t bin_id_min = gmat.cut.Ptrs()[0];
        const size_t bin_id_max = gmat.cut.Ptrs()[1];

        // attempt to split at different bins
        for (size_t split = 0; split < 4; split++) {
          size_t left_cnt = 0, right_cnt = 0;

          // manually compute how many samples go left or right
          for (size_t rid = 0; rid < num_row; ++rid) {
            for (size_t offset = gmat.row_ptr[rid]; offset < gmat.row_ptr[rid + 1]; ++offset) {
              const size_t bin_id = gmat.index[offset];
              if (bin_id >= bin_id_min && bin_id < bin_id_max) {
                if (bin_id <= split) {
                  left_cnt++;
                } else {
                  right_cnt++;
                }
              }
            }
          }

          // if any were missing due to sparsity, we add them to the left or to the right
          size_t missing = kNRows - left_cnt - right_cnt;
          if (tree[0].DefaultLeft()) {
            left_cnt += missing;
          } else {
            right_cnt += missing;
          }

          // have one node with kNRows (=8 at the moment) rows, just one task
          RealImpl::partition_builder_.Init(1, 1, [&](size_t node_in_set) {
            return 1;
          });
          this->template PartitionKernel<uint8_t>(0, 0, common::Range1d(0, kNRows),
                                                  split, cm, tree);
          RealImpl::partition_builder_.CalculateRowOffsets();
          ASSERT_EQ(RealImpl::partition_builder_.GetNLeftElems(0), left_cnt);
          ASSERT_EQ(RealImpl::partition_builder_.GetNRightElems(0), right_cnt);
        }
      }
    }
  };

  int static constexpr kNRows = 8, kNCols = 16;
  std::shared_ptr<xgboost::DMatrix> dmat_;
  const std::vector<std::pair<std::string, std::string> > cfg_;
  std::shared_ptr<BuilderMock<float> > float_builder_;
  std::shared_ptr<BuilderMock<double> > double_builder_;

 public:
  explicit QuantileHistMock(
      const std::vector<std::pair<std::string, std::string> >& args,
      const bool single_precision_histogram = false, bool batch = true) :
      cfg_{args} {
    QuantileHistMaker::Configure(args);
    spliteval_->Init(&param_);
    dmat_ = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
    if (single_precision_histogram) {
      float_builder_.reset(
          new BuilderMock<float>(
              param_,
              std::move(pruner_),
              std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
              int_constraint_,
              dmat_.get()));
      if (batch) {
        float_builder_->SetHistSynchronizer(new BatchHistSynchronizer<float>());
        float_builder_->SetHistRowsAdder(new BatchHistRowsAdder<float>());
      } else {
        float_builder_->SetHistSynchronizer(new DistributedHistSynchronizer<float>());
        float_builder_->SetHistRowsAdder(new DistributedHistRowsAdder<float>());
      }
    } else {
      double_builder_.reset(
          new BuilderMock<double>(
              param_,
              std::move(pruner_),
              std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
              int_constraint_,
              dmat_.get()));
      if (batch) {
        double_builder_->SetHistSynchronizer(new BatchHistSynchronizer<double>());
        double_builder_->SetHistRowsAdder(new BatchHistRowsAdder<double>());
      } else {
        double_builder_->SetHistSynchronizer(new DistributedHistSynchronizer<double>());
        double_builder_->SetHistRowsAdder(new DistributedHistRowsAdder<double>());
      }
    }
  }
  ~QuantileHistMock() override = default;

  static size_t GetNumColumns() { return kNCols; }

  void TestInitData() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init(dmat_.get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestInitData(gmat, gpair, dmat_.get(), tree);
    } else {
      float_builder_->TestInitData(gmat, gpair, dmat_.get(), tree);
    }
  }

  void TestInitDataSampling() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init(dmat_.get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestInitDataSampling(gmat, gpair, dmat_.get(), tree);
    } else {
      float_builder_->TestInitDataSampling(gmat, gpair, dmat_.get(), tree);
    }
  }

  void TestAddHistRows() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init(dmat_.get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestAddHistRows(gmat, gpair, dmat_.get(), &tree);
    } else {
      float_builder_->TestAddHistRows(gmat, gpair, dmat_.get(), &tree);
    }
  }

  void TestSyncHistograms() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init(dmat_.get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestSyncHistograms(gmat, gpair, dmat_.get(), &tree);
    } else {
      float_builder_->TestSyncHistograms(gmat, gpair, dmat_.get(), &tree);
    }
  }


  void TestBuildHist() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrix gmat;
    gmat.Init(dmat_.get(), kMaxBins);
    if (double_builder_) {
      double_builder_->TestBuildHist(0, gmat, *dmat_, tree);
    } else {
      float_builder_->TestBuildHist(0, gmat, *dmat_, tree);
    }
  }

  void TestEvaluateSplit() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    if (double_builder_) {
      double_builder_->TestEvaluateSplit(gmatb_, tree);
    } else {
      float_builder_->TestEvaluateSplit(gmatb_, tree);
    }
  }

  void TestApplySplit() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    if (double_builder_) {
      double_builder_->TestApplySplit(gmatb_, tree);
    } else {
      float_builder_->TestEvaluateSplit(gmatb_, tree);
    }
  }
};

TEST(QuantileHist, InitData) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg);
  maker.TestInitData();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestInitData();
}

TEST(QuantileHist, InitDataSampling) {
  const float subsample = 0.5;
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"subsample", std::to_string(subsample)}};
  QuantileHistMock maker(cfg);
  maker.TestInitDataSampling();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestInitDataSampling();
}

TEST(QuantileHist, AddHistRows) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg);
  maker.TestAddHistRows();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestAddHistRows();
}

TEST(QuantileHist, SyncHistograms) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg);
  maker.TestSyncHistograms();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestSyncHistograms();
}

TEST(QuantileHist, DistributedAddHistRows) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg, false);
  maker.TestAddHistRows();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestAddHistRows();
}

TEST(QuantileHist, DistributedSyncHistograms) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())}};
  QuantileHistMock maker(cfg, false);
  maker.TestSyncHistograms();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestSyncHistograms();
}

TEST(QuantileHist, BuildHist) {
  // Don't enable feature grouping
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"enable_feature_grouping", std::to_string(0)}};
  QuantileHistMock maker(cfg);
  maker.TestBuildHist();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestBuildHist();
}

TEST(QuantileHist, EvalSplits) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"split_evaluator", "elastic_net"},
       {"reg_lambda", "0"}, {"reg_alpha", "0"}, {"max_delta_step", "0"},
       {"min_child_weight", "0"}};
  QuantileHistMock maker(cfg);
  maker.TestEvaluateSplit();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestEvaluateSplit();
}

TEST(QuantileHist, ApplySplit) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMock::GetNumColumns())},
       {"split_evaluator", "elastic_net"},
       {"reg_lambda", "0"}, {"reg_alpha", "0"}, {"max_delta_step", "0"},
       {"min_child_weight", "0"}};
  QuantileHistMock maker(cfg);
  maker.TestApplySplit();
  const bool single_precision_histogram = true;
  QuantileHistMock maker_float(cfg, single_precision_histogram);
  maker_float.TestApplySplit();
}

}  // namespace tree
}  // namespace xgboost
