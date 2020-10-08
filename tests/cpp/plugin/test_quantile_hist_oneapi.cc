/*!
 * Copyright 2018-2020 by Contributors
 */
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <string>

#include "../helpers.h"
#include "../../../plugin/updater_oneapi/updater_quantile_hist_oneapi.h"
#include "../../../plugin/updater_oneapi/split_evaluator_oneapi.h"
#include "xgboost/data.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace tree {

class QuantileHistMockOneAPI : public GPUQuantileHistMakerOneAPI {
  static double constexpr kEps = 1e-6;

  template <typename GradientSumT>
  struct BuilderMock : public GPUQuantileHistMakerOneAPI::Builder<GradientSumT> {
    using RealImpl = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;
    using ExpandEntryT = typename RealImpl::ExpandEntry;
    using GHistRowT = typename RealImpl::GHistRowT;

    BuilderMock(cl::sycl::queue qu,
                const TrainParam& param,
                std::unique_ptr<TreeUpdater> pruner,
                FeatureInteractionConstraintHost int_constraint,
                DMatrix const* fmat)
        : RealImpl(qu, param, std::move(pruner),
          std::move(int_constraint), fmat) {}

   public:
    void TestInitData(const GHistIndexMatrixOneAPI& gmat,
                      const std::vector<GradientPair>& gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      USMVector<GradientPair> gpair_device(this->qu_, gpair);

      RealImpl::InitData(gmat, gpair, gpair_device, *p_fmat, tree);
      ASSERT_EQ(this->data_layout_, RealImpl::kSparseData);

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

      /* Validate GHistIndexMatrixOneAPI */
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
            // Each entry of GHistIndexMatrixOneAPI represents a bin ID
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

    void TestInitDataSampling(const GHistIndexMatrixOneAPI& gmat,
                      const std::vector<GradientPair>& gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      const size_t nthreads = omp_get_num_threads();
      // save state of global rng engine
      auto initial_rnd = common::GlobalRandom();
      USMVector<GradientPair> gpair_device(this->qu_, gpair);
      RealImpl::InitData(gmat, gpair, gpair_device, *p_fmat, tree);
      USMVector<size_t>& row_indices_initial = this->row_set_collection_.Data();

      for (size_t i_nthreads = 1; i_nthreads < 4; ++i_nthreads) {
        omp_set_num_threads(i_nthreads);
        // return initial state of global rng engine
        common::GlobalRandom() = initial_rnd;
        RealImpl::InitData(gmat, gpair, gpair_device, *p_fmat, tree);
        USMVector<size_t>& row_indices = this->row_set_collection_.Data();
        ASSERT_EQ(row_indices_initial.Size(), row_indices.Size());
        for (size_t i = 0; i < row_indices_initial.Size(); ++i) {
          ASSERT_EQ(row_indices_initial[i], row_indices[i]);
        }
      }
      omp_set_num_threads(nthreads);
    }

    void TestAddHistRows(const GHistIndexMatrixOneAPI& gmat,
                         const std::vector<GradientPair>& gpair,
                         DMatrix* p_fmat,
                         RegTree* tree) {
      USMVector<GradientPair> gpair_device(this->qu_, gpair);

      RealImpl::InitData(gmat, gpair, gpair_device, *p_fmat, *tree);

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


    void TestSyncHistograms(const GHistIndexMatrixOneAPI& gmat,
                            const std::vector<GradientPair>& gpair,
                            DMatrix* p_fmat,
                            RegTree* tree) {
      // init
      USMVector<GradientPair> gpair_device(this->qu_, gpair);

      RealImpl::InitData(gmat, gpair, gpair_device, *p_fmat, *tree);

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
      ASSERT_EQ(n_nodes, 2ul);
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
        GradientSumT* p_hist = reinterpret_cast<GradientSumT*>(this_hist.Data());
        for (size_t bin_id = 0; bin_id < 2*nbins; ++bin_id) {
          p_hist[bin_id] = 2*bin_id;
        }
      }
      n_ids[0] = 3;
      n_ids[1] = 5;
      for (size_t i : n_ids) {
        auto this_hist = this->hist_[i];
        GradientSumT* p_hist = reinterpret_cast<GradientSumT*>(this_hist.Data());
        for (size_t bin_id = 0; bin_id < 2*nbins; ++bin_id) {
          p_hist[bin_id] = bin_id;
        }
      }

      this->hist_buffer_.Reset(256);
      // sync hist
      this->hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, tree);

      auto check_hist = [] (const GHistRowT parent, const GHistRowT left,
                            const GHistRowT right, size_t begin, size_t end) {
        const GradientSumT* p_parent = reinterpret_cast<const GradientSumT*>(parent.DataConst());
        const GradientSumT* p_left = reinterpret_cast<const GradientSumT*>(left.DataConst());
        const GradientSumT* p_right = reinterpret_cast<const GradientSumT*>(right.DataConst());
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
                       const GHistIndexMatrixOneAPI& gmat,
                       const DMatrix& fmat,
                       const RegTree& tree) {
      const std::vector<GradientPair> gpair =
          { {0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {0.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {0.47f, 0.49f}, {0.57f, 0.59f} };
      USMVector<GradientPair> gpair_device(this->qu_, gpair);

      RealImpl::InitData(gmat, gpair, gpair_device, fmat, tree);
      this->hist_.AddHistRow(nid);
      this->hist_buffer_.Reset(256);
      this->BuildHist(gpair, gpair_device, this->row_set_collection_[nid],
                gmat, this->hist_[nid], this->hist_buffer_.GetDeviceBuffer());

      // Check if number of histogram bins is correct
      ASSERT_EQ(this->hist_[nid].Size(), gmat.cut.Ptrs().back());
      std::vector<GradientPairPrecise> histogram_expected(this->hist_[nid].Size());

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
      for (size_t i = 0; i < this->hist_[nid].Size(); ++i) {
        GradientPairPrecise sol = histogram_expected[i];
        ASSERT_NEAR(sol.GetGrad(), this->hist_[nid][i].GetGrad(), kEps);
        ASSERT_NEAR(sol.GetHess(), this->hist_[nid][i].GetHess(), kEps);
      }
    }

    void TestEvaluateSplit(const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;
      auto dmat = RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
      // dense, no missing values

      common::GHistIndexMatrixOneAPI gmat;
      DeviceMatrixOneAPI dmat_device(this->qu_, dmat.get());
      gmat.Init(this->qu_, dmat_device, kMaxBins);

      USMVector<GradientPair> row_gpairs_device(this->qu_, row_gpairs);
      RealImpl::InitData(gmat, row_gpairs, row_gpairs_device, *dmat, tree);
      this->hist_.AddHistRow(0);
      RealImpl::hist_buffer_.Reset(256);
      this->BuildHist(row_gpairs, row_gpairs_device, this->row_set_collection_[0],
                      gmat, this->hist_[0], RealImpl::hist_buffer_.GetDeviceBuffer());

      RealImpl::InitNewNode(0, gmat, row_gpairs, row_gpairs_device, *dmat, tree);

      /* Compute correct split (best_split) using the computed histogram */
      const size_t num_row = dmat->Info().num_row_;
      const size_t num_feature = dmat->Info().num_col_;
      CHECK_EQ(num_row, row_gpairs.size());
      // Compute total gradient for all data points
      GradientPairPrecise total_gpair;
      for (const auto& e : row_gpairs) {
        total_gpair += GradientPairPrecise(e);
      }
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
          auto evaluator = this->tree_evaluator_.GetEvaluator();
          const auto split_gain = evaluator.CalcSplitGain(0, fid, GradStatsOneAPI(left_sum), GradStatsOneAPI(right_sum));
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

    void TestEvaluateSplitParallel(const RegTree &tree) {
      omp_set_num_threads(2);
      TestEvaluateSplit(tree);
      omp_set_num_threads(1);
    }

    void TestApplySplit(const RegTree& tree) {
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;

      // try out different sparsity to get different number of missing values
      for (double sparsity : {0.0, 0.1, 0.2}) {
        // kNRows samples with kNCols features
        auto dmat = RandomDataGenerator(kNRows, kNCols, sparsity).Seed(3).GenerateDMatrix();

        common::GHistIndexMatrixOneAPI gmat;
        DeviceMatrixOneAPI dmat_device(this->qu_, dmat.get());
        gmat.Init(this->qu_, dmat_device, kMaxBins);
        ColumnMatrixOneAPI cm;

        // treat everything as dense, as this is what we intend to test here
        cm.Init(this->qu_, gmat, dmat_device, 0.0);

        USMVector<GradientPair> row_gpairs_device(this->qu_, row_gpairs);

        RealImpl::InitData(gmat, row_gpairs, row_gpairs_device, *dmat, tree);
        this->hist_.AddHistRow(0);

        RealImpl::InitNewNode(0, gmat, row_gpairs, row_gpairs_device, *dmat, tree);

        const size_t num_row = dmat->Info().num_row_;
        // split by feature 0
        const size_t bin_id_min = gmat.cut.Ptrs()[0];
        const size_t bin_id_max = gmat.cut.Ptrs()[1];
        LOG(INFO) << "partition bins " << bin_id_min << " " << bin_id_max;

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
          LOG(INFO) << "partition " << left_cnt << " " << right_cnt << " " << missing;
          if (tree[0].DefaultLeft()) {
            left_cnt += missing;
          } else {
            right_cnt += missing;
          }

          // have one node with kNRows (=8 at the moment) rows, just one task
          RealImpl::partition_builder_.Init(this->qu_, 1, [&](size_t node_in_set) {
            return num_row;
          });
          this->template PartitionKernel<uint8_t>(0, 0, common::Range1d(0, kNRows),
                                                  split, cm, tree);
          RealImpl::partition_builder_.CalculateRowOffsets();
          LOG(INFO) << "partition  get, " << RealImpl::partition_builder_.GetNLeftElems(0) << " " << RealImpl::partition_builder_.GetNRightElems(0);
          LOG(INFO) << "partition real, " << left_cnt << " " << right_cnt;
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
  explicit QuantileHistMockOneAPI(
      const std::vector<std::pair<std::string, std::string> >& args,
      const bool single_precision_histogram = false, bool batch = true) :
      cfg_{args} {
    GPUQuantileHistMakerOneAPI::Configure(args);
    dmat_ = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
    if (single_precision_histogram) {
      float_builder_.reset(
          new BuilderMock<float>(
              qu_,
              param_,
              std::move(pruner_),
              int_constraint_,
              dmat_.get()));
      if (batch) {
        float_builder_->SetHistSynchronizer(new BatchHistSynchronizerOneAPI<float>());
        float_builder_->SetHistRowsAdder(new BatchHistRowsAdderOneAPI<float>());
      } else {
//        float_builder_->SetHistSynchronizer(new DistributedHistSynchronizerOneAPI<float>());
//        float_builder_->SetHistRowsAdder(new DistributedHistRowsAdderOneAPI<float>());
      }
    } else {
      double_builder_.reset(
          new BuilderMock<double>(
              qu_,
              param_,
              std::move(pruner_),
              int_constraint_,
              dmat_.get()));
      if (batch) {
        double_builder_->SetHistSynchronizer(new BatchHistSynchronizerOneAPI<double>());
        double_builder_->SetHistRowsAdder(new BatchHistRowsAdderOneAPI<double>());
      } else {
//        double_builder_->SetHistSynchronizer(new DistributedHistSynchronizerOneAPI<double>());
//        double_builder_->SetHistRowsAdder(new DistributedHistRowsAdderOneAPI<double>());
      }
    }
  }
  ~QuantileHistMockOneAPI() override = default;

  static size_t GetNumColumns() { return kNCols; }

  void TestInitData() {
    size_t constexpr kMaxBins = 4;
    common::GHistIndexMatrixOneAPI gmat;
    DeviceMatrixOneAPI dmat_device(qu_, dmat_.get());
    gmat.Init(qu_, dmat_device, kMaxBins);

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
    common::GHistIndexMatrixOneAPI gmat;
    DeviceMatrixOneAPI dmat_device(qu_, dmat_.get());
    gmat.Init(qu_, dmat_device, kMaxBins);

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
    common::GHistIndexMatrixOneAPI gmat;
    DeviceMatrixOneAPI dmat_device(qu_, dmat_.get());
    gmat.Init(qu_, dmat_device, kMaxBins);

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
    common::GHistIndexMatrixOneAPI gmat;
    DeviceMatrixOneAPI dmat_device(qu_, dmat_.get());
    gmat.Init(qu_, dmat_device, kMaxBins);

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
    common::GHistIndexMatrixOneAPI gmat;
    DeviceMatrixOneAPI dmat_device(qu_, dmat_.get());
    gmat.Init(qu_, dmat_device, kMaxBins);
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
      double_builder_->TestEvaluateSplit(tree);
    } else {
      float_builder_->TestEvaluateSplit(tree);
    }
  }

  void TestApplySplit() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    if (double_builder_) {
      double_builder_->TestApplySplit(tree);
    } else {
      float_builder_->TestEvaluateSplit(tree);
    }
  }
};

TEST(Plugin, GPUQuantileHistOneAPIInitData) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestInitData();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestInitData();
}

TEST(Plugin, GPUQuantileHistOneAPIInitDataSampling) {
  const float subsample = 0.5;
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())},
       {"subsample", std::to_string(subsample)}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestInitDataSampling();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestInitDataSampling();
}

TEST(Plugin, GPUQuantileHistOneAPIAddHistRows) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestAddHistRows();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestAddHistRows();
}

TEST(Plugin, GPUQuantileHistOneAPISyncHistograms) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestSyncHistograms();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestSyncHistograms();
}

TEST(Plugin, GPUQuantileHistOneAPIBuildHist) {
  // Don't enable feature grouping
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())},
       {"enable_feature_grouping", std::to_string(0)}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestBuildHist();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestBuildHist();
}

TEST(Plugin, GPUQuantileHistOneAPIEvalSplits) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())},
       {"split_evaluator", "elastic_net"},
       {"reg_lambda", "0"}, {"reg_alpha", "0"}, {"max_delta_step", "0"},
       {"min_child_weight", "0"}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestEvaluateSplit();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestEvaluateSplit();
}

TEST(Plugin, QuantileHistOneAPIApplySplit) {
  std::vector<std::pair<std::string, std::string>> cfg
      {{"num_feature", std::to_string(QuantileHistMockOneAPI::GetNumColumns())},
       {"split_evaluator", "elastic_net"},
       {"reg_lambda", "0"}, {"reg_alpha", "0"}, {"max_delta_step", "0"},
       {"min_child_weight", "0"}};
  QuantileHistMockOneAPI maker(cfg);
  maker.TestApplySplit();
  const bool single_precision_histogram = true;
  QuantileHistMockOneAPI maker_float(cfg, single_precision_histogram);
  maker_float.TestApplySplit();
}

}  // namespace tree
}  // namespace xgboost
