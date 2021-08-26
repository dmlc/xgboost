/*!
 * Copyright 2018-2021 by Contributors
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
    using GHistRowT = typename RealImpl::GHistRowT;

    BuilderMock(const TrainParam &param, std::unique_ptr<TreeUpdater> pruner,
                DMatrix const *fmat)
        : RealImpl(1, param, std::move(pruner), fmat) {}

   public:
    void TestInitData(const GHistIndexMatrix& gmat,
                      std::vector<GradientPair>* gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      ColumnMatrix column_matrix;
      column_matrix.Init(gmat, 1);
      RealImpl::InitData(gmat, column_matrix, *p_fmat, tree, gpair);
      ASSERT_EQ(this->data_layout_, RealImpl::DataLayout::kSparseData);

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
        auto page = batch.GetView();
        for (size_t i = 0; i < batch.Size(); ++i) {
          const size_t rid = batch.base_rowid + i;
          ASSERT_LT(rid, num_row);
          const size_t gmat_row_offset = gmat.row_ptr[rid];
          ASSERT_LT(gmat_row_offset, gmat.index.Size());
          SparsePage::Inst inst = page[i];
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
                      std::vector<GradientPair>* gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      // check SimpleSkip
      size_t initial_seed = 777;
      RandomReplace::EngineT eng_first(initial_seed);
      for (size_t i = 0; i < 100; ++i) {
        eng_first();
      }
      uint64_t initial_seed_th = RandomReplace::SimpleSkip(100, initial_seed, 16807, RandomReplace::kMod);
      RandomReplace::EngineT eng_second(initial_seed_th);
      ASSERT_EQ(eng_first(), eng_second());

      const size_t nthreads = omp_get_num_threads();
      // save state of global rng engine
      auto initial_rnd = common::GlobalRandom();
      auto initial_rnd_copy = initial_rnd;
      initial_seed = initial_rnd_copy();
      RandomReplace::EngineT test_eng(initial_seed);
      std::vector<size_t> unused_rows;
      std::bernoulli_distribution coin_flip(RealImpl::param_.subsample);

      for (size_t i = 0; i < gpair->size(); ++i) {
        if (!coin_flip(test_eng)) {
          unused_rows.push_back(i);
        }
      }

      for (size_t i_nthreads = 1; i_nthreads < 4; ++i_nthreads) {
        omp_set_num_threads(i_nthreads);
        // return initial state of global rng engine
        common::GlobalRandom() = initial_rnd;

        ColumnMatrix column_matrix;
        column_matrix.Init(gmat, 1);
        RealImpl::InitData(gmat, column_matrix, *p_fmat, tree, gpair);
        for (const size_t unused_row : unused_rows) {
          ASSERT_EQ((*gpair)[unused_row], GradientPair(0));
        }
      }
      omp_set_num_threads(nthreads);
    }

    void TestApplySplit(const RegTree& tree) {
      const size_t initial_nthreads = omp_get_num_threads();
      omp_set_num_threads(1);
      std::vector<GradientPair> row_gpairs =
          { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
            {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
      size_t constexpr kMaxBins = 4;
      CPUExpandEntry node(CPUExpandEntry::kRootNid, 0, 0.0f);
      std::vector<uint16_t> nodes(1, 0);
      std::vector<uint16_t> curr_level_nodes(2);
      std::vector<uint32_t> split_nodes(1, 0);
      curr_level_nodes[0] = 1;
      curr_level_nodes[1] = 2;
      const bst_uint fid = tree[node.nid].SplitIndex();
      // let's left is small
      std::vector<bool> smalest_nodes_mask(3, false);
      smalest_nodes_mask[1] = true;

      // try out different sparsity to get different number of missing values
      for (double sparsity : {0.0, 0.1, 0.2}) {
        // kNRows samples with kNCols features
        auto dmat = RandomDataGenerator(kNRows, kNCols, sparsity).Seed(3).GenerateDMatrix();

        GHistIndexMatrix gmat(dmat.get(), kMaxBins);
        ColumnMatrix cm;

        // treat everything as dense, as this is what we intend to test here
        cm.Init(gmat, 0.0);
        const uint8_t* data = reinterpret_cast<const uint8_t*>(cm.GetIndexData());
        RealImpl::InitData(gmat, cm,  *dmat, tree, &row_gpairs);
        const size_t num_row = dmat->Info().num_row_;
        // split by feature 0
        const size_t bin_id_min = gmat.cut.Ptrs()[0];
        const size_t bin_id_max = gmat.cut.Ptrs()[1];

        RealImpl::opt_partition_builder_.SetSlice(node.nid, 0, num_row);
        const size_t depth_begin = RealImpl::opt_partition_builder_.DepthBegin(nodes,
                                                                     &tree, false);
        const size_t depth_size = RealImpl::opt_partition_builder_.DepthSize(gmat, nodes,
                                                                   &tree, false);
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

          std::vector<int32_t> split_conditions(1, split - gmat.cut.Ptrs()[fid]);
          std::vector<uint64_t> split_ind(1, fid*((gmat.IsDense() ? num_row : 1)));
          std::vector<uint16_t> node_ids(num_row, 0);
          if (gmat.IsDense()) {
            RealImpl::opt_partition_builder_.template CommonPartition<
              uint8_t, false, true>(0, depth_begin,
                                        depth_size, data,
                                        node_ids.data(),
                                        &split_conditions,
                                        &split_ind,
                                        smalest_nodes_mask, row_gpairs,
                                        &curr_level_nodes,
                                        cm, split_nodes);
          } else {
            RealImpl::opt_partition_builder_.template CommonPartition<
              uint8_t, false, false>(0, depth_begin,
                                        depth_size, data,
                                        node_ids.data(),
                                        &split_conditions,
                                        &split_ind,
                                        smalest_nodes_mask, row_gpairs,
                                        &curr_level_nodes,
                                        cm, split_nodes);

          }

          RealImpl::opt_partition_builder_.UpdateRowBuffer(node_ids, &tree,
                                                           gmat, gmat.cut.Ptrs().size() - 1,
                                                           0, node_ids, false);
          ASSERT_EQ(RealImpl::opt_partition_builder_.summ_size, left_cnt);
          ASSERT_EQ(num_row - RealImpl::opt_partition_builder_.summ_size, right_cnt);
        }
      }
      omp_set_num_threads(initial_nthreads);
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
    dmat_ = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
    if (single_precision_histogram) {
      float_builder_.reset(
          new BuilderMock<float>(
              param_,
              std::move(pruner_),
              dmat_.get()));
    } else {
      double_builder_.reset(
          new BuilderMock<double>(
              param_,
              std::move(pruner_),
              dmat_.get()));
    }
  }
  ~QuantileHistMock() override = default;

  static size_t GetNumColumns() { return kNCols; }

  void TestInitData() {
    size_t constexpr kMaxBins = 4;
    GHistIndexMatrix gmat(dmat_.get(), kMaxBins);
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestInitData(gmat, &gpair, dmat_.get(), tree);
    } else {
      float_builder_->TestInitData(gmat, &gpair, dmat_.get(), tree);
    }
  }

  void TestInitDataSampling() {
    size_t constexpr kMaxBins = 4;
    GHistIndexMatrix gmat(dmat_.get(), kMaxBins);

    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);

    std::vector<GradientPair> gpair =
        { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
          {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
    if (double_builder_) {
      double_builder_->TestInitDataSampling(gmat, &gpair, dmat_.get(), tree);
    } else {
      float_builder_->TestInitDataSampling(gmat, &gpair, dmat_.get(), tree);
    }
  }

  void TestApplySplit() {
    RegTree tree = RegTree();
    tree.param.UpdateAllowUnknown(cfg_);
    if (double_builder_) {
      double_builder_->TestApplySplit(tree);
    } else {
      float_builder_->TestApplySplit(tree);
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
