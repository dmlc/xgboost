/*!
 * Copyright 2018-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../../src/tree/param.h"
#include "../../../src/tree/split_evaluator.h"
#include "../../../src/tree/updater_quantile_hist.h"
#include "../helpers.h"
#include "test_partitioner.h"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {

class QuantileHistMock : public QuantileHistMaker {
  static double constexpr kEps = 1e-6;

  template <typename GradientSumT>
  struct BuilderMock : public QuantileHistMaker::Builder<GradientSumT> {
    using RealImpl = QuantileHistMaker::Builder<GradientSumT>;

    BuilderMock(const TrainParam &param, std::unique_ptr<TreeUpdater> pruner,
                DMatrix const *fmat, GenericParameter const* ctx)
        : RealImpl(1, param, std::move(pruner), fmat, ObjInfo{ObjInfo::kRegression}, ctx) {}

   public:
    void TestInitData(const GHistIndexMatrix& gmat,
                      std::vector<GradientPair>* gpair,
                      DMatrix* p_fmat,
                      const RegTree& tree) {
      RealImpl::InitData(gmat, *p_fmat, tree, gpair);
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
  };

  int static constexpr kNRows = 8, kNCols = 16;
  std::shared_ptr<xgboost::DMatrix> dmat_;
  GenericParameter ctx_;
  const std::vector<std::pair<std::string, std::string> > cfg_;
  std::shared_ptr<BuilderMock<float> > float_builder_;
  std::shared_ptr<BuilderMock<double> > double_builder_;

 public:
  explicit QuantileHistMock(
      const std::vector<std::pair<std::string, std::string> >& args,
      const bool single_precision_histogram = false, bool batch = true) :
      QuantileHistMaker{ObjInfo{ObjInfo::kRegression}}, cfg_{args} {
    QuantileHistMaker::Configure(args);
    dmat_ = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
    ctx_.UpdateAllowUnknown(Args{});
    if (single_precision_histogram) {
      float_builder_.reset(new BuilderMock<float>(param_, std::move(pruner_), dmat_.get(), &ctx_));
    } else {
      double_builder_.reset(
          new BuilderMock<double>(param_, std::move(pruner_), dmat_.get(), &ctx_));
    }
  }
  ~QuantileHistMock() override = default;

  static size_t GetNumColumns() { return kNCols; }

  void TestInitData() {
    int32_t constexpr kMaxBins = 4;
    GHistIndexMatrix gmat{dmat_.get(), kMaxBins, 0.0f, false, common::OmpGetNumThreads(0)};

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

TEST(QuantileHist, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});

  HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  auto grad = GenerateRandomGradients(n_samples);
  std::vector<float> hess(grad.Size());
  std::transform(grad.HostVector().cbegin(), grad.HostVector().cend(), hess.begin(),
                 [](auto gpair) { return gpair.GetHess(); });

  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>({64, 0.5})) {
    bst_feature_t const split_ind = 0;
    common::ColumnMatrix column_indices;
    column_indices.Init(page, 0.5, ctx.Threads());
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition<false, true>(&ctx, page, column_indices, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      GetSplit(&tree, split_value, &candidates);
      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      partitioner.UpdatePosition<false, true>(&ctx, page, column_indices, candidates, &tree);

      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = page.cut.Values().at(page.index[*it]);
        ASSERT_LE(value, split_value);
      }
      auto right_nidx = tree[RegTree::kRoot].RightChild();
      elem = partitioner[right_nidx];
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = page.cut.Values().at(page.index[*it]);
        ASSERT_GT(value, split_value) << *it;
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
