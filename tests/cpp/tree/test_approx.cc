/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/updater_approx.h"
#include "../helpers.h"

namespace xgboost {
namespace tree {
TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  ApproxRowPartitioner partitioner{n_samples, base_rowid};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  for (auto const &page : Xy->GetBatches<GHistIndexMatrix>({GenericParameter::kCpuId, 64})) {
    bst_feature_t split_ind = 0;
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/0, /*split_index=*/0, /*split_value=*/min_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      ApproxRowPartitioner partitioner{n_samples, base_rowid};
      candidates.front().split.split_value = min_value;
      candidates.front().split.sindex = 0;
      candidates.front().split.sindex |= (1U << 31);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      ApproxRowPartitioner partitioner{n_samples, base_rowid};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/RegTree::kRoot, /*split_index=*/split_ind,
          /*split_value=*/split_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      candidates.front().split.split_value = split_value;
      candidates.front().split.sindex = 0;
      candidates.front().split.sindex |= (1U << 31);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);

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

TEST(Approx, PredictionCache) {
  size_t n_samples = 2048, n_features = 13;
  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);

  {
    omp_set_num_threads(1);
    GenericParameter ctx;
    ctx.InitAllowUnknown(Args{{"nthread", "8"}});
    std::unique_ptr<TreeUpdater> approx{
        TreeUpdater::Create("grow_histmaker", &ctx, ObjInfo{ObjInfo::kRegression})};
    RegTree tree;
    std::vector<RegTree *> trees{&tree};
    auto gpair = GenerateRandomGradients(n_samples);
    approx->Configure(Args{{"max_bin", "64"}});
    approx->Update(&gpair, Xy.get(), trees);
    HostDeviceVector<float> out_prediction_cached;
    out_prediction_cached.Resize(n_samples);
    auto cache = linalg::VectorView<float>{
        out_prediction_cached.HostSpan(), {out_prediction_cached.Size()}, GenericParameter::kCpuId};
    ASSERT_TRUE(approx->UpdatePredictionCache(Xy.get(), cache));
  }

  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  learner->SetParam("tree_method", "approx");
  learner->SetParam("nthread", "0");
  learner->Configure();

  for (size_t i = 0; i < 8; ++i) {
    learner->UpdateOneIter(i, Xy);
  }

  HostDeviceVector<float> out_prediction_cached;
  learner->Predict(Xy, false, &out_prediction_cached, 0, 0);

  Json model{Object()};
  learner->SaveModel(&model);

  HostDeviceVector<float> out_prediction;
  {
    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    learner->LoadModel(model);
    learner->Predict(Xy, false, &out_prediction, 0, 0);
  }

  auto const h_predt_cached = out_prediction_cached.ConstHostSpan();
  auto const h_predt = out_prediction.ConstHostSpan();

  ASSERT_EQ(h_predt.size(), h_predt_cached.size());
  for (size_t i = 0; i < h_predt.size(); ++i) {
    ASSERT_NEAR(h_predt[i], h_predt_cached[i], kRtEps);
  }
}
}  // namespace tree
}  // namespace xgboost
