/**
 * Copyright 2021-2024 by XGBoost contributors.
 */
#pragma once

#include <gtest/gtest.h>

#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>

#include <memory>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"
#include "xgboost/task.h"             // for ObjInfo

namespace xgboost {
class TestPredictionCache : public ::testing::Test {
  std::shared_ptr<DMatrix> Xy_;
  std::size_t n_samples_{2048};

 protected:
  void SetUp() override {
    std::size_t n_features = 13;
    bst_target_t n_targets = 3;
    Xy_ = RandomDataGenerator{n_samples_, n_features, 0}.Targets(n_targets).GenerateDMatrix(true);
  }

  void RunLearnerTest(Context const* ctx, std::string updater_name, float subsample,
                      std::string const& grow_policy, std::string const& strategy) {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("device", ctx->DeviceName());
    learner->SetParam("updater", updater_name);
    learner->SetParam("multi_strategy", strategy);
    learner->SetParam("grow_policy", grow_policy);
    learner->SetParam("subsample", std::to_string(subsample));
    learner->SetParam("nthread", "0");
    learner->Configure();

    for (size_t i = 0; i < 8; ++i) {
      learner->UpdateOneIter(i, Xy_);
    }

    HostDeviceVector<float> out_prediction_cached;
    learner->Predict(Xy_, false, &out_prediction_cached, 0, 0);

    Json model{Object()};
    learner->SaveModel(&model);

    HostDeviceVector<float> out_prediction;
    {
      std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
      learner->LoadModel(model);
      learner->Predict(Xy_, false, &out_prediction, 0, 0);
    }

    auto const h_predt_cached = out_prediction_cached.ConstHostSpan();
    auto const h_predt = out_prediction.ConstHostSpan();

    ASSERT_EQ(h_predt.size(), h_predt_cached.size());
    for (size_t i = 0; i < h_predt.size(); ++i) {
      ASSERT_NEAR(h_predt[i], h_predt_cached[i], kRtEps);
    }
  }

  void RunTest(Context* ctx, std::string const& updater_name, std::string const& strategy) {
    {
      ctx->InitAllowUnknown(Args{{"nthread", "8"}});

      ObjInfo task{ObjInfo::kRegression};
      std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create(updater_name, ctx, &task)};
      RegTree tree;
      std::vector<RegTree*> trees{&tree};
      auto gpair = GenerateRandomGradients(ctx, n_samples_, 1);
      tree::TrainParam param;
      param.UpdateAllowUnknown(Args{{"max_bin", "64"}});

      updater->Configure(Args{});
      std::vector<HostDeviceVector<bst_node_t>> position(1);
      updater->Update(&param, &gpair, Xy_.get(), position, trees);
      HostDeviceVector<float> out_prediction_cached;
      out_prediction_cached.SetDevice(ctx->Device());
      out_prediction_cached.Resize(n_samples_);
      auto cache =
          linalg::MakeTensorView(ctx, &out_prediction_cached, out_prediction_cached.Size(), 1);
      ASSERT_TRUE(updater->UpdatePredictionCache(Xy_.get(), cache));
    }

    for (auto policy : {"depthwise", "lossguide"}) {
      for (auto subsample : {1.0f, 0.4f}) {
        this->RunLearnerTest(ctx, updater_name, subsample, policy, strategy);
        this->RunLearnerTest(ctx, updater_name, subsample, policy, strategy);
      }
    }
  }
};
}  // namespace xgboost
