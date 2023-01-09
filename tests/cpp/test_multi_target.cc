/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>     // bst_target_t
#include <xgboost/data.h>     // DMatrix
#include <xgboost/json.h>     // Json,Object,Number,get
#include <xgboost/learner.h>  // Learner

#include <cstddef>            // size_t
#include <memory>             // shared_ptr,unique_ptr
#include <numeric>
#include <string>             // stod
#include <vector>

#include "../../src/common/linalg_op.h"  // cbegin,cend
#include "../../src/common/stats.h"      // Median
#include "helpers.h"                     // RandomDataGenerator
#include "xgboost/linalg.h"

namespace xgboost {

class TestL1MultiTarget : public ::testing::Test {
  std::shared_ptr<DMatrix> Xy_;
  std::shared_ptr<DMatrix> Xyw_;
  std::vector<std::shared_ptr<DMatrix>> single_;
  std::vector<std::shared_ptr<DMatrix>> single_w_;

 public:
  void SetUp() override {
    std::size_t constexpr kRows{256}, kCols{5}, kTargets{3};
    auto make_fmat = [&](bool weighted) {
      if (weighted) {
        auto p_fmat =
            RandomDataGenerator{kRows, kCols, 0.5f}.Targets(kTargets).GenerateDMatrix(true);
        p_fmat->Info().weights_.Resize(kRows);
        RandomDataGenerator{kRows, 1, 0.0f}.GenerateDense(&p_fmat->Info().weights_);
        return p_fmat;
      } else {
        return RandomDataGenerator{kRows, kCols, 0.5f}.Targets(kTargets).GenerateDMatrix(true);
      }
    };

    Xy_ = make_fmat(false);
    Xyw_ = make_fmat(true);
    ASSERT_EQ(Xy_->Info().labels.Shape(1), kTargets);
    ASSERT_EQ(Xyw_->Info().labels.Shape(1), kTargets);

    single_.clear();
    single_w_.clear();
    for (bst_target_t t{0}; t < kTargets; ++t) {
      {
        single_.emplace_back(make_fmat(false));
        single_[t]->Info().labels.Reshape(kRows, 1);
        auto h_labels = single_[t]->Info().labels.HostView();
        auto in_labels = Xy_->Info().labels.HostView().Slice(linalg::All(), t);
        std::copy(linalg::cbegin(in_labels), linalg::cend(in_labels), linalg::begin(h_labels));
      }
      {
        single_w_.emplace_back(make_fmat(true));
        single_w_[t]->Info().labels.Reshape(kRows, 1);
        auto h_labels = single_w_[t]->Info().labels.HostView();
        auto in_labels = Xyw_->Info().labels.HostView().Slice(linalg::All(), t);
        std::copy(linalg::cbegin(in_labels), linalg::cend(in_labels), linalg::begin(h_labels));
      }
    }
  }

  void RunTest(std::string const& tree_method, bool weight) {
    auto p_fmat = weight ? Xyw_ : Xy_;
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->SetParams(Args{{"tree_method", tree_method}, {"objective", "reg:absoluteerror"}});
    learner->Configure();
    for (auto i = 0; i < 4; ++i) {
      learner->UpdateOneIter(i, p_fmat);
    }
    ASSERT_EQ(learner->Groups(), 3);

    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score =
        std::stod(get<String const>(config["learner"]["learner_model_param"]["base_score"]));

    std::vector<float> base_scores;
    for (bst_target_t t{0}; t < p_fmat->Info().labels.Shape(1); ++t) {
      auto t_Xy = weight ? single_w_[t] : single_[t];
      std::unique_ptr<Learner> sl{Learner::Create({t_Xy})};
      sl->SetParams(Args{{"tree_method", tree_method}, {"objective", "reg:absoluteerror"}});
      sl->Configure();
      sl->UpdateOneIter(0, t_Xy);
      Json s_config{Object{}};
      sl->SaveConfig(&s_config);
      auto s_base_score =
          std::stod(get<String const>(s_config["learner"]["learner_model_param"]["base_score"]));
      linalg::Vector<float> out;
      common::Median(sl->Ctx(), t_Xy->Info().labels, t_Xy->Info().weights_, &out);
      ASSERT_FLOAT_EQ(s_base_score, out(0));
      base_scores.push_back(s_base_score);
    }
    auto mean = std::accumulate(base_scores.cbegin(), base_scores.cend(), .0f) /
                static_cast<float>(base_scores.size());
    ASSERT_FLOAT_EQ(mean, base_score);
  }

  void RunTest(std::string const& tree_method) {
    this->RunTest(tree_method, false);
    this->RunTest(tree_method, true);
  }
};

TEST_F(TestL1MultiTarget, Hist) { this->RunTest("hist"); }

TEST_F(TestL1MultiTarget, Exact) { this->RunTest("exact"); }

TEST_F(TestL1MultiTarget, Approx) { this->RunTest("approx"); }

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestL1MultiTarget, GpuHist) { this->RunTest("gpu_hist"); }
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
