/*!
 * Copyright 2020 XGBoost contributors
 */
#include <dmlc/filesystem.h>
#include <fstream>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/learner.h>

#include "gtest/gtest.h"

namespace xgboost {
namespace tree {

class GpuHistSamplingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    constexpr size_t kRows = 1000;
    constexpr size_t kCols = 1;
    constexpr size_t kPageSize = 1024;

    temp_dir = new dmlc::TemporaryDirectory();
    const std::string tmp_file = temp_dir->path + "/random.libsvm";
    {
      std::ofstream fo(tmp_file.c_str());

      std::mt19937 gen{2020}; // NOLINT
      std::normal_distribution<> rnd{0, 1};

      for (size_t i = 0; i < kRows; i++) {
        std::stringstream row;
        row << rnd(gen);

        for (size_t j = 0; j < kCols; j++) {
          row << " " << j << ":" << rnd(gen);
        }
        fo << row.str() << "\n";
      }
    }

    dmat = std::shared_ptr<DMatrix>(DMatrix::Load(tmp_file, true, false));
    const std::string ext_mem_file = tmp_file + "#" + tmp_file + ".cache";
    dmat_ext = std::shared_ptr<DMatrix>(
        DMatrix::Load(ext_mem_file, true, false, "auto", kPageSize));
  }

  static void TearDownTestCase() {
    dmat.reset();
    dmat_ext.reset();
    delete temp_dir;
  }

  static void VerifyPredictionMean(const std::shared_ptr<DMatrix>& dtrain,
                                   float subsample = 1.0f,
                                   const std::string& sampling_method = "uniform") {
    std::vector<std::shared_ptr<DMatrix>> cache_mats{dtrain};
    std::unique_ptr<Learner> learner(Learner::Create(cache_mats));
    Args args {
        {"tree_method", "gpu_hist"},
        {"max_depth", "1"},
        {"subsample", std::to_string(subsample)},
        {"sampling_method", sampling_method},

        {"learning_rate", "1"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
    };
    learner->SetParams(args);

    constexpr int kNumRound = 10;
    for (int i = 0; i < kNumRound; ++i) {
      learner->UpdateOneIter(i, dtrain.get());
    }

    HostDeviceVector<bst_float> preds;
    learner->Predict(dtrain.get(), true, &preds);
    auto h_preds = preds.ConstHostVector();
    float mean = std::accumulate(h_preds.begin(), h_preds.end(), 0.0f) / h_preds.size();
    EXPECT_NEAR(mean, 0.0f, 2e-2) << "subsample=" << subsample;
  }

  static dmlc::TemporaryDirectory* temp_dir;
  static std::shared_ptr<DMatrix> dmat;
  static std::shared_ptr<DMatrix> dmat_ext;
};

dmlc::TemporaryDirectory* GpuHistSamplingTest::temp_dir;
std::shared_ptr<DMatrix> GpuHistSamplingTest::dmat;
std::shared_ptr<DMatrix> GpuHistSamplingTest::dmat_ext;

TEST_F(GpuHistSamplingTest, NoSampling) {
  VerifyPredictionMean(dmat);
}

TEST_F(GpuHistSamplingTest, NoSampling_ExternalMemory) {
  VerifyPredictionMean(dmat_ext);
}

TEST_F(GpuHistSamplingTest, UniformSampling) {
  for (int i = 1; i < 10; i++) {
    float subsample = static_cast<float>(i) / 10.0f;
    VerifyPredictionMean(dmat, subsample);
  }
}

TEST_F(GpuHistSamplingTest, UniformSampling_ExternalMemory) {
  for (int i = 1; i < 10; i++) {
    float subsample = static_cast<float>(i) / 10.0f;
    VerifyPredictionMean(dmat_ext, subsample);
  }
}

TEST_F(GpuHistSamplingTest, GradientBasedSampling) {
  for (int i = 1; i < 10; i++) {
    float subsample = static_cast<float>(i) / 10.0f;
    VerifyPredictionMean(dmat, subsample, "gradient_based");
  }
}

TEST_F(GpuHistSamplingTest, GradientBasedSampling_ExternalMemory) {
  for (int i = 1; i < 10; i++) {
    float subsample = static_cast<float>(i) / 10.0f;
    VerifyPredictionMean(dmat_ext, subsample, "gradient_based");
  }
}

}  // namespace tree
}  // namespace xgboost
