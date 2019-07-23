// Copyright by Contributors
#include <gtest/gtest.h>
#include <vector>
#include "helpers.h"

#include "xgboost/learner.h"
#include "dmlc/filesystem.h"

namespace xgboost {

TEST(Learner, Basic) {
  using Arg = std::pair<std::string, std::string>;
  auto args = {Arg("tree_method", "exact")};
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams(args);

  delete mat_ptr;
}

TEST(Learner, CheckGroup) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  size_t constexpr kNumCols = 15;

  auto pp_mat = CreateDMatrix(kNumRows, kNumCols, 0);
  auto& p_mat = *pp_mat;
  std::vector<bst_float> weight(kNumGroups);
  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;
  std::vector<bst_float> labels (kNumRows);
  for (size_t i = 0; i < kNumRows; ++i) {
    labels[i] = i % 2;
  }

  p_mat->Info().SetInfo(
      "weight", static_cast<void*>(weight.data()), DataType::kFloat32, kNumGroups);
  p_mat->Info().SetInfo(
      "group", group.data(), DataType::kUInt32, kNumGroups);
  p_mat->Info().SetInfo("label", labels.data(), DataType::kFloat32, kNumRows);

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_mat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams({Arg{"objective", "rank:pairwise"}});
  EXPECT_NO_THROW(learner->UpdateOneIter(0, p_mat.get()));

  group.resize(kNumGroups+1);
  group[3] = 4;
  group[4] = 1;
  p_mat->Info().SetInfo("group", group.data(), DataType::kUInt32, kNumGroups+1);
  EXPECT_ANY_THROW(learner->UpdateOneIter(0, p_mat.get()));

  delete pp_mat;
}

TEST(Learner, SLOW_CheckMultiBatch) {
  using Arg = std::pair<std::string, std::string>;
  // Create sufficiently large data to make two row pages
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/big.libsvm";
  CreateBigTestData(tmp_file, 5000000);
  std::shared_ptr<DMatrix> dmat(xgboost::DMatrix::Load( tmp_file + "#" + tmp_file + ".cache", true, false));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));
  EXPECT_FALSE(dmat->SingleColBlock());
  size_t num_row = dmat->Info().num_row_;
  std::vector<bst_float> labels(num_row);
  for (size_t i = 0; i < num_row; ++i) {
    labels[i] = i % 2;
  }
  dmat->Info().SetInfo("label", labels.data(), DataType::kFloat32, num_row);
  std::vector<std::shared_ptr<DMatrix>> mat{dmat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams({Arg{"objective", "binary:logistic"}, Arg{"verbosity", "3"}});
  learner->UpdateOneIter(0, dmat.get());
  ASSERT_EQ(learner->GetGenericParameter().external_memory, true);
}

TEST(Learner, Configuration) {
  std::string const emetric = "eval_metric";
  {
    std::unique_ptr<Learner> learner { Learner::Create({nullptr}) };
    learner->SetParam(emetric, "auc");
    learner->SetParam(emetric, "rmsle");
    learner->SetParam("foo", "bar");

    // eval_metric is not part of configuration
    auto attr_names = learner->GetConfigurationArguments();
    ASSERT_EQ(attr_names.size(), 1);
    ASSERT_EQ(attr_names.find(emetric), attr_names.cend());
    ASSERT_EQ(attr_names.at("foo"), "bar");
  }

  {
    std::unique_ptr<Learner> learner { Learner::Create({nullptr}) };
    learner->SetParams({{"foo", "bar"}, {emetric, "auc"}, {emetric, "entropy"}, {emetric, "KL"}});
    auto attr_names = learner->GetConfigurationArguments();
    ASSERT_EQ(attr_names.size(), 1);
    ASSERT_EQ(attr_names.at("foo"), "bar");
  }
}

#if defined(XGBOOST_USE_CUDA)

TEST(Learner, IO) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kRows = 10;
  auto pp_dmat = CreateDMatrix(kRows, 10, 0);
  auto p_dmat = *pp_dmat;

  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;
  std::vector<std::shared_ptr<DMatrix>> mat {p_dmat};

  std::unique_ptr<Learner> learner {Learner::Create(mat)};
  learner->SetParams({Arg{"tree_method", "auto"},
                      Arg{"predictor", "gpu_predictor"},
                      Arg{"n_gpus", "-1"}});
  learner->UpdateOneIter(0, p_dmat.get());
  ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  ASSERT_EQ(learner->GetGenericParameter().n_gpus, -1);
  ASSERT_EQ(learner->GetGenericParameter().external_memory, false);

  dmlc::TemporaryDirectory tempdir;
  const std::string fname = tempdir.path + "/model.bst";

  {
    // Create a scope to close the stream before next read.
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    learner->Save(fo.get());
  }

  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
  learner->Load(fi.get());
  ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  ASSERT_EQ(learner->GetGenericParameter().n_gpus, 0);

  delete pp_dmat;
}

// Tests for automatic GPU configuration.
TEST(Learner, GPUConfiguration) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kRows = 10;
  auto pp_dmat = CreateDMatrix(kRows, 10, 0);
  auto p_dmat = *pp_dmat;
  std::vector<std::shared_ptr<DMatrix>> mat {p_dmat};
  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"booster", "gblinear"},
                        Arg{"updater", "gpu_coord_descent"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 1);
  }
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "gpu_exact"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 1);
  }
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "gpu_hist"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 1);
  }
  {
    // with CPU algorithm
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 0);
  }
  {
    // with CPU algorithm, but `n_gpus` takes priority
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"n_gpus", "1"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 1);
  }
  {
    // With CPU algorithm but GPU Predictor, this is to simulate when
    // XGBoost is only used for prediction, so tree method is not
    // specified.
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"predictor", "gpu_predictor"}});
    learner->UpdateOneIter(0, p_dmat.get());
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
    ASSERT_EQ(learner->GetGenericParameter().n_gpus, 1);
  }

  delete pp_dmat;
}

#endif  // XGBOOST_USE_CUDA

}  // namespace xgboost
