/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include "helpers.h"
#include <dmlc/filesystem.h>

#include <xgboost/learner.h>
#include <xgboost/version_config.h>
#include "xgboost/json.h"
#include "../../src/common/io.h"
#include "../../src/common/random.h"

namespace xgboost {

TEST(Learner, Basic) {
  using Arg = std::pair<std::string, std::string>;
  auto args = {Arg("tree_method", "exact")};
  auto mat_ptr = RandomDataGenerator{10, 10, 0.0f}.GenerateDMatrix();
  auto learner = std::unique_ptr<Learner>(Learner::Create({mat_ptr}));
  learner->SetParams(args);


  auto major = XGBOOST_VER_MAJOR;
  auto minor = XGBOOST_VER_MINOR;
  auto patch = XGBOOST_VER_PATCH;

  static_assert(std::is_integral<decltype(major)>::value, "Wrong major version type");
  static_assert(std::is_integral<decltype(minor)>::value, "Wrong minor version type");
  static_assert(std::is_integral<decltype(patch)>::value, "Wrong patch version type");
}

TEST(Learner, ParameterValidation) {
  ConsoleLogger::Configure({{"verbosity", "2"}});
  size_t constexpr kRows = 1;
  size_t constexpr kCols = 1;
  auto p_mat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
  learner->SetParam("validate_parameters", "1");
  learner->SetParam("Knock Knock", "Who's there?");
  learner->SetParam("Silence", "....");
  learner->SetParam("tree_method", "exact");

  testing::internal::CaptureStderr();
  learner->Configure();
  std::string output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(output.find("Parameters: { Knock Knock, Silence }") != std::string::npos);
}

TEST(Learner, CheckGroup) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  bst_feature_t constexpr kNumCols = 15;

  std::shared_ptr<DMatrix> p_mat{
      RandomDataGenerator{kNumRows, kNumCols, 0.0f}.GenerateDMatrix()};
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
  EXPECT_NO_THROW(learner->UpdateOneIter(0, p_mat));

  group.resize(kNumGroups+1);
  group[3] = 4;
  group[4] = 1;
  p_mat->Info().SetInfo("group", group.data(), DataType::kUInt32, kNumGroups+1);
  EXPECT_ANY_THROW(learner->UpdateOneIter(0, p_mat));
}

TEST(Learner, SLOW_CheckMultiBatch) {  // NOLINT
  // Create sufficiently large data to make two row pages
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/big.libsvm";
  CreateBigTestData(tmp_file, 50000);
  std::shared_ptr<DMatrix> dmat(xgboost::DMatrix::Load(
      tmp_file + "#" + tmp_file + ".cache", true, false, "auto", 100));
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
  learner->SetParams(Args{{"objective", "binary:logistic"}});
  learner->UpdateOneIter(0, dmat);
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
    ASSERT_EQ(attr_names.size(), 1ul);
    ASSERT_EQ(attr_names.find(emetric), attr_names.cend());
    ASSERT_EQ(attr_names.at("foo"), "bar");
  }

  {
    std::unique_ptr<Learner> learner { Learner::Create({nullptr}) };
    learner->SetParams({{"foo", "bar"}, {emetric, "auc"}, {emetric, "entropy"}, {emetric, "KL"}});
    auto attr_names = learner->GetConfigurationArguments();
    ASSERT_EQ(attr_names.size(), 1ul);
    ASSERT_EQ(attr_names.at("foo"), "bar");
  }
}

TEST(Learner, JsonModelIO) {
  // Test of comparing JSON object directly.
  size_t constexpr kRows = 8;
  int32_t constexpr kIters = 4;

  std::shared_ptr<DMatrix> p_dmat{
    RandomDataGenerator{kRows, 10, 0}.GenerateDMatrix()};
  p_dmat->Info().labels_.Resize(kRows);
  CHECK_NE(p_dmat->Info().num_col_, 0);

  {
    std::unique_ptr<Learner> learner { Learner::Create({p_dmat}) };
    learner->Configure();
    Json out { Object() };
    learner->SaveModel(&out);

    dmlc::TemporaryDirectory tmpdir;

    std::ofstream fout (tmpdir.path + "/model.json");
    fout << out;
    fout.close();

    auto loaded_str = common::LoadSequentialFile(tmpdir.path + "/model.json");
    Json loaded = Json::Load(StringView{loaded_str.c_str(), loaded_str.size()});

    learner->LoadModel(loaded);
    learner->Configure();

    Json new_in { Object() };
    learner->SaveModel(&new_in);
    ASSERT_EQ(new_in, out);
  }

  {
    std::unique_ptr<Learner> learner { Learner::Create({p_dmat}) };
    for (int32_t iter = 0; iter < kIters; ++iter) {
      learner->UpdateOneIter(iter, p_dmat);
    }
    learner->SetAttr("best_score", "15.2");

    Json out { Object() };
    learner->SaveModel(&out);

    learner->LoadModel(out);
    Json new_in { Object() };
    learner->Configure();
    learner->SaveModel(&new_in);

    ASSERT_TRUE(IsA<Object>(out["learner"]["attributes"]));
    ASSERT_EQ(get<Object>(out["learner"]["attributes"]).size(), 1ul);
    ASSERT_EQ(out, new_in);
  }
}

// Crashes the test runner if there are race condiditions.
//
// Build with additional cmake flags to enable thread sanitizer
// which definitely catches problems. Note that OpenMP needs to be
// disabled, otherwise thread sanitizer will also report false
// positives.
//
// ```
// -DUSE_SANITIZER=ON -DENABLED_SANITIZERS=thread -DUSE_OPENMP=OFF
// ```
TEST(Learner, MultiThreadedPredict) {
  size_t constexpr kRows = 1000;
  size_t constexpr kCols = 1000;

  std::shared_ptr<DMatrix> p_dmat{
      RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix()};
  p_dmat->Info().labels_.Resize(kRows);
  CHECK_NE(p_dmat->Info().num_col_, 0);

  std::shared_ptr<DMatrix> p_data{
      RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix()};
  CHECK_NE(p_data->Info().num_col_, 0);

  std::shared_ptr<Learner> learner{Learner::Create({p_dmat})};
  learner->Configure();

  std::vector<std::thread> threads;
  for (uint32_t thread_id = 0;
       thread_id < 2 * std::thread::hardware_concurrency(); ++thread_id) {
    threads.emplace_back([learner, p_data] {
      size_t constexpr kIters = 10;
      auto &entry = learner->GetThreadLocal().prediction_entry;
      for (size_t iter = 0; iter < kIters; ++iter) {
        learner->Predict(p_data, false, &entry.predictions);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST(Learner, BinaryModelIO) {
  size_t constexpr kRows = 8;
  int32_t constexpr kIters = 4;
  auto p_dmat = RandomDataGenerator{kRows, 10, 0}.GenerateDMatrix();
  p_dmat->Info().labels_.Resize(kRows);

  std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
  learner->SetParam("eval_metric", "rmsle");
  learner->Configure();
  for (int32_t iter = 0; iter < kIters; ++iter) {
    learner->UpdateOneIter(iter, p_dmat);
  }
  dmlc::TemporaryDirectory tempdir;
  std::string const fname = tempdir.path + "binary_model_io.bin";
  {
    // Make sure the write is complete before loading.
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    learner->SaveModel(fo.get());
  }

  learner.reset(Learner::Create({p_dmat}));
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
  learner->LoadModel(fi.get());
  learner->Configure();
  Json config { Object() };
  learner->SaveConfig(&config);
  std::string config_str;
  Json::Dump(config, &config_str);
  ASSERT_NE(config_str.find("rmsle"), std::string::npos);
  ASSERT_EQ(config_str.find("WARNING"), std::string::npos);
}

#if defined(XGBOOST_USE_CUDA)
// Tests for automatic GPU configuration.
TEST(Learner, GPUConfiguration) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kRows = 10;
  auto p_dmat = RandomDataGenerator(kRows, 10, 0).GenerateDMatrix();
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
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  }
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "gpu_hist"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  }
  {
    // with CPU algorithm
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, -1);
  }
  {
    // with CPU algorithm, but `gpu_id` takes priority
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"gpu_id", "0"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  }
  {
    // With CPU algorithm but GPU Predictor, this is to simulate when
    // XGBoost is only used for prediction, so tree method is not
    // specified.
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"predictor", "gpu_predictor"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->GetGenericParameter().gpu_id, 0);
  }
}
#endif  // defined(XGBOOST_USE_CUDA)

TEST(Learner, Seed) {
  auto m = RandomDataGenerator{10, 10, 0}.GenerateDMatrix();
  std::unique_ptr<Learner> learner {
    Learner::Create({m})
  };
  auto seed = std::numeric_limits<int64_t>::max();
  learner->SetParam("seed", std::to_string(seed));
  learner->Configure();
  Json config { Object() };
  learner->SaveConfig(&config);
  ASSERT_EQ(std::to_string(seed),
            get<String>(config["learner"]["generic_param"]["seed"]));

  seed = std::numeric_limits<int64_t>::min();
  learner->SetParam("seed", std::to_string(seed));
  learner->Configure();
  learner->SaveConfig(&config);
  ASSERT_EQ(std::to_string(seed),
            get<String>(config["learner"]["generic_param"]["seed"]));
}

TEST(Learner, ConstantSeed) {
  auto m = RandomDataGenerator{10, 10, 0}.GenerateDMatrix(true);
  std::unique_ptr<Learner> learner{Learner::Create({m})};
  learner->Configure();  // seed the global random

  std::uniform_real_distribution<float> dist;
  auto& rng = common::GlobalRandom();
  float v_0 = dist(rng);

  learner->SetParam("", "");
  learner->Configure();  // check configure doesn't change the seed.
  float v_1 = dist(rng);
  CHECK_NE(v_0, v_1);

  {
    rng.seed(GenericParameter::kDefaultSeed);
    std::uniform_real_distribution<float> dist;
    float v_2 = dist(rng);
    CHECK_EQ(v_0, v_2);
  }
}
}  // namespace xgboost
