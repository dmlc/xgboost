/**
 * Copyright (c) 2017-2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/learner.h>                        // for Learner
#include <xgboost/logging.h>                        // for LogCheck_NE, CHECK_NE, LogCheck_EQ
#include <xgboost/objective.h>                      // for ObjFunction
#include <xgboost/version_config.h>                 // for XGBOOST_VER_MAJOR, XGBOOST_VER_MINOR

#include <algorithm>                                // for equal, transform
#include <cinttypes>                                // for int32_t, int64_t, uint32_t
#include <cstddef>                                  // for size_t
#include <iosfwd>                                   // for ofstream
#include <iterator>                                 // for back_insert_iterator, back_inserter
#include <limits>                                   // for numeric_limits
#include <map>                                      // for map
#include <memory>                                   // for unique_ptr, shared_ptr, __shared_ptr_...
#include <random>                                   // for uniform_real_distribution
#include <string>                                   // for allocator, basic_string, string, oper...
#include <thread>                                   // for thread
#include <type_traits>                              // for is_integral
#include <utility>                                  // for pair
#include <vector>                                   // for vector

#include "../../src/collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../../src/common/api_entry.h"             // for XGBAPIThreadLocalEntry
#include "../../src/common/io.h"                    // for LoadSequentialFile
#include "../../src/common/linalg_op.h"             // for ElementWiseTransformHost, begin, end
#include "../../src/common/random.h"                // for GlobalRandom
#include "../../src/common/transform_iterator.h"    // for IndexTransformIter
#include "dmlc/io.h"                                // for Stream
#include "dmlc/omp.h"                               // for omp_get_max_threads
#include "dmlc/registry.h"                          // for Registry
#include "filesystem.h"                             // for TemporaryDirectory
#include "helpers.h"                                // for GetBaseScore, RandomDataGenerator
#include "objective_helpers.h"                      // for MakeObjNamesForTest, ObjTestNameGenerator
#include "xgboost/base.h"                           // for bst_float, Args, bst_feature_t, bst_int
#include "xgboost/context.h"                        // for Context
#include "xgboost/data.h"                           // for DMatrix, MetaInfo, DataType
#include "xgboost/host_device_vector.h"             // for HostDeviceVector
#include "xgboost/json.h"                           // for Json, Object, get, String, IsA, opera...
#include "xgboost/linalg.h"                         // for Tensor, TensorView
#include "xgboost/logging.h"                        // for ConsoleLogger
#include "xgboost/predictor.h"                      // for PredictionCacheEntry
#include "xgboost/span.h"                           // for Span, operator!=, SpanIterator
#include "xgboost/string_view.h"                    // for StringView

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
  learner->SetParam("Knock-Knock", "Who's-there?");
  learner->SetParam("Silence", "....");
  learner->SetParam("tree_method", "exact");

  testing::internal::CaptureStderr();
  learner->Configure();
  std::string output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(output.find(R"(Parameters: { "Knock-Knock", "Silence" })") != std::string::npos);

  // whitespace
  learner->SetParam("tree method", "exact");
  EXPECT_THROW(learner->Configure(), dmlc::Error);
}

TEST(Learner, CheckGroup) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  bst_feature_t constexpr kNumCols = 15;

  std::shared_ptr<DMatrix> p_mat{
      RandomDataGenerator{kNumRows, kNumCols, 0.0f}.GenerateDMatrix()};
  std::vector<bst_float> weight(kNumGroups, 1);
  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;
  std::vector<bst_float> labels (kNumRows);
  for (size_t i = 0; i < kNumRows; ++i) {
    labels[i] = i % 2;
  }

  p_mat->SetInfo("weight", static_cast<void *>(weight.data()), DataType::kFloat32, kNumGroups);
  p_mat->SetInfo("group", group.data(), DataType::kUInt32, kNumGroups);
  p_mat->SetInfo("label", labels.data(), DataType::kFloat32, kNumRows);

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_mat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams({Arg{"objective", "rank:pairwise"}});
  EXPECT_NO_THROW(learner->UpdateOneIter(0, p_mat));

  group.resize(kNumGroups+1);
  group[3] = 4;
  group[4] = 1;
  p_mat->SetInfo("group", group.data(), DataType::kUInt32, kNumGroups+1);
  EXPECT_ANY_THROW(learner->UpdateOneIter(0, p_mat));
}

TEST(Learner, SLOW_CheckMultiBatch) {  // NOLINT
  // Create sufficiently large data to make two row pages
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/big.libsvm";
  CreateBigTestData(tmp_file, 50000);
  std::shared_ptr<DMatrix> dmat(
      xgboost::DMatrix::Load(tmp_file + "?format=libsvm" + "#" + tmp_file + ".cache"));
  EXPECT_FALSE(dmat->SingleColBlock());
  size_t num_row = dmat->Info().num_row_;
  std::vector<bst_float> labels(num_row);
  for (size_t i = 0; i < num_row; ++i) {
    labels[i] = i % 2;
  }
  dmat->SetInfo("label", labels.data(), DataType::kFloat32, num_row);
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

  std::shared_ptr<DMatrix> p_dmat{RandomDataGenerator{kRows, 10, 0}.GenerateDMatrix()};
  p_dmat->Info().labels.Reshape(kRows);
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
  size_t constexpr kCols = 100;

  std::shared_ptr<DMatrix> p_dmat{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix()};
  p_dmat->Info().labels.Reshape(kRows);
  CHECK_NE(p_dmat->Info().num_col_, 0);

  std::shared_ptr<DMatrix> p_data{RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix()};
  CHECK_NE(p_data->Info().num_col_, 0);

  std::shared_ptr<Learner> learner{Learner::Create({p_dmat})};
  learner->Configure();

  std::vector<std::thread> threads;
  for (uint32_t thread_id = 0;
       thread_id < 2 * std::thread::hardware_concurrency(); ++thread_id) {
    threads.emplace_back([learner, p_data] {
      size_t constexpr kIters = 10;
      auto &entry = learner->GetThreadLocal().prediction_entry;
      HostDeviceVector<float> predictions;
      for (size_t iter = 0; iter < kIters; ++iter) {
        learner->Predict(p_data, false, &entry.predictions, 0, 0);

        learner->Predict(p_data, false, &predictions, 0, 0, false, true);  // leaf
        learner->Predict(p_data, false, &predictions, 0, 0, false, false, true);  // contribs
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
  p_dmat->Info().labels.Reshape(kRows);

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
  p_dmat->Info().labels.Data()->HostVector() = labels;
  p_dmat->Info().labels.Reshape(kRows);
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"booster", "gblinear"},
                        Arg{"updater", "gpu_coord_descent"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, 0);
  }
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "gpu_hist"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, 0);
  }
  {
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "gpu_hist"},
                        Arg{"gpu_id", "-1"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, 0);
  }
  {
    // with CPU algorithm
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, -1);
  }
  {
    // with CPU algorithm, but `gpu_id` takes priority
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"gpu_id", "0"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, 0);
  }
  {
    // With CPU algorithm but GPU Predictor, this is to simulate when
    // XGBoost is only used for prediction, so tree method is not
    // specified.
    std::unique_ptr<Learner> learner {Learner::Create(mat)};
    learner->SetParams({Arg{"tree_method", "hist"},
                        Arg{"predictor", "gpu_predictor"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->gpu_id, 0);
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
    rng.seed(Context::kDefaultSeed);
    std::uniform_real_distribution<float> dist;
    float v_2 = dist(rng);
    CHECK_EQ(v_0, v_2);
  }
}

TEST(Learner, FeatureInfo) {
  size_t constexpr kCols = 10;
  auto m = RandomDataGenerator{10, kCols, 0}.GenerateDMatrix(true);
  std::vector<std::string> names(kCols);
  for (size_t i = 0; i < kCols; ++i) {
    names[i] = ("f" + std::to_string(i));
  }

  std::vector<std::string> types(kCols);
  for (size_t i = 0; i < kCols; ++i) {
    types[i] = "q";
  }
  types[8] = "f";
  types[0] = "int";
  types[3] = "i";
  types[7] = "i";

  std::vector<char const*> c_names(kCols);
  for (size_t i = 0; i < names.size(); ++i) {
    c_names[i] = names[i].c_str();
  }
  std::vector<char const*> c_types(kCols);
  for (size_t i = 0; i < types.size(); ++i) {
    c_types[i] = names[i].c_str();
  }

  std::vector<std::string> out_names;
  std::vector<std::string> out_types;

  Json model{Object()};
  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    learner->Configure();
    learner->SetFeatureNames(names);
    learner->GetFeatureNames(&out_names);

    learner->SetFeatureTypes(types);
    learner->GetFeatureTypes(&out_types);

    ASSERT_TRUE(std::equal(out_names.begin(), out_names.end(), names.begin()));
    ASSERT_TRUE(std::equal(out_types.begin(), out_types.end(), types.begin()));

    learner->SaveModel(&model);
  }

  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    learner->LoadModel(model);

    learner->GetFeatureNames(&out_names);
    learner->GetFeatureTypes(&out_types);
    ASSERT_TRUE(std::equal(out_names.begin(), out_names.end(), names.begin()));
    ASSERT_TRUE(std::equal(out_types.begin(), out_types.end(), types.begin()));
  }
}

TEST(Learner, MultiTarget) {
  size_t constexpr kRows{128}, kCols{10}, kTargets{3};
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  m->Info().labels.Reshape(kRows, kTargets);
  linalg::ElementWiseTransformHost(m->Info().labels.HostView(), omp_get_max_threads(),
                                   [](auto i, auto) { return i; });

  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    learner->Configure();

    Json model{Object()};
    learner->SaveModel(&model);
    ASSERT_EQ(get<String>(model["learner"]["learner_model_param"]["num_target"]),
              std::to_string(kTargets));
  }
  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    learner->SetParam("objective", "multi:softprob");
    // unsupported objective.
    EXPECT_THROW({ learner->Configure(); }, dmlc::Error);
  }
}

/**
 * Test the model initialization sequence is correctly performed.
 */
class InitBaseScore : public ::testing::Test {
 protected:
  std::size_t static constexpr Cols() { return 10; }
  std::shared_ptr<DMatrix> Xy_;

  void SetUp() override { Xy_ = RandomDataGenerator{10, Cols(), 0}.GenerateDMatrix(true); }

 public:
  void TestUpdateConfig() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("objective", "reg:absoluteerror");
    learner->UpdateOneIter(0, Xy_);
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_NE(base_score, ObjFunction::DefaultBaseScore());

    // already initialized
    auto Xy1 = RandomDataGenerator{100, Cols(), 0}.Seed(321).GenerateDMatrix(true);
    learner->UpdateOneIter(1, Xy1);
    learner->SaveConfig(&config);
    auto base_score1 = GetBaseScore(config);
    ASSERT_EQ(base_score, base_score1);

    Json model{Object{}};
    learner->SaveModel(&model);
    learner.reset(Learner::Create({}));
    learner->LoadModel(model);
    learner->Configure();
    learner->UpdateOneIter(2, Xy1);
    learner->SaveConfig(&config);
    auto base_score2 = GetBaseScore(config);
    ASSERT_EQ(base_score, base_score2);
  }

  void TestBoostFromAvgParam() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("objective", "reg:absoluteerror");
    learner->SetParam("base_score", "1.3");
    Json config(Object{});
    learner->Configure();
    learner->SaveConfig(&config);

    auto base_score = GetBaseScore(config);
    // no change
    ASSERT_FLOAT_EQ(base_score, 1.3);

    HostDeviceVector<float> predt;
    learner->Predict(Xy_, false, &predt, 0, 0);
    auto h_predt = predt.ConstHostSpan();
    for (auto v : h_predt) {
      ASSERT_FLOAT_EQ(v, 1.3);
    }
    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    // no change
    ASSERT_FLOAT_EQ(base_score, 1.3);

    auto from_avg = std::stoi(
        get<String const>(config["learner"]["learner_model_param"]["boost_from_average"]));
    // from_avg is disabled when base score is set
    ASSERT_EQ(from_avg, 0);
    // in the future when we can deprecate the binary model, user can set the parameter directly.
    learner->SetParam("boost_from_average", "1");
    learner->Configure();
    learner->SaveConfig(&config);
    from_avg = std::stoi(
        get<String const>(config["learner"]["learner_model_param"]["boost_from_average"]));
    ASSERT_EQ(from_avg, 1);
  }

  void TestInitAfterLoad() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("objective", "reg:absoluteerror");
    learner->Configure();

    Json model{Object{}};
    learner->SaveModel(&model);
    auto base_score = GetBaseScore(model);
    ASSERT_EQ(base_score, ObjFunction::DefaultBaseScore());

    learner.reset(Learner::Create({Xy_}));
    learner->LoadModel(model);
    Json config(Object{});
    learner->Configure();
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, ObjFunction::DefaultBaseScore());

    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    ASSERT_NE(base_score, ObjFunction::DefaultBaseScore());
  }

  void TestInitWithPredt() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("objective", "reg:absoluteerror");
    HostDeviceVector<float> predt;
    learner->Predict(Xy_, false, &predt, 0, 0);

    auto h_predt = predt.ConstHostSpan();
    for (auto v : h_predt) {
      ASSERT_EQ(v, ObjFunction::DefaultBaseScore());
    }

    Json config(Object{});
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, ObjFunction::DefaultBaseScore());

    // since prediction is not used for trianing, the train procedure still runs estimation
    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    ASSERT_NE(base_score, ObjFunction::DefaultBaseScore());
  }

  void TestUpdateProcess() {
    // Check that when training continuation is performed with update, the base score is
    // not re-evaluated.
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->SetParam("objective", "reg:absoluteerror");
    learner->Configure();

    learner->UpdateOneIter(0, Xy_);
    Json model{Object{}};
    learner->SaveModel(&model);
    auto base_score = GetBaseScore(model);

    auto Xy1 = RandomDataGenerator{100, Cols(), 0}.Seed(321).GenerateDMatrix(true);
    learner.reset(Learner::Create({Xy1}));
    learner->LoadModel(model);
    learner->SetParam("process_type", "update");
    learner->SetParam("updater", "refresh");
    learner->UpdateOneIter(1, Xy1);

    Json config(Object{});
    learner->SaveConfig(&config);
    auto base_score1 = GetBaseScore(config);
    ASSERT_EQ(base_score, base_score1);
  }
};

TEST_F(InitBaseScore, TestUpdateConfig) { this->TestUpdateConfig(); }

TEST_F(InitBaseScore, FromAvgParam) { this->TestBoostFromAvgParam(); }

TEST_F(InitBaseScore, InitAfterLoad) { this->TestInitAfterLoad(); }

TEST_F(InitBaseScore, InitWithPredict) { this->TestInitWithPredt(); }

TEST_F(InitBaseScore, UpdateProcess) { this->TestUpdateProcess(); }

class TestColumnSplit : public ::testing::TestWithParam<std::string> {
  static auto MakeFmat(std::string const& obj) {
    auto constexpr kRows = 10, kCols = 10;
    auto p_fmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
    auto& h_upper = p_fmat->Info().labels_upper_bound_.HostVector();
    auto& h_lower = p_fmat->Info().labels_lower_bound_.HostVector();
    h_lower.resize(kRows);
    h_upper.resize(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      h_lower[i] = 1;
      h_upper[i] = 10;
    }
    if (obj.find("rank:") != std::string::npos) {
      auto h_label = p_fmat->Info().labels.HostView();
      std::size_t k = 0;
      for (auto& v : h_label) {
        v = k % 2 == 0;
        ++k;
      }
    }
    return p_fmat;
  };

  void TestBaseScore(std::string objective, float expected_base_score, Json expected_model) {
    auto const world_size = collective::GetWorldSize();
    auto const rank = collective::GetRank();

    auto p_fmat = MakeFmat(objective);
    std::shared_ptr<DMatrix> sliced{p_fmat->SliceCol(world_size, rank)};
    std::unique_ptr<Learner> learner{Learner::Create({sliced})};
    learner->SetParam("tree_method", "approx");
    learner->SetParam("objective", objective);
    if (objective.find("quantile") != std::string::npos) {
      learner->SetParam("quantile_alpha", "0.5");
    }
    if (objective.find("multi") != std::string::npos) {
      learner->SetParam("num_class", "3");
    }
    learner->UpdateOneIter(0, sliced);
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, expected_base_score);

    Json model{Object{}};
    learner->SaveModel(&model);
    ASSERT_EQ(model, expected_model);
  }

 public:
  void Run(std::string objective) {
    auto p_fmat = MakeFmat(objective);
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->SetParam("tree_method", "approx");
    learner->SetParam("objective", objective);
    if (objective.find("quantile") != std::string::npos) {
      learner->SetParam("quantile_alpha", "0.5");
    }
    if (objective.find("multi") != std::string::npos) {
      learner->SetParam("num_class", "3");
    }
    learner->UpdateOneIter(0, p_fmat);

    Json config{Object{}};
    learner->SaveConfig(&config);

    Json model{Object{}};
    learner->SaveModel(&model);

    auto constexpr kWorldSize{3};
    auto call = [this, &objective](auto&... args) { TestBaseScore(objective, args...); };
    auto score = GetBaseScore(config);
    RunWithInMemoryCommunicator(kWorldSize, call, score, model);
  }
};

TEST_P(TestColumnSplit, Objective) {
  std::string objective = GetParam();
  this->Run(objective);
}

INSTANTIATE_TEST_SUITE_P(ColumnSplitObjective, TestColumnSplit,
                         ::testing::ValuesIn(MakeObjNamesForTest()),
                         [](const ::testing::TestParamInfo<TestColumnSplit::ParamType>& info) {
                           return ObjTestNameGenerator(info);
                         });
}  // namespace xgboost
