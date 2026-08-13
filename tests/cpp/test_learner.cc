/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <xgboost/learner.h>         // for Learner
#include <xgboost/logging.h>         // for LogCheck_NE, CHECK_NE, LogCheck_EQ
#include <xgboost/objective.h>       // for ObjFunction
#include <xgboost/version_config.h>  // for XGBOOST_VER_MAJOR, XGBOOST_VER_MINOR

#include <algorithm>    // for equal, transform
#include <cstddef>      // for size_t
#include <iosfwd>       // for ofstream
#include <limits>       // for numeric_limits
#include <map>          // for map
#include <memory>       // for unique_ptr, shared_ptr, __shared_ptr_...
#include <random>       // for uniform_real_distribution
#include <string>       // for allocator, basic_string, string, oper...
#include <thread>       // for thread
#include <type_traits>  // for is_integral
#include <utility>      // for pair
#include <vector>       // for vector

#include "../../src/collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../../src/common/api_entry.h"             // for XGBAPIThreadLocalEntry
#include "../../src/common/io.h"                    // for LoadSequentialFile
#include "../../src/common/linalg_op.h"             // for ElementWiseTransformHost, begin, end
#include "./collective/test_worker.h"               // for TestDistributedGlobal
#include "dmlc/omp.h"                               // for omp_get_max_threads
#include "filesystem.h"                             // for TemporaryDirectory
#include "helpers.h"                                // for GetBaseScore, RandomDataGenerator
#include "objective_helpers.h"                      // for MakeObjNamesForTest, ObjTestNameGenerator
#include "test_serialization.h"                     // for CompareJsonModels
#include "xgboost/base.h"                           // for bst_float, Args, bst_feature_t, bst_int
#include "xgboost/context.h"                        // for Context, DeviceOrd
#include "xgboost/data.h"                           // for DMatrix, MetaInfo, DataType
#include "xgboost/host_device_vector.h"             // for HostDeviceVector
#include "xgboost/json.h"                           // for Json, Object, get, String, IsA, opera...
#include "xgboost/linalg.h"                         // for Tensor, TensorView
#include "xgboost/logging.h"                        // for ConsoleLogger
#include "xgboost/string_view.h"                    // for StringView

namespace xgboost {
TEST(LearnerModelState, Initialization) {
  LearnerModelState state;
  EXPECT_TRUE(state.NeedsInitialization());
  state.num_feature = 1;
  state.num_output_group = 1;
  EXPECT_TRUE(state.NeedsInitialization());

  auto initialized = MakeMP(1, 0.5f, 1);
  EXPECT_FALSE(initialized.NeedsInitialization());
  EXPECT_TRUE(initialized.Initialized());
}

TEST(Learner, Basic) {
  using Arg = std::pair<std::string, std::string>;
  auto args = {Arg("tree_method", "exact")};
  auto mat_ptr = RandomDataGenerator{10, 10, 0.0f}.GenerateDMatrix();
  auto learner = std::unique_ptr<Learner>(Learner::Create({mat_ptr}));
  learner->Configure(args);

  auto major = XGBOOST_VER_MAJOR;
  auto minor = XGBOOST_VER_MINOR;
  auto patch = XGBOOST_VER_PATCH;

  static_assert(std::is_integral_v<decltype(major)>, "Wrong major version type");
  static_assert(std::is_integral_v<decltype(minor)>, "Wrong minor version type");
  static_assert(std::is_integral_v<decltype(patch)>, "Wrong patch version type");
}

TEST(Learner, ConfigureArguments) {
  auto p_mat = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix();
  auto learner = std::unique_ptr<Learner>{Learner::Create({p_mat})};

  learner->Configure(
      {{"objective", "reg:absoluteerror"}, {"eval_metric", "mae"}, {"eval_metric", "rmse"}});

  Json config{Object{}};
  learner->SaveConfig(&config);
  EXPECT_EQ(get<String const>(config["learner"]["objective"]["name"]), "reg:absoluteerror");
  EXPECT_EQ(get<Array const>(config["learner"]["metrics"]).size(), 2);
}

TEST(Learner, ParameterValidation) {
  ConsoleLogger::Configure({{"verbosity", "2"}});
  size_t constexpr kRows = 1;
  size_t constexpr kCols = 1;
  auto p_mat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));

  testing::internal::CaptureStderr();
  learner->Configure(Args{{"validate_parameters", "1"},
                          {"Knock-Knock", "Who's-there?"},
                          {"Silence", "...."},
                          {"tree_method", "exact"}});
  std::string output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(output.find(R"(Parameters: { "Knock-Knock", "Silence" })") != std::string::npos);

  // whitespace
  ASSERT_THAT([&] { learner->Configure({{"tree method", "exact"}}); },
              GMockThrow(R"("tree method" contains whitespace)"));
}

TEST(Learner, ParameterValidationUsesConsumedParameters) {
  auto p_mat = RandomDataGenerator{1, 1, 0}.GenerateDMatrix(true);
  auto configure = [&p_mat](Args params) {
    auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
    params.emplace_back("validate_parameters", "1");
    params.emplace_back("verbosity", "1");
    testing::internal::CaptureStderr();
    learner->Configure(params);
    return testing::internal::GetCapturedStderr();
  };

  // Report the spelling supplied by the user, including aliases.
  auto output = configure({{"eta", "0.3"},
                           {"lambda", "1.0"},
                           {"alpha", "0.0"},
                           {"gamma", "0.0"},
                           {"random_state", "0"},
                           {"n_jobs", "1"}});
  EXPECT_EQ(output.find("Parameters:"), std::string::npos);

  // Collect parameters from the active model and updater recursively.
  output = configure({{"tree_method", "hist"}, {"num_parallel_tree", "2"}, {"max_bin", "64"}});
  EXPECT_EQ(output.find("Parameters:"), std::string::npos);

  // A parameter for an inactive component is not consumed.
  output = configure({{"booster", "gblinear"}, {"max_depth", "3"}});
  EXPECT_NE(output.find(R"(Parameters: { "max_depth" })"), std::string::npos);

  // More than one active component can consume the same parameter.
  output = configure(
      {{"objective", "reg:quantileerror"}, {"eval_metric", "quantile"}, {"quantile_alpha", "0.5"}});
  EXPECT_EQ(output.find("Parameters:"), std::string::npos);
}

TEST(Learner, DeprecatedGblinearBooster) {
  auto p_mat = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix();

  std::unique_ptr<Learner> learner{Learner::Create({p_mat})};

  testing::internal::CaptureStderr();
  learner->Configure({{"booster", "gblinear"}, {"verbosity", "2"}});
  auto output = testing::internal::GetCapturedStderr();

  ASSERT_NE(output.find("`booster=gblinear` is deprecated"), std::string::npos);
}

TEST(Learner, CheckGroup) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  bst_feature_t constexpr kNumCols = 15;

  std::shared_ptr<DMatrix> p_mat{RandomDataGenerator{kNumRows, kNumCols, 0.0f}.GenerateDMatrix()};
  std::vector<bst_float> weight(kNumGroups, 1);
  std::vector<bst_group_t> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;
  std::vector<bst_float> labels(kNumRows);
  for (size_t i = 0; i < kNumRows; ++i) {
    labels[i] = i % 2;
  }

  p_mat->SetInfo("weight", Make1dInterfaceTest(weight.data(), kNumGroups));
  p_mat->SetInfo("group", Make1dInterfaceTest(group.data(), kNumGroups));
  p_mat->SetInfo("label", Make1dInterfaceTest(labels.data(), kNumRows));

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_mat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure({Arg{"objective", "rank:pairwise"}});
  EXPECT_NO_THROW(learner->UpdateOneIter(0, p_mat));

  group.resize(kNumGroups + 1);
  group[3] = 4;
  group[4] = 1;
  p_mat->SetInfo("group", Make1dInterfaceTest(group.data(), kNumGroups + 1));
  EXPECT_ANY_THROW(learner->UpdateOneIter(0, p_mat));
}

TEST(Learner, CheckMultiBatch) {
  auto p_fmat =
      RandomDataGenerator{512, 128, 0.8}.Batches(4).GenerateSparsePageDMatrix("temp", true);
  ASSERT_FALSE(p_fmat->SingleColBlock());

  std::vector<std::shared_ptr<DMatrix>> mat{p_fmat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(Args{{"objective", "binary:logistic"}});
  learner->UpdateOneIter(0, p_fmat);
}

TEST(Learner, Configuration) {
  std::string const emetric = "eval_metric";
  {
    std::unique_ptr<Learner> learner{Learner::Create({nullptr})};
    learner->Configure({{emetric, "auc"}});
    learner->Configure({{emetric, "rmsle"}});
    learner->Configure({{"foo", "bar"}});

    Json config{Object{}};
    learner->SaveConfig(&config);
    ASSERT_EQ(get<Array const>(config["learner"]["metrics"]).size(), 2);
  }

  {
    auto p_mat = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix();
    std::unique_ptr<Learner> learner{Learner::Create({p_mat})};
    learner->Configure({{emetric, "auc"}, {emetric, "rmse"}, {emetric, "mae"}});

    Json config{Object{}};
    learner->SaveConfig(&config);
    ASSERT_EQ(get<Array const>(config["learner"]["metrics"]).size(), 3);
  }
}

TEST(Learner, PoissonMaxDeltaStepIsGeneric) {
  auto p_mat = RandomDataGenerator{1, 1, 0.0f}.GenerateDMatrix();
  std::unique_ptr<Learner> learner{Learner::Create({p_mat})};
  learner->Configure({{"objective", "count:poisson"}});

  auto max_delta_step = [&] {
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto const& value = config["learner"]["gradient_booster"]["tree_train_param"]["max_delta_step"];
    return std::stof(get<String const>(value));
  };
  ASSERT_FLOAT_EQ(max_delta_step(), 0.0f);

  learner->Configure({{"max_delta_step", "0.5"}});
  ASSERT_FLOAT_EQ(max_delta_step(), 0.5f);
}

TEST(Learner, ModelInitializedByTrainingData) {
  auto train = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix(true);
  auto eval = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix(true);
  auto learner = std::unique_ptr<Learner>{Learner::Create({train, eval})};
  learner->Configure({{"objective", "reg:absoluteerror"}});

  EXPECT_EQ(learner->GetNumFeature(), 0);
  Json config{Object{}};
  EXPECT_NO_THROW(learner->SaveConfig(&config));

  std::string snapshot;
  common::MemoryBufferStream out{&snapshot};
  EXPECT_NO_THROW(learner->Save(&out));
  auto restored = std::unique_ptr<Learner>{Learner::Create({train})};
  common::MemoryBufferStream in{&snapshot};
  EXPECT_NO_THROW(restored->Load(&in));
  EXPECT_EQ(restored->GetNumFeature(), 0);
  Json restored_config{Object{}};
  restored->SaveConfig(&restored_config);
  EXPECT_EQ(config, restored_config);
  restored->UpdateOneIter(0, train);
  EXPECT_EQ(restored->GetNumFeature(), train->Info().num_col_);

  HostDeviceVector<float> predt;
  EXPECT_THROW(learner->Predict(eval, false, &predt, 0, 0), dmlc::Error);
  EXPECT_EQ(learner->GetNumFeature(), 0);

  learner->UpdateOneIter(0, train);
  EXPECT_EQ(learner->GetNumFeature(), train->Info().num_col_);

  learner.reset(Learner::Create({train}));
  learner->Configure();
  EXPECT_NO_THROW(learner->Predict(train, false, &predt, 0, 0, true));
  EXPECT_EQ(learner->GetNumFeature(), train->Info().num_col_);
}

TEST(Learner, ResetInitializesCachedModel) {
  auto train = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix(true);
  auto learner = std::unique_ptr<Learner>{Learner::Create({train})};
  learner->Configure({{"objective", "reg:absoluteerror"}});
  EXPECT_EQ(learner->GetNumFeature(), 0);

  EXPECT_NO_THROW(learner->Reset());
  EXPECT_EQ(learner->GetNumFeature(), train->Info().num_col_);
  Json model{Object{}};
  EXPECT_NO_THROW(learner->SaveModel(&model));
}

TEST(Learner, LoadPendingModelInputsFromOldSnapshot) {
  auto train = RandomDataGenerator{8, 4, 0.0f}.GenerateDMatrix(true);
  auto learner = std::unique_ptr<Learner>{Learner::Create({train})};
  learner->Configure({{"objective", "reg:absoluteerror"}, {"base_score", "1.3"}});

  std::string snapshot;
  common::MemoryBufferStream out{&snapshot};
  learner->Save(&out);

  auto memory_snapshot = Json::Load(StringView{snapshot}, std::ios::binary);
  auto& train_param = get<Object>(memory_snapshot["Config"]["learner"]["learner_train_param"]);
  for (auto key : {"base_score", "num_class", "num_target", "boost_from_average"}) {
    train_param.erase(key);
  }
  std::vector<char> serialized;
  Json::Dump(memory_snapshot, &serialized, std::ios::binary);
  std::string old_snapshot{serialized.cbegin(), serialized.cend()};

  auto restored = std::unique_ptr<Learner>{Learner::Create({train})};
  common::MemoryBufferStream in{&old_snapshot};
  restored->Load(&in);
  EXPECT_EQ(restored->GetNumFeature(), 0);
  restored->UpdateOneIter(0, train);

  Json config{Object{}};
  restored->SaveConfig(&config);
  auto base_score = GetBaseScore(config);
  ASSERT_EQ(base_score.size(), 1);
  EXPECT_FLOAT_EQ(base_score.front(), 1.3);
  EXPECT_EQ(get<String const>(config["learner"]["learner_model_param"]["boost_from_average"]), "0");
}

TEST(Learner, JsonModelIO) {
  // Test of comparing JSON object directly.
  size_t constexpr kRows = 8;
  int32_t constexpr kIters = 4;

  std::shared_ptr<DMatrix> p_dmat{RandomDataGenerator{kRows, 10, 0}.GenerateDMatrix()};
  p_dmat->Info().labels.Reshape(kRows);
  CHECK_NE(p_dmat->Info().num_col_, 0);

  {
    std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
    learner->Configure();
    Json uninitialized{Object()};
    EXPECT_THROW(learner->SaveModel(&uninitialized), dmlc::Error);
    learner->UpdateOneIter(0, p_dmat);
    Json out{Object()};
    learner->SaveModel(&out);

    common::TemporaryDirectory tmpdir;

    std::ofstream fout(tmpdir.Path() / "model.json");
    fout << out;
    fout.close();

    auto loaded_str = common::LoadSequentialFile(tmpdir.Str() + "/model.json");
    Json loaded = Json::Load(StringView{loaded_str.data(), loaded_str.size()});

    learner->LoadModel(loaded);
    Json new_in{Object()};
    learner->SaveModel(&new_in);
    ASSERT_EQ(new_in, out);
  }

  {
    std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
    for (int32_t iter = 0; iter < kIters; ++iter) {
      learner->UpdateOneIter(iter, p_dmat);
    }
    learner->SetAttr("best_score", "15.2");

    Json out{Object()};
    learner->SaveModel(&out);

    learner->LoadModel(out);
    Json new_in{Object{}};
    learner->SaveModel(&new_in);

    ASSERT_TRUE(IsA<Object>(out["learner"]["attributes"]));
    ASSERT_EQ(get<Object>(out["learner"]["attributes"]).size(), 1ul);
    ASSERT_EQ(out, new_in);
  }
}

TEST(Learner, ConfigIO) {
  bst_idx_t n_samples = 128;
  bst_feature_t n_features = 12;
  std::shared_ptr<DMatrix> p_fmat{
      RandomDataGenerator{n_samples, n_features, 0}.Classes(2).GenerateDMatrix(true)};

  Json config{Object{}};
  auto serialised_model_tmp = std::string{};
  std::string eval_res_0;
  std::string eval_res_1;
  {
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->Configure(Args{{"eval_metric", "ndcg"}, {"eval_metric", "map"}});
    learner->Configure();
    learner->UpdateOneIter(0, p_fmat);
    eval_res_0 = learner->EvalOneIter(0, {p_fmat}, {"Train"});
    learner->SaveConfig(&config);
    common::MemoryBufferStream fo(&serialised_model_tmp);
    learner->Save(&fo);
  }

  {
    common::MemoryBufferStream fi(&serialised_model_tmp);
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->Load(&fi);
    eval_res_1 = learner->EvalOneIter(0, {p_fmat}, {"Train"});
  }
  ASSERT_EQ(eval_res_0, eval_res_1);

  {
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->LoadConfig(config);

    Json loaded{Object{}};
    learner->SaveConfig(&loaded);
    ASSERT_EQ(get<Array const>(loaded["learner"]["metrics"]).size(), 2);
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
  learner->UpdateOneIter(0, p_dmat);

  std::vector<std::thread> threads;

#if defined(__linux__)
  auto n_threads = std::thread::hardware_concurrency() * 4u;
#else
  auto n_threads = std::thread::hardware_concurrency();
#endif

  for (decltype(n_threads) thread_id = 0; thread_id < n_threads; ++thread_id) {
    threads.emplace_back([learner, p_data] {
      size_t constexpr kIters = 10;
      auto& out_predictions = learner->GetThreadLocal().predictions;
      HostDeviceVector<float> predictions;
      for (size_t iter = 0; iter < kIters; ++iter) {
        learner->Predict(p_data, false, &out_predictions, 0, 0);

        learner->Predict(p_data, false, &predictions, 0, 0, false, true);         // leaf
        learner->Predict(p_data, false, &predictions, 0, 0, false, false, true);  // contribs
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

#if defined(XGBOOST_USE_CUDA)
// Tests for automatic GPU configuration.
TEST(Learner, GPUConfiguration) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kRows = 10;
  auto p_dmat = RandomDataGenerator(kRows, 10, 0).GenerateDMatrix();
  std::vector<std::shared_ptr<DMatrix>> mat{p_dmat};
  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels.Data()->HostVector() = labels;
  p_dmat->Info().labels.Reshape(kRows);
  {
    std::unique_ptr<Learner> learner{Learner::Create(mat)};
    learner->Configure(
        {Arg{"booster", "gblinear"}, Arg{"updater", "coord_descent"}, Arg{"device", "cuda"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->Device(), DeviceOrd::CUDA(0));
  }
  {
    std::unique_ptr<Learner> learner{Learner::Create(mat)};
    learner->Configure({Arg{"tree_method", "hist"}, {"device", "cuda"}});
    learner->Configure();
    ASSERT_EQ(learner->Ctx()->Device(), DeviceOrd::CUDA(0));
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->Device(), DeviceOrd::CUDA(0));
  }
  {
    std::unique_ptr<Learner> learner{Learner::Create(mat)};
    learner->Configure({Arg{"tree_method", "hist"}, Arg{"device", "cuda"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->Device(), DeviceOrd::CUDA(0));
  }
  {
    // with CPU algorithm
    std::unique_ptr<Learner> learner{Learner::Create(mat)};
    learner->Configure({Arg{"tree_method", "hist"}});
    learner->UpdateOneIter(0, p_dmat);
    ASSERT_EQ(learner->Ctx()->Device(), DeviceOrd::CPU());
  }
}
#endif  // defined(XGBOOST_USE_CUDA)

TEST(Learner, Seed) {
  auto m = RandomDataGenerator{10, 10, 0}.GenerateDMatrix();
  std::unique_ptr<Learner> learner{Learner::Create({m})};
  auto seed = std::numeric_limits<int64_t>::max();
  learner->Configure({{"seed", std::to_string(seed)}});
  learner->Configure();
  Json config{Object()};
  learner->SaveConfig(&config);
  ASSERT_EQ(std::to_string(seed), get<String>(config["learner"]["generic_param"]["seed"]));

  seed = std::numeric_limits<int64_t>::min();
  learner->Configure({{"seed", std::to_string(seed)}});
  learner->Configure();
  learner->SaveConfig(&config);
  ASSERT_EQ(std::to_string(seed), get<String>(config["learner"]["generic_param"]["seed"]));
}

TEST(Learner, ConstantSeed) {
  auto m = RandomDataGenerator{10, 10, 0}.GenerateDMatrix(true);
  std::unique_ptr<Learner> learner{Learner::Create({m})};
  // Use exact as it doesn't initialize column sampler at construction, which alters the rng.
  learner->Configure({{"tree_method", "exact"}});
  learner->Configure();

  std::uniform_real_distribution<float> dist;
  auto& rng = learner->Ctx()->Rng();
  float v_0 = dist(rng);

  learner->Configure({{"", ""}});
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
    learner->UpdateOneIter(0, m);
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
  linalg::cpu_impl::TransformIdxKernel(m->Info().labels.HostView(), omp_get_max_threads(),
                                       [](auto i, auto) { return i; });

  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    learner->Configure();
    learner->UpdateOneIter(0, m);

    Json model{Object()};
    learner->SaveModel(&model);
    ASSERT_EQ(get<String>(model["learner"]["learner_model_param"]["num_target"]),
              std::to_string(kTargets));
  }
  {
    std::unique_ptr<Learner> learner{Learner::Create({m})};
    // unsupported objective.
    EXPECT_THROW({ learner->Configure({{"objective", "multi:softprob"}}); }, dmlc::Error);
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
    learner->Configure({{"objective", "reg:absoluteerror"}});
    learner->UpdateOneIter(0, Xy_);
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score.size(), 1);
    ASSERT_NE(base_score[0], ObjFunction::DefaultBaseScore());

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

    // Unrelated parameters don't rematerialize model state from stale user inputs.
    learner->Configure({{"max_depth", "2"}});
    learner->SaveConfig(&config);
    ASSERT_EQ(base_score, GetBaseScore(config));

    // Explicit model input updates are applied to initialized state.
    learner->Configure({{"base_score", "1.3"}});
    learner->SaveConfig(&config);
    auto updated_base_score = GetBaseScore(config);
    ASSERT_EQ(updated_base_score.size(), 1);
    ASSERT_FLOAT_EQ(updated_base_score[0], 1.3);
  }

  void TestBoostFromAvgParam() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->Configure({{"objective", "reg:absoluteerror"}});
    learner->Configure({{"base_score", "1.3"}});
    Json config(Object{});
    learner->Configure();
    learner->SaveConfig(&config);

    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score.size(), 1);
    // no change
    ASSERT_FLOAT_EQ(base_score[0], 1.3);

    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    ASSERT_EQ(base_score.size(), 1);
    // no change
    ASSERT_FLOAT_EQ(base_score[0], 1.3);

    auto from_avg = std::stoi(
        get<String const>(config["learner"]["learner_model_param"]["boost_from_average"]));
    // from_avg is disabled when base score is set
    ASSERT_EQ(from_avg, 0);
    // in the future when we can deprecate the binary model, user can set the parameter directly.
    learner->Configure({{"boost_from_average", "1"}});
    learner->Configure();
    learner->SaveConfig(&config);
    from_avg = std::stoi(
        get<String const>(config["learner"]["learner_model_param"]["boost_from_average"]));
    ASSERT_EQ(from_avg, 1);
  }

  void TestInitAfterLoad() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->Configure({{"objective", "reg:absoluteerror"}});
    learner->Configure();

    Json model{Object{}};
    EXPECT_THROW(learner->SaveModel(&model), dmlc::Error);

    learner->UpdateOneIter(0, Xy_);
    learner->SaveModel(&model);
    auto base_score = GetBaseScore(model);
    ASSERT_EQ(base_score.size(), 1);
    ASSERT_FALSE(std::isnan(base_score[0]));
    ASSERT_NE(base_score[0], ObjFunction::DefaultBaseScore());

    learner.reset(Learner::Create({Xy_}));
    learner->LoadModel(model);
    Json config(Object{});
    learner->SaveConfig(&config);
    auto loaded_base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, loaded_base_score);

    learner->UpdateOneIter(1, Xy_);
    learner->SaveConfig(&config);
    loaded_base_score = GetBaseScore(config);
    ASSERT_EQ(base_score, loaded_base_score);
  }

  void TestInitWithPredt() {
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->Configure({{"objective", "reg:absoluteerror"}});
    HostDeviceVector<float> predt;
    EXPECT_THROW(learner->Predict(Xy_, false, &predt, 0, 0), dmlc::Error);
    EXPECT_EQ(learner->GetNumFeature(), 0);

    Json config(Object{});
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_TRUE(base_score.empty());

    // Prediction does not initialize the model; training still estimates the intercept.
    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    base_score = GetBaseScore(config);
    ASSERT_EQ(base_score.size(), 1);
    ASSERT_FALSE(std::isnan(base_score[0]));
    ASSERT_NE(base_score[0], ObjFunction::DefaultBaseScore());

    learner.reset(Learner::Create({Xy_}));
    learner->Configure({{"objective", "reg:absoluteerror"}});
    learner->Predict(Xy_, false, &predt, 0, 0, true);
    learner->SaveConfig(&config);
    auto training_base_score = GetBaseScore(config);
    ASSERT_EQ(training_base_score.size(), 1);
    ASSERT_EQ(training_base_score[0], ObjFunction::DefaultBaseScore());

    // The first training operation commits the intercept choice.
    learner->UpdateOneIter(0, Xy_);
    learner->SaveConfig(&config);
    ASSERT_EQ(training_base_score, GetBaseScore(config));
  }

  void TestUpdateProcess() {
    // Check that when training continuation is performed with update, the base score is
    // not re-evaluated.
    std::unique_ptr<Learner> learner{Learner::Create({Xy_})};
    learner->Configure({{"objective", "reg:absoluteerror"}});
    learner->Configure();

    learner->UpdateOneIter(0, Xy_);
    Json model{Object{}};
    learner->SaveModel(&model);
    auto base_score = GetBaseScore(model);
    ASSERT_EQ(base_score.size(), 1);
    ASSERT_FALSE(std::isnan(base_score[0]));

    auto Xy1 = RandomDataGenerator{100, Cols(), 0}.Seed(321).GenerateDMatrix(true);
    learner.reset(Learner::Create({Xy1}));
    learner->LoadModel(model);
    learner->Configure({{"process_type", "update"}});
    learner->Configure({{"updater", "refresh"}});
    learner->UpdateOneIter(1, Xy1);

    Json config(Object{});
    learner->SaveConfig(&config);
    auto base_score1 = GetBaseScore(config);
    ASSERT_EQ(base_score1.size(), 1);
    ASSERT_FALSE(std::isnan(base_score1[0]));
    ASSERT_EQ(base_score, base_score1);
  }
};

TEST_F(InitBaseScore, TestUpdateConfig) { this->TestUpdateConfig(); }

TEST_F(InitBaseScore, FromAvgParam) { this->TestBoostFromAvgParam(); }

TEST_F(InitBaseScore, InitAfterLoad) { this->TestInitAfterLoad(); }

TEST_F(InitBaseScore, InitWithPredict) { this->TestInitWithPredt(); }

TEST_F(InitBaseScore, UpdateProcess) { this->TestUpdateProcess(); }

}  // namespace xgboost
