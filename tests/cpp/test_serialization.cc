/**
 * Copyright (c) 2019-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/feature_map.h>  // for FeatureMap
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <string>

#include "../../src/common/io.h"
#include "../../src/common/random.h"
#include "filesystem.h"  // dmlc::TemporaryDirectory
#include "helpers.h"

namespace xgboost {
template <typename Array>
void CompareIntArray(Json l, Json r) {
  auto const& l_arr = get<Array const>(l);
  auto const& r_arr = get<Array const>(r);
  ASSERT_EQ(l_arr.size(), r_arr.size());
  for (size_t i = 0; i < l_arr.size(); ++i) {
    ASSERT_EQ(l_arr[i], r_arr[i]);
  }
}

void CompareJSON(Json l, Json r) {
  switch (l.GetValue().Type()) {
  case Value::ValueKind::kString: {
    ASSERT_EQ(l, r);
    break;
  }
  case Value::ValueKind::kNumber: {
    ASSERT_NEAR(get<Number>(l), get<Number>(r), kRtEps);
    break;
  }
  case Value::ValueKind::kInteger: {
    ASSERT_EQ(l, r);
    break;
  }
  case Value::ValueKind::kObject: {
    auto const &l_obj = get<Object const>(l);
    auto const &r_obj = get<Object const>(r);
    ASSERT_EQ(l_obj.size(), r_obj.size());

    for (auto const& kv : l_obj) {
      ASSERT_NE(r_obj.find(kv.first), r_obj.cend());
      CompareJSON(l_obj.at(kv.first), r_obj.at(kv.first));
    }
    break;
  }
  case Value::ValueKind::kArray: {
    auto const& l_arr = get<Array const>(l);
    auto const& r_arr = get<Array const>(r);
    ASSERT_EQ(l_arr.size(), r_arr.size());
    for (size_t i = 0; i < l_arr.size(); ++i) {
      CompareJSON(l_arr[i], r_arr[i]);
    }
    break;
  }
  case Value::ValueKind::kNumberArray: {
    auto const& l_arr = get<F32Array const>(l);
    auto const& r_arr = get<F32Array const>(r);
    ASSERT_EQ(l_arr.size(), r_arr.size());
    for (size_t i = 0; i < l_arr.size(); ++i) {
      ASSERT_NEAR(l_arr[i], r_arr[i], kRtEps);
    }
    break;
  }
  case Value::ValueKind::kU8Array: {
    CompareIntArray<U8Array>(l, r);
    break;
  }
  case Value::ValueKind::kI32Array: {
    CompareIntArray<I32Array>(l, r);
    break;
  }
  case Value::ValueKind::kI64Array: {
    CompareIntArray<I64Array>(l, r);
    break;
  }
  case Value::ValueKind::kBoolean: {
    ASSERT_EQ(l, r);
    break;
  }
  case Value::ValueKind::kNull: {
    ASSERT_EQ(l, r);
    break;
  }
  }
}

void TestLearnerSerialization(Args args, FeatureMap const& fmap, std::shared_ptr<DMatrix> p_dmat) {
  for (auto& batch : p_dmat->GetBatches<SparsePage>()) {
    batch.data.HostVector();
    batch.offset.HostVector();
  }

  int32_t constexpr kIters = 2;

  dmlc::TemporaryDirectory tempdir;
  std::string const fname = tempdir.path + "/model";

  std::vector<std::string> dumped_0;
  std::string model_at_kiter;

  // Train for kIters.
  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    std::unique_ptr<Learner> learner {Learner::Create({p_dmat})};
    learner->SetParams(args);
    for (int32_t iter = 0; iter < kIters; ++iter) {
      learner->UpdateOneIter(iter, p_dmat);
    }
    dumped_0 = learner->DumpModel(fmap, true, "json");
    learner->Save(fo.get());

    common::MemoryBufferStream mem_out(&model_at_kiter);
    learner->Save(&mem_out);
  }

  // Assert dumped model is same after loading
  std::vector<std::string> dumped_1;
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    std::unique_ptr<Learner> learner {Learner::Create({p_dmat})};
    learner->Load(fi.get());
    learner->Configure();
    dumped_1 = learner->DumpModel(fmap, true, "json");
  }
  ASSERT_EQ(dumped_0, dumped_1);

  std::string model_at_2kiter;

  // Test training continuation with data from host
  {
    std::string continued_model;
    {
      // Continue the previous training with another kIters
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(fname.c_str(), "r"));
      std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
      learner->Load(fi.get());
      learner->Configure();

      // verify the loaded model doesn't change.
      std::string serialised_model_tmp;
      common::MemoryBufferStream mem_out(&serialised_model_tmp);
      learner->Save(&mem_out);
      ASSERT_EQ(model_at_kiter, serialised_model_tmp);

      for (auto &batch : p_dmat->GetBatches<SparsePage>()) {
        batch.data.HostVector();
        batch.offset.HostVector();
      }

      for (int32_t iter = kIters; iter < 2 * kIters; ++iter) {
        learner->UpdateOneIter(iter, p_dmat);
      }
      common::MemoryBufferStream fo(&continued_model);
      learner->Save(&fo);
    }

    {
      // Train 2 * kIters in one go
      std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
      learner->SetParams(args);
      for (int32_t iter = 0; iter < 2 * kIters; ++iter) {
        learner->UpdateOneIter(iter, p_dmat);

        // Verify model is same at the same iteration during two training
        // sessions.
        if (iter == kIters - 1) {
          std::string reproduced_model;
          common::MemoryBufferStream fo(&reproduced_model);
          learner->Save(&fo);
          ASSERT_EQ(model_at_kiter, reproduced_model);
        }
      }
      common::MemoryBufferStream fo(&model_at_2kiter);
      learner->Save(&fo);
    }

    Json m_0 = Json::Load(StringView{continued_model}, std::ios::binary);
    Json m_1 = Json::Load(StringView{model_at_2kiter}, std::ios::binary);

    CompareJSON(m_0, m_1);
  }

  // Test training continuation with data from device.
  {
    // Continue the previous training but on data from device.
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
    learner->Load(fi.get());
    learner->Configure();

    // verify the loaded model doesn't change.
    std::string serialised_model_tmp;
    common::MemoryBufferStream mem_out(&serialised_model_tmp);
    learner->Save(&mem_out);
    ASSERT_EQ(model_at_kiter, serialised_model_tmp);

    for (auto const& [key, value] : args) {
      if (key == "tree_method" && value == "gpu_hist") {
        learner->SetParam("gpu_id", "0");
      }
    }
    // Pull data to device
    for (auto &batch : p_dmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(0);
      batch.data.DeviceSpan();
      batch.offset.SetDevice(0);
      batch.offset.DeviceSpan();
    }

    for (int32_t iter = kIters; iter < 2 * kIters; ++iter) {
      learner->UpdateOneIter(iter, p_dmat);
    }
    serialised_model_tmp = std::string{};
    common::MemoryBufferStream fo(&serialised_model_tmp);
    learner->Save(&fo);

    Json m_0 = Json::Load(StringView{model_at_2kiter}, std::ios::binary);
    Json m_1 = Json::Load(StringView{serialised_model_tmp}, std::ios::binary);
    // GPU ID is changed as data is coming from device.
    ASSERT_EQ(get<Object>(m_0["Config"]["learner"]["generic_param"]).erase("gpu_id"),
              get<Object>(m_1["Config"]["learner"]["generic_param"]).erase("gpu_id"));
  }
}

// Binary is not tested, as it is NOT reproducible.
class SerializationTest : public ::testing::Test {
 protected:
  size_t constexpr static kRows = 15;
  size_t constexpr static kCols = 15;
  std::shared_ptr<DMatrix> p_dmat_;
  FeatureMap fmap_;

 protected:
  ~SerializationTest() override = default;
  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix();

    p_dmat_->Info().labels.Reshape(kRows);
    auto& h_labels = p_dmat_->Info().labels.Data()->HostVector();

    xgboost::SimpleLCG gen(0);
    SimpleRealUniformDistribution<float> dis(0.0f, 1.0f);

    for (auto& v : h_labels) { v = dis(&gen); }

    for (size_t i = 0; i < kCols; ++i) {
      std::string name = "feat_" + std::to_string(i);
      fmap_.PushBack(i, name.c_str(), "q");
    }
  }
};

size_t constexpr SerializationTest::kRows;
size_t constexpr SerializationTest::kCols;

TEST_F(SerializationTest, Exact) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"base_score", "3.14195265"},
                            {"max_depth", "2"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"base_score", "3.14195265"},
                            {"max_depth", "2"},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"base_score", "3.14195265"},
                            {"max_depth", "2"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);
}

TEST_F(SerializationTest, Approx) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);
}

TEST_F(SerializationTest, Hist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);
}

TEST_F(SerializationTest, CPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "coord_descent"}},
                           fmap_, p_dmat_);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(SerializationTest, GpuHist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);
}

TEST_F(SerializationTest, ConfigurationCount) {
  auto& p_dmat = p_dmat_;
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_dmat};

  xgboost::ConsoleLogger::Configure({{"verbosity", "3"}});

  testing::internal::CaptureStderr();

  std::string model_str;
  {
    auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

    learner->SetParam("tree_method", "gpu_hist");

    for (size_t i = 0; i < 10; ++i) {
      learner->UpdateOneIter(i, p_dmat);
    }
    common::MemoryBufferStream fo(&model_str);
    learner->Save(&fo);
  }

  {
    common::MemoryBufferStream fi(&model_str);
    auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
    learner->Load(&fi);
    for (size_t i = 0; i < 10; ++i) {
      learner->UpdateOneIter(i, p_dmat);
    }
  }

  std::string output = testing::internal::GetCapturedStderr();
  std::string target = "[GPU Hist]: Configure";
  ASSERT_NE(output.find(target), std::string::npos);

  size_t occureences = 0;
  size_t pos = 0;
  // Should run configuration exactly 2 times, one for each learner.
  while ((pos = output.find("[GPU Hist]: Configure", pos)) != std::string::npos) {
    occureences ++;
    pos += target.size();
  }
  ASSERT_EQ(occureences, 2ul);

  xgboost::ConsoleLogger::Configure({{"verbosity", "2"}});
}

TEST_F(SerializationTest, GPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "gpu_coord_descent"}},
                           fmap_, p_dmat_);
}
#endif  // defined(XGBOOST_USE_CUDA)

class L1SerializationTest : public SerializationTest {};

TEST_F(L1SerializationTest, Exact) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "reg:absoluteerror"},
                            {"seed", "0"},
                            {"max_depth", "2"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);
}

TEST_F(L1SerializationTest, Approx) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "reg:absoluteerror"},
                            {"seed", "0"},
                            {"max_depth", "2"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);
}

TEST_F(L1SerializationTest, Hist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "reg:absoluteerror"},
                            {"seed", "0"},
                            {"max_depth", "2"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(L1SerializationTest, GpuHist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "reg:absoluteerror"},
                            {"seed", "0"},
                            {"max_depth", "2"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);
}
#endif  //  defined(XGBOOST_USE_CUDA)

class LogitSerializationTest : public SerializationTest {
 protected:
  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix();

    std::shared_ptr<DMatrix> p_dmat{p_dmat_};
    p_dmat->Info().labels.Reshape(kRows);
    auto& h_labels = p_dmat->Info().labels.Data()->HostVector();

    std::bernoulli_distribution flip(0.5);
    auto& rnd = common::GlobalRandom();
    rnd.seed(0);

    for (auto& v : h_labels) { v = flip(rnd); }

    for (size_t i = 0; i < kCols; ++i) {
      std::string name = "feat_" + std::to_string(i);
      fmap_.PushBack(i, name.c_str(), "q");
    }
  }
};

TEST_F(LogitSerializationTest, Exact) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);
}

TEST_F(LogitSerializationTest, Approx) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);
}

TEST_F(LogitSerializationTest, Hist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);
}

TEST_F(LogitSerializationTest, CPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "coord_descent"}},
                           fmap_, p_dmat_);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(LogitSerializationTest, GpuHist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", "2"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);
}

TEST_F(LogitSerializationTest, GPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"objective", "binary:logistic"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "gpu_coord_descent"}},
                           fmap_, p_dmat_);
}
#endif  // defined(XGBOOST_USE_CUDA)

class MultiClassesSerializationTest : public SerializationTest {
 protected:
  size_t constexpr static kClasses = 4;

  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix();

    std::shared_ptr<DMatrix> p_dmat{p_dmat_};
    p_dmat->Info().labels.Reshape(kRows);
    auto &h_labels = p_dmat->Info().labels.Data()->HostVector();

    std::uniform_int_distribution<size_t> categorical(0, kClasses - 1);
    auto& rnd = common::GlobalRandom();
    rnd.seed(0);

    for (auto& v : h_labels) { v = categorical(rnd); }

    for (size_t i = 0; i < kCols; ++i) {
      std::string name = "feat_" + std::to_string(i);
      fmap_.PushBack(i, name.c_str(), "q");
    }
  }
};

TEST_F(MultiClassesSerializationTest, Exact) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "exact"}},
                           fmap_, p_dmat_);
}

TEST_F(MultiClassesSerializationTest, Approx) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "approx"}},
                           fmap_, p_dmat_);
}

TEST_F(MultiClassesSerializationTest, Hist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"num_parallel_tree", "4"},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "hist"}},
                           fmap_, p_dmat_);
}

TEST_F(MultiClassesSerializationTest, CPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "coord_descent"}},
                           fmap_, p_dmat_);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(MultiClassesSerializationTest, GpuHist) {
  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            // Somehow rebuilding the cache can generate slightly
                            // different result (1e-7) with CPU predictor for some
                            // entries.
                            {"predictor", "gpu_predictor"},
                            // Mitigate the difference caused by hardware fused multiply
                            // add to tree weight during update prediction cache.
                            {"learning_rate", "1.0"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "gbtree"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"max_depth", std::to_string(kClasses)},
                            // GPU_Hist has higher floating point error. 1e-6 doesn't work
                            // after num_parallel_tree goes to 4
                            {"num_parallel_tree", "4"},
                            {"learning_rate", "1.0"},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);

  TestLearnerSerialization({{"booster", "dart"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"learning_rate", "1.0"},
                            {"max_depth", std::to_string(kClasses)},
                            {"tree_method", "gpu_hist"}},
                           fmap_, p_dmat_);
}

TEST_F(MultiClassesSerializationTest, GPUCoordDescent) {
  TestLearnerSerialization({{"booster", "gblinear"},
                            {"num_class", std::to_string(kClasses)},
                            {"seed", "0"},
                            {"nthread", "1"},
                            {"updater", "gpu_coord_descent"}},
                           fmap_, p_dmat_);
}
#endif  // defined(XGBOOST_USE_CUDA)
}       // namespace xgboost
