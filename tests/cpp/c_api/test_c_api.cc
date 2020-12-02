/*!
 * Copyright 2019-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/version_config.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/learner.h>

#include "../helpers.h"
#include "../../../src/common/io.h"

#include "../../../src/c_api/c_api_error.h"

TEST(CAPI, XGDMatrixCreateFromMatDT) {
  std::vector<int> col0 = {0, -1, 3};
  std::vector<float> col1 = {-4.0f, 2.0f, 0.0f};
  const char *col0_type = "int32";
  const char *col1_type = "float32";
  std::vector<void *> data = {col0.data(), col1.data()};
  std::vector<const char *> types = {col0_type, col1_type};
  DMatrixHandle handle;
  XGDMatrixCreateFromDT(data.data(), types.data(), 3, 2, &handle,
                        0);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, 2ul);
  ASSERT_EQ(info.num_row_, 3ul);
  ASSERT_EQ(info.num_nonzero_, 6ul);

  for (const auto &batch : (*dmat)->GetBatches<xgboost::SparsePage>()) {
    ASSERT_EQ(batch[0][0].fvalue, 0.0f);
    ASSERT_EQ(batch[0][1].fvalue, -4.0f);
    ASSERT_EQ(batch[2][0].fvalue, 3.0f);
    ASSERT_EQ(batch[2][1].fvalue, 0.0f);
  }

  delete dmat;
}

TEST(CAPI, XGDMatrixCreateFromMatOmp) {
  std::vector<bst_ulong> num_rows = {100, 11374, 15000};
  for (auto row : num_rows) {
    bst_ulong num_cols = 50;
    int num_missing = 5;
    DMatrixHandle handle;
    std::vector<float> data(num_cols * row, 1.5);
    for (int i = 0; i < num_missing; i++) {
      data[i] = std::numeric_limits<float>::quiet_NaN();
    }

    XGDMatrixCreateFromMat_omp(data.data(), row, num_cols,
                               std::numeric_limits<float>::quiet_NaN(), &handle,
                               0);

    std::shared_ptr<xgboost::DMatrix> *dmat =
        static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
    xgboost::MetaInfo &info = (*dmat)->Info();
    ASSERT_EQ(info.num_col_, num_cols);
    ASSERT_EQ(info.num_row_, row);
    ASSERT_EQ(info.num_nonzero_, num_cols * row - num_missing);

    for (const auto &batch : (*dmat)->GetBatches<xgboost::SparsePage>()) {
      for (size_t i = 0; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (auto e : inst) {
          ASSERT_EQ(e.fvalue, 1.5);
        }
      }
    }
    delete dmat;
  }
}

namespace xgboost {

TEST(CAPI, Version) {
  int patch {0};
  XGBoostVersion(NULL, NULL, &patch);  // NOLINT
  ASSERT_EQ(patch, XGBOOST_VER_PATCH);
}

TEST(CAPI, ConfigIO) {
  size_t constexpr kRows = 10;
  auto p_dmat = RandomDataGenerator(kRows, 10, 0).GenerateDMatrix();
  std::vector<std::shared_ptr<DMatrix>> mat {p_dmat};
  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;

  std::shared_ptr<Learner> learner { Learner::Create(mat) };

  BoosterHandle handle = learner.get();
  learner->UpdateOneIter(0, p_dmat);

  char const* out[1];
  bst_ulong len {0};
  XGBoosterSaveJsonConfig(handle, &len, out);

  std::string config_str_0 { out[0] };
  auto config_0 = Json::Load({config_str_0.c_str(), config_str_0.size()});
  XGBoosterLoadJsonConfig(handle, out[0]);

  bst_ulong len_1 {0};
  std::string config_str_1 { out[0] };
  XGBoosterSaveJsonConfig(handle, &len_1, out);
  auto config_1 = Json::Load({config_str_1.c_str(), config_str_1.size()});

  ASSERT_EQ(config_0, config_1);
}

TEST(CAPI, JsonModelIO) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  dmlc::TemporaryDirectory tempdir;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  std::vector<std::shared_ptr<DMatrix>> mat {p_dmat};
  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;

  std::shared_ptr<Learner> learner { Learner::Create(mat) };

  learner->UpdateOneIter(0, p_dmat);
  BoosterHandle handle = learner.get();

  std::string modelfile_0 = tempdir.path + "/model_0.json";
  XGBoosterSaveModel(handle, modelfile_0.c_str());
  XGBoosterLoadModel(handle, modelfile_0.c_str());

  bst_ulong num_feature {0};
  ASSERT_EQ(XGBoosterGetNumFeature(handle, &num_feature), 0);
  ASSERT_EQ(num_feature, kCols);

  std::string modelfile_1 = tempdir.path + "/model_1.json";
  XGBoosterSaveModel(handle, modelfile_1.c_str());

  auto model_str_0 = common::LoadSequentialFile(modelfile_0);
  auto model_str_1 = common::LoadSequentialFile(modelfile_1);

  ASSERT_EQ(model_str_0.front(), '{');
  ASSERT_EQ(model_str_0, model_str_1);
}

TEST(CAPI, CatchDMLCError) {
  DMatrixHandle out;
  ASSERT_EQ(XGDMatrixCreateFromFile("foo", 0, &out), -1);
  EXPECT_THROW({ dmlc::Stream::Create("foo", "r"); },  dmlc::Error);
}

TEST(CAPI, DMatrixSetFeatureName) {
  size_t constexpr kRows = 10;
  bst_feature_t constexpr kCols = 2;

  DMatrixHandle handle;
  std::vector<float> data(kCols * kRows, 1.5);

  XGDMatrixCreateFromMat_omp(data.data(), kRows, kCols,
                             std::numeric_limits<float>::quiet_NaN(), &handle,
                             0);
  std::vector<std::string> feature_names;
  for (bst_feature_t i = 0; i < kCols; ++i) {
    feature_names.emplace_back(std::to_string(i));
  }
  std::vector<char const*> c_feature_names;
  c_feature_names.resize(feature_names.size());
  std::transform(feature_names.cbegin(), feature_names.cend(),
                 c_feature_names.begin(),
                 [](auto const &str) { return str.c_str(); });
  XGDMatrixSetStrFeatureInfo(handle, u8"feature_name", c_feature_names.data(),
                             c_feature_names.size());
  bst_ulong out_len = 0;
  char const **c_out_features;
  XGDMatrixGetStrFeatureInfo(handle, u8"feature_name", &out_len,
                             &c_out_features);

  CHECK_EQ(out_len, kCols);
  std::vector<std::string> out_features;
  for (bst_ulong i = 0; i < out_len; ++i) {
    ASSERT_EQ(std::to_string(i), c_out_features[i]);
  }

  char const* feat_types [] {"i", "q"};
  static_assert(sizeof(feat_types)/ sizeof(feat_types[0]) == kCols, "");
  XGDMatrixSetStrFeatureInfo(handle, "feature_type", feat_types, kCols);
  char const **c_out_types;
  XGDMatrixGetStrFeatureInfo(handle, u8"feature_type", &out_len,
                             &c_out_types);
  for (bst_ulong i = 0; i < out_len; ++i) {
    ASSERT_STREQ(feat_types[i], c_out_types[i]);
  }

  XGDMatrixFree(handle);
}

int TestExceptionCatching() {
  API_BEGIN();
  throw std::bad_alloc();
  API_END();
}

TEST(CAPI, Exception) {
  ASSERT_NO_THROW({TestExceptionCatching();});
  ASSERT_EQ(TestExceptionCatching(), -1);
  auto error = XGBGetLastError();
  // Not null
  ASSERT_TRUE(error);
}
}  // namespace xgboost
