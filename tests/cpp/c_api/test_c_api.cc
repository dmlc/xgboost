/**
 * Copyright 2019-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/json.h>  // Json
#include <xgboost/learner.h>
#include <xgboost/version_config.h>

#include <cstddef>  // std::size_t
#include <limits>   // std::numeric_limits
#include <string>   // std::string
#include <vector>

#include "../../../src/c_api/c_api_error.h"
#include "../../../src/common/io.h"
#include "../helpers.h"

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
    auto page = batch.GetView();
    ASSERT_EQ(page[0][0].fvalue, 0.0f);
    ASSERT_EQ(page[0][1].fvalue, -4.0f);
    ASSERT_EQ(page[2][0].fvalue, 3.0f);
    ASSERT_EQ(page[2][1].fvalue, 0.0f);
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
      auto page = batch.GetView();
      for (size_t i = 0; i < batch.Size(); i++) {
        auto inst = page[i];
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

TEST(CAPI, XGDMatrixCreateFromCSR) {
  HostDeviceVector<std::size_t> indptr{0, 3};
  HostDeviceVector<double> data{0.0, 1.0, 2.0};
  HostDeviceVector<std::size_t> indices{0, 1, 2};
  auto indptr_arr = GetArrayInterface(&indptr, 2, 1);
  auto indices_arr = GetArrayInterface(&indices, 3, 1);
  auto data_arr = GetArrayInterface(&data, 3, 1);
  std::string sindptr, sindices, sdata, sconfig;
  Json::Dump(indptr_arr, &sindptr);
  Json::Dump(indices_arr, &sindices);
  Json::Dump(data_arr, &sdata);
  Json config{Object{}};
  config["missing"] = Number{std::numeric_limits<float>::quiet_NaN()};
  Json::Dump(config, &sconfig);

  DMatrixHandle handle;
  XGDMatrixCreateFromCSR(sindptr.c_str(), sindices.c_str(), sdata.c_str(), 3, sconfig.c_str(),
                         &handle);
  bst_ulong n;
  ASSERT_EQ(XGDMatrixNumRow(handle, &n), 0);
  ASSERT_EQ(n, 1);
  ASSERT_EQ(XGDMatrixNumCol(handle, &n), 0);
  ASSERT_EQ(n, 3);
  ASSERT_EQ(XGDMatrixNumNonMissing(handle, &n), 0);
  ASSERT_EQ(n, 3);

  std::shared_ptr<xgboost::DMatrix> *pp_fmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  ASSERT_EQ((*pp_fmat)->Ctx()->Threads(), AllThreadsForTest());

  XGDMatrixFree(handle);
}

TEST(CAPI, ConfigIO) {
  size_t constexpr kRows = 10;
  auto p_dmat = RandomDataGenerator(kRows, 10, 0).GenerateDMatrix();
  std::vector<std::shared_ptr<DMatrix>> mat {p_dmat};
  std::vector<bst_float> labels(kRows);
  for (size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels.Data()->HostVector() = labels;
  p_dmat->Info().labels.Reshape(kRows);

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
  p_dmat->Info().labels.Data()->HostVector() = labels;
  p_dmat->Info().labels.Reshape(kRows);

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

  /**
   * In memory
   */
  bst_ulong len{0};
  char const *data;
  XGBoosterSaveModelToBuffer(handle, R"({"format": "ubj"})", &len, &data);
  ASSERT_GT(len, 3);

  XGBoosterLoadModelFromBuffer(handle, data, len);
  char const *saved;
  bst_ulong saved_len{0};
  XGBoosterSaveModelToBuffer(handle, R"({"format": "ubj"})", &saved_len, &saved);
  ASSERT_EQ(len, saved_len);
  auto l = StringView{data, len};
  auto r = StringView{saved, saved_len};
  ASSERT_EQ(l.size(), r.size());
  ASSERT_EQ(l, r);

  std::string buffer;
  Json::Dump(Json::Load(l, std::ios::binary), &buffer);
  ASSERT_EQ(model_str_0.size() - 1, buffer.size());
  ASSERT_EQ(model_str_0.back(), '\0');
  ASSERT_TRUE(std::equal(model_str_0.begin(), model_str_0.end() - 1, buffer.begin()));

  ASSERT_EQ(XGBoosterSaveModelToBuffer(handle, R"({})", &len, &data), -1);
  ASSERT_EQ(XGBoosterSaveModelToBuffer(handle, R"({"format": "foo"})", &len, &data), -1);
}

TEST(CAPI, CatchDMLCError) {
  DMatrixHandle out;
  ASSERT_EQ(XGDMatrixCreateFromFile("foo", 0, &out), -1);
  EXPECT_THROW({ dmlc::Stream::Create("foo", "r"); },  dmlc::Error);
}

TEST(CAPI, CatchDMLCErrorURI) {
  Json config{Object()};
  config["uri"] = String{"foo"};
  config["silent"] = Integer{0};
  std::string config_str;
  Json::Dump(config, &config_str);
  DMatrixHandle out;
  ASSERT_EQ(XGDMatrixCreateFromURI(config_str.c_str(), &out), -1);
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
  static_assert(sizeof(feat_types) / sizeof(feat_types[0]) == kCols);
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

TEST(CAPI, XGBGlobalConfig) {
  int ret;
  {
    const char *config_str = R"json(
    {
      "verbosity": 0,
      "use_rmm": false
    }
  )json";
    ret = XGBSetGlobalConfig(config_str);
    ASSERT_EQ(ret, 0);
    const char *updated_config_cstr;
    ret = XGBGetGlobalConfig(&updated_config_cstr);
    ASSERT_EQ(ret, 0);

    std::string updated_config_str{updated_config_cstr};
    auto updated_config =
        Json::Load({updated_config_str.data(), updated_config_str.size()});
    ASSERT_EQ(get<Integer>(updated_config["verbosity"]), 0);
    ASSERT_EQ(get<Boolean>(updated_config["use_rmm"]), false);
  }
  {
    const char *config_str = R"json(
    {
      "use_rmm": true
    }
  )json";
    ret = XGBSetGlobalConfig(config_str);
    ASSERT_EQ(ret, 0);
    const char *updated_config_cstr;
    ret = XGBGetGlobalConfig(&updated_config_cstr);
    ASSERT_EQ(ret, 0);

    std::string updated_config_str{updated_config_cstr};
    auto updated_config =
        Json::Load({updated_config_str.data(), updated_config_str.size()});
    ASSERT_EQ(get<Boolean>(updated_config["use_rmm"]), true);
  }
  {
    const char *config_str = R"json(
    {
      "foo": 0
    }
  )json";
    ret = XGBSetGlobalConfig(config_str);
    ASSERT_EQ(ret , -1);
    auto err = std::string{XGBGetLastError()};
    ASSERT_NE(err.find("foo"), std::string::npos);
  }
  {
    const char *config_str = R"json(
    {
      "foo": 0,
      "verbosity": 0
    }
  )json";
    ret = XGBSetGlobalConfig(config_str);
    ASSERT_EQ(ret , -1);
    auto err = std::string{XGBGetLastError()};
    ASSERT_NE(err.find("foo"), std::string::npos);
    ASSERT_EQ(err.find("verbosity"), std::string::npos);
  }
}

TEST(CAPI, BuildInfo) {
  char const* out;
  XGBuildInfo(&out);
  auto loaded = Json::Load(StringView{out});
  ASSERT_TRUE(get<Object const>(loaded).find("USE_OPENMP") != get<Object const>(loaded).cend());
  ASSERT_TRUE(get<Object const>(loaded).find("USE_CUDA") != get<Object const>(loaded).cend());
  ASSERT_TRUE(get<Object const>(loaded).find("USE_NCCL") != get<Object const>(loaded).cend());
}

TEST(CAPI, NullPtr) {
  ASSERT_EQ(XGBSetGlobalConfig(nullptr), -1);
  auto const *err = XGBGetLastError();
  auto pos = std::string{err}.find("Invalid pointer argument: json_str");
  ASSERT_NE(pos, std::string::npos);
  XGBAPISetLastError("");
}

TEST(CAPI, JArgs) {
  {
    Json args{Object{}};
    args["key"] = String{"value"};
    args["null"] = Null{};
    auto value = OptionalArg<String>(args, "key", std::string{"foo"});
    ASSERT_EQ(value, "value");
    value = OptionalArg<String const>(args, "key", std::string{"foo"});
    ASSERT_EQ(value, "value");

    ASSERT_THROW({ OptionalArg<Number>(args, "key", 0.0f); }, dmlc::Error);
    value = OptionalArg<String const>(args, "bar", std::string{"foo"});
    ASSERT_EQ(value, "foo");
    value = OptionalArg<String const>(args, "null", std::string{"foo"});
    ASSERT_EQ(value, "foo");
  }

  {
    Json args{Object{}};
    args["key"] = String{"value"};
    args["null"] = Null{};
    auto value = RequiredArg<String>(args, "key", __func__);
    ASSERT_EQ(value, "value");
    value = RequiredArg<String const>(args, "key", __func__);
    ASSERT_EQ(value, "value");

    ASSERT_THROW({ RequiredArg<Integer>(args, "key", __func__); }, dmlc::Error);
    ASSERT_THROW({ RequiredArg<String const>(args, "foo", __func__); }, dmlc::Error);
    ASSERT_THROW({ RequiredArg<String>(args, "null", __func__); }, dmlc::Error);
  }
}
}  // namespace xgboost
