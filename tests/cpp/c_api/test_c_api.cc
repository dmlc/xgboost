/**
 * Copyright 2019-2024 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/json.h>  // Json
#include <xgboost/learner.h>
#include <xgboost/version_config.h>

#include <array>       // for array
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem
#include <limits>      // std::numeric_limits
#include <string>      // std::string
#include <vector>

#include "../../../src/c_api/c_api_error.h"
#include "../../../src/common/io.h"
#include "../../../src/data/adapter.h"              // for ArrayAdapter
#include "../../../src/data/array_interface.h"      // for ArrayInterface
#include "../../../src/data/batch_utils.h"          // for MatchingPageBytes
#include "../../../src/data/gradient_index.h"       // for GHistIndexMatrix
#include "../../../src/data/iterative_dmatrix.h"    // for IterativeDMatrix
#include "../../../src/data/sparse_page_dmatrix.h"  // for SparsePageDMatrix
#include "../helpers.h"

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
  config["data_split_mode"] = Integer{static_cast<int64_t>(DataSplitMode::kCol)};
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
  ASSERT_EQ(XGDMatrixDataSplitMode(handle, &n), 0);
  ASSERT_EQ(n, static_cast<int64_t>(DataSplitMode::kCol));

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

  std::array<char const* , 1> out;
  bst_ulong len {0};
  XGBoosterSaveJsonConfig(handle, &len, out.data());

  std::string config_str_0 { out[0] };
  auto config_0 = Json::Load({config_str_0.c_str(), config_str_0.size()});
  XGBoosterLoadJsonConfig(handle, out[0]);

  bst_ulong len_1 {0};
  std::string config_str_1 { out[0] };
  XGBoosterSaveJsonConfig(handle, &len_1, out.data());
  auto config_1 = Json::Load({config_str_1.c_str(), config_str_1.size()});

  ASSERT_EQ(config_0, config_1);
}

TEST(CAPI, JsonModelIO) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  auto tempdir = std::filesystem::temp_directory_path();

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

  auto modelfile_0 = tempdir / std::filesystem::u8path(u8"모델_0.json");
  XGBoosterSaveModel(handle, modelfile_0.u8string().c_str());
  XGBoosterLoadModel(handle, modelfile_0.u8string().c_str());

  bst_ulong num_feature {0};
  ASSERT_EQ(XGBoosterGetNumFeature(handle, &num_feature), 0);
  ASSERT_EQ(num_feature, kCols);

  auto modelfile_1 = tempdir / "model_1.json";
  XGBoosterSaveModel(handle, modelfile_1.u8string().c_str());

  auto model_str_0 = common::LoadSequentialFile(modelfile_0.u8string());
  auto model_str_1 = common::LoadSequentialFile(modelfile_1.u8string());

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
  auto l = StringView{data, static_cast<size_t>(len)};
  auto r = StringView{saved, static_cast<size_t>(saved_len)};
  ASSERT_EQ(l.size(), r.size());
  ASSERT_EQ(l, r);

  std::string buffer;
  Json::Dump(Json::Load(l, std::ios::binary), &buffer);
  ASSERT_EQ(model_str_0.size(), buffer.size());
  ASSERT_EQ(model_str_0.back(), '}');
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

  std::array<char const *, 2> feat_types{"i", "q"};
  static_assert(sizeof(feat_types) / sizeof(feat_types[0]) == kCols);
  XGDMatrixSetStrFeatureInfo(handle, "feature_type", feat_types.data(), kCols);
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

namespace {
void MakeLabelForTest(std::shared_ptr<DMatrix> Xy, DMatrixHandle cxy) {
  auto n_samples = Xy->Info().num_row_;
  std::vector<float> y(n_samples);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = static_cast<float>(i);
  }

  Xy->Info().labels.Reshape(n_samples);
  Xy->Info().labels.Data()->HostVector() = y;

  auto y_int = GetArrayInterface(Xy->Info().labels.Data(), n_samples, 1);
  std::string s_y_int;
  Json::Dump(y_int, &s_y_int);

  XGDMatrixSetInfoFromInterface(cxy, "label", s_y_int.c_str());
}

auto MakeSimpleDMatrixForTest(bst_idx_t n_samples, bst_feature_t n_features, Json dconfig) {
  HostDeviceVector<float> storage;
  auto arr_int = RandomDataGenerator{n_samples, n_features, 0.5f}.GenerateArrayInterface(&storage);

  data::ArrayAdapter adapter{StringView{arr_int}};
  std::shared_ptr<DMatrix> Xy{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), Context{}.Threads())};

  DMatrixHandle p_fmat;
  std::string s_dconfig;
  Json::Dump(dconfig, &s_dconfig);
  CHECK_EQ(XGDMatrixCreateFromDense(arr_int.c_str(), s_dconfig.c_str(), &p_fmat), 0);

  MakeLabelForTest(Xy, p_fmat);
  return std::pair{p_fmat, Xy};
}

auto MakeQDMForTest(Context const *ctx, bst_idx_t n_samples, bst_feature_t n_features,
                    Json dconfig) {
  bst_bin_t n_bins{16};
  dconfig["max_bin"] = Integer{n_bins};

  std::size_t n_batches{4};
  std::unique_ptr<ArrayIterForTest> iter_0;
  if (ctx->IsCUDA()) {
    iter_0 = std::make_unique<CudaArrayIterForTest>(0.0f, n_samples, n_features, n_batches);
  } else {
    iter_0 = std::make_unique<NumpyArrayIterForTest>(0.0f, n_samples, n_features, n_batches);
  }
  std::string s_dconfig;
  Json::Dump(dconfig, &s_dconfig);
  DMatrixHandle p_fmat;
  CHECK_EQ(XGQuantileDMatrixCreateFromCallback(static_cast<DataIterHandle>(iter_0.get()),
                                               iter_0->Proxy(), nullptr, Reset, Next,
                                               s_dconfig.c_str(), &p_fmat),
           0);

  std::unique_ptr<ArrayIterForTest> iter_1;
  if (ctx->IsCUDA()) {
    iter_1 = std::make_unique<CudaArrayIterForTest>(0.0f, n_samples, n_features, n_batches);
  } else {
    iter_1 = std::make_unique<NumpyArrayIterForTest>(0.0f, n_samples, n_features, n_batches);
  }
  auto Xy = std::make_shared<data::IterativeDMatrix>(
      iter_1.get(), iter_1->Proxy(), nullptr, Reset, Next, std::numeric_limits<float>::quiet_NaN(),
      0, n_bins, std::numeric_limits<std::int64_t>::max());
  return std::pair{p_fmat, Xy};
}

auto MakeExtMemForTest(bst_idx_t n_samples, bst_feature_t n_features, Json dconfig) {
  std::size_t n_batches{4};
  NumpyArrayIterForTest iter_0{0.0f, n_samples, n_features, n_batches};
  std::string s_dconfig;
  dconfig["cache_prefix"] = String{"cache"};
  Json::Dump(dconfig, &s_dconfig);
  DMatrixHandle p_fmat;
  CHECK_EQ(XGDMatrixCreateFromCallback(static_cast<DataIterHandle>(&iter_0), iter_0.Proxy(), Reset,
                                       Next, s_dconfig.c_str(), &p_fmat),
           0);

  NumpyArrayIterForTest iter_1{0.0f, n_samples, n_features, n_batches};
  auto config = ExtMemConfig{"",
                             false,
                             cuda_impl::MatchingPageBytes(),
                             std::numeric_limits<float>::quiet_NaN(),
                             cuda_impl::MaxNumDevicePages(),
                             0};
  auto Xy = std::make_shared<data::SparsePageDMatrix>(&iter_1, iter_1.Proxy(), Reset, Next, config);
  MakeLabelForTest(Xy, p_fmat);
  return std::pair{p_fmat, Xy};
}

template <typename Page>
void CheckResult(Context const *ctx, bst_feature_t n_features, std::shared_ptr<DMatrix> Xy,
                 float const *out_data, std::uint64_t const *out_indptr) {
  for (auto const &page : Xy->GetBatches<Page>(ctx, BatchParam{16, 0.2})) {
    auto const &cut = page.Cuts();
    auto const &ptrs = cut.Ptrs();
    auto const &vals = cut.Values();
    auto const &mins = cut.MinValues();
    for (bst_feature_t f = 0; f < Xy->Info().num_col_; ++f) {
      ASSERT_EQ(ptrs[f] + f, out_indptr[f]);
      ASSERT_EQ(mins[f], out_data[out_indptr[f]]);
      auto beg = out_indptr[f];
      auto end = out_indptr[f + 1];
      auto val_beg = ptrs[f];
      for (std::uint64_t i = beg + 1, j = val_beg; i < end; ++i, ++j) {
        ASSERT_EQ(vals[j], out_data[i]);
      }
    }

    ASSERT_EQ(ptrs[n_features] + n_features, out_indptr[n_features]);
  }
}

void TestXGDMatrixGetQuantileCut(Context const *ctx) {
  bst_idx_t n_samples{1024};
  bst_feature_t n_features{16};

  Json dconfig{Object{}};
  dconfig["ntread"] = Integer{Context{}.Threads()};
  dconfig["missing"] = Number{std::numeric_limits<float>::quiet_NaN()};

  auto check_result = [n_features, &ctx](std::shared_ptr<DMatrix> Xy, StringView s_out_data,
                                         StringView s_out_indptr) {
    auto i_out_data = ArrayInterface<1, false>{s_out_data};
    ASSERT_EQ(i_out_data.type, ArrayInterfaceHandler::kF4);
    auto out_data = static_cast<float const *>(i_out_data.data);
    ASSERT_TRUE(out_data);

    auto i_out_indptr = ArrayInterface<1, false>{s_out_indptr};
    ASSERT_EQ(i_out_indptr.type, ArrayInterfaceHandler::kU8);
    auto out_indptr = static_cast<std::uint64_t const *>(i_out_indptr.data);
    ASSERT_TRUE(out_data);

    if (ctx->IsCPU()) {
      CheckResult<GHistIndexMatrix>(ctx, n_features, Xy, out_data, out_indptr);
    } else {
      CheckResult<EllpackPage>(ctx, n_features, Xy, out_data, out_indptr);
    }
  };

  Json config{Null{}};
  std::string s_config;
  Json::Dump(config, &s_config);
  char const *out_indptr;
  char const *out_data;

  {
    // SimpleDMatrix
    auto [p_fmat, Xy] = MakeSimpleDMatrixForTest(n_samples, n_features, dconfig);
    // assert fail, we don't have the quantile yet.
    ASSERT_EQ(XGDMatrixGetQuantileCut(p_fmat, s_config.c_str(), &out_indptr, &out_data), -1);

    std::array<DMatrixHandle, 1> mats{p_fmat};
    BoosterHandle booster;
    ASSERT_EQ(XGBoosterCreate(mats.data(), 1, &booster), 0);
    ASSERT_EQ(XGBoosterSetParam(booster, "max_bin", "16"), 0);
    if (ctx->IsCUDA()) {
      ASSERT_EQ(XGBoosterSetParam(booster, "device", ctx->DeviceName().c_str()), 0);
    }
    ASSERT_EQ(XGBoosterUpdateOneIter(booster, 0, p_fmat), 0);
    ASSERT_EQ(XGDMatrixGetQuantileCut(p_fmat, s_config.c_str(), &out_indptr, &out_data), 0);

    check_result(Xy, out_data, out_indptr);

    XGDMatrixFree(p_fmat);
    XGBoosterFree(booster);
  }

  {
    // IterativeDMatrix
    auto [p_fmat, Xy] = MakeQDMForTest(ctx, n_samples, n_features, dconfig);
    ASSERT_EQ(XGDMatrixGetQuantileCut(p_fmat, s_config.c_str(), &out_indptr, &out_data), 0);

    check_result(Xy, out_data, out_indptr);
    XGDMatrixFree(p_fmat);
  }

  {
    // SparsePageDMatrix
    auto [p_fmat, Xy] = MakeExtMemForTest(n_samples, n_features, dconfig);
    // assert fail, we don't have the quantile yet.
    ASSERT_EQ(XGDMatrixGetQuantileCut(p_fmat, s_config.c_str(), &out_indptr, &out_data), -1);

    std::array<DMatrixHandle, 1> mats{p_fmat};
    BoosterHandle booster;
    ASSERT_EQ(XGBoosterCreate(mats.data(), 1, &booster), 0);
    ASSERT_EQ(XGBoosterSetParam(booster, "max_bin", "16"), 0);
    if (ctx->IsCUDA()) {
      ASSERT_EQ(XGBoosterSetParam(booster, "device", ctx->DeviceName().c_str()), 0);
    }
    ASSERT_EQ(XGBoosterUpdateOneIter(booster, 0, p_fmat), 0);
    ASSERT_EQ(XGDMatrixGetQuantileCut(p_fmat, s_config.c_str(), &out_indptr, &out_data), 0);

    XGDMatrixFree(p_fmat);
    XGBoosterFree(booster);
  }
}
}  // namespace

TEST(CAPI, XGDMatrixGetQuantileCut) {
  Context ctx;
  TestXGDMatrixGetQuantileCut(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(CAPI, GPUXGDMatrixGetQuantileCut) {
  auto ctx = MakeCUDACtx(0);
  TestXGDMatrixGetQuantileCut(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
