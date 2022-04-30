// Copyright (c) 2014-2022 by Contributors
#include <rabit/rabit.h>
#include <rabit/c_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"
#include "xgboost/c_api.h"
#include "xgboost/logging.h"
#include "xgboost/version_config.h"
#include "xgboost/json.h"
#include "xgboost/global_config.h"

#include "c_api_error.h"
#include "c_api_utils.h"
#include "../common/io.h"
#include "../common/charconv.h"
#include "../data/adapter.h"
#include "../data/simple_dmatrix.h"
#include "../data/proxy_dmatrix.h"

using namespace xgboost; // NOLINT(*);

XGB_DLL void XGBoostVersion(int* major, int* minor, int* patch) {
  if (major) {
    *major = XGBOOST_VER_MAJOR;
  }
  if (minor) {
    *minor = XGBOOST_VER_MINOR;
  }
  if (patch) {
    *patch = XGBOOST_VER_PATCH;
  }
}

using GlobalConfigAPIThreadLocalStore = dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry>;

#if !defined(XGBOOST_USE_CUDA)
namespace xgboost {
void XGBBuildInfoDevice(Json *p_info) {
  auto &info = *p_info;
  info["USE_CUDA"] = Boolean{false};
  info["USE_NCCL"] = Boolean{false};
  info["USE_RMM"] = Boolean{false};
}
}  // namespace xgboost
#endif

XGB_DLL int XGBuildInfo(char const **out) {
  API_BEGIN();
  CHECK(out) << "Invalid input pointer";
  Json info{Object{}};

#if defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  info["BUILTIN_PREFETCH_PRESENT"] = Boolean{true};
#else
  info["BUILTIN_PREFETCH_PRESENT"] = Boolean{false};
#endif

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  info["MM_PREFETCH_PRESENT"] = Boolean{true};
#else
  info["MM_PREFETCH_PRESENT"] = Boolean{false};
#endif

#if defined(_OPENMP)
  info["USE_OPENMP"] = Boolean{true};
#else
  info["USE_OPENMP"] = Boolean{false};
#endif

#if defined(__GNUC__) && !defined(__clang__)
  info["GCC_VERSION"] = std::vector<Json>{Json{Integer{__GNUC__}}, Json{Integer{__GNUC_MINOR__}},
                                          Json{Integer{__GNUC_PATCHLEVEL__}}};
#endif

#if defined(__clang__)
  info["CLANG_VERSION"] =
      std::vector<Json>{Json{Integer{__clang_major__}}, Json{Integer{__clang_minor__}},
                        Json{Integer{__clang_patchlevel__}}};
#endif

#if !defined(NDEBUG)
  info["DEBUG"] = Boolean{true};
#else
  info["DEBUG"] = Boolean{false};
#endif

  XGBBuildInfoDevice(&info);

  auto &out_str = GlobalConfigAPIThreadLocalStore::Get()->ret_str;
  Json::Dump(info, &out_str);
  *out = out_str.c_str();

  API_END();
}

XGB_DLL int XGBRegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN_UNGUARD();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->Register(callback);
  API_END();
}

XGB_DLL int XGBSetGlobalConfig(const char* json_str) {
  API_BEGIN();
  Json config{Json::Load(StringView{json_str})};
  for (auto& items : get<Object>(config)) {
    switch (items.second.GetValue().Type()) {
    case xgboost::Value::ValueKind::kInteger: {
      items.second = String{std::to_string(get<Integer const>(items.second))};
      break;
    }
    case xgboost::Value::ValueKind::kBoolean: {
      if (get<Boolean const>(items.second)) {
        items.second = String{"true"};
      } else {
        items.second = String{"false"};
      }
      break;
    }
    case xgboost::Value::ValueKind::kNumber: {
      auto n = get<Number const>(items.second);
      char chars[NumericLimits<float>::kToCharsSize];
      auto ec = to_chars(chars, chars + sizeof(chars), n).ec;
      CHECK(ec == std::errc());
      items.second = String{chars};
      break;
    }
    default:
      break;
    }
  }
  auto unknown = FromJson(config, GlobalConfigThreadLocalStore::Get());
  if (!unknown.empty()) {
    std::stringstream ss;
    ss << "Unknown global parameters: { ";
    size_t i = 0;
    for (auto const& item : unknown) {
      ss << item.first;
      i++;
      if (i != unknown.size()) {
        ss << ", ";
      }
    }
    LOG(FATAL) << ss.str()  << " }";
  }
  API_END();
}

XGB_DLL int XGBGetGlobalConfig(const char** json_str) {
  API_BEGIN();
  auto const& global_config = *GlobalConfigThreadLocalStore::Get();
  Json config {ToJson(global_config)};
  auto const* mgr = global_config.__MANAGER__();

  for (auto& item : get<Object>(config)) {
    auto const &str = get<String const>(item.second);
    auto const &name = item.first;
    auto e = mgr->Find(name);
    CHECK(e);

    if (dynamic_cast<dmlc::parameter::FieldEntry<int32_t> const*>(e) ||
        dynamic_cast<dmlc::parameter::FieldEntry<int64_t> const*>(e) ||
        dynamic_cast<dmlc::parameter::FieldEntry<uint32_t> const*>(e) ||
        dynamic_cast<dmlc::parameter::FieldEntry<uint64_t> const*>(e)) {
      auto i = std::strtoimax(str.data(), nullptr, 10);
      CHECK_LE(i, static_cast<intmax_t>(std::numeric_limits<int64_t>::max()));
      item.second = Integer(static_cast<int64_t>(i));
    } else if (dynamic_cast<dmlc::parameter::FieldEntry<float> const *>(e) ||
               dynamic_cast<dmlc::parameter::FieldEntry<double> const *>(e)) {
      float f;
      auto ec = from_chars(str.data(), str.data() + str.size(), f).ec;
      CHECK(ec == std::errc());
      item.second = Number(f);
    } else if (dynamic_cast<dmlc::parameter::FieldEntry<bool> const *>(e)) {
      item.second = Boolean(str != "0");
    }
  }

  auto& local = *GlobalConfigAPIThreadLocalStore::Get();
  Json::Dump(config, &local.ret_str);
  *json_str = local.ret_str.c_str();
  API_END();
}

XGB_DLL int XGDMatrixCreateFromFile(const char *fname,
                                    int silent,
                                    DMatrixHandle *out) {
  API_BEGIN();
  bool load_row_split = false;
  if (rabit::IsDistributed()) {
    LOG(CONSOLE) << "XGBoost distributed mode detected, "
                 << "will split data among workers";
    load_row_split = true;
  }
  *out = new std::shared_ptr<DMatrix>(DMatrix::Load(fname, silent != 0, load_row_split));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromDataIter(
    void *data_handle,                  // a Java iterator
    XGBCallbackDataIterNext *callback,  // C++ callback defined in xgboost4j.cpp
    const char *cache_info, DMatrixHandle *out) {
  API_BEGIN();

  std::string scache;
  if (cache_info != nullptr) {
    scache = cache_info;
  }
  xgboost::data::IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext,
                                 XGBoostBatchCSR> adapter(data_handle, callback);
  *out = new std::shared_ptr<DMatrix> {
    DMatrix::Create(
        &adapter, std::numeric_limits<float>::quiet_NaN(),
        1, scache
    )
  };
  API_END();
}

#ifndef XGBOOST_USE_CUDA
XGB_DLL int XGDMatrixCreateFromCudaColumnar(char const *data,
                                            char const* c_json_config,
                                            DMatrixHandle *out) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCudaArrayInterface(char const *data,
                                                  char const* c_json_config,
                                                  DMatrixHandle *out) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}

#endif

// Create from data iterator
XGB_DLL int XGDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                        DataIterResetCallback *reset, XGDMatrixCallbackNext *next,
                                        char const *c_json_config, DMatrixHandle *out) {
  API_BEGIN();
  auto config = Json::Load(StringView{c_json_config});
  auto missing = GetMissing(config);
  std::string cache = RequiredArg<String>(config, "cache_prefix", __func__);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", common::OmpGetNumThreads(0));
  *out = new std::shared_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Create(iter, proxy, reset, next, missing, n_threads, cache)};
  API_END();
}

XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallback(
    DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
    XGDMatrixCallbackNext *next, float missing, int nthread,
    int max_bin, DMatrixHandle *out) {
  API_BEGIN();
  *out = new std::shared_ptr<xgboost::DMatrix>{
    xgboost::DMatrix::Create(iter, proxy, reset, next, missing, nthread, max_bin)};
  API_END();
}

XGB_DLL int XGProxyDMatrixCreate(DMatrixHandle* out) {
  API_BEGIN();
  *out = new std::shared_ptr<xgboost::DMatrix>(new xgboost::data::DMatrixProxy);;
  API_END();
}

XGB_DLL int
XGProxyDMatrixSetDataCudaArrayInterface(DMatrixHandle handle,
                                        char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m =   static_cast<xgboost::data::DMatrixProxy*>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetData(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataCudaColumnar(DMatrixHandle handle,
                                              char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m =   static_cast<xgboost::data::DMatrixProxy*>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetData(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataDense(DMatrixHandle handle,
                                       char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m =   static_cast<xgboost::data::DMatrixProxy*>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetArrayData(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataCSR(DMatrixHandle handle, char const *indptr,
                                     char const *indices, char const *data,
                                     xgboost::bst_ulong ncol) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m =   static_cast<xgboost::data::DMatrixProxy*>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetCSRData(indptr, indices, data, ncol, true);
  API_END();
}

// End Create from data iterator

XGB_DLL int XGDMatrixCreateFromCSREx(const size_t* indptr,
                                     const unsigned* indices,
                                     const bst_float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_col,
                                     DMatrixHandle* out) {
  API_BEGIN();
  data::CSRAdapter adapter(indptr, indices, data, nindptr - 1, nelem, num_col);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, std::nan(""), 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSR(char const *indptr,
                                   char const *indices, char const *data,
                                   xgboost::bst_ulong ncol,
                                   char const* c_json_config,
                                   DMatrixHandle* out) {
  API_BEGIN();
  data::CSRArrayAdapter adapter(StringView{indptr}, StringView{indices},
                                StringView{data}, ncol);
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", common::OmpGetNumThreads(0));
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromDense(char const *data,
                                     char const *c_json_config,
                                     DMatrixHandle *out) {
  API_BEGIN();
  xgboost::data::ArrayAdapter adapter{
      xgboost::data::ArrayAdapter(StringView{data})};
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", common::OmpGetNumThreads(0));
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t* col_ptr,
                                     const unsigned* indices,
                                     const bst_float* data,
                                     size_t nindptr,
                                     size_t,
                                     size_t num_row,
                                     DMatrixHandle* out) {
  API_BEGIN();
  data::CSCAdapter adapter(col_ptr, indices, data, nindptr - 1, num_row);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, std::nan(""), 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromMat(const bst_float* data,
                                   xgboost::bst_ulong nrow,
                                   xgboost::bst_ulong ncol, bst_float missing,
                                   DMatrixHandle* out) {
  API_BEGIN();
  data::DenseAdapter adapter(data, nrow, ncol);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromMat_omp(const bst_float* data,  // NOLINT
                                       xgboost::bst_ulong nrow,
                                       xgboost::bst_ulong ncol,
                                       bst_float missing, DMatrixHandle* out,
                                       int nthread) {
  API_BEGIN();
  data::DenseAdapter adapter(data, nrow, ncol);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, nthread));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromDT(void** data, const char** feature_stypes,
                                  xgboost::bst_ulong nrow,
                                  xgboost::bst_ulong ncol, DMatrixHandle* out,
                                  int nthread) {
  API_BEGIN();
  data::DataTableAdapter adapter(data, feature_stypes, nrow, ncol);
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, std::nan(""), nthread));
  API_END();
}

XGB_DLL int XGImportArrowRecordBatch(DataIterHandle data_handle, void *ptr_array,
                                     void *ptr_schema) {
  API_BEGIN();
  static_cast<data::RecordBatchesIterAdapter *>(data_handle)
      ->SetData(static_cast<struct ArrowArray *>(ptr_array),
                static_cast<struct ArrowSchema *>(ptr_schema));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromArrowCallback(XGDMatrixCallbackNext *next, char const *json_config,
                                             DMatrixHandle *out) {
  API_BEGIN();
  auto config = Json::Load(StringView{json_config});
  auto missing = GetMissing(config);
  int32_t n_threads = get<Integer const>(config["nthread"]);
  n_threads = common::OmpGetNumThreads(n_threads);
  data::RecordBatchesIterAdapter adapter(next, n_threads);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                                  const int* idxset,
                                  xgboost::bst_ulong len,
                                  DMatrixHandle* out) {
  return XGDMatrixSliceDMatrixEx(handle, idxset, len, out, 0);
}

XGB_DLL int XGDMatrixSliceDMatrixEx(DMatrixHandle handle,
                                    const int* idxset,
                                    xgboost::bst_ulong len,
                                    DMatrixHandle* out,
                                    int allow_groups) {
  API_BEGIN();
  CHECK_HANDLE();
  if (!allow_groups) {
    CHECK_EQ(static_cast<std::shared_ptr<DMatrix>*>(handle)
                 ->get()
                 ->Info()
                 .group_ptr_.size(),
             0U)
        << "slice does not support group structure";
  }
  DMatrix* dmat = static_cast<std::shared_ptr<DMatrix>*>(handle)->get();
  *out = new std::shared_ptr<DMatrix>(
      dmat->Slice({idxset, static_cast<std::size_t>(len)}));
  API_END();
}

XGB_DLL int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  delete static_cast<std::shared_ptr<DMatrix>*>(handle);
  API_END();
}

XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle, const char* fname,
                                int) {
  API_BEGIN();
  CHECK_HANDLE();
  auto dmat = static_cast<std::shared_ptr<DMatrix>*>(handle)->get();
  if (data::SimpleDMatrix* derived = dynamic_cast<data::SimpleDMatrix*>(dmat)) {
    derived->SaveToLocalFile(fname);
  } else {
    LOG(FATAL) << "binary saving only supported by SimpleDMatrix";
  }
  API_END();
}

XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle, const char *field, const bst_float *info,
                                  xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  auto const& p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, info, xgboost::DataType::kFloat32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle, char const *field,
                                          char const *interface_c_str) {
  API_BEGIN();
  CHECK_HANDLE();
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, interface_c_str);
  API_END();
}

XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle, const char *field, const unsigned *info,
                                 xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, info, xgboost::DataType::kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       const char **c_info,
                                       const xgboost::bst_ulong size) {
  API_BEGIN();
  CHECK_HANDLE();
  auto &info = static_cast<std::shared_ptr<DMatrix> *>(handle)->get()->Info();
  info.SetFeatureInfo(field, c_info, size);
  API_END();
}

XGB_DLL int XGDMatrixGetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       xgboost::bst_ulong *len,
                                       const char ***out_features) {
  API_BEGIN();
  CHECK_HANDLE();
  auto m = *static_cast<std::shared_ptr<DMatrix>*>(handle);
  auto &info = static_cast<std::shared_ptr<DMatrix> *>(handle)->get()->Info();

  std::vector<const char *> &charp_vecs = m->GetThreadLocal().ret_vec_charp;
  std::vector<std::string> &str_vecs = m->GetThreadLocal().ret_vec_str;

  info.GetFeatureInfo(field, &str_vecs);

  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out_features = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGDMatrixSetDenseInfo(DMatrixHandle handle, const char *field, void const *data,
                                  xgboost::bst_ulong size, int type) {
  API_BEGIN();
  CHECK_HANDLE();
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  CHECK(type >= 1 && type <= 4);
  p_fmat->SetInfo(field, data, static_cast<DataType>(type), size);
  API_END();
}

XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle, const unsigned *group, xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  LOG(WARNING) << "XGDMatrixSetGroup is deprecated, use `XGDMatrixSetUIntInfo` instead.";
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo("group", group, xgboost::DataType::kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char* field,
                                  xgboost::bst_ulong* out_len,
                                  const bst_float** out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  info.GetInfo(field, out_len, DataType::kFloat32, reinterpret_cast<void const**>(out_dptr));
  API_END();
}

XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 xgboost::bst_ulong *out_len,
                                 const unsigned **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  info.GetInfo(field, out_len, DataType::kUInt32, reinterpret_cast<void const**>(out_dptr));
  API_END();
}

XGB_DLL int XGDMatrixNumRow(const DMatrixHandle handle,
                            xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  *out = static_cast<xgboost::bst_ulong>(
      static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info().num_row_);
  API_END();
}

XGB_DLL int XGDMatrixNumCol(const DMatrixHandle handle,
                            xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  *out = static_cast<xgboost::bst_ulong>(
      static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info().num_col_);
  API_END();
}

// xgboost implementation
XGB_DLL int XGBoosterCreate(const DMatrixHandle dmats[],
                            xgboost::bst_ulong len,
                            BoosterHandle *out) {
  API_BEGIN();
  std::vector<std::shared_ptr<DMatrix> > mats;
  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    mats.push_back(*static_cast<std::shared_ptr<DMatrix>*>(dmats[i]));
  }
  *out = Learner::Create(mats);
  API_END();
}

XGB_DLL int XGBoosterFree(BoosterHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  delete static_cast<Learner*>(handle);
  API_END();
}

XGB_DLL int XGBoosterSetParam(BoosterHandle handle,
                              const char *name,
                              const char *value) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<Learner*>(handle)->SetParam(name, value);
  API_END();
}

XGB_DLL int XGBoosterGetNumFeature(BoosterHandle handle,
                                   xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<Learner*>(handle)->Configure();
  *out = static_cast<Learner*>(handle)->GetNumFeature();
  API_END();
}

XGB_DLL int XGBoosterBoostedRounds(BoosterHandle handle, int* out) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<Learner*>(handle)->Configure();
  *out = static_cast<Learner*>(handle)->BoostedRounds();
  API_END();
}

XGB_DLL int XGBoosterLoadJsonConfig(BoosterHandle handle, char const* json_parameters) {
  API_BEGIN();
  CHECK_HANDLE();
  Json config { Json::Load(StringView{json_parameters}) };
  static_cast<Learner*>(handle)->LoadConfig(config);
  API_END();
}

XGB_DLL int XGBoosterSaveJsonConfig(BoosterHandle handle,
                                    xgboost::bst_ulong *out_len,
                                    char const** out_str) {
  API_BEGIN();
  CHECK_HANDLE();
  Json config { Object() };
  auto* learner = static_cast<Learner*>(handle);
  learner->Configure();
  learner->SaveConfig(&config);
  std::string& raw_str = learner->GetThreadLocal().ret_str;
  Json::Dump(config, &raw_str);
  *out_str = raw_str.c_str();
  *out_len = static_cast<xgboost::bst_ulong>(raw_str.length());
  API_END();
}

XGB_DLL int XGBoosterUpdateOneIter(BoosterHandle handle,
                                   int iter,
                                   DMatrixHandle dtrain) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* bst = static_cast<Learner*>(handle);
  auto *dtr =
      static_cast<std::shared_ptr<DMatrix>*>(dtrain);

  bst->UpdateOneIter(iter, *dtr);
  API_END();
}

XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle,
                                  DMatrixHandle dtrain,
                                  bst_float *grad,
                                  bst_float *hess,
                                  xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  HostDeviceVector<GradientPair> tmp_gpair;
  auto* bst = static_cast<Learner*>(handle);
  auto* dtr =
      static_cast<std::shared_ptr<DMatrix>*>(dtrain);
  tmp_gpair.Resize(len);
  std::vector<GradientPair>& tmp_gpair_h = tmp_gpair.HostVector();
  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    tmp_gpair_h[i] = GradientPair(grad[i], hess[i]);
  }

  bst->BoostOneIter(0, *dtr, &tmp_gpair);
  API_END();
}

XGB_DLL int XGBoosterEvalOneIter(BoosterHandle handle,
                                 int iter,
                                 DMatrixHandle dmats[],
                                 const char* evnames[],
                                 xgboost::bst_ulong len,
                                 const char** out_str) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* bst = static_cast<Learner*>(handle);
  std::string& eval_str = bst->GetThreadLocal().ret_str;

  std::vector<std::shared_ptr<DMatrix>> data_sets;
  std::vector<std::string> data_names;

  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    data_sets.push_back(*static_cast<std::shared_ptr<DMatrix>*>(dmats[i]));
    data_names.emplace_back(evnames[i]);
  }

  eval_str = bst->EvalOneIter(iter, data_sets, data_names);
  *out_str = eval_str.c_str();
  API_END();
}

XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             int training,
                             xgboost::bst_ulong *len,
                             const bst_float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner*>(handle);
  auto& entry = learner->GetThreadLocal().prediction_entry;
  auto iteration_end = GetIterationFromTreeLimit(ntree_limit, learner);
  learner->Predict(*static_cast<std::shared_ptr<DMatrix> *>(dmat),
                   (option_mask & 1) != 0, &entry.predictions, 0, iteration_end,
                   static_cast<bool>(training), (option_mask & 2) != 0,
                   (option_mask & 4) != 0, (option_mask & 8) != 0,
                   (option_mask & 16) != 0);
  *out_result = dmlc::BeginPtr(entry.predictions.ConstHostVector());
  *len = static_cast<xgboost::bst_ulong>(entry.predictions.Size());
  API_END();
}

XGB_DLL int XGBoosterPredictFromDMatrix(BoosterHandle handle,
                                        DMatrixHandle dmat,
                                        char const* c_json_config,
                                        xgboost::bst_ulong const **out_shape,
                                        xgboost::bst_ulong *out_dim,
                                        bst_float const **out_result) {
  API_BEGIN();
  if (handle == nullptr) {
    LOG(FATAL) << "Booster has not been initialized or has already been disposed.";
  }
  if (dmat == nullptr) {
    LOG(FATAL) << "DMatrix has not been initialized or has already been disposed.";
  }
  auto config = Json::Load(StringView{c_json_config});

  auto *learner = static_cast<Learner*>(handle);
  auto& entry = learner->GetThreadLocal().prediction_entry;
  auto p_m = *static_cast<std::shared_ptr<DMatrix> *>(dmat);

  auto type = PredictionType(RequiredArg<Integer>(config, "type", __func__));
  auto iteration_begin = RequiredArg<Integer>(config, "iteration_begin", __func__);
  auto iteration_end = RequiredArg<Integer>(config, "iteration_end", __func__);

  auto const& j_config = get<Object const>(config);
  auto ntree_limit_it = j_config.find("ntree_limit");
  if (ntree_limit_it != j_config.cend() && !IsA<Null>(ntree_limit_it->second) &&
      get<Integer const>(ntree_limit_it->second) != 0) {
    CHECK(iteration_end == 0) <<
        "Only one of the `ntree_limit` or `iteration_range` can be specified.";
    LOG(WARNING) << "`ntree_limit` is deprecated, use `iteration_range` instead.";
    iteration_end = GetIterationFromTreeLimit(get<Integer const>(ntree_limit_it->second), learner);
  }

  bool approximate = type == PredictionType::kApproxContribution ||
                     type == PredictionType::kApproxInteraction;
  bool contribs = type == PredictionType::kContribution ||
                  type == PredictionType::kApproxContribution;
  bool interactions = type == PredictionType::kInteraction ||
                      type == PredictionType::kApproxInteraction;
  bool training = RequiredArg<Boolean>(config, "training", __func__);
  learner->Predict(p_m, type == PredictionType::kMargin, &entry.predictions,
                   iteration_begin, iteration_end, training,
                   type == PredictionType::kLeaf, contribs, approximate,
                   interactions);
  *out_result = dmlc::BeginPtr(entry.predictions.ConstHostVector());
  auto &shape = learner->GetThreadLocal().prediction_shape;
  auto chunksize = p_m->Info().num_row_ == 0 ? 0 : entry.predictions.Size() / p_m->Info().num_row_;
  auto rounds = iteration_end - iteration_begin;
  rounds = rounds == 0 ? learner->BoostedRounds() : rounds;
  // Determine shape
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);
  CalcPredictShape(strict_shape, type, p_m->Info().num_row_,
                   p_m->Info().num_col_, chunksize, learner->Groups(), rounds,
                   &shape, out_dim);
  *out_shape = dmlc::BeginPtr(shape);
  API_END();
}

template <typename T>
void InplacePredictImpl(std::shared_ptr<T> x, std::shared_ptr<DMatrix> p_m,
                        char const *c_json_config, Learner *learner,
                        size_t n_rows, size_t n_cols,
                        xgboost::bst_ulong const **out_shape,
                        xgboost::bst_ulong *out_dim, const float **out_result) {
  auto config = Json::Load(StringView{c_json_config});
  CHECK_EQ(get<Integer const>(config["cache_id"]), 0) << "Cache ID is not supported yet";

  HostDeviceVector<float>* p_predt { nullptr };
  auto type = PredictionType(RequiredArg<Integer>(config, "type", __func__));
  float missing = GetMissing(config);
  learner->InplacePredict(x, p_m, type, missing, &p_predt,
                          RequiredArg<Integer>(config, "iteration_begin", __func__),
                          RequiredArg<Integer>(config, "iteration_end", __func__));
  CHECK(p_predt);
  auto &shape = learner->GetThreadLocal().prediction_shape;
  auto chunksize = n_rows == 0 ? 0 : p_predt->Size() / n_rows;
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);
  CalcPredictShape(strict_shape, type, n_rows, n_cols, chunksize, learner->Groups(),
                   learner->BoostedRounds(), &shape, out_dim);
  *out_result = dmlc::BeginPtr(p_predt->HostVector());
  *out_shape = dmlc::BeginPtr(shape);
}

// A hidden API as cache id is not being supported yet.
XGB_DLL int XGBoosterPredictFromDense(BoosterHandle handle,
                                      char const *array_interface,
                                      char const *c_json_config,
                                      DMatrixHandle m,
                                      xgboost::bst_ulong const **out_shape,
                                      xgboost::bst_ulong *out_dim,
                                      const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  std::shared_ptr<xgboost::data::ArrayAdapter> x{
      new xgboost::data::ArrayAdapter(StringView{array_interface})};
  std::shared_ptr<DMatrix> p_m {nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  auto *learner = static_cast<xgboost::Learner *>(handle);
  InplacePredictImpl(x, p_m, c_json_config, learner, x->NumRows(),
                     x->NumColumns(), out_shape, out_dim, out_result);
  API_END();
}

// A hidden API as cache id is not being supported yet.
XGB_DLL int XGBoosterPredictFromCSR(BoosterHandle handle, char const *indptr,
                                    char const *indices, char const *data,
                                    xgboost::bst_ulong cols,
                                    char const *c_json_config, DMatrixHandle m,
                                    xgboost::bst_ulong const **out_shape,
                                    xgboost::bst_ulong *out_dim,
                                    const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  std::shared_ptr<xgboost::data::CSRArrayAdapter> x{
      new xgboost::data::CSRArrayAdapter{StringView{indptr},
                                         StringView{indices}, StringView{data},
                                         static_cast<size_t>(cols)}};
  std::shared_ptr<DMatrix> p_m {nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  auto *learner = static_cast<xgboost::Learner *>(handle);
  InplacePredictImpl(x, p_m, c_json_config, learner, x->NumRows(),
                     x->NumColumns(), out_shape, out_dim, out_result);
  API_END();
}

#if !defined(XGBOOST_USE_CUDA)
XGB_DLL int XGBoosterPredictFromCUDAArray(
    BoosterHandle handle, char const *c_json_strs, char const *c_json_config,
    DMatrixHandle m, xgboost::bst_ulong const **out_shape, xgboost::bst_ulong *out_dim,
    const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  common::AssertGPUSupport();
  API_END();
}

XGB_DLL int XGBoosterPredictFromCUDAColumnar(
    BoosterHandle handle, char const *c_json_strs, char const *c_json_config,
    DMatrixHandle m, xgboost::bst_ulong const **out_shape, xgboost::bst_ulong *out_dim,
    const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  common::AssertGPUSupport();
  API_END();
}
#endif  // !defined(XGBOOST_USE_CUDA)

XGB_DLL int XGBoosterLoadModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  CHECK_HANDLE();
  auto read_file = [&]() {
    auto str = common::LoadSequentialFile(fname);
    CHECK_GE(str.size(), 3);  // "{}\0"
    CHECK_EQ(str[0], '{');
    CHECK_EQ(str[str.size() - 2], '}');
    return str;
  };
  if (common::FileExtension(fname) == "json") {
    auto str = read_file();
    Json in{Json::Load(StringView{str})};
    static_cast<Learner*>(handle)->LoadModel(in);
  } else if (common::FileExtension(fname) == "ubj") {
    auto str = read_file();
    Json in = Json::Load(StringView{str}, std::ios::binary);
    static_cast<Learner *>(handle)->LoadModel(in);
  } else {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
    static_cast<Learner*>(handle)->LoadModel(fi.get());
  }
  API_END();
}

namespace {
void WarnOldModel() {
  if (XGBOOST_VER_MAJOR >= 2) {
    LOG(WARNING) << "Saving into deprecated binary model format, please consider using `json` or "
                    "`ubj`. Model format will default to JSON in XGBoost 2.2 if not specified.";
  }
}
}  // anonymous namespace

XGB_DLL int XGBoosterSaveModel(BoosterHandle handle, const char *c_fname) {
  API_BEGIN();
  CHECK_HANDLE();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(c_fname, "w"));
  auto *learner = static_cast<Learner *>(handle);
  learner->Configure();
  auto save_json = [&](std::ios::openmode mode) {
    Json out{Object()};
    learner->SaveModel(&out);
    std::vector<char> str;
    Json::Dump(out, &str, mode);
    fo->Write(str.data(), str.size());
  };
  if (common::FileExtension(c_fname) == "json") {
    save_json(std::ios::out);
  } else if (common::FileExtension(c_fname) == "ubj") {
    save_json(std::ios::binary);
  } else if (XGBOOST_VER_MAJOR == 2 && XGBOOST_VER_MINOR >= 2) {
    LOG(WARNING) << "Saving model to JSON as default.  You can use file extension `json`, `ubj` or "
                    "`deprecated` to choose between formats.";
    save_json(std::ios::out);
  } else {
    WarnOldModel();
    auto *bst = static_cast<Learner *>(handle);
    bst->SaveModel(fo.get());
  }
  API_END();
}

XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle, const void *buf,
                                         xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  common::MemoryFixSizeBuffer fs((void *)buf, len);  // NOLINT(*)
  static_cast<Learner *>(handle)->LoadModel(&fs);
  API_END();
}

XGB_DLL int XGBoosterSaveModelToBuffer(BoosterHandle handle, char const *json_config,
                                       xgboost::bst_ulong *out_len, char const **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  auto config = Json::Load(StringView{json_config});
  auto format = RequiredArg<String>(config, "format", __func__);

  auto *learner = static_cast<Learner *>(handle);
  learner->Configure();

  auto save_json = [&](std::ios::openmode mode) {
    std::vector<char> &raw_char_vec = learner->GetThreadLocal().ret_char_vec;
    Json out{Object{}};
    learner->SaveModel(&out);
    Json::Dump(out, &raw_char_vec, mode);
    *out_dptr = dmlc::BeginPtr(raw_char_vec);
    *out_len = static_cast<xgboost::bst_ulong>(raw_char_vec.size());
  };

  Json out{Object{}};
  if (format == "json") {
    save_json(std::ios::out);
  } else if (format == "ubj") {
    save_json(std::ios::binary);
  } else if (format == "deprecated") {
    WarnOldModel();
    auto &raw_str = learner->GetThreadLocal().ret_str;
    raw_str.clear();
    common::MemoryBufferStream fo(&raw_str);
    learner->SaveModel(&fo);
    *out_dptr = dmlc::BeginPtr(raw_str);
    *out_len = static_cast<xgboost::bst_ulong>(raw_str.size());
  } else {
    LOG(FATAL) << "Unknown format: `" << format << "`";
  }

  API_END();
}

XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle,
                                 xgboost::bst_ulong* out_len,
                                 const char** out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner*>(handle);
  std::string& raw_str = learner->GetThreadLocal().ret_str;
  raw_str.resize(0);

  common::MemoryBufferStream fo(&raw_str);
  LOG(WARNING) << "`" << __func__
               << "` is deprecated, please use `XGBoosterSaveModelToBuffer` instead.";

  learner->Configure();
  learner->SaveModel(&fo);
  *out_dptr = dmlc::BeginPtr(raw_str);
  *out_len = static_cast<xgboost::bst_ulong>(raw_str.length());
  API_END();
}

// The following two functions are `Load` and `Save` for memory based
// serialization methods. E.g. Python pickle.
XGB_DLL int XGBoosterSerializeToBuffer(BoosterHandle handle,
                                       xgboost::bst_ulong *out_len,
                                       const char **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner*>(handle);
  std::string &raw_str = learner->GetThreadLocal().ret_str;
  raw_str.resize(0);
  common::MemoryBufferStream fo(&raw_str);
  learner->Configure();
  learner->Save(&fo);
  *out_dptr = dmlc::BeginPtr(raw_str);
  *out_len = static_cast<xgboost::bst_ulong>(raw_str.length());
  API_END();
}

XGB_DLL int XGBoosterUnserializeFromBuffer(BoosterHandle handle,
                                           const void *buf,
                                           xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
  static_cast<Learner*>(handle)->Load(&fs);
  API_END();
}

XGB_DLL int XGBoosterLoadRabitCheckpoint(BoosterHandle handle,
                                         int* version) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* bst = static_cast<Learner*>(handle);
  *version = rabit::LoadCheckPoint(bst);
  if (*version != 0) {
    bst->Configure();
  }
  API_END();
}

XGB_DLL int XGBoosterSaveRabitCheckpoint(BoosterHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* learner = static_cast<Learner*>(handle);
  learner->Configure();
  if (learner->AllowLazyCheckPoint()) {
    rabit::LazyCheckPoint(learner);
  } else {
    rabit::CheckPoint(learner);
  }
  API_END();
}

XGB_DLL int XGBoosterSlice(BoosterHandle handle, int begin_layer,
                           int end_layer, int step,
                           BoosterHandle *out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* learner = static_cast<Learner*>(handle);
  bool out_of_bound = false;
  auto p_out = learner->Slice(begin_layer, end_layer, step, &out_of_bound);
  if (out_of_bound) {
    return -2;
  }
  CHECK(p_out);
  *out = p_out;
  API_END();
}

inline void XGBoostDumpModelImpl(BoosterHandle handle, FeatureMap* fmap,
                                 int with_stats, const char *format,
                                 xgboost::bst_ulong *len,
                                 const char ***out_models) {
  auto *bst = static_cast<Learner*>(handle);
  bst->Configure();
  GenerateFeatureMap(bst, {}, bst->GetNumFeature(), fmap);

  std::vector<std::string>& str_vecs = bst->GetThreadLocal().ret_vec_str;
  std::vector<const char*>& charp_vecs = bst->GetThreadLocal().ret_vec_charp;
  str_vecs = bst->DumpModel(*fmap, with_stats != 0, format);
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out_models = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
}

XGB_DLL int XGBoosterDumpModel(BoosterHandle handle,
                               const char* fmap,
                               int with_stats,
                               xgboost::bst_ulong* len,
                               const char*** out_models) {
  API_BEGIN();
  CHECK_HANDLE();
  return XGBoosterDumpModelEx(handle, fmap, with_stats, "text", len, out_models);
  API_END();
}

XGB_DLL int XGBoosterDumpModelEx(BoosterHandle handle,
                                 const char* fmap,
                                 int with_stats,
                                 const char *format,
                                 xgboost::bst_ulong* len,
                                 const char*** out_models) {
  API_BEGIN();
  CHECK_HANDLE();
  std::string uri{fmap};
  FeatureMap featmap = LoadFeatureMap(uri);
  XGBoostDumpModelImpl(handle, &featmap, with_stats, format, len, out_models);
  API_END();
}

XGB_DLL int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                           int fnum,
                                           const char** fname,
                                           const char** ftype,
                                           int with_stats,
                                           xgboost::bst_ulong* len,
                                           const char*** out_models) {
  return XGBoosterDumpModelExWithFeatures(handle, fnum, fname, ftype,
                                          with_stats, "text", len, out_models);
}

XGB_DLL int XGBoosterDumpModelExWithFeatures(BoosterHandle handle,
                                             int fnum,
                                             const char** fname,
                                             const char** ftype,
                                             int with_stats,
                                             const char *format,
                                             xgboost::bst_ulong* len,
                                             const char*** out_models) {
  API_BEGIN();
  CHECK_HANDLE();
  FeatureMap featmap;
  for (int i = 0; i < fnum; ++i) {
    featmap.PushBack(i, fname[i], ftype[i]);
  }
  XGBoostDumpModelImpl(handle, &featmap, with_stats, format, len, out_models);
  API_END();
}

XGB_DLL int XGBoosterGetAttr(BoosterHandle handle,
                     const char* key,
                     const char** out,
                     int* success) {
  auto* bst = static_cast<Learner*>(handle);
  std::string& ret_str = bst->GetThreadLocal().ret_str;
  API_BEGIN();
  CHECK_HANDLE();
  if (bst->GetAttr(key, &ret_str)) {
    *out = ret_str.c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
  }
  API_END();
}

XGB_DLL int XGBoosterSetAttr(BoosterHandle handle,
                             const char* key,
                             const char* value) {
  API_BEGIN();
  CHECK_HANDLE();
  auto* bst = static_cast<Learner*>(handle);
  if (value == nullptr) {
    bst->DelAttr(key);
  } else {
    bst->SetAttr(key, value);
  }
  API_END();
}

XGB_DLL int XGBoosterGetAttrNames(BoosterHandle handle,
                                  xgboost::bst_ulong* out_len,
                                  const char*** out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner *>(handle);
  std::vector<std::string> &str_vecs = learner->GetThreadLocal().ret_vec_str;
  std::vector<const char *> &charp_vecs =
      learner->GetThreadLocal().ret_vec_charp;
  str_vecs = learner->GetAttrNames();
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out = dmlc::BeginPtr(charp_vecs);
  *out_len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGBoosterSetStrFeatureInfo(BoosterHandle handle, const char *field,
                                       const char **features,
                                       const xgboost::bst_ulong size) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner *>(handle);
  std::vector<std::string> feature_info;
  for (size_t i = 0; i < size; ++i) {
    feature_info.emplace_back(features[i]);
  }
  if (!std::strcmp(field, "feature_name")) {
    learner->SetFeatureNames(feature_info);
  } else if (!std::strcmp(field, "feature_type")) {
    learner->SetFeatureTypes(feature_info);
  } else {
    LOG(FATAL) << "Unknown field for Booster feature info:" << field;
  }
  API_END();
}

XGB_DLL int XGBoosterGetStrFeatureInfo(BoosterHandle handle, const char *field,
                                       xgboost::bst_ulong *len,
                                       const char ***out_features) {
  API_BEGIN();
  CHECK_HANDLE();
  auto const *learner = static_cast<Learner const *>(handle);
  std::vector<const char *> &charp_vecs =
      learner->GetThreadLocal().ret_vec_charp;
  std::vector<std::string> &str_vecs = learner->GetThreadLocal().ret_vec_str;
  if (!std::strcmp(field, "feature_name")) {
    learner->GetFeatureNames(&str_vecs);
  } else if (!std::strcmp(field, "feature_type")) {
    learner->GetFeatureTypes(&str_vecs);
  } else {
    LOG(FATAL) << "Unknown field for Booster feature info:" << field;
  }
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out_features = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGBoosterFeatureScore(BoosterHandle handle, char const *json_config,
                                  xgboost::bst_ulong *out_n_features,
                                  char const ***out_features,
                                  bst_ulong *out_dim,
                                  bst_ulong const **out_shape,
                                  float const **out_scores) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner *>(handle);
  auto config = Json::Load(StringView{json_config});

  auto importance = RequiredArg<String>(config, "importance_type", __func__);
  std::string feature_map_uri;
  if (!IsA<Null>(config["feature_map"])) {
    feature_map_uri = get<String const>(config["feature_map"]);
  }
  FeatureMap feature_map = LoadFeatureMap(feature_map_uri);
  std::vector<Json> custom_feature_names;
  if (!IsA<Null>(config["feature_names"])) {
    custom_feature_names = get<Array const>(config["feature_names"]);
  }

  std::vector<int32_t> tree_idx;
  if (!IsA<Null>(config["tree_idx"])) {
    auto j_tree_idx = get<Array const>(config["tree_idx"]);
    for (auto const &idx : j_tree_idx) {
      tree_idx.push_back(get<Integer const>(idx));
    }
  }

  auto &scores = learner->GetThreadLocal().ret_vec_float;
  std::vector<bst_feature_t> features;
  learner->CalcFeatureScore(importance, common::Span<int32_t const>(tree_idx), &features, &scores);

  auto n_features = learner->GetNumFeature();
  GenerateFeatureMap(learner, custom_feature_names, n_features, &feature_map);

  auto& feature_names = learner->GetThreadLocal().ret_vec_str;
  feature_names.resize(features.size());
  auto& feature_names_c = learner->GetThreadLocal().ret_vec_charp;
  feature_names_c.resize(features.size());

  for (bst_feature_t i = 0; i < features.size(); ++i) {
    feature_names[i] = feature_map.Name(features[i]);
    feature_names_c[i] = feature_names[i].data();
  }
  *out_n_features = feature_names.size();

  CHECK_LE(features.size(), scores.size());
  auto &shape = learner->GetThreadLocal().prediction_shape;
  if (scores.size() > features.size()) {
    // Linear model multi-class model
    CHECK_EQ(scores.size() % features.size(), 0ul);
    auto n_classes = scores.size() / features.size();
    *out_dim = 2;
    shape = {n_features, n_classes};
  } else {
    CHECK_EQ(features.size(), scores.size());
    *out_dim = 1;
    shape.resize(1);
    shape.front() = scores.size();
  }

  *out_shape = dmlc::BeginPtr(shape);
  *out_scores = scores.data();
  *out_features = dmlc::BeginPtr(feature_names_c);
  API_END();
}

// force link rabit
static DMLC_ATTRIBUTE_UNUSED int XGBOOST_LINK_RABIT_C_API_ = RabitLinkTag();
