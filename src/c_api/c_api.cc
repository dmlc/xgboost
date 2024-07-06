/**
 * Copyright 2014-2024, XGBoost Contributors
 */
#include "xgboost/c_api.h"

#include <algorithm>                         // for copy, transform
#include <cinttypes>                         // for strtoimax
#include <cmath>                             // for nan
#include <cstring>                           // for strcmp
#include <limits>                            // for numeric_limits
#include <map>                               // for operator!=, _Rb_tree_const_iterator, _Rb_tre...
#include <memory>                            // for shared_ptr, allocator, __shared_ptr_access
#include <string>                            // for char_traits, basic_string, operator==, string
#include <system_error>                      // for errc
#include <utility>                           // for pair
#include <vector>                            // for vector

#include "../common/api_entry.h"             // for XGBAPIThreadLocalEntry
#include "../common/charconv.h"              // for from_chars, to_chars, NumericLimits, from_ch...
#include "../common/error_msg.h"             // for NoFederated
#include "../common/hist_util.h"             // for HistogramCuts
#include "../common/io.h"                    // for FileExtension, LoadSequentialFile, MemoryBuf...
#include "../common/threading_utils.h"       // for OmpGetNumThreads, ParallelFor
#include "../data/adapter.h"                 // for ArrayAdapter, DenseAdapter, RecordBatchesIte...
#include "../data/ellpack_page.h"            // for EllpackPage
#include "../data/proxy_dmatrix.h"           // for DMatrixProxy
#include "../data/simple_dmatrix.h"          // for SimpleDMatrix
#include "c_api_error.h"                     // for xgboost_CHECK_C_ARG_PTR, API_END, API_BEGIN
#include "c_api_utils.h"                     // for RequiredArg, OptionalArg, GetMissing, CastDM...
#include "dmlc/base.h"                       // for BeginPtr
#include "dmlc/io.h"                         // for Stream
#include "dmlc/parameter.h"                  // for FieldAccessEntry, FieldEntry, ParamManager
#include "dmlc/thread_local.h"               // for ThreadLocalStore
#include "xgboost/base.h"                    // for bst_ulong, bst_float, GradientPair, bst_feat...
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for DMatrix, MetaInfo, DataType, ExtSparsePage
#include "xgboost/feature_map.h"             // for FeatureMap
#include "xgboost/global_config.h"           // for GlobalConfiguration, GlobalConfigThreadLocal...
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/json.h"                    // for Json, get, Integer, IsA, Boolean, String
#include "xgboost/learner.h"                 // for Learner, PredictionType
#include "xgboost/logging.h"                 // for LOG_FATAL, LogMessageFatal, CHECK, LogCheck_EQ
#include "xgboost/predictor.h"               // for PredictionCacheEntry
#include "xgboost/span.h"                    // for Span
#include "xgboost/string_view.h"             // for StringView, operator<<
#include "xgboost/version_config.h"          // for XGBOOST_VER_MAJOR, XGBOOST_VER_MINOR, XGBOOS...

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

static_assert(DMLC_CXX11_THREAD_LOCAL, "XGBoost depends on thread-local storage.");
using GlobalConfigAPIThreadLocalStore = dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry>;

#if !defined(XGBOOST_USE_CUDA)
namespace xgboost {
void XGBBuildInfoDevice(Json *p_info) {
  auto &info = *p_info;
  info["USE_CUDA"] = Boolean{false};
  info["USE_NCCL"] = Boolean{false};
  info["USE_RMM"] = Boolean{false};
  info["USE_DLOPEN_NCCL"] = Boolean{false};
}
}  // namespace xgboost
#endif

XGB_DLL int XGBuildInfo(char const **out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
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

#if defined(XGBOOST_USE_FEDERATED)
  info["USE_FEDERATED"] = Boolean{true};
#else
  info["USE_FEDERATED"] = Boolean{false};
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

  xgboost_CHECK_C_ARG_PTR(json_str);
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

  xgboost_CHECK_C_ARG_PTR(json_str);
  *json_str = local.ret_str.c_str();
  API_END();
}

XGB_DLL int XGDMatrixCreateFromFile(const char *fname, int silent, DMatrixHandle *out) {
  xgboost_CHECK_C_ARG_PTR(fname);
  xgboost_CHECK_C_ARG_PTR(out);

  Json config{Object()};
  config["uri"] = std::string{fname};
  config["silent"] = silent;
  std::string config_str;
  Json::Dump(config, &config_str);
  return XGDMatrixCreateFromURI(config_str.c_str(), out);
}

XGB_DLL int XGDMatrixCreateFromURI(const char *config, DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(config);
  xgboost_CHECK_C_ARG_PTR(out);

  auto jconfig = Json::Load(StringView{config});
  std::string uri = RequiredArg<String>(jconfig, "uri", __func__);
  auto silent = static_cast<bool>(OptionalArg<Integer, int64_t>(jconfig, "silent", 1));
  auto data_split_mode =
      static_cast<DataSplitMode>(OptionalArg<Integer, int64_t>(jconfig, "data_split_mode", 0));

  *out = new std::shared_ptr<DMatrix>(DMatrix::Load(uri, silent, data_split_mode));
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
  xgboost::data::IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR> adapter(
      data_handle, callback);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix> {
    DMatrix::Create(
        &adapter, std::numeric_limits<float>::quiet_NaN(),
        1, scache
    )
  };
  API_END();
}

#ifndef XGBOOST_USE_CUDA
XGB_DLL int XGDMatrixCreateFromCudaColumnar(char const *, char const *, DMatrixHandle *) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCudaArrayInterface(char const *, char const *, DMatrixHandle *) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}

#endif

// Create from data iterator
XGB_DLL int XGDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                        DataIterResetCallback *reset, XGDMatrixCallbackNext *next,
                                        char const *config, DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(config);

  auto jconfig = Json::Load(StringView{config});
  auto missing = GetMissing(jconfig);
  std::string cache = RequiredArg<String>(jconfig, "cache_prefix", __func__);
  auto n_threads = OptionalArg<Integer, int64_t>(jconfig, "nthread", 0);
  auto on_host = OptionalArg<Boolean, bool>(jconfig, "on_host", false);

  xgboost_CHECK_C_ARG_PTR(next);
  xgboost_CHECK_C_ARG_PTR(reset);
  xgboost_CHECK_C_ARG_PTR(out);

  *out = new std::shared_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Create(iter, proxy, reset, next, missing, n_threads, cache, on_host)};
  API_END();
}

XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                                      DataIterResetCallback *reset,
                                                      XGDMatrixCallbackNext *next, float missing,
                                                      int nthread, int max_bin,
                                                      DMatrixHandle *out) {
  API_BEGIN();
  LOG(WARNING) << error::DeprecatedFunc(__func__, "1.7.0", "XGQuantileDMatrixCreateFromCallback");
  *out = new std::shared_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Create(iter, proxy, nullptr, reset, next, missing, nthread, max_bin)};
  API_END();
}

XGB_DLL int XGQuantileDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                                DataIterHandle ref, DataIterResetCallback *reset,
                                                XGDMatrixCallbackNext *next, char const *config,
                                                DMatrixHandle *out) {
  API_BEGIN();
  std::shared_ptr<DMatrix> _ref{nullptr};
  if (ref) {
    auto pp_ref = static_cast<std::shared_ptr<xgboost::DMatrix> *>(ref);
    StringView err{"Invalid handle to ref."};
    CHECK(pp_ref) << err;
    _ref = *pp_ref;
    CHECK(_ref) << err;
  }

  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});
  auto missing = GetMissing(jconfig);
  auto n_threads = OptionalArg<Integer, int64_t>(jconfig, "nthread", 0);
  auto max_bin = OptionalArg<Integer, int64_t>(jconfig, "max_bin", 256);

  xgboost_CHECK_C_ARG_PTR(next);
  xgboost_CHECK_C_ARG_PTR(reset);
  xgboost_CHECK_C_ARG_PTR(out);

  *out = new std::shared_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Create(iter, proxy, _ref, reset, next, missing, n_threads, max_bin)};
  API_END();
}

XGB_DLL int XGProxyDMatrixCreate(DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<xgboost::DMatrix>(new xgboost::data::DMatrixProxy);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataCudaArrayInterface(DMatrixHandle handle,
                                                    char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(c_interface_str);
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m = static_cast<xgboost::data::DMatrixProxy *>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetCUDAArray(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataCudaColumnar(DMatrixHandle handle, char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(c_interface_str);
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m = static_cast<xgboost::data::DMatrixProxy *>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetCUDAArray(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataColumnar(DMatrixHandle handle, char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(c_interface_str);
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m = static_cast<xgboost::data::DMatrixProxy *>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetColumnarData(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataDense(DMatrixHandle handle, char const *c_interface_str) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(c_interface_str);
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m = static_cast<xgboost::data::DMatrixProxy *>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetArrayData(c_interface_str);
  API_END();
}

XGB_DLL int XGProxyDMatrixSetDataCSR(DMatrixHandle handle, char const *indptr, char const *indices,
                                     char const *data, xgboost::bst_ulong ncol) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(indptr);
  xgboost_CHECK_C_ARG_PTR(indices);
  xgboost_CHECK_C_ARG_PTR(data);
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  auto m = static_cast<xgboost::data::DMatrixProxy *>(p_m->get());
  CHECK(m) << "Current DMatrix type does not support set data.";
  m->SetCSRData(indptr, indices, data, ncol, true);
  API_END();
}

// End Create from data iterator

XGB_DLL int XGDMatrixCreateFromCSREx(const size_t *indptr, const unsigned *indices,
                                     const bst_float *data, size_t nindptr, size_t nelem,
                                     size_t num_col, DMatrixHandle *out) {
  API_BEGIN();
  LOG(WARNING) << error::DeprecatedFunc(__func__, "2.0.0", "XGDMatrixCreateFromCSR");
  data::CSRAdapter adapter(indptr, indices, data, nindptr - 1, nelem, num_col);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, std::nan(""), 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromColumnar(char const *data, char const *c_json_config,
                                        DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  xgboost_CHECK_C_ARG_PTR(data);

  auto config = Json::Load(c_json_config);
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, std::int64_t>(config, "nthread", 0);
  auto data_split_mode =
      static_cast<DataSplitMode>(OptionalArg<Integer, int64_t>(config, "data_split_mode", 0));

  data::ColumnarAdapter adapter{data};
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, missing, n_threads, "", data_split_mode));

  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSR(char const *indptr, char const *indices, char const *data,
                                   xgboost::bst_ulong ncol, char const *c_json_config,
                                   DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(indptr);
  xgboost_CHECK_C_ARG_PTR(indices);
  xgboost_CHECK_C_ARG_PTR(data);
  data::CSRArrayAdapter adapter(StringView{indptr}, StringView{indices}, StringView{data}, ncol);
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", 0);
  auto data_split_mode =
      static_cast<DataSplitMode>(OptionalArg<Integer, int64_t>(config, "data_split_mode", 0));
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, missing, n_threads, "", data_split_mode));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromDense(char const *data,
                                     char const *c_json_config,
                                     DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(data);
  xgboost::data::ArrayAdapter adapter{xgboost::data::ArrayAdapter(StringView{data})};
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", 0);
  auto data_split_mode =
      static_cast<DataSplitMode>(OptionalArg<Integer, int64_t>(config, "data_split_mode", 0));
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, missing, n_threads, "", data_split_mode));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSC(char const *indptr, char const *indices, char const *data,
                                   xgboost::bst_ulong nrow, char const *c_json_config,
                                   DMatrixHandle *out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(indptr);
  xgboost_CHECK_C_ARG_PTR(indices);
  xgboost_CHECK_C_ARG_PTR(data);
  data::CSCArrayAdapter adapter{StringView{indptr}, StringView{indices}, StringView{data},
                                static_cast<std::size_t>(nrow)};
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, int64_t>(config, "nthread", common::OmpGetNumThreads(0));
  auto data_split_mode =
      static_cast<DataSplitMode>(OptionalArg<Integer, int64_t>(config, "data_split_mode", 0));
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, missing, n_threads, "", data_split_mode));

  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t *col_ptr, const unsigned *indices,
                                     const bst_float *data, size_t nindptr, size_t, size_t num_row,
                                     DMatrixHandle *out) {
  API_BEGIN();
  LOG(WARNING) << error::DeprecatedFunc(__func__, "2.0.0", "XGDMatrixCreateFromCSC");
  data::CSCAdapter adapter(col_ptr, indices, data, nindptr - 1, num_row);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, std::nan(""), 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromMat(const bst_float* data,
                                   xgboost::bst_ulong nrow,
                                   xgboost::bst_ulong ncol, bst_float missing,
                                   DMatrixHandle* out) {
  API_BEGIN();
  data::DenseAdapter adapter(data, nrow, ncol);
  xgboost_CHECK_C_ARG_PTR(out);
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
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, nthread));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromDT(void** data, const char** feature_stypes,
                                  xgboost::bst_ulong nrow,
                                  xgboost::bst_ulong ncol, DMatrixHandle* out,
                                  int nthread) {
  API_BEGIN();
  data::DataTableAdapter adapter(data, feature_stypes, nrow, ncol);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, std::nan(""), nthread));
  API_END();
}

XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle, const int *idxset, xgboost::bst_ulong len,
                                  DMatrixHandle *out) {
  xgboost_CHECK_C_ARG_PTR(out);
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
  xgboost_CHECK_C_ARG_PTR(fname);
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
  xgboost_CHECK_C_ARG_PTR(field);
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, linalg::Make1dInterface(info, len));
  API_END();
}

XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle, char const *field,
                                          char const *interface_c_str) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(field);
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, interface_c_str);
  API_END();
}

XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle, const char *field, const unsigned *info,
                                 xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(field);
  LOG(WARNING) << error::DeprecatedFunc(__func__, "2.1.0", "XGDMatrixSetInfoFromInterface");
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  p_fmat->SetInfo(field, linalg::Make1dInterface(info, len));
  API_END();
}

XGB_DLL int XGDMatrixSetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       const char **c_info,
                                       const xgboost::bst_ulong size) {
  API_BEGIN();
  CHECK_HANDLE();
  auto &info = static_cast<std::shared_ptr<DMatrix> *>(handle)->get()->Info();
  xgboost_CHECK_C_ARG_PTR(field);
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

  xgboost_CHECK_C_ARG_PTR(field);
  info.GetFeatureInfo(field, &str_vecs);

  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  xgboost_CHECK_C_ARG_PTR(out_features);
  xgboost_CHECK_C_ARG_PTR(len);
  *out_features = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGDMatrixSetDenseInfo(DMatrixHandle handle, const char *field, void const *data,
                                  xgboost::bst_ulong size, int type) {
  API_BEGIN();
  CHECK_HANDLE();
  LOG(WARNING) << error::DeprecatedFunc(__func__, "2.1.0", "XGDMatrixSetInfoFromInterface");
  auto const &p_fmat = *static_cast<std::shared_ptr<DMatrix> *>(handle);
  CHECK(type >= 1 && type <= 4);
  xgboost_CHECK_C_ARG_PTR(field);

  Context ctx;
  auto dtype = static_cast<DataType>(type);
  std::string str;
  auto proc = [&](auto cast_d_ptr) {
    using T = std::remove_pointer_t<decltype(cast_d_ptr)>;
    auto t = linalg::TensorView<T, 1>(
        common::Span<T>{cast_d_ptr, static_cast<typename common::Span<T>::index_type>(size)},
        {size}, DeviceOrd::CPU());
    CHECK(t.CContiguous());
    Json iface{linalg::ArrayInterface(t)};
    CHECK(ArrayInterface<1>{iface}.is_contiguous);
    str = Json::Dump(iface);
    return str;
  };

  // Legacy code using XGBoost dtype, which is a small subset of array interface types.
  switch (dtype) {
    case xgboost::DataType::kFloat32: {
      auto cast_ptr = reinterpret_cast<const float *>(data);
      p_fmat->Info().SetInfo(ctx, field, proc(cast_ptr));
      break;
    }
    case xgboost::DataType::kDouble: {
      auto cast_ptr = reinterpret_cast<const double *>(data);
      p_fmat->Info().SetInfo(ctx, field, proc(cast_ptr));
      break;
    }
    case xgboost::DataType::kUInt32: {
      auto cast_ptr = reinterpret_cast<const uint32_t *>(data);
      p_fmat->Info().SetInfo(ctx, field, proc(cast_ptr));
      break;
    }
    case xgboost::DataType::kUInt64: {
      auto cast_ptr = reinterpret_cast<const uint64_t *>(data);
      p_fmat->Info().SetInfo(ctx, field, proc(cast_ptr));
      break;
    }
    default:
      LOG(FATAL) << "Unknown data type" << static_cast<uint8_t>(dtype);
  }

  API_END();
}

XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char* field,
                                  xgboost::bst_ulong* out_len,
                                  const bst_float** out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(field);
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_dptr);
  info.GetInfo(field, out_len, DataType::kFloat32, reinterpret_cast<void const**>(out_dptr));
  API_END();
}

XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 xgboost::bst_ulong *out_len,
                                 const unsigned **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(field);
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_dptr);
  info.GetInfo(field, out_len, DataType::kUInt32, reinterpret_cast<void const**>(out_dptr));
  API_END();
}

XGB_DLL int XGDMatrixNumRow(DMatrixHandle handle, xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = CastDMatrixHandle(handle);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<xgboost::bst_ulong>(p_m->Info().num_row_);
  API_END();
}

XGB_DLL int XGDMatrixNumCol(DMatrixHandle handle, xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = CastDMatrixHandle(handle);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<xgboost::bst_ulong>(p_m->Info().num_col_);
  API_END();
}

// We name the function non-missing instead of non-zero since zero is perfectly valid for XGBoost.
XGB_DLL int XGDMatrixNumNonMissing(DMatrixHandle const handle, xgboost::bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = CastDMatrixHandle(handle);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<xgboost::bst_ulong>(p_m->Info().num_nonzero_);
  API_END();
}

XGB_DLL int XGDMatrixDataSplitMode(DMatrixHandle handle, bst_ulong *out) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_m = CastDMatrixHandle(handle);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<xgboost::bst_ulong>(p_m->Info().data_split_mode);
  API_END();
}

XGB_DLL int XGDMatrixGetDataAsCSR(DMatrixHandle const handle, char const *config,
                                  xgboost::bst_ulong *out_indptr, unsigned *out_indices,
                                  float *out_data) {
  API_BEGIN();
  CHECK_HANDLE();

  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});

  auto p_m = CastDMatrixHandle(handle);

  xgboost_CHECK_C_ARG_PTR(out_indptr);
  xgboost_CHECK_C_ARG_PTR(out_indices);
  xgboost_CHECK_C_ARG_PTR(out_data);

  CHECK_LE(p_m->Info().num_col_, std::numeric_limits<unsigned>::max());

  for (auto const &page : p_m->GetBatches<ExtSparsePage>(p_m->Ctx(), BatchParam{})) {
    CHECK(page.page);
    auto const &h_offset = page.page->offset.ConstHostVector();
    std::copy(h_offset.cbegin(), h_offset.cend(), out_indptr);
    auto pv = page.page->GetView();
    common::ParallelFor(page.page->data.Size(), p_m->Ctx()->Threads(), [&](std::size_t i) {
      auto fvalue = pv.data[i].fvalue;
      auto findex = pv.data[i].index;
      out_data[i] = fvalue;
      out_indices[i] = findex;
    });
  }

  API_END();
}

namespace {
template <typename Page>
void GetCutImpl(Context const *ctx, std::shared_ptr<DMatrix> p_m,
                std::vector<std::uint64_t> *p_indptr, std::vector<float> *p_data) {
  auto &indptr = *p_indptr;
  auto &data = *p_data;
  for (auto const &page : p_m->GetBatches<Page>(ctx, {})) {
    auto const &cut = page.Cuts();

    auto const &ptrs = cut.Ptrs();
    indptr.resize(ptrs.size());

    auto const &vals = cut.Values();
    auto const &mins = cut.MinValues();

    bst_feature_t n_features = p_m->Info().num_col_;
    auto ft = p_m->Info().feature_types.ConstHostSpan();
    std::size_t n_categories = std::count_if(ft.cbegin(), ft.cend(),
                                             [](auto t) { return t == FeatureType::kCategorical; });
    data.resize(vals.size() + n_features - n_categories);  // |vals| + |mins|
    std::size_t i{0}, n_numeric{0};
    for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
      CHECK_LT(i, data.size());
      bool is_numeric = !common::IsCat(ft, fidx);
      if (is_numeric) {
        data[i] = mins[fidx];
        i++;
      }
      auto beg = ptrs[fidx];
      auto end = ptrs[fidx + 1];
      CHECK_LE(end, data.size());
      std::copy(vals.cbegin() + beg, vals.cbegin() + end, data.begin() + i);
      i += (end - beg);
      // shift by min values.
      indptr[fidx] = ptrs[fidx] + n_numeric;
      if (is_numeric) {
        n_numeric++;
      }
    }
    CHECK_EQ(n_numeric, n_features - n_categories);

    indptr.back() = data.size();
    CHECK_EQ(indptr.back(), vals.size() + mins.size() - n_categories);
    break;
  }
}
}  // namespace

XGB_DLL int XGDMatrixGetQuantileCut(DMatrixHandle const handle, char const *config,
                                    char const **out_indptr, char const **out_data) {
  API_BEGIN();
  CHECK_HANDLE();

  auto p_m = CastDMatrixHandle(handle);

  xgboost_CHECK_C_ARG_PTR(config);
  xgboost_CHECK_C_ARG_PTR(out_indptr);
  xgboost_CHECK_C_ARG_PTR(out_data);

  auto jconfig = Json::Load(StringView{config});

  if (!p_m->PageExists<GHistIndexMatrix>() && !p_m->PageExists<EllpackPage>()) {
    LOG(FATAL) << "The quantile cut hasn't been generated yet. Unless this is a `QuantileDMatrix`, "
                  "quantile cut is generated during training.";
  }
  // Get return buffer
  auto &data = p_m->GetThreadLocal().ret_vec_float;
  auto &indptr = p_m->GetThreadLocal().ret_vec_u64;

  if (p_m->PageExists<GHistIndexMatrix>()) {
    auto ctx = p_m->Ctx()->IsCPU() ? *p_m->Ctx() : p_m->Ctx()->MakeCPU();
    GetCutImpl<GHistIndexMatrix>(&ctx, p_m, &indptr, &data);
  } else {
    auto ctx = p_m->Ctx()->IsCUDA() ? *p_m->Ctx() : p_m->Ctx()->MakeCUDA(0);
    GetCutImpl<EllpackPage>(&ctx, p_m, &indptr, &data);
  }

  // Create a CPU context
  Context ctx;
  // Get return buffer
  auto &ret_vec_str = p_m->GetThreadLocal().ret_vec_str;
  ret_vec_str.clear();

  ret_vec_str.emplace_back(linalg::ArrayInterfaceStr(
      linalg::MakeTensorView(&ctx, common::Span{indptr.data(), indptr.size()}, indptr.size())));
  ret_vec_str.emplace_back(linalg::ArrayInterfaceStr(
      linalg::MakeTensorView(&ctx, common::Span{data.data(), data.size()}, data.size())));

  auto &charp_vecs = p_m->GetThreadLocal().ret_vec_charp;
  charp_vecs.resize(ret_vec_str.size());
  std::transform(ret_vec_str.cbegin(), ret_vec_str.cend(), charp_vecs.begin(),
                 [](auto const &str) { return str.c_str(); });

  *out_indptr = charp_vecs[0];
  *out_data = charp_vecs[1];
  API_END();
}

// xgboost implementation
XGB_DLL int XGBoosterCreate(const DMatrixHandle dmats[],
                            xgboost::bst_ulong len,
                            BoosterHandle *out) {
  API_BEGIN();
  std::vector<std::shared_ptr<DMatrix> > mats;
  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    xgboost_CHECK_C_ARG_PTR(dmats);
    mats.push_back(*static_cast<std::shared_ptr<DMatrix>*>(dmats[i]));
  }
  xgboost_CHECK_C_ARG_PTR(out);
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
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<Learner*>(handle)->GetNumFeature();
  API_END();
}

XGB_DLL int XGBoosterBoostedRounds(BoosterHandle handle, int* out) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<Learner*>(handle)->Configure();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<Learner*>(handle)->BoostedRounds();
  API_END();
}

XGB_DLL int XGBoosterLoadJsonConfig(BoosterHandle handle, char const* json_parameters) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(json_parameters);
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

  xgboost_CHECK_C_ARG_PTR(out_str);
  xgboost_CHECK_C_ARG_PTR(out_len);

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
  xgboost_CHECK_C_ARG_PTR(dtrain);
  auto *dtr = static_cast<std::shared_ptr<DMatrix> *>(dtrain);
  CHECK(dtr);
  bst->UpdateOneIter(iter, *dtr);
  API_END();
}

XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle, DMatrixHandle dtrain, bst_float *grad,
                                  bst_float *hess, xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  LOG(WARNING) << error::DeprecatedFunc(__func__, "2.1.0", "XGBoosterTrainOneIter");
  auto *learner = static_cast<Learner *>(handle);
  auto ctx = learner->Ctx()->MakeCPU();

  auto t_grad = linalg::MakeTensorView(&ctx, common::Span{grad, static_cast<size_t>(len)}, len);
  auto t_hess = linalg::MakeTensorView(&ctx, common::Span{hess, static_cast<size_t>(len)}, len);

  auto s_grad = linalg::ArrayInterfaceStr(t_grad);
  auto s_hess = linalg::ArrayInterfaceStr(t_hess);

  return XGBoosterTrainOneIter(handle, dtrain, 0, s_grad.c_str(), s_hess.c_str());
  API_END();
}

namespace xgboost {
// copy user-supplied CUDA gradient arrays
void CopyGradientFromCUDAArrays(Context const *, ArrayInterface<2, false> const &,
                                ArrayInterface<2, false> const &, linalg::Matrix<GradientPair> *)
#if !defined(XGBOOST_USE_CUDA)
{
  common::AssertGPUSupport();
}
#else
;  // NOLINT
#endif
}  // namespace xgboost

XGB_DLL int XGBoosterTrainOneIter(BoosterHandle handle, DMatrixHandle dtrain, int iter,
                                  char const *grad, char const *hess) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(grad);
  xgboost_CHECK_C_ARG_PTR(hess);
  auto p_fmat = CastDMatrixHandle(dtrain);
  ArrayInterface<2, false> i_grad{StringView{grad}};
  ArrayInterface<2, false> i_hess{StringView{hess}};
  StringView msg{"Mismatched shape between the gradient and hessian."};
  CHECK_EQ(i_grad.Shape(0), i_hess.Shape(0)) << msg;
  CHECK_EQ(i_grad.Shape(1), i_hess.Shape(1)) << msg;
  linalg::Matrix<GradientPair> gpair;
  auto grad_is_cuda = ArrayInterfaceHandler::IsCudaPtr(i_grad.data);
  auto hess_is_cuda = ArrayInterfaceHandler::IsCudaPtr(i_hess.data);
  CHECK_EQ(i_grad.Shape(0), p_fmat->Info().num_row_)
      << "Mismatched size between the gradient and training data.";
  CHECK_EQ(grad_is_cuda, hess_is_cuda) << "gradient and hessian should be on the same device.";
  auto *learner = static_cast<Learner *>(handle);
  auto ctx = learner->Ctx();
  if (!grad_is_cuda) {
    gpair.Reshape(i_grad.Shape(0), i_grad.Shape(1));
    auto const shape = gpair.Shape();
    auto h_gpair = gpair.HostView();
    DispatchDType(i_grad, DeviceOrd::CPU(), [&](auto &&t_grad) {
      DispatchDType(i_hess, DeviceOrd::CPU(), [&](auto &&t_hess) {
        common::ParallelFor(h_gpair.Size(), ctx->Threads(),
                            detail::CustomGradHessOp{t_grad, t_hess, h_gpair});
      });
    });
  } else {
    CopyGradientFromCUDAArrays(ctx, i_grad, i_hess, &gpair);
  }
  learner->BoostOneIter(iter, p_fmat, &gpair);
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
    xgboost_CHECK_C_ARG_PTR(dmats);
    data_sets.push_back(*static_cast<std::shared_ptr<DMatrix>*>(dmats[i]));
    xgboost_CHECK_C_ARG_PTR(evnames);
    data_names.emplace_back(evnames[i]);
  }

  eval_str = bst->EvalOneIter(iter, data_sets, data_names);
  xgboost_CHECK_C_ARG_PTR(out_str);
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

  xgboost_CHECK_C_ARG_PTR(len);
  xgboost_CHECK_C_ARG_PTR(out_result);

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
  xgboost_CHECK_C_ARG_PTR(c_json_config);
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

  xgboost_CHECK_C_ARG_PTR(out_result);
  *out_result = dmlc::BeginPtr(entry.predictions.ConstHostVector());

  auto &shape = learner->GetThreadLocal().prediction_shape;
  auto chunksize = p_m->Info().num_row_ == 0 ? 0 : entry.predictions.Size() / p_m->Info().num_row_;
  auto rounds = iteration_end - iteration_begin;
  rounds = rounds == 0 ? learner->BoostedRounds() : rounds;
  // Determine shape
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);

  xgboost_CHECK_C_ARG_PTR(out_dim);
  xgboost_CHECK_C_ARG_PTR(out_shape);

  CalcPredictShape(strict_shape, type, p_m->Info().num_row_,
                   p_m->Info().num_col_, chunksize, learner->Groups(), rounds,
                   &shape, out_dim);
  *out_shape = dmlc::BeginPtr(shape);
  API_END();
}

void InplacePredictImpl(std::shared_ptr<DMatrix> p_m, char const *c_json_config, Learner *learner,
                        xgboost::bst_ulong const **out_shape, xgboost::bst_ulong *out_dim,
                        const float **out_result) {
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  auto config = Json::Load(StringView{c_json_config});

  HostDeviceVector<float> *p_predt{nullptr};
  auto type = PredictionType(RequiredArg<Integer>(config, "type", __func__));
  float missing = GetMissing(config);
  learner->InplacePredict(p_m, type, missing, &p_predt,
                          RequiredArg<Integer>(config, "iteration_begin", __func__),
                          RequiredArg<Integer>(config, "iteration_end", __func__));
  CHECK(p_predt);
  auto &shape = learner->GetThreadLocal().prediction_shape;
  auto const &info = p_m->Info();
  auto n_samples = info.num_row_;
  auto n_features = info.num_col_;
  auto chunksize = n_samples == 0 ? 0 : p_predt->Size() / n_samples;
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);

  xgboost_CHECK_C_ARG_PTR(out_dim);
  CalcPredictShape(strict_shape, type, n_samples, n_features, chunksize, learner->Groups(),
                   learner->BoostedRounds(), &shape, out_dim);
  CHECK_GE(p_predt->Size(), n_samples);

  xgboost_CHECK_C_ARG_PTR(out_result);
  xgboost_CHECK_C_ARG_PTR(out_shape);

  *out_result = dmlc::BeginPtr(p_predt->HostVector());
  *out_shape = dmlc::BeginPtr(shape);
}

XGB_DLL int XGBoosterPredictFromDense(BoosterHandle handle, char const *array_interface,
                                      char const *c_json_config, DMatrixHandle m,
                                      xgboost::bst_ulong const **out_shape,
                                      xgboost::bst_ulong *out_dim, const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  std::shared_ptr<DMatrix> p_m{nullptr};
  if (!m) {
    p_m.reset(new data::DMatrixProxy);
  } else {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
  CHECK(proxy) << "Invalid input type for inplace predict.";
  xgboost_CHECK_C_ARG_PTR(array_interface);
  proxy->SetArrayData(array_interface);
  auto *learner = static_cast<xgboost::Learner *>(handle);
  InplacePredictImpl(p_m, c_json_config, learner, out_shape, out_dim, out_result);
  API_END();
}

XGB_DLL int XGBoosterPredictFromColumnar(BoosterHandle handle, char const *array_interface,
                                         char const *c_json_config, DMatrixHandle m,
                                         xgboost::bst_ulong const **out_shape,
                                         xgboost::bst_ulong *out_dim, const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  std::shared_ptr<DMatrix> p_m{nullptr};
  if (!m) {
    p_m.reset(new data::DMatrixProxy);
  } else {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
  CHECK(proxy) << "Invalid input type for inplace predict.";
  xgboost_CHECK_C_ARG_PTR(array_interface);
  proxy->SetColumnarData(array_interface);
  auto *learner = static_cast<xgboost::Learner *>(handle);
  InplacePredictImpl(p_m, c_json_config, learner, out_shape, out_dim, out_result);
  API_END();
}

XGB_DLL int XGBoosterPredictFromCSR(BoosterHandle handle, char const *indptr, char const *indices,
                                    char const *data, xgboost::bst_ulong cols,
                                    char const *c_json_config, DMatrixHandle m,
                                    xgboost::bst_ulong const **out_shape,
                                    xgboost::bst_ulong *out_dim, const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  std::shared_ptr<DMatrix> p_m{nullptr};
  if (!m) {
    p_m.reset(new data::DMatrixProxy);
  } else {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
  CHECK(proxy) << "Invalid input type for inplace predict.";
  xgboost_CHECK_C_ARG_PTR(indptr);
  proxy->SetCSRData(indptr, indices, data, cols, true);
  auto *learner = static_cast<xgboost::Learner *>(handle);
  InplacePredictImpl(p_m, c_json_config, learner, out_shape, out_dim, out_result);
  API_END();
}

#if !defined(XGBOOST_USE_CUDA)
XGB_DLL int XGBoosterPredictFromCUDAArray(BoosterHandle handle, char const *, char const *,
                                          DMatrixHandle, xgboost::bst_ulong const **,
                                          xgboost::bst_ulong *, const float **) {
  API_BEGIN();
  CHECK_HANDLE();
  common::AssertGPUSupport();
  API_END();
}

XGB_DLL int XGBoosterPredictFromCUDAColumnar(BoosterHandle handle, char const *, char const *,
                                             DMatrixHandle, xgboost::bst_ulong const **,
                                             xgboost::bst_ulong *, const float **) {
  API_BEGIN();
  CHECK_HANDLE();
  common::AssertGPUSupport();
  API_END();
}
#endif  // !defined(XGBOOST_USE_CUDA)

XGB_DLL int XGBoosterLoadModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(fname);
  auto read_file = [&]() {
    auto str = common::LoadSequentialFile(fname);
    CHECK_GE(str.size(), 3);  // "{}\0"
    CHECK_EQ(str[0], '{');
    return str;
  };
  if (common::FileExtension(fname) == "json") {
    auto buffer = read_file();
    Json in{Json::Load(StringView{buffer.data(), buffer.size()})};
    static_cast<Learner*>(handle)->LoadModel(in);
  } else if (common::FileExtension(fname) == "ubj") {
    auto buffer = read_file();
    Json in = Json::Load(StringView{buffer.data(), buffer.size()}, std::ios::binary);
    static_cast<Learner *>(handle)->LoadModel(in);
  } else {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
    static_cast<Learner*>(handle)->LoadModel(fi.get());
  }
  API_END();
}

namespace {
void WarnOldModel() {
  LOG(WARNING) << "Saving into deprecated binary model format, please consider using `json` or "
                  "`ubj`. Model format is default to UBJSON in XGBoost 2.1 if not specified.";
}
}  // anonymous namespace

XGB_DLL int XGBoosterSaveModel(BoosterHandle handle, const char *fname) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(fname);

  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  auto *learner = static_cast<Learner *>(handle);
  learner->Configure();
  auto save_json = [&](std::ios::openmode mode) {
    Json out{Object()};
    learner->SaveModel(&out);
    std::vector<char> str;
    Json::Dump(out, &str, mode);
    fo->Write(str.data(), str.size());
  };
  if (common::FileExtension(fname) == "json") {
    save_json(std::ios::out);
  } else if (common::FileExtension(fname) == "ubj") {
    save_json(std::ios::binary);
  } else if (common::FileExtension(fname) == "deprecated") {
    WarnOldModel();
    auto *bst = static_cast<Learner *>(handle);
    bst->SaveModel(fo.get());
  } else {
    LOG(WARNING) << "Saving model in the UBJSON format as default.  You can use file extension:"
                    " `json`, `ubj` or `deprecated` to choose between formats.";
    save_json(std::ios::binary);
  }
  API_END();
}

XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle, const void *buf,
                                         xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(buf);

  common::MemoryFixSizeBuffer fs((void *)buf, len);  // NOLINT(*)
  static_cast<Learner *>(handle)->LoadModel(&fs);
  API_END();
}

XGB_DLL int XGBoosterSaveModelToBuffer(BoosterHandle handle, char const *json_config,
                                       xgboost::bst_ulong *out_len, char const **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();

  xgboost_CHECK_C_ARG_PTR(json_config);
  xgboost_CHECK_C_ARG_PTR(out_dptr);
  xgboost_CHECK_C_ARG_PTR(out_len);

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

// The following two functions are `Load` and `Save` for memory based
// serialization methods. E.g. Python pickle.
XGB_DLL int XGBoosterSerializeToBuffer(BoosterHandle handle, xgboost::bst_ulong *out_len,
                                       const char **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();

  auto *learner = static_cast<Learner *>(handle);
  std::string &raw_str = learner->GetThreadLocal().ret_str;
  raw_str.resize(0);
  common::MemoryBufferStream fo(&raw_str);
  learner->Configure();
  learner->Save(&fo);

  xgboost_CHECK_C_ARG_PTR(out_dptr);
  xgboost_CHECK_C_ARG_PTR(out_len);

  *out_dptr = dmlc::BeginPtr(raw_str);
  *out_len = static_cast<xgboost::bst_ulong>(raw_str.length());
  API_END();
}

XGB_DLL int XGBoosterUnserializeFromBuffer(BoosterHandle handle,
                                           const void *buf,
                                           xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(buf);

  common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
  static_cast<Learner*>(handle)->Load(&fs);
  API_END();
}

XGB_DLL int XGBoosterSlice(BoosterHandle handle, int begin_layer, int end_layer, int step,
                           BoosterHandle *out) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(out);

  auto *learner = static_cast<Learner *>(handle);
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

  xgboost_CHECK_C_ARG_PTR(out_models);
  xgboost_CHECK_C_ARG_PTR(len);

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

  xgboost_CHECK_C_ARG_PTR(fmap);
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
    xgboost_CHECK_C_ARG_PTR(fname);
    xgboost_CHECK_C_ARG_PTR(ftype);
    featmap.PushBack(i, fname[i], ftype[i]);
  }
  XGBoostDumpModelImpl(handle, &featmap, with_stats, format, len, out_models);
  API_END();
}

XGB_DLL int XGBoosterGetAttr(BoosterHandle handle, const char *key, const char **out,
                             int *success) {
  auto* bst = static_cast<Learner*>(handle);
  std::string& ret_str = bst->GetThreadLocal().ret_str;
  API_BEGIN();
  CHECK_HANDLE();

  xgboost_CHECK_C_ARG_PTR(out);
  xgboost_CHECK_C_ARG_PTR(success);

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
  xgboost_CHECK_C_ARG_PTR(key);
  if (value == nullptr) {
    bst->DelAttr(key);
  } else {
    xgboost_CHECK_C_ARG_PTR(value);
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

  xgboost_CHECK_C_ARG_PTR(out);
  xgboost_CHECK_C_ARG_PTR(out_len);

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
  if (size > 0) {
    xgboost_CHECK_C_ARG_PTR(features);
  }
  for (size_t i = 0; i < size; ++i) {
    feature_info.emplace_back(features[i]);
  }

  xgboost_CHECK_C_ARG_PTR(field);
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

  xgboost_CHECK_C_ARG_PTR(out_features);
  xgboost_CHECK_C_ARG_PTR(len);

  *out_features = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGBoosterFeatureScore(BoosterHandle handle, char const *config,
                                  xgboost::bst_ulong *out_n_features, char const ***out_features,
                                  bst_ulong *out_dim, bst_ulong const **out_shape,
                                  float const **out_scores) {
  API_BEGIN();
  CHECK_HANDLE();
  auto *learner = static_cast<Learner *>(handle);
  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});

  auto importance = RequiredArg<String>(jconfig, "importance_type", __func__);
  std::string feature_map_uri;
  if (!IsA<Null>(jconfig["feature_map"])) {
    feature_map_uri = get<String const>(jconfig["feature_map"]);
  }
  FeatureMap feature_map = LoadFeatureMap(feature_map_uri);
  std::vector<Json> custom_feature_names;
  if (!IsA<Null>(jconfig["feature_names"])) {
    custom_feature_names = get<Array const>(jconfig["feature_names"]);
  }

  std::vector<int32_t> tree_idx;
  if (!IsA<Null>(jconfig["tree_idx"])) {
    auto j_tree_idx = get<Array const>(jconfig["tree_idx"]);
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
  xgboost_CHECK_C_ARG_PTR(out_n_features);
  *out_n_features = feature_names.size();

  CHECK_LE(features.size(), scores.size());
  auto &shape = learner->GetThreadLocal().prediction_shape;

  xgboost_CHECK_C_ARG_PTR(out_dim);
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

  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_scores);
  xgboost_CHECK_C_ARG_PTR(out_features);

  *out_shape = dmlc::BeginPtr(shape);
  *out_scores = scores.data();
  *out_features = dmlc::BeginPtr(feature_names_c);
  API_END();
}
