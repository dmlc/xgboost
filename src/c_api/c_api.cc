// Copyright (c) 2014-2020 by Contributors
#include <dmlc/thread_local.h>
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
#include "xgboost/learner.h"
#include "xgboost/c_api.h"
#include "xgboost/logging.h"
#include "xgboost/version_config.h"
#include "xgboost/json.h"

#include "c_api_error.h"
#include "../common/io.h"
#include "../data/adapter.h"
#include "../data/simple_dmatrix.h"

using namespace xgboost; // NOLINT(*);

/*! \brief entry to to easily hold returning information */
struct XGBAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief returning float vector. */
  std::vector<bst_float> ret_vec_float;
  /*! \brief temp variable of gradient pairs. */
  std::vector<GradientPair> tmp_gpair;
};

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

// define the threadlocal store.
using XGBAPIThreadLocalStore = dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry>;

int XGBRegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->Register(callback);
  API_END();
}

int XGDMatrixCreateFromFile(const char *fname,
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
    void *data_handle,                  // a Java interator
    XGBCallbackDataIterNext *callback,  // C++ callback defined in xgboost4j.cpp
    const char *cache_info, DMatrixHandle *out) {
  API_BEGIN();

  std::string scache;
  if (cache_info != nullptr) {
    scache = cache_info;
  }
  xgboost::data::IteratorAdapter adapter(data_handle, callback);
  *out = new std::shared_ptr<DMatrix> {
    DMatrix::Create(
        &adapter, std::numeric_limits<float>::quiet_NaN(),
        1, scache
    )
  };
  API_END();
}

#ifndef XGBOOST_USE_CUDA
XGB_DLL int XGDMatrixCreateFromArrayInterfaceColumns(char const* c_json_strs,
                                                     bst_float missing,
                                                     int nthread,
                                                     DMatrixHandle* out) {
  API_BEGIN();
  LOG(FATAL) << "Xgboost not compiled with cuda";
  API_END();
}

XGB_DLL int XGDMatrixCreateFromArrayInterface(char const* c_json_strs,
                                                     bst_float missing,
                                                     int nthread,
                                                     DMatrixHandle* out) {
  API_BEGIN();
  LOG(FATAL) << "Xgboost not compiled with cuda";
  API_END();
}

#endif

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

XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t* col_ptr,
                                     const unsigned* indices,
                                     const bst_float* data,
                                     size_t nindptr,
                                     size_t nelem,
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
  CHECK(dynamic_cast<data::SimpleDMatrix*>(dmat))
      << "Slice only supported for SimpleDMatrix currently.";
  data::DMatrixSliceAdapter adapter(dmat, {idxset, static_cast<size_t>(len)});
  *out = new std::shared_ptr<DMatrix>(
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1));
  API_END();
}

XGB_DLL int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  delete static_cast<std::shared_ptr<DMatrix>*>(handle);
  API_END();
}

XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle, const char* fname,
                                int silent) {
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

XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                                  const char* field,
                                  const bst_float* info,
                                  xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo(field, info, xgboost::DataType::kFloat32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle,
                                          char const* field,
                                          char const* interface_c_str) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo(field, interface_c_str);
  API_END();
}

XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                                 const char* field,
                                 const unsigned* info,
                                 xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo(field, info, xgboost::DataType::kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned* group,
                              xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  LOG(WARNING) << "XGDMatrixSetGroup is deprecated, use `XGDMatrixSetUIntInfo` instead.";
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo("group", group, xgboost::DataType::kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char* field,
                                  xgboost::bst_ulong* out_len,
                                  const bst_float** out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  const std::vector<bst_float>* vec = nullptr;
  if (!std::strcmp(field, "label")) {
    vec = &info.labels_.HostVector();
  } else if (!std::strcmp(field, "weight")) {
    vec = &info.weights_.HostVector();
  } else if (!std::strcmp(field, "base_margin")) {
    vec = &info.base_margin_.HostVector();
  } else {
    LOG(FATAL) << "Unknown float field name " << field;
  }
  *out_len = static_cast<xgboost::bst_ulong>(vec->size());  // NOLINT
  *out_dptr = dmlc::BeginPtr(*vec);
  API_END();
}

XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 xgboost::bst_ulong *out_len,
                                 const unsigned **out_dptr) {
  API_BEGIN();
  CHECK_HANDLE();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  const std::vector<unsigned>* vec = nullptr;
  if (!std::strcmp(field, "group_ptr")) {
    vec = &info.group_ptr_;
  } else {
    LOG(FATAL) << "Unknown comp uint field name " << field
      << " with comparison " << std::strcmp(field, "group_ptr");
  }
  *out_len = static_cast<xgboost::bst_ulong>(vec->size());
  *out_dptr = dmlc::BeginPtr(*vec);
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

XGB_DLL int XGBoosterLoadJsonConfig(BoosterHandle handle, char const* json_parameters) {
  API_BEGIN();
  CHECK_HANDLE();
  std::string str {json_parameters};
  Json config { Json::Load(StringView{str.c_str(), str.size()}) };
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
  std::string& raw_str = XGBAPIThreadLocalStore::Get()->ret_str;
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
  HostDeviceVector<GradientPair> tmp_gpair;
  API_BEGIN();
  CHECK_HANDLE();
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
  std::string& eval_str = XGBAPIThreadLocalStore::Get()->ret_str;
  API_BEGIN();
  CHECK_HANDLE();
  auto* bst = static_cast<Learner*>(handle);
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
                             int32_t training,
                             xgboost::bst_ulong *len,
                             const bst_float **out_result) {
  std::vector<bst_float>& preds =
      XGBAPIThreadLocalStore::Get()->ret_vec_float;
  API_BEGIN();
  CHECK_HANDLE();
  auto *bst = static_cast<Learner*>(handle);
  HostDeviceVector<bst_float> tmp_preds;
  bst->Predict(
      *static_cast<std::shared_ptr<DMatrix>*>(dmat),
      (option_mask & 1) != 0,
      &tmp_preds, ntree_limit,
      static_cast<bool>(training),
      (option_mask & 2) != 0,
      (option_mask & 4) != 0,
      (option_mask & 8) != 0,
      (option_mask & 16) != 0);
  preds = tmp_preds.HostVector();
  *out_result = dmlc::BeginPtr(preds);
  *len = static_cast<xgboost::bst_ulong>(preds.size());
  API_END();
}

XGB_DLL int XGBoosterLoadModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  CHECK_HANDLE();
  if (common::FileExtension(fname) == "json") {
    auto str = common::LoadSequentialFile(fname);
    CHECK_GT(str.size(), 2);
    CHECK_EQ(str[0], '{');
    Json in { Json::Load({str.c_str(), str.size()}) };
    static_cast<Learner*>(handle)->LoadModel(in);
  } else {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
    static_cast<Learner*>(handle)->LoadModel(fi.get());
  }
  API_END();
}

XGB_DLL int XGBoosterSaveModel(BoosterHandle handle, const char* c_fname) {
  API_BEGIN();
  CHECK_HANDLE();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(c_fname, "w"));
  auto *learner = static_cast<Learner *>(handle);
  learner->Configure();
  if (common::FileExtension(c_fname) == "json") {
    Json out { Object() };
    learner->SaveModel(&out);
    std::string str;
    Json::Dump(out, &str);
    fo->Write(str.c_str(), str.size());
  } else {
    auto *bst = static_cast<Learner*>(handle);
    bst->SaveModel(fo.get());
  }
  API_END();
}

XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                         const void* buf,
                                         xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
  static_cast<Learner*>(handle)->LoadModel(&fs);
  API_END();
}

XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle,
                                 xgboost::bst_ulong* out_len,
                                 const char** out_dptr) {
  std::string& raw_str = XGBAPIThreadLocalStore::Get()->ret_str;
  raw_str.resize(0);

  API_BEGIN();
  CHECK_HANDLE();
  common::MemoryBufferStream fo(&raw_str);
  auto *learner = static_cast<Learner*>(handle);
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
  std::string &raw_str = XGBAPIThreadLocalStore::Get()->ret_str;
  raw_str.resize(0);

  API_BEGIN();
  CHECK_HANDLE();
  common::MemoryBufferStream fo(&raw_str);
  auto *learner = static_cast<Learner*>(handle);
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

inline void XGBoostDumpModelImpl(
    BoosterHandle handle,
    const FeatureMap& fmap,
    int with_stats,
    const char *format,
    xgboost::bst_ulong* len,
    const char*** out_models) {
  std::vector<std::string>& str_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_str;
  std::vector<const char*>& charp_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_charp;
  auto *bst = static_cast<Learner*>(handle);
  bst->Configure();
  str_vecs = bst->DumpModel(fmap, with_stats != 0, format);
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
  return XGBoosterDumpModelEx(handle, fmap, with_stats, "text", len, out_models);
}

XGB_DLL int XGBoosterDumpModelEx(BoosterHandle handle,
                                 const char* fmap,
                                 int with_stats,
                                 const char *format,
                                 xgboost::bst_ulong* len,
                                 const char*** out_models) {
  API_BEGIN();
  CHECK_HANDLE();
  FeatureMap featmap;
  if (strlen(fmap) != 0) {
    std::unique_ptr<dmlc::Stream> fs(
        dmlc::Stream::Create(fmap, "r"));
    dmlc::istream is(fs.get());
    featmap.LoadText(is);
  }
  XGBoostDumpModelImpl(handle, featmap, with_stats, format, len, out_models);
  API_END();
}

XGB_DLL int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                           int fnum,
                                           const char** fname,
                                           const char** ftype,
                                           int with_stats,
                                           xgboost::bst_ulong* len,
                                           const char*** out_models) {
  return XGBoosterDumpModelExWithFeatures(handle, fnum, fname, ftype, with_stats,
                                          "text", len, out_models);
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
  XGBoostDumpModelImpl(handle, featmap, with_stats, format, len, out_models);
  API_END();
}

XGB_DLL int XGBoosterGetAttr(BoosterHandle handle,
                     const char* key,
                     const char** out,
                     int* success) {
  auto* bst = static_cast<Learner*>(handle);
  std::string& ret_str = XGBAPIThreadLocalStore::Get()->ret_str;
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
  auto* bst = static_cast<Learner*>(handle);
  API_BEGIN();
  CHECK_HANDLE();
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
  std::vector<std::string>& str_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_str;
  std::vector<const char*>& charp_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_charp;
  auto *bst = static_cast<Learner*>(handle);
  API_BEGIN();
  CHECK_HANDLE();
  str_vecs = bst->GetAttrNames();
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out = dmlc::BeginPtr(charp_vecs);
  *out_len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

// force link rabit
static DMLC_ATTRIBUTE_UNUSED int XGBOOST_LINK_RABIT_C_API_ = RabitLinkTag();
