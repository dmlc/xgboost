// Copyright (c) 2014-2019 by Contributors
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


#include "xgboost/data.h"
#include "xgboost/learner.h"
#include "xgboost/c_api.h"
#include "xgboost/logging.h"
#include "xgboost/version_config.h"
#include "xgboost/json.h"

#include "c_api_error.h"
#include "../data/simple_csr_source.h"
#include "../common/io.h"
#include "../data/adapter.h"

namespace xgboost {
// declare the data callback.
XGB_EXTERN_C int XGBoostNativeDataIterSetData(
    void *handle, XGBoostBatchCSR batch);

/*! \brief Native data iterator that takes callback to return data */
class NativeDataIter : public dmlc::Parser<uint32_t> {
 public:
  NativeDataIter(DataIterHandle data_handle,
                 XGBCallbackDataIterNext* next_callback)
      :  at_first_(true), bytes_read_(0),
         data_handle_(data_handle), next_callback_(next_callback) {
  }

  // override functions
  void BeforeFirst() override {
    CHECK(at_first_) << "cannot reset NativeDataIter";
  }

  bool Next() override {
    if ((*next_callback_)(data_handle_,
                          XGBoostNativeDataIterSetData,
                          this) != 0) {
      at_first_ = false;
      return true;
    } else {
      return false;
    }
  }

  const dmlc::RowBlock<uint32_t>& Value() const override {
    return block_;
  }

  size_t BytesRead() const override {
    return bytes_read_;
  }

  // callback to set the data
  void SetData(const XGBoostBatchCSR& batch) {
    offset_.clear();
    label_.clear();
    weight_.clear();
    index_.clear();
    value_.clear();
    offset_.insert(offset_.end(), batch.offset, batch.offset + batch.size + 1);
    if (batch.label != nullptr) {
      label_.insert(label_.end(), batch.label, batch.label + batch.size);
    }
    if (batch.weight != nullptr) {
      weight_.insert(weight_.end(), batch.weight, batch.weight + batch.size);
    }
    if (batch.index != nullptr) {
      index_.insert(index_.end(), batch.index + offset_[0], batch.index + offset_.back());
    }
    if (batch.value != nullptr) {
      value_.insert(value_.end(), batch.value + offset_[0], batch.value + offset_.back());
    }
    if (offset_[0] != 0) {
      size_t base = offset_[0];
      for (size_t& item : offset_) {
        item -= base;
      }
    }
    block_.size = batch.size;
    block_.offset = dmlc::BeginPtr(offset_);
    block_.label = dmlc::BeginPtr(label_);
    block_.weight = dmlc::BeginPtr(weight_);
    block_.qid = nullptr;
    block_.field = nullptr;
    block_.index = dmlc::BeginPtr(index_);
    block_.value = dmlc::BeginPtr(value_);
    bytes_read_ += offset_.size() * sizeof(size_t) +
        label_.size() * sizeof(dmlc::real_t) +
        weight_.size() * sizeof(dmlc::real_t) +
        index_.size() * sizeof(uint32_t) +
        value_.size() * sizeof(dmlc::real_t);
  }

 private:
  // at the beinning.
  bool at_first_;
  // bytes that is read.
  size_t bytes_read_;
  // handle to the iterator,
  DataIterHandle data_handle_;
  // call back to get the data.
  XGBCallbackDataIterNext* next_callback_;
  // internal offset
  std::vector<size_t> offset_;
  // internal label data
  std::vector<dmlc::real_t> label_;
  // internal weight data
  std::vector<dmlc::real_t> weight_;
  // internal index.
  std::vector<uint32_t> index_;
  // internal value.
  std::vector<dmlc::real_t> value_;
  // internal Rowblock
  dmlc::RowBlock<uint32_t> block_;
};

int XGBoostNativeDataIterSetData(
    void *handle, XGBoostBatchCSR batch) {
  API_BEGIN();
  static_cast<xgboost::NativeDataIter*>(handle)->SetData(batch);
  API_END();
}
}  // namespace xgboost

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

int XGDMatrixCreateFromDataIter(
    void* data_handle,
    XGBCallbackDataIterNext* callback,
    const char *cache_info,
    DMatrixHandle *out) {
  API_BEGIN();

  std::string scache;
  if (cache_info != nullptr) {
    scache = cache_info;
  }
  NativeDataIter parser(data_handle, callback);
  data::FileAdapter adapter(&parser);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(
      &adapter, std::numeric_limits<float>::quiet_NaN(), 1, scache));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromArrayInterfaces(
    char const* c_json_strs, bst_int has_missing, bst_float missing, DMatrixHandle* out) {
  API_BEGIN();
  std::string json_str {c_json_strs};
  std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
  source->CopyFrom(json_str, has_missing, missing);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

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
  data::DenseAdapter adapter(data, nrow, nrow * ncol, ncol);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, 1));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromMat_omp(const bst_float* data,  // NOLINT
                                       xgboost::bst_ulong nrow,
                                       xgboost::bst_ulong ncol,
                                       bst_float missing, DMatrixHandle* out,
                                       int nthread) {
  API_BEGIN();
  data::DenseAdapter adapter(data, nrow, nrow * ncol, ncol);
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
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  CHECK_HANDLE();
  data::SimpleCSRSource src;
  src.CopyFrom(static_cast<std::shared_ptr<DMatrix>*>(handle)->get());
  data::SimpleCSRSource& ret = *source;

  if (!allow_groups) {
    CHECK_EQ(src.info.group_ptr_.size(), 0U)
      << "slice does not support group structure";
  }

  ret.Clear();
  ret.info.num_row_ = len;
  ret.info.num_col_ = src.info.num_col_;

  auto iter = &src;
  iter->BeforeFirst();
  CHECK(iter->Next());

  const auto& batch = iter->Value();
  const auto& src_labels = src.info.labels_.ConstHostVector();
  const auto& src_weights = src.info.weights_.ConstHostVector();
  const auto& src_base_margin = src.info.base_margin_.ConstHostVector();
  auto& ret_labels = ret.info.labels_.HostVector();
  auto& ret_weights = ret.info.weights_.HostVector();
  auto& ret_base_margin = ret.info.base_margin_.HostVector();
  auto& offset_vec = ret.page_.offset.HostVector();
  auto& data_vec = ret.page_.data.HostVector();

  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    const int ridx = idxset[i];
    auto inst = batch[ridx];
    CHECK_LT(static_cast<xgboost::bst_ulong>(ridx), batch.Size());
    data_vec.insert(data_vec.end(), inst.data(),
                    inst.data() + inst.size());
    offset_vec.push_back(offset_vec.back() + inst.size());
    ret.info.num_nonzero_ += inst.size();

    if (src_labels.size() != 0) {
      ret_labels.push_back(src_labels[ridx]);
    }
    if (src_weights.size() != 0) {
      ret_weights.push_back(src_weights[ridx]);
    }
    if (src_base_margin.size() != 0) {
      ret_base_margin.push_back(src_base_margin[ridx]);
    }
  }
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

XGB_DLL int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  delete static_cast<std::shared_ptr<DMatrix>*>(handle);
  API_END();
}

XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle,
                                const char* fname,
                                int silent) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->SaveToLocalFile(fname);
  API_END();
}

XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                                  const char* field,
                                  const bst_float* info,
                                  xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo(field, info, kFloat32, len);
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
      ->get()->Info().SetInfo(field, info, kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned* group,
                              xgboost::bst_ulong len) {
  API_BEGIN();
  CHECK_HANDLE();
  LOG(WARNING) << "XGDMatrixSetGroup is deprecated, use `XGDMatrixSetUIntInfo` instead.";
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->Info().SetInfo("group", group, kUInt32, len);
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

  bst->UpdateOneIter(iter, dtr->get());
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

  bst->BoostOneIter(0, dtr->get(), &tmp_gpair);
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
  std::vector<DMatrix*> data_sets;
  std::vector<std::string> data_names;

  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    data_sets.push_back(static_cast<std::shared_ptr<DMatrix>*>(dmats[i])->get());
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
                             xgboost::bst_ulong *len,
                             const bst_float **out_result) {
  std::vector<bst_float>&preds =
    XGBAPIThreadLocalStore::Get()->ret_vec_float;
  API_BEGIN();
  CHECK_HANDLE();
  auto *bst = static_cast<Learner*>(handle);
  HostDeviceVector<bst_float> tmp_preds;
  bst->Predict(
      static_cast<std::shared_ptr<DMatrix>*>(dmat)->get(),
      (option_mask & 1) != 0,
      &tmp_preds, ntree_limit,
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
