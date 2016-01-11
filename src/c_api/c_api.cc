// Copyright (c) 2014 by Contributors

#include <xgboost/data.h>
#include <xgboost/learner.h>
#include <xgboost/c_api.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

#include "./c_api_error.h"
#include "../data/simple_csr_source.h"
#include "../common/thread_local.h"
#include "../common/math.h"
#include "../common/io.h"
#include "../common/group_data.h"

namespace xgboost {

// booster wrapper for backward compatible reason.
class Booster {
 public:
  explicit Booster(const std::vector<DMatrix*>& cache_mats)
      : configured_(false),
        initialized_(false),
        learner_(Learner::Create(cache_mats)) {}

  inline Learner* learner() {
    return learner_.get();
  }

  inline void SetParam(const std::string& name, const std::string& val) {
    cfg_.push_back(std::make_pair(name, val));
    if (configured_) {
      learner_->Configure(cfg_);
    }
  }

  inline void LazyInit() {
    if (!configured_) {
      learner_->Configure(cfg_);
      configured_ = true;
    }
    if (!initialized_) {
      learner_->InitModel();
      initialized_ = true;
    }
  }

  inline void LoadModel(dmlc::Stream* fi) {
    learner_->Load(fi);
    initialized_ = true;
  }

 public:
  bool configured_;
  bool initialized_;
  std::unique_ptr<Learner> learner_;
  std::vector<std::pair<std::string, std::string> > cfg_;
};
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
  std::vector<float> ret_vec_float;
  /*! \brief temp variable of gradient pairs. */
  std::vector<bst_gpair> tmp_gpair;
};

// define the threadlocal store.
typedef xgboost::common::ThreadLocalStore<XGBAPIThreadLocalEntry> XGBAPIThreadLocalStore;

int XGDMatrixCreateFromFile(const char *fname,
                            int silent,
                            DMatrixHandle *out) {
  API_BEGIN();
  *out = DMatrix::Load(
      fname, silent != 0, false);
  API_END();
}

int XGDMatrixCreateFromCSR(const bst_ulong* indptr,
                           const unsigned *indices,
                           const float* data,
                           bst_ulong nindptr,
                           bst_ulong nelem,
                           DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource& mat = *source;
  mat.row_ptr_.resize(nindptr);
  for (bst_ulong i = 0; i < nindptr; ++i) {
    mat.row_ptr_[i] = static_cast<size_t>(indptr[i]);
  }
  mat.row_data_.resize(nelem);
  for (bst_ulong i = 0; i < nelem; ++i) {
    mat.row_data_[i] = RowBatch::Entry(indices[i], data[i]);
    mat.info.num_col = std::max(mat.info.num_col,
                                static_cast<uint64_t>(indices[i] + 1));
  }
  mat.info.num_row = nindptr - 1;
  mat.info.num_nonzero = static_cast<uint64_t>(nelem);
  *out  = DMatrix::Create(std::move(source));
  API_END();
}

int XGDMatrixCreateFromCSC(const bst_ulong* col_ptr,
                           const unsigned* indices,
                           const float* data,
                           bst_ulong nindptr,
                           bst_ulong nelem,
                           DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  int nthread;
  #pragma omp parallel
  {
    nthread = omp_get_num_threads();
  }
  data::SimpleCSRSource& mat = *source;
  common::ParallelGroupBuilder<RowBatch::Entry> builder(&mat.row_ptr_, &mat.row_data_);
  builder.InitBudget(0, nthread);
  long ncol = static_cast<long>(nindptr - 1);  // NOLINT(*)
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < ncol; ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    for (unsigned j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
      builder.AddBudget(indices[j], tid);
    }
  }
  builder.InitStorage();
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < ncol; ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    for (unsigned j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
      builder.Push(indices[j],
                   RowBatch::Entry(static_cast<bst_uint>(i), data[j]),
                   tid);
    }
}
  mat.info.num_row = mat.row_ptr_.size() - 1;
  mat.info.num_col = static_cast<uint64_t>(ncol);
  mat.info.num_nonzero = nelem;
  *out  = DMatrix::Create(std::move(source));
  API_END();
}

int XGDMatrixCreateFromMat(const float* data,
                           bst_ulong nrow,
                           bst_ulong ncol,
                           float  missing,
                           DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource& mat = *source;
  bool nan_missing = common::CheckNAN(missing);
  mat.info.num_row = nrow;
  mat.info.num_col = ncol;
  for (bst_ulong i = 0; i < nrow; ++i, data += ncol) {
    bst_ulong nelem = 0;
    for (bst_ulong j = 0; j < ncol; ++j) {
      if (common::CheckNAN(data[j])) {
        CHECK(nan_missing)
            << "There are NAN in the matrix, however, you did not set missing=NAN";
      } else {
        if (nan_missing || data[j] != missing) {
          mat.row_data_.push_back(RowBatch::Entry(j, data[j]));
          ++nelem;
        }
      }
    }
    mat.row_ptr_.push_back(mat.row_ptr_.back() + nelem);
  }
  mat.info.num_nonzero = mat.row_data_.size();
  *out  = DMatrix::Create(std::move(source));
  API_END();
}

int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                          const int* idxset,
                          bst_ulong len,
                          DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource src;
  src.CopyFrom(static_cast<DMatrix*>(handle));
  data::SimpleCSRSource& ret = *source;

  CHECK_EQ(src.info.group_ptr.size(), 0)
      << "slice does not support group structure";

  ret.Clear();
  ret.info.num_row = len;
  ret.info.num_col = src.info.num_col;

  dmlc::DataIter<RowBatch>* iter = &src;
  iter->BeforeFirst();
  CHECK(iter->Next());

  const RowBatch& batch = iter->Value();
  for (bst_ulong i = 0; i < len; ++i) {
    const int ridx = idxset[i];
    RowBatch::Inst inst = batch[ridx];
    CHECK_LT(static_cast<bst_ulong>(ridx), batch.size);
    ret.row_data_.resize(ret.row_data_.size() + inst.length);
    std::memcpy(dmlc::BeginPtr(ret.row_data_) + ret.row_ptr_.back(), inst.data,
                sizeof(RowBatch::Entry) * inst.length);
    ret.row_ptr_.push_back(ret.row_ptr_.back() + inst.length);
    ret.info.num_nonzero += inst.length;

    if (src.info.labels.size() != 0) {
      ret.info.labels.push_back(src.info.labels[ridx]);
    }
    if (src.info.weights.size() != 0) {
      ret.info.weights.push_back(src.info.weights[ridx]);
    }
    if (src.info.root_index.size() != 0) {
      ret.info.root_index.push_back(src.info.root_index[ridx]);
    }
  }
  *out  = DMatrix::Create(std::move(source));
  API_END();
}

int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<DMatrix*>(handle);
  API_END();
}

int XGDMatrixSaveBinary(DMatrixHandle handle,
                        const char* fname,
                        int silent) {
  API_BEGIN();
  static_cast<DMatrix*>(handle)->SaveToLocalFile(fname);
  API_END();
}

int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                          const char* field,
                          const float* info,
                          bst_ulong len) {
  API_BEGIN();
  static_cast<DMatrix*>(handle)->info().SetInfo(field, info, kFloat32, len);
  API_END();
}

int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                         const char* field,
                         const unsigned* info,
                         bst_ulong len) {
  API_BEGIN();
  static_cast<DMatrix*>(handle)->info().SetInfo(field, info, kUInt32, len);
  API_END();
}

int XGDMatrixSetGroup(DMatrixHandle handle,
                      const unsigned* group,
                      bst_ulong len) {
  API_BEGIN();
  DMatrix *pmat = static_cast<DMatrix*>(handle);
  MetaInfo& info = pmat->info();
  info.group_ptr.resize(len + 1);
  info.group_ptr[0] = 0;
  for (uint64_t i = 0; i < len; ++i) {
    info.group_ptr[i + 1] = info.group_ptr[i] + group[i];
  }
  API_END();
}

int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                          const char* field,
                          bst_ulong* out_len,
                          const float** out_dptr) {
  API_BEGIN();
  const MetaInfo& info = static_cast<const DMatrix*>(handle)->info();
  const std::vector<float>* vec = nullptr;
  if (!std::strcmp(field, "label")) {
    vec = &info.labels;
  } else if (!std::strcmp(field, "weight")) {
    vec = &info.weights;
  } else if (!std::strcmp(field, "base_margin")) {
    vec = &info.base_margin;
  } else {
    LOG(FATAL) << "Unknown float field name " << field;
  }
  *out_len = static_cast<bst_ulong>(vec->size());
  *out_dptr = dmlc::BeginPtr(*vec);
  API_END();
}

int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                         const char *field,
                         bst_ulong *out_len,
                         const unsigned **out_dptr) {
  API_BEGIN();
  const MetaInfo& info = static_cast<const DMatrix*>(handle)->info();
  const std::vector<unsigned>* vec = nullptr;
  if (!std::strcmp(field, "root_index")) {
    vec = &info.root_index;
  } else {
    LOG(FATAL) << "Unknown uint field name " << field;
  }
  *out_len = static_cast<bst_ulong>(vec->size());
  *out_dptr = dmlc::BeginPtr(*vec);
  API_END();
}

int XGDMatrixNumRow(const DMatrixHandle handle,
                    bst_ulong *out) {
  API_BEGIN();
  *out = static_cast<bst_ulong>(static_cast<const DMatrix*>(handle)->info().num_row);
  API_END();
}

int XGDMatrixNumCol(const DMatrixHandle handle,
                    bst_ulong *out) {
  API_BEGIN();
  *out = static_cast<size_t>(static_cast<const DMatrix*>(handle)->info().num_col);
  API_END();
}

// xgboost implementation
int XGBoosterCreate(DMatrixHandle dmats[],
                    bst_ulong len,
                    BoosterHandle *out) {
  API_BEGIN();
  std::vector<DMatrix*> mats;
  for (bst_ulong i = 0; i < len; ++i) {
    mats.push_back(static_cast<DMatrix*>(dmats[i]));
  }
  *out = new Booster(mats);
  API_END();
}

int XGBoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete static_cast<Booster*>(handle);
  API_END();
}

int XGBoosterSetParam(BoosterHandle handle,
                      const char *name,
                      const char *value) {
  API_BEGIN();
  static_cast<Booster*>(handle)->SetParam(name, value);
  API_END();
}

int XGBoosterUpdateOneIter(BoosterHandle handle,
                           int iter,
                           DMatrixHandle dtrain) {
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  DMatrix *dtr = static_cast<DMatrix*>(dtrain);

  bst->LazyInit();
  bst->learner()->UpdateOneIter(iter, dtr);
  API_END();
}

int XGBoosterBoostOneIter(BoosterHandle handle,
                          DMatrixHandle dtrain,
                          float *grad,
                          float *hess,
                          bst_ulong len) {
  std::vector<bst_gpair>& tmp_gpair = XGBAPIThreadLocalStore::Get()->tmp_gpair;
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  DMatrix* dtr = static_cast<DMatrix*>(dtrain);
  tmp_gpair.resize(len);
  for (bst_ulong i = 0; i < len; ++i) {
    tmp_gpair[i] = bst_gpair(grad[i], hess[i]);
  }

  bst->LazyInit();
  bst->learner()->BoostOneIter(0, dtr, &tmp_gpair);
  API_END();
}

int XGBoosterEvalOneIter(BoosterHandle handle,
                         int iter,
                         DMatrixHandle dmats[],
                         const char* evnames[],
                         bst_ulong len,
                         const char** out_str) {
  std::string& eval_str = XGBAPIThreadLocalStore::Get()->ret_str;
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  std::vector<DMatrix*> data_sets;
  std::vector<std::string> data_names;

  for (bst_ulong i = 0; i < len; ++i) {
    data_sets.push_back(static_cast<DMatrix*>(dmats[i]));
    data_names.push_back(std::string(evnames[i]));
  }

  bst->LazyInit();
  eval_str = bst->learner()->EvalOneIter(iter, data_sets, data_names);
  *out_str = eval_str.c_str();
  API_END();
}

int XGBoosterPredict(BoosterHandle handle,
                     DMatrixHandle dmat,
                     int option_mask,
                     unsigned ntree_limit,
                     bst_ulong *len,
                     const float **out_result) {
  std::vector<float>& preds = XGBAPIThreadLocalStore::Get()->ret_vec_float;
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Predict(
      static_cast<DMatrix*>(dmat),
      (option_mask & 1) != 0,
      &preds, ntree_limit,
      (option_mask & 2) != 0);
  *out_result = dmlc::BeginPtr(preds);
  *len = static_cast<bst_ulong>(preds.size());
  API_END();
}

int XGBoosterLoadModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  static_cast<Booster*>(handle)->LoadModel(fi.get());
  API_END();
}

int XGBoosterSaveModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Save(fo.get());
  API_END();
}

int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                 const void* buf,
                                 bst_ulong len) {
  API_BEGIN();
  common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
  static_cast<Booster*>(handle)->LoadModel(&fs);
  API_END();
}

int XGBoosterGetModelRaw(BoosterHandle handle,
                         bst_ulong* out_len,
                         const char** out_dptr) {
  std::string& raw_str = XGBAPIThreadLocalStore::Get()->ret_str;
  raw_str.resize(0);

  API_BEGIN();
  common::MemoryBufferStream fo(&raw_str);
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Save(&fo);
  *out_dptr = dmlc::BeginPtr(raw_str);
  *out_len = static_cast<bst_ulong>(raw_str.length());
  API_END();
}

inline void XGBoostDumpModelImpl(
    BoosterHandle handle,
    const FeatureMap& fmap,
    int with_stats,
    bst_ulong* len,
    const char*** out_models) {
  std::vector<std::string>& str_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_str;
  std::vector<const char*>& charp_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_charp;
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  str_vecs = bst->learner()->Dump2Text(fmap, with_stats != 0);
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out_models = dmlc::BeginPtr(charp_vecs);
  *len = static_cast<bst_ulong>(charp_vecs.size());
}
int XGBoosterDumpModel(BoosterHandle handle,
                       const char* fmap,
                       int with_stats,
                       bst_ulong* len,
                       const char*** out_models) {
  API_BEGIN();
  FeatureMap featmap;
  if (strlen(fmap) != 0) {
    std::unique_ptr<dmlc::Stream> fs(
        dmlc::Stream::Create(fmap, "r"));
    dmlc::istream is(fs.get());
    featmap.LoadText(is);
  }
  XGBoostDumpModelImpl(handle, featmap, with_stats, len, out_models);
  API_END();
}

int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                   int fnum,
                                   const char** fname,
                                   const char** ftype,
                                   int with_stats,
                                   bst_ulong* len,
                                   const char*** out_models) {
  API_BEGIN();
  FeatureMap featmap;
  for (int i = 0; i < fnum; ++i) {
    featmap.PushBack(i, fname[i], ftype[i]);
  }
  XGBoostDumpModelImpl(handle, featmap, with_stats, len, out_models);
  API_END();
}
