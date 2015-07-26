// Copyright (c) 2014 by Contributors
// implementations in ctypes
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <exception>
// include all std functions
using namespace std;
#include "./xgboost_wrapper.h"
#include "../src/data.h"
#include "../src/learner/learner-inl.hpp"
#include "../src/io/io.h"
#include "../src/utils/utils.h"
#include "../src/utils/math.h"
#include "../src/utils/group_data.h"
#include "../src/io/simple_dmatrix-inl.hpp"

using namespace xgboost;
using namespace xgboost::io;

namespace xgboost {
namespace wrapper {
// booster wrapper class
class Booster: public learner::BoostLearner {
 public:
  explicit Booster(const std::vector<DataMatrix*>& mats) {
    this->silent = 1;
    this->init_model = false;
    this->SetCacheData(mats);
  }
  inline const float *Pred(const DataMatrix &dmat, int option_mask,
                           unsigned ntree_limit, bst_ulong *len) {
    this->CheckInitModel();
    this->Predict(dmat, (option_mask&1) != 0, &this->preds_,
                  ntree_limit, (option_mask&2) != 0);
    *len = static_cast<bst_ulong>(this->preds_.size());
    return BeginPtr(this->preds_);
  }
  inline void BoostOneIter(const DataMatrix &train,
                           float *grad, float *hess, bst_ulong len) {
    this->gpair_.resize(len);
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(len);
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      gpair_[j] = bst_gpair(grad[j], hess[j]);
    }
    gbm_->DoBoost(train.fmat(), this->FindBufferOffset(train), train.info.info, &gpair_);
  }
  inline void CheckInitModel(void) {
    if (!init_model) {
      this->InitModel(); init_model = true;
    }
  }
  inline void LoadModel(const char *fname) {
    learner::BoostLearner::LoadModel(fname);
    this->init_model = true;
  }
  inline void LoadModelFromBuffer(const void *buf, size_t size) {
    utils::MemoryFixSizeBuffer fs((void*)buf, size);  // NOLINT(*)
    learner::BoostLearner::LoadModel(fs, true);
    this->init_model = true;
  }
  inline const char *GetModelRaw(bst_ulong *out_len) {
    this->CheckInitModel();
    model_str.resize(0);
    utils::MemoryBufferStream fs(&model_str);
    learner::BoostLearner::SaveModel(fs, false);
    *out_len = static_cast<bst_ulong>(model_str.length());
    if (*out_len == 0) {
      return NULL;
    } else {
      return &model_str[0];
    }
  }
  inline const char** GetModelDump(const utils::FeatMap& fmap, bool with_stats, bst_ulong *len) {
    model_dump = this->DumpModel(fmap, with_stats);
    model_dump_cptr.resize(model_dump.size());
    for (size_t i = 0; i < model_dump.size(); ++i) {
      model_dump_cptr[i] = model_dump[i].c_str();
    }
    *len = static_cast<bst_ulong>(model_dump.size());
    return BeginPtr(model_dump_cptr);
  }
  // temporal fields
  // temporal data to save evaluation dump
  std::string eval_str;
  // temporal data to save model dump
  std::string model_str;
  // temporal space to save model dump
  std::vector<std::string> model_dump;
  std::vector<const char*> model_dump_cptr;

 private:
  bool init_model;
};
}  // namespace wrapper
}  // namespace xgboost

using namespace xgboost::wrapper;

#ifndef XGBOOST_STRICT_CXX98_
namespace xgboost {
namespace wrapper {
// helper to support threadlocal
struct ThreadLocalStore {
  std::vector<std::string*> data;
  // allocate a string
  inline std::string *Alloc() {
    mutex.Lock();
    data.push_back(new std::string());
    std::string *ret = data.back();
    mutex.Unlock();
    return ret;
  }
  ThreadLocalStore() {
    mutex.Init();
  }
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data.size(); ++i) {
      delete data[i];
    }
    mutex.Destroy();
  }
  utils::Mutex mutex;
};

static ThreadLocalStore thread_local_store;
}  // namespace wrapper
}  // namespace xgboost

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*!
 * \brief every function starts with API_BEGIN(); and finishes with API_END();
 * \param Finalize optionally put in a finalizer
 */
#define API_END_FINALIZE(Finalize) } catch(std::exception &e) {  \
    Finalize; return XGBHandleException(e);             \
  } return 0;
/*! \brief API End with no finalization */
#define API_END() API_END_FINALIZE(;)

// do not use threadlocal on OSX since it is not always available
#ifndef DISABLE_THREAD_LOCAL
#ifdef __GNUC__
  #define XGB_TREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
  #define XGB_TREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
  #define XGB_TREAD_LOCAL __declspec(thread)
#endif
#endif

#ifndef XGB_TREAD_LOCAL
#pragma message("Warning: Threadlocal not enabled, used single thread error handling")
#define XGB_TREAD_LOCAL
#endif

/*!
 * \brief a helper function for error handling
 *  will set the last error to be str_set when it is not NULL
 * \param str_set the error to set
 * \return a pointer message to last error
 */
const char *XGBSetGetLastError_(const char *str_set) {
  // use last_error to record last error
  static XGB_TREAD_LOCAL std::string *last_error = NULL;
  if (last_error == NULL) {
    last_error = thread_local_store.Alloc();
  }
  if (str_set != NULL) {
    *last_error = str_set;
  }
  return last_error->c_str();
}
#else
// crippled implementation for solaris case
// exception handling is not needed for R, so it is OK.
#define API_BEGIN()
#define API_END_FINALIZE(Finalize) return 0
#define API_END() return 0

const char *XGBSetGetLastError_(const char *str_set) {
  return NULL;
}
#endif  // XGBOOST_STRICT_CXX98_

/*! \brief return str message of the last error */
const char *XGBGetLastError() {
  return XGBSetGetLastError_(NULL);
}

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int XGBHandleException(const std::exception &e) {
  XGBSetGetLastError_(e.what());
  return -1;
}

int XGDMatrixCreateFromFile(const char *fname,
                            int silent,
                            DMatrixHandle *out) {
  API_BEGIN();
  *out = LoadDataMatrix(fname, silent != 0, false, false);
  API_END();
}

int XGDMatrixCreateFromCSR(const bst_ulong *indptr,
                           const unsigned *indices,
                           const float *data,
                           bst_ulong nindptr,
                           bst_ulong nelem,
                           DMatrixHandle *out) {
  DMatrixSimple *p_mat = NULL;
  API_BEGIN();
  p_mat = new DMatrixSimple();
  DMatrixSimple &mat = *p_mat;
  mat.row_ptr_.resize(nindptr);
  for (bst_ulong i = 0; i < nindptr; ++i) {
    mat.row_ptr_[i] = static_cast<size_t>(indptr[i]);
  }
  mat.row_data_.resize(nelem);
  for (bst_ulong i = 0; i < nelem; ++i) {
    mat.row_data_[i] = RowBatch::Entry(indices[i], data[i]);
    mat.info.info.num_col = std::max(mat.info.info.num_col,
                                     static_cast<size_t>(indices[i]+1));
  }
  mat.info.info.num_row = nindptr - 1;
  *out = p_mat;
  API_END_FINALIZE(delete p_mat);
}

int XGDMatrixCreateFromCSC(const bst_ulong *col_ptr,
                           const unsigned *indices,
                           const float *data,
                           bst_ulong nindptr,
                           bst_ulong nelem,
                           DMatrixHandle *out) {
  DMatrixSimple *p_mat = NULL;
  API_BEGIN();
  int nthread;
  #pragma omp parallel
  {
    nthread = omp_get_num_threads();
  }
  p_mat = new DMatrixSimple();
  DMatrixSimple &mat = *p_mat;
  utils::ParallelGroupBuilder<RowBatch::Entry> builder(&mat.row_ptr_, &mat.row_data_);
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
  mat.info.info.num_row = mat.row_ptr_.size() - 1;
  mat.info.info.num_col = static_cast<size_t>(ncol);
  *out = p_mat;
  API_END_FINALIZE(delete p_mat);
}

int XGDMatrixCreateFromMat(const float *data,
                           bst_ulong nrow,
                           bst_ulong ncol,
                           float  missing,
                           DMatrixHandle *out) {
  DMatrixSimple *p_mat = NULL;
  API_BEGIN();
  p_mat = new DMatrixSimple();
  bool nan_missing = utils::CheckNAN(missing);
  DMatrixSimple &mat = *p_mat;
  mat.info.info.num_row = nrow;
  mat.info.info.num_col = ncol;
  for (bst_ulong i = 0; i < nrow; ++i, data += ncol) {
    bst_ulong nelem = 0;
    for (bst_ulong j = 0; j < ncol; ++j) {
      if (utils::CheckNAN(data[j])) {
        utils::Check(nan_missing,
                     "There are NAN in the matrix, however, you did not set missing=NAN");
      } else {
        if (nan_missing || data[j] != missing) {
          mat.row_data_.push_back(RowBatch::Entry(j, data[j]));
          ++nelem;
        }
      }
    }
    mat.row_ptr_.push_back(mat.row_ptr_.back() + nelem);
  }
  *out = p_mat;
  API_END_FINALIZE(delete p_mat);
}

int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                          const int *idxset,
                          bst_ulong len,
                          DMatrixHandle *out) {
  DMatrixSimple *p_ret = NULL;
  API_BEGIN();
  DMatrixSimple tmp;
  DataMatrix &dsrc = *static_cast<DataMatrix*>(handle);
  if (dsrc.magic != DMatrixSimple::kMagic) {
    tmp.CopyFrom(dsrc);
  }
  DataMatrix &src = (dsrc.magic == DMatrixSimple::kMagic ?
                     *static_cast<DMatrixSimple*>(handle): tmp);
  p_ret = new DMatrixSimple();
  DMatrixSimple &ret = *p_ret;

  utils::Check(src.info.group_ptr.size() == 0,
               "slice does not support group structure");
  ret.Clear();
  ret.info.info.num_row = len;
  ret.info.info.num_col = src.info.num_col();

  utils::IIterator<RowBatch> *iter = src.fmat()->RowIterator();
  iter->BeforeFirst();
  utils::Assert(iter->Next(), "slice");
  const RowBatch &batch = iter->Value();
  for (bst_ulong i = 0; i < len; ++i) {
    const int ridx = idxset[i];
    RowBatch::Inst inst = batch[ridx];
    utils::Check(static_cast<bst_ulong>(ridx) < batch.size, "slice index exceed number of rows");
    ret.row_data_.resize(ret.row_data_.size() + inst.length);
    memcpy(&ret.row_data_[ret.row_ptr_.back()], inst.data,
           sizeof(RowBatch::Entry) * inst.length);
    ret.row_ptr_.push_back(ret.row_ptr_.back() + inst.length);
    if (src.info.labels.size() != 0) {
      ret.info.labels.push_back(src.info.labels[ridx]);
    }
    if (src.info.weights.size() != 0) {
      ret.info.weights.push_back(src.info.weights[ridx]);
    }
    if (src.info.info.root_index.size() != 0) {
      ret.info.info.root_index.push_back(src.info.info.root_index[ridx]);
    }
    if (src.info.info.fold_index.size() != 0) {
      ret.info.info.fold_index.push_back(src.info.info.fold_index[ridx]);
    }
  }
  *out = p_ret;
  API_END_FINALIZE(delete p_ret);
}

int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<DataMatrix*>(handle);
  API_END();
}

int XGDMatrixSaveBinary(DMatrixHandle handle,
                        const char *fname,
                        int silent) {
  API_BEGIN();
  SaveDataMatrix(*static_cast<DataMatrix*>(handle), fname, silent != 0);
  API_END();
}

int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                          const char *field,
                          const float *info,
                          bst_ulong len) {
  API_BEGIN();
  std::vector<float> &vec =
      static_cast<DataMatrix*>(handle)->info.GetFloatInfo(field);
  vec.resize(len);
  memcpy(BeginPtr(vec), info, sizeof(float) * len);
  API_END();
}

int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                         const char *field,
                         const unsigned *info,
                         bst_ulong len) {
  API_BEGIN();
  std::vector<unsigned> &vec =
      static_cast<DataMatrix*>(handle)->info.GetUIntInfo(field);
  vec.resize(len);
  memcpy(BeginPtr(vec), info, sizeof(unsigned) * len);
  API_END();
}

int XGDMatrixSetGroup(DMatrixHandle handle,
                      const unsigned *group,
                      bst_ulong len) {
  API_BEGIN();
  DataMatrix *pmat = static_cast<DataMatrix*>(handle);
  pmat->info.group_ptr.resize(len + 1);
  pmat->info.group_ptr[0] = 0;
  for (uint64_t i = 0; i < len; ++i) {
    pmat->info.group_ptr[i+1] = pmat->info.group_ptr[i] + group[i];
  }
  API_END();
}

int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                          const char *field,
                          bst_ulong *out_len,
                          const float **out_dptr) {
  API_BEGIN();
  const std::vector<float> &vec =
      static_cast<const DataMatrix*>(handle)->info.GetFloatInfo(field);
  *out_len = static_cast<bst_ulong>(vec.size());
  *out_dptr = BeginPtr(vec);
  API_END();
}

int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                         const char *field,
                         bst_ulong *out_len,
                         const unsigned **out_dptr) {
  API_BEGIN();
  const std::vector<unsigned> &vec =
      static_cast<const DataMatrix*>(handle)->info.GetUIntInfo(field);
  *out_len = static_cast<bst_ulong>(vec.size());
  *out_dptr = BeginPtr(vec);
  API_END();
}
int XGDMatrixNumRow(const DMatrixHandle handle,
                    bst_ulong *out) {
  API_BEGIN();
  *out = static_cast<bst_ulong>(static_cast<const DataMatrix*>(handle)->info.num_row());
  API_END();
}

// xgboost implementation
int XGBoosterCreate(DMatrixHandle dmats[],
                    bst_ulong len,
                    BoosterHandle *out) {
  API_BEGIN();
  std::vector<DataMatrix*> mats;
  for (bst_ulong i = 0; i < len; ++i) {
    DataMatrix *dtr = static_cast<DataMatrix*>(dmats[i]);
    mats.push_back(dtr);
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
                      const char *name, const char *value) {
  API_BEGIN();
  static_cast<Booster*>(handle)->SetParam(name, value);
  API_END();
}

int XGBoosterUpdateOneIter(BoosterHandle handle,
                           int iter,
                           DMatrixHandle dtrain) {
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  DataMatrix *dtr = static_cast<DataMatrix*>(dtrain);
  bst->CheckInitModel();
  bst->CheckInit(dtr);
  bst->UpdateOneIter(iter, *dtr);
  API_END();
}

int XGBoosterBoostOneIter(BoosterHandle handle,
                          DMatrixHandle dtrain,
                          float *grad,
                          float *hess,
                          bst_ulong len) {
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  DataMatrix *dtr = static_cast<DataMatrix*>(dtrain);
  bst->CheckInitModel();
  bst->CheckInit(dtr);
  bst->BoostOneIter(*dtr, grad, hess, len);
  API_END();
}

int XGBoosterEvalOneIter(BoosterHandle handle,
                         int iter,
                         DMatrixHandle dmats[],
                         const char *evnames[],
                         bst_ulong len,
                         const char **out_str) {
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  std::vector<std::string> names;
  std::vector<const DataMatrix*> mats;
  for (bst_ulong i = 0; i < len; ++i) {
    mats.push_back(static_cast<DataMatrix*>(dmats[i]));
    names.push_back(std::string(evnames[i]));
  }
  bst->CheckInitModel();
  bst->eval_str = bst->EvalOneIter(iter, mats, names);
  *out_str = bst->eval_str.c_str();
  API_END();
}

int XGBoosterPredict(BoosterHandle handle,
                     DMatrixHandle dmat,
                     int option_mask,
                     unsigned ntree_limit,
                     bst_ulong *len,
                     const float **out_result) {
  API_BEGIN();
  *out_result = static_cast<Booster*>(handle)->
      Pred(*static_cast<DataMatrix*>(dmat),
           option_mask, ntree_limit, len);
  API_END();
}

int XGBoosterLoadModel(BoosterHandle handle, const char *fname) {
  API_BEGIN();
  static_cast<Booster*>(handle)->LoadModel(fname);
  API_END();
}

int XGBoosterSaveModel(BoosterHandle handle, const char *fname) {
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  bst->CheckInitModel();
  bst->SaveModel(fname, false);
  API_END();
}

int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                 const void *buf,
                                 bst_ulong len) {
  API_BEGIN();
  static_cast<Booster*>(handle)->LoadModelFromBuffer(buf, len);
  API_END();
}

int XGBoosterGetModelRaw(BoosterHandle handle,
                         bst_ulong *out_len,
                         const char **out_dptr) {
  API_BEGIN();
  *out_dptr = static_cast<Booster*>(handle)->GetModelRaw(out_len);
  API_END();
}

int XGBoosterDumpModel(BoosterHandle handle,
                       const char *fmap,
                       int with_stats,
                       bst_ulong *len,
                       const char ***out_models) {
  API_BEGIN();
  utils::FeatMap featmap;
  if (strlen(fmap) != 0) {
    featmap.LoadText(fmap);
  }
  *out_models = static_cast<Booster*>(handle)->GetModelDump(
      featmap, with_stats != 0, len);
  API_END();
}
