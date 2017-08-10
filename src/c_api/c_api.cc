// Copyright (c) 2014 by Contributors

#include <xgboost/data.h>
#include <xgboost/learner.h>
#include <xgboost/c_api.h>
#include <xgboost/logging.h>
#include <dmlc/thread_local.h>
#include <rabit/rabit.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

#include "./c_api_error.h"
#include "../data/simple_csr_source.h"
#include "../common/math.h"
#include "../common/io.h"
#include "../common/group_data.h"

namespace xgboost {
// booster wrapper for backward compatible reason.
class Booster {
 public:
  explicit Booster(const std::vector<std::shared_ptr<DMatrix> >& cache_mats)
      : configured_(false),
        initialized_(false),
        learner_(Learner::Create(cache_mats)) {}

  inline Learner* learner() {
    return learner_.get();
  }

  inline void SetParam(const std::string& name, const std::string& val) {
    auto it = std::find_if(cfg_.begin(), cfg_.end(),
      [&name, &val](decltype(*cfg_.begin()) &x) {
        if (name == "eval_metric") {
          return x.first == name && x.second == val;
        }
        return x.first == name;
      });
    if (it == cfg_.end()) {
      cfg_.push_back(std::make_pair(name, val));
    } else {
      (*it).second = val;
    }
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
    if ((*next_callback_)(
            data_handle_,
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
  std::vector<bst_gpair> tmp_gpair;
};

// define the threadlocal store.
typedef dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry> XGBAPIThreadLocalStore;

int XGDMatrixCreateFromFile(const char *fname,
                            int silent,
                            DMatrixHandle *out) {
  API_BEGIN();
  if (rabit::IsDistributed()) {
    LOG(CONSOLE) << "XGBoost distributed mode detected, "
                 << "will split data among workers";
  }
  *out = new std::shared_ptr<DMatrix>(DMatrix::Load(fname, silent != 0, true));
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
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&parser, scache));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSREx(const size_t* indptr,
                                     const unsigned* indices,
                                     const bst_float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_col,
                                     DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource& mat = *source;
  mat.row_ptr_.reserve(nindptr);
  mat.row_data_.reserve(nelem);
  mat.row_ptr_.resize(1);
  mat.row_ptr_[0] = 0;
  size_t num_column = 0;
  for (size_t i = 1; i < nindptr; ++i) {
    for (size_t j = indptr[i - 1]; j < indptr[i]; ++j) {
      if (!common::CheckNAN(data[j])) {
        // automatically skip nan.
        mat.row_data_.emplace_back(RowBatch::Entry(indices[j], data[j]));
        num_column = std::max(num_column, static_cast<size_t>(indices[j] + 1));
      }
    }
    mat.row_ptr_.push_back(mat.row_data_.size());
  }

  mat.info.num_col = num_column;
  if (num_col > 0) {
    CHECK_LE(mat.info.num_col, num_col)
        << "num_col=" << num_col << " vs " << mat.info.num_col;
    mat.info.num_col = num_col;
  }
  mat.info.num_row = nindptr - 1;
  mat.info.num_nonzero = mat.row_data_.size();
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSR(const xgboost::bst_ulong* indptr,
                                   const unsigned *indices,
                                   const bst_float* data,
                                   xgboost::bst_ulong nindptr,
                                   xgboost::bst_ulong nelem,
                                   DMatrixHandle* out) {
  std::vector<size_t> indptr_(nindptr);
  for (xgboost::bst_ulong i = 0; i < nindptr; ++i) {
    indptr_[i] = static_cast<size_t>(indptr[i]);
  }
  return XGDMatrixCreateFromCSREx(&indptr_[0], indices, data,
    static_cast<size_t>(nindptr), static_cast<size_t>(nelem), 0, out);
}

XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t* col_ptr,
                                     const unsigned* indices,
                                     const bst_float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_row,
                                     DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  // FIXME: User should be able to control number of threads
  const int nthread = omp_get_max_threads();
  data::SimpleCSRSource& mat = *source;
  common::ParallelGroupBuilder<RowBatch::Entry> builder(&mat.row_ptr_, &mat.row_data_);
  builder.InitBudget(0, nthread);
  size_t ncol = nindptr - 1;  // NOLINT(*)
  #pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(ncol); ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    for (size_t j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
      if (!common::CheckNAN(data[j])) {
        builder.AddBudget(indices[j], tid);
      }
    }
  }
  builder.InitStorage();
  #pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(ncol); ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    for (size_t j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
      if (!common::CheckNAN(data[j])) {
        builder.Push(indices[j],
                     RowBatch::Entry(static_cast<bst_uint>(i), data[j]),
                     tid);
      }
    }
  }
  mat.info.num_row = mat.row_ptr_.size() - 1;
  if (num_row > 0) {
    CHECK_LE(mat.info.num_row, num_row);
    mat.info.num_row = num_row;
  }
  mat.info.num_col = ncol;
  mat.info.num_nonzero = nelem;
  *out  = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCSC(const xgboost::bst_ulong* col_ptr,
                                   const unsigned* indices,
                                   const bst_float* data,
                                   xgboost::bst_ulong nindptr,
                                   xgboost::bst_ulong nelem,
                                   DMatrixHandle* out) {
  std::vector<size_t> col_ptr_(nindptr);
  for (xgboost::bst_ulong i = 0; i < nindptr; ++i) {
    col_ptr_[i] = static_cast<size_t>(col_ptr[i]);
  }
  return XGDMatrixCreateFromCSCEx(&col_ptr_[0], indices, data,
    static_cast<size_t>(nindptr), static_cast<size_t>(nelem), 0, out);
}

XGB_DLL int XGDMatrixCreateFromMat(const bst_float* data,
                                   xgboost::bst_ulong nrow,
                                   xgboost::bst_ulong ncol,
                                   bst_float missing,
                                   DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource& mat = *source;
  mat.row_ptr_.resize(1+nrow);
  bool nan_missing = common::CheckNAN(missing);
  mat.info.num_row = nrow;
  mat.info.num_col = ncol;
  const bst_float* data0 = data;

  // count elements for sizing data
  data = data0;
  for (xgboost::bst_ulong i = 0; i < nrow; ++i, data += ncol) {
    xgboost::bst_ulong nelem = 0;
    for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
      if (common::CheckNAN(data[j])) {
        CHECK(nan_missing)
          << "There are NAN in the matrix, however, you did not set missing=NAN";
      } else {
        if (nan_missing || data[j] != missing) {
          ++nelem;
        }
      }
    }
    mat.row_ptr_[i+1] = mat.row_ptr_[i] + nelem;
  }
  mat.row_data_.resize(mat.row_data_.size() + mat.row_ptr_.back());

  data = data0;
  for (xgboost::bst_ulong i = 0; i < nrow; ++i, data += ncol) {
    xgboost::bst_ulong matj = 0;
    for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
      if (common::CheckNAN(data[j])) {
      } else {
        if (nan_missing || data[j] != missing) {
          mat.row_data_[mat.row_ptr_[i] + matj] = RowBatch::Entry(j, data[j]);
          ++matj;
        }
      }
    }
  }

  mat.info.num_nonzero = mat.row_data_.size();
  *out  = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

void prefixsum_inplace(size_t *x, size_t N) {
  size_t *suma;
#pragma omp parallel
  {
    const int ithread = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#pragma omp single
    {
      suma = new size_t[nthreads+1];
      suma[0] = 0;
    }
    size_t sum = 0;
#pragma omp for schedule(static)
    for (omp_ulong i = 0; i < N; i++) {
      sum += x[i];
      x[i] = sum;
    }
    suma[ithread+1] = sum;
#pragma omp barrier
    size_t offset = 0;
    for (omp_ulong i = 0; i < static_cast<omp_ulong>(ithread+1); i++) {
      offset += suma[i];
    }
#pragma omp for schedule(static)
    for (omp_ulong i = 0; i < N; i++) {
      x[i] += offset;
    }
  }
  delete[] suma;
}


XGB_DLL int XGDMatrixCreateFromMat_omp(const bst_float* data,
                                       xgboost::bst_ulong nrow,
                                       xgboost::bst_ulong ncol,
                                       bst_float missing,
                                       DMatrixHandle* out,
                                       int nthread) {
  // avoid openmp unless enough data to be worth it to avoid overhead costs
  if (nrow*ncol <= 10000*50) {
    return(XGDMatrixCreateFromMat(data, nrow, ncol, missing, out));
  }

  API_BEGIN();
  const int nthreadmax = std::max(omp_get_num_procs() / 2 - 1, 1);
  //  const int nthreadmax = omp_get_max_threads();
  if (nthread <= 0) nthread=nthreadmax;
  omp_set_num_threads(nthread);

  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
  data::SimpleCSRSource& mat = *source;
  mat.row_ptr_.resize(1+nrow);
  mat.info.num_row = nrow;
  mat.info.num_col = ncol;

  // Check for errors in missing elements
  // Count elements per row (to avoid otherwise need to copy)
  bool nan_missing = common::CheckNAN(missing);
  int *badnan;
  badnan = new int[nthread];
  for (int i = 0; i < nthread; i++) {
    badnan[i] = 0;
  }

#pragma omp parallel num_threads(nthread)
  {
    int ithread  = omp_get_thread_num();

    // Count elements per row
#pragma omp for schedule(static)
    for (omp_ulong i = 0; i < nrow; ++i) {
      xgboost::bst_ulong nelem = 0;
      for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
        if (common::CheckNAN(data[ncol*i + j]) && !nan_missing) {
          badnan[ithread] = 1;
        } else if (common::CheckNAN(data[ncol * i + j])) {
        } else if (nan_missing || data[ncol * i + j] != missing) {
          ++nelem;
        }
      }
      mat.row_ptr_[i+1] = nelem;
    }
  }
  // Inform about any NaNs and resize data matrix
  for (int i = 0; i < nthread; i++) {
    CHECK(!badnan[i]) << "There are NAN in the matrix, however, you did not set missing=NAN";
  }

  // do cumulative sum (to avoid otherwise need to copy)
  prefixsum_inplace(&mat.row_ptr_[0], mat.row_ptr_.size());
  mat.row_data_.resize(mat.row_data_.size() + mat.row_ptr_.back());

  // Fill data matrix (now that know size, no need for slow push_back())
#pragma omp parallel num_threads(nthread)
  {
#pragma omp for schedule(static)
    for (omp_ulong i = 0; i < nrow; ++i) {
      xgboost::bst_ulong matj = 0;
      for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
        if (common::CheckNAN(data[ncol * i + j])) {
        } else if (nan_missing || data[ncol * i + j] != missing) {
          mat.row_data_[mat.row_ptr_[i] + matj] =
              RowBatch::Entry(j, data[ncol * i + j]);
          ++matj;
        }
      }
    }
  }

  mat.info.num_nonzero = mat.row_data_.size();
  *out  = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                                  const int* idxset,
                                  xgboost::bst_ulong len,
                                  DMatrixHandle* out) {
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

  API_BEGIN();
  data::SimpleCSRSource src;
  src.CopyFrom(static_cast<std::shared_ptr<DMatrix>*>(handle)->get());
  data::SimpleCSRSource& ret = *source;

  CHECK_EQ(src.info.group_ptr.size(), 0U)
      << "slice does not support group structure";

  ret.Clear();
  ret.info.num_row = len;
  ret.info.num_col = src.info.num_col;

  dmlc::DataIter<RowBatch>* iter = &src;
  iter->BeforeFirst();
  CHECK(iter->Next());

  const RowBatch& batch = iter->Value();
  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    const int ridx = idxset[i];
    RowBatch::Inst inst = batch[ridx];
    CHECK_LT(static_cast<xgboost::bst_ulong>(ridx), batch.size);
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
    if (src.info.base_margin.size() != 0) {
      ret.info.base_margin.push_back(src.info.base_margin[ridx]);
    }
    if (src.info.root_index.size() != 0) {
      ret.info.root_index.push_back(src.info.root_index[ridx]);
    }
  }
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

XGB_DLL int XGDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<std::shared_ptr<DMatrix>*>(handle);
  API_END();
}

XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle,
                                const char* fname,
                                int silent) {
  API_BEGIN();
  static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->SaveToLocalFile(fname);
  API_END();
}

XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                          const char* field,
                          const bst_float* info,
                          xgboost::bst_ulong len) {
  API_BEGIN();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->info().SetInfo(field, info, kFloat32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                         const char* field,
                         const unsigned* info,
                         xgboost::bst_ulong len) {
  API_BEGIN();
  static_cast<std::shared_ptr<DMatrix>*>(handle)
      ->get()->info().SetInfo(field, info, kUInt32, len);
  API_END();
}

XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned* group,
                              xgboost::bst_ulong len) {
  API_BEGIN();
  std::shared_ptr<DMatrix> *pmat = static_cast<std::shared_ptr<DMatrix>*>(handle);
  MetaInfo& info = pmat->get()->info();
  info.group_ptr.resize(len + 1);
  info.group_ptr[0] = 0;
  for (uint64_t i = 0; i < len; ++i) {
    info.group_ptr[i + 1] = info.group_ptr[i] + group[i];
  }
  API_END();
}

XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char* field,
                                  xgboost::bst_ulong* out_len,
                                  const bst_float** out_dptr) {
  API_BEGIN();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->info();
  const std::vector<bst_float>* vec = nullptr;
  if (!std::strcmp(field, "label")) {
    vec = &info.labels;
  } else if (!std::strcmp(field, "weight")) {
    vec = &info.weights;
  } else if (!std::strcmp(field, "base_margin")) {
    vec = &info.base_margin;
  } else {
    LOG(FATAL) << "Unknown float field name " << field;
  }
  *out_len = static_cast<xgboost::bst_ulong>(vec->size());
  *out_dptr = dmlc::BeginPtr(*vec);
  API_END();
}

XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 xgboost::bst_ulong *out_len,
                                 const unsigned **out_dptr) {
  API_BEGIN();
  const MetaInfo& info = static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->info();
  const std::vector<unsigned>* vec = nullptr;
  if (!std::strcmp(field, "root_index")) {
    vec = &info.root_index;
  } else {
    LOG(FATAL) << "Unknown uint field name " << field;
  }
  *out_len = static_cast<xgboost::bst_ulong>(vec->size());
  *out_dptr = dmlc::BeginPtr(*vec);
  API_END();
}

XGB_DLL int XGDMatrixNumRow(const DMatrixHandle handle,
                            xgboost::bst_ulong *out) {
  API_BEGIN();
  *out = static_cast<xgboost::bst_ulong>(
      static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->info().num_row);
  API_END();
}

XGB_DLL int XGDMatrixNumCol(const DMatrixHandle handle,
                            xgboost::bst_ulong *out) {
  API_BEGIN();
  *out = static_cast<size_t>(
      static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->info().num_col);
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
  *out = new Booster(mats);
  API_END();
}

XGB_DLL int XGBoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete static_cast<Booster*>(handle);
  API_END();
}

XGB_DLL int XGBoosterSetParam(BoosterHandle handle,
                              const char *name,
                              const char *value) {
  API_BEGIN();
  static_cast<Booster*>(handle)->SetParam(name, value);
  API_END();
}

XGB_DLL int XGBoosterUpdateOneIter(BoosterHandle handle,
                                   int iter,
                                   DMatrixHandle dtrain) {
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  std::shared_ptr<DMatrix> *dtr =
      static_cast<std::shared_ptr<DMatrix>*>(dtrain);

  bst->LazyInit();
  bst->learner()->UpdateOneIter(iter, dtr->get());
  API_END();
}

XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle,
                                  DMatrixHandle dtrain,
                                  bst_float *grad,
                                  bst_float *hess,
                                  xgboost::bst_ulong len) {
  std::vector<bst_gpair>& tmp_gpair = XGBAPIThreadLocalStore::Get()->tmp_gpair;
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  std::shared_ptr<DMatrix>* dtr =
      static_cast<std::shared_ptr<DMatrix>*>(dtrain);
  tmp_gpair.resize(len);
  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    tmp_gpair[i] = bst_gpair(grad[i], hess[i]);
  }

  bst->LazyInit();
  bst->learner()->BoostOneIter(0, dtr->get(), &tmp_gpair);
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
  Booster* bst = static_cast<Booster*>(handle);
  std::vector<DMatrix*> data_sets;
  std::vector<std::string> data_names;

  for (xgboost::bst_ulong i = 0; i < len; ++i) {
    data_sets.push_back(static_cast<std::shared_ptr<DMatrix>*>(dmats[i])->get());
    data_names.push_back(std::string(evnames[i]));
  }

  bst->LazyInit();
  eval_str = bst->learner()->EvalOneIter(iter, data_sets, data_names);
  *out_str = eval_str.c_str();
  API_END();
}

XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             xgboost::bst_ulong *len,
                             const bst_float **out_result) {
  std::vector<bst_float>& preds = XGBAPIThreadLocalStore::Get()->ret_vec_float;
  API_BEGIN();
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Predict(
      static_cast<std::shared_ptr<DMatrix>*>(dmat)->get(),
      (option_mask & 1) != 0,
      &preds, ntree_limit,
      (option_mask & 2) != 0,
      (option_mask & 4) != 0,
      (option_mask & 8) != 0);
  *out_result = dmlc::BeginPtr(preds);
  *len = static_cast<xgboost::bst_ulong>(preds.size());
  API_END();
}

XGB_DLL int XGBoosterLoadModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  static_cast<Booster*>(handle)->LoadModel(fi.get());
  API_END();
}

XGB_DLL int XGBoosterSaveModel(BoosterHandle handle, const char* fname) {
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Save(fo.get());
  API_END();
}

XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                 const void* buf,
                                 xgboost::bst_ulong len) {
  API_BEGIN();
  common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
  static_cast<Booster*>(handle)->LoadModel(&fs);
  API_END();
}

XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle,
                         xgboost::bst_ulong* out_len,
                         const char** out_dptr) {
  std::string& raw_str = XGBAPIThreadLocalStore::Get()->ret_str;
  raw_str.resize(0);

  API_BEGIN();
  common::MemoryBufferStream fo(&raw_str);
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  bst->learner()->Save(&fo);
  *out_dptr = dmlc::BeginPtr(raw_str);
  *out_len = static_cast<xgboost::bst_ulong>(raw_str.length());
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
  Booster *bst = static_cast<Booster*>(handle);
  bst->LazyInit();
  str_vecs = bst->learner()->DumpModel(fmap, with_stats != 0, format);
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
  Booster* bst = static_cast<Booster*>(handle);
  std::string& ret_str = XGBAPIThreadLocalStore::Get()->ret_str;
  API_BEGIN();
  if (bst->learner()->GetAttr(key, &ret_str)) {
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
  Booster* bst = static_cast<Booster*>(handle);
  API_BEGIN();
  if (value == nullptr) {
    bst->learner()->DelAttr(key);
  } else {
    bst->learner()->SetAttr(key, value);
  }
  API_END();
}

XGB_DLL int XGBoosterGetAttrNames(BoosterHandle handle,
                     xgboost::bst_ulong* out_len,
                     const char*** out) {
  std::vector<std::string>& str_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_str;
  std::vector<const char*>& charp_vecs = XGBAPIThreadLocalStore::Get()->ret_vec_charp;
  Booster *bst = static_cast<Booster*>(handle);
  API_BEGIN();
  str_vecs = bst->learner()->GetAttrNames();
  charp_vecs.resize(str_vecs.size());
  for (size_t i = 0; i < str_vecs.size(); ++i) {
    charp_vecs[i] = str_vecs[i].c_str();
  }
  *out = dmlc::BeginPtr(charp_vecs);
  *out_len = static_cast<xgboost::bst_ulong>(charp_vecs.size());
  API_END();
}

XGB_DLL int XGBoosterLoadRabitCheckpoint(BoosterHandle handle,
                                 int* version) {
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  *version = rabit::LoadCheckPoint(bst->learner());
  if (*version != 0) {
    bst->initialized_ = true;
  }
  API_END();
}

XGB_DLL int XGBoosterSaveRabitCheckpoint(BoosterHandle handle) {
  API_BEGIN();
  Booster* bst = static_cast<Booster*>(handle);
  if (bst->learner()->AllowLazyCheckPoint()) {
    rabit::LazyCheckPoint(bst->learner());
  } else {
    rabit::CheckPoint(bst->learner());
  }
  API_END();
}

// force link rabit
static DMLC_ATTRIBUTE_UNUSED int XGBOOST_LINK_RABIT_C_API_ = RabitLinkTag();
