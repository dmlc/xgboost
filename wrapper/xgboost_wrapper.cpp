// implementations in ctypes
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include "./xgboost_wrapper.h"
#include "../src/data.h"
#include "../src/learner/learner-inl.hpp"
#include "../src/io/io.h"
#include "../src/io/simple_dmatrix-inl.hpp"

using namespace xgboost;
using namespace xgboost::io;

namespace xgboost {
namespace wrapper {
// booster wrapper class
class Booster: public learner::BoostLearner<FMatrixS> {
 public:
  explicit Booster(const std::vector<DataMatrix*>& mats) {
    this->silent = 1;
    this->init_model = false;
    this->SetCacheData(mats);
  }
  const float *Pred(const DataMatrix &dmat, int output_margin, uint64_t *len) {
    this->CheckInitModel();
    this->Predict(dmat, output_margin, &this->preds_);
    *len = this->preds_.size();
    return &this->preds_[0];
  }
  inline void BoostOneIter(const DataMatrix &train,
                           float *grad, float *hess, uint64_t len) {
    this->gpair_.resize(len);
    const unsigned ndata = static_cast<unsigned>(len);
    #pragma omp parallel for schedule(static)
    for (unsigned j = 0; j < ndata; ++j) {
      gpair_[j] = bst_gpair(grad[j], hess[j]);
    }
    gbm_->DoBoost(train.fmat, train.info.info, &gpair_);
  }
  inline void CheckInitModel(void) {
    if (!init_model) {
      this->InitModel(); init_model = true;
    }
  }
  inline void LoadModel(const char *fname) {
    learner::BoostLearner<FMatrixS>::LoadModel(fname);
    this->init_model = true;
  }
  inline const char** GetModelDump(const utils::FeatMap& fmap, bool with_stats, uint64_t *len) {
    model_dump = this->DumpModel(fmap, with_stats);
    model_dump_cptr.resize(model_dump.size());
    for (size_t i = 0; i < model_dump.size(); ++i) {
      model_dump_cptr[i] = model_dump[i].c_str();
    }
    *len = model_dump.size();
    return &model_dump_cptr[0];
  }
  // temporal fields
  // temporal data to save evaluation dump
  std::string eval_str;
  // temporal space to save model dump
  std::vector<std::string> model_dump;
  std::vector<const char*> model_dump_cptr;

 private:
  bool init_model;
};
}  // namespace wrapper
}  // namespace xgboost

using namespace xgboost::wrapper;

extern "C"{
  void* XGDMatrixCreateFromFile(const char *fname, int silent) {
    return LoadDataMatrix(fname, silent, false);
  }
  void* XGDMatrixCreateFromCSR(const uint64_t *indptr,
                               const unsigned *indices,
                               const float *data,
                               uint64_t nindptr,
                               uint64_t nelem) {
    DMatrixSimple *p_mat = new DMatrixSimple();
    DMatrixSimple &mat = *p_mat;
    mat.row_ptr_.resize(nindptr);
    for (uint64_t i = 0; i < nindptr; ++i) {
      mat.row_ptr_[i] = static_cast<size_t>(indptr[i]);
    }
    mat.row_data_.resize(nelem);
    for (uint64_t i = 0; i < nelem; ++i) {
      mat.row_data_[i] = SparseBatch::Entry(indices[i], data[i]);
      mat.info.info.num_col = std::max(mat.info.info.num_col,
                                       static_cast<uint64_t>(indices[i]+1));
    }
    mat.info.info.num_row = nindptr - 1;
    return p_mat;
  }
  void* XGDMatrixCreateFromMat(const float *data,
                               uint64_t nrow,
                               uint64_t ncol,
                               float  missing) {
    DMatrixSimple *p_mat = new DMatrixSimple();
    DMatrixSimple &mat = *p_mat;
    mat.info.info.num_row = nrow;
    mat.info.info.num_col = ncol;
    for (uint64_t i = 0; i < nrow; ++i, data += ncol) {
      uint64_t nelem = 0;
      for (uint64_t j = 0; j < ncol; ++j) {
        if (data[j] != missing) {
          mat.row_data_.push_back(SparseBatch::Entry(j, data[j]));
          ++nelem;
        }
      }
      mat.row_ptr_.push_back(mat.row_ptr_.back() + nelem);
    }
    return p_mat;
  }
  void* XGDMatrixSliceDMatrix(void *handle,
                              const int *idxset,
                              uint64_t len) {
    DMatrixSimple tmp;
    DataMatrix &dsrc = *static_cast<DataMatrix*>(handle);
    if (dsrc.magic != DMatrixSimple::kMagic) {
      tmp.CopyFrom(dsrc);
    }
    DataMatrix &src = (dsrc.magic == DMatrixSimple::kMagic ?
                       *static_cast<DMatrixSimple*>(handle): tmp);
    DMatrixSimple *p_ret = new DMatrixSimple();
    DMatrixSimple &ret = *p_ret;

    utils::Check(src.info.group_ptr.size() == 0,
                 "slice does not support group structure");
    ret.Clear();
    ret.info.info.num_row = len;
    ret.info.info.num_col = src.info.num_col();

    utils::IIterator<SparseBatch> *iter = src.fmat.RowIterator();
    iter->BeforeFirst();
    utils::Assert(iter->Next(), "slice");
    const SparseBatch &batch = iter->Value();
    for (uint64_t i = 0; i < len; ++i) {
      const int ridx = idxset[i];
      SparseBatch::Inst inst = batch[ridx];
      utils::Check(static_cast<uint64_t>(ridx) < batch.size, "slice index exceed number of rows");
      ret.row_data_.resize(ret.row_data_.size() + inst.length);
      memcpy(&ret.row_data_[ret.row_ptr_.back()], inst.data,
             sizeof(SparseBatch::Entry) * inst.length);
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
    }
    return p_ret;
  }
  void XGDMatrixFree(void *handle) {
    delete static_cast<DataMatrix*>(handle);
  }
  void XGDMatrixSaveBinary(void *handle, const char *fname, int silent) {
    SaveDataMatrix(*static_cast<DataMatrix*>(handle), fname, silent);
  }
  void XGDMatrixSetFloatInfo(void *handle, const char *field, const float *info, uint64_t len) {
    std::vector<float> &vec = 
        static_cast<DataMatrix*>(handle)->info.GetFloatInfo(field);
    vec.resize(len);
    memcpy(&vec[0], info, sizeof(float) * len);
  }
  void XGDMatrixSetUIntInfo(void *handle, const char *field, const unsigned *info, uint64_t len) {
    std::vector<unsigned> &vec =
        static_cast<DataMatrix*>(handle)->info.GetUIntInfo(field);
    vec.resize(len);
    memcpy(&vec[0], info, sizeof(unsigned) * len);
  }
  void XGDMatrixSetGroup(void *handle, const unsigned *group, uint64_t len) {
    DataMatrix *pmat = static_cast<DataMatrix*>(handle);
    pmat->info.group_ptr.resize(len + 1);
    pmat->info.group_ptr[0] = 0;
    for (uint64_t i = 0; i < len; ++i) {
      pmat->info.group_ptr[i+1] = pmat->info.group_ptr[i]+group[i];
    }
  }
  const float* XGDMatrixGetFloatInfo(const void *handle, const char *field, uint64_t* len) {
    const std::vector<float> &vec =
        static_cast<const DataMatrix*>(handle)->info.GetFloatInfo(field);
    *len = vec.size();
    return &vec[0];
  }
  const unsigned* XGDMatrixGetUIntInfo(const void *handle, const char *field, uint64_t* len) {
    const std::vector<unsigned> &vec =
        static_cast<const DataMatrix*>(handle)->info.GetUIntInfo(field);
    *len = vec.size();
    return &vec[0];
  }
  uint64_t XGDMatrixNumRow(const void *handle) {
    return static_cast<const DataMatrix*>(handle)->info.num_row();
  }

  // xgboost implementation
  void *XGBoosterCreate(void *dmats[], uint64_t len) {
    std::vector<DataMatrix*> mats;
    for (uint64_t i = 0; i < len; ++i) {
      DataMatrix *dtr = static_cast<DataMatrix*>(dmats[i]);
      mats.push_back(dtr);
    }
    return new Booster(mats);
  }
  void XGBoosterFree(void *handle) {
    delete static_cast<Booster*>(handle);
  }
  void XGBoosterSetParam(void *handle, const char *name, const char *value) {
    static_cast<Booster*>(handle)->SetParam(name, value);
  }
  void XGBoosterUpdateOneIter(void *handle, int iter, void *dtrain) {
    Booster *bst = static_cast<Booster*>(handle);
    DataMatrix *dtr = static_cast<DataMatrix*>(dtrain);
    bst->CheckInitModel();
    bst->CheckInit(dtr);
    bst->UpdateOneIter(iter, *dtr);
  }
  void XGBoosterBoostOneIter(void *handle, void *dtrain,
                             float *grad, float *hess, uint64_t len) {
    Booster *bst = static_cast<Booster*>(handle);
    DataMatrix *dtr = static_cast<DataMatrix*>(dtrain);
    bst->CheckInitModel();
    bst->CheckInit(dtr);
    bst->BoostOneIter(*dtr, grad, hess, len);
  }
  const char* XGBoosterEvalOneIter(void *handle, int iter, void *dmats[],
                                   const char *evnames[], uint64_t len) {
    Booster *bst = static_cast<Booster*>(handle);
    std::vector<std::string> names;
    std::vector<const DataMatrix*> mats;
    for (uint64_t i = 0; i < len; ++i) {
      mats.push_back(static_cast<DataMatrix*>(dmats[i]));
      names.push_back(std::string(evnames[i]));
    }
    bst->CheckInitModel();
    bst->eval_str = bst->EvalOneIter(iter, mats, names);
    return bst->eval_str.c_str();
  }
  const float *XGBoosterPredict(void *handle, void *dmat, int output_margin, uint64_t *len) {
    return static_cast<Booster*>(handle)->Pred(*static_cast<DataMatrix*>(dmat), output_margin, len);
  }
  void XGBoosterLoadModel(void *handle, const char *fname) {
    static_cast<Booster*>(handle)->LoadModel(fname);
  }
  void XGBoosterSaveModel(const void *handle, const char *fname) {
    static_cast<const Booster*>(handle)->SaveModel(fname);
  }
  const char** XGBoosterDumpModel(void *handle, const char *fmap, uint64_t *len){
    utils::FeatMap featmap;
    if (strlen(fmap) != 0) {
      featmap.LoadText(fmap);
    }
    return static_cast<Booster*>(handle)->GetModelDump(featmap, false, len);
  }
};
