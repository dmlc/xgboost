#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include "xgboost_R.h"
#include "xgboost_wrapper.h"
#include "../src/utils/utils.h"
#include "../src/utils/omp.h"
#include "../src/utils/matrix_csr.h"

using namespace xgboost;
// implements error handling
namespace xgboost {
namespace utils {
void HandleAssertError(const char *msg) {
  error("%s", msg);
}
void HandleCheckError(const char *msg) {
  error("%s", msg);
}
}  // namespace utils
}  // namespace xgboost

extern "C" {
  void _DMatrixFinalizer(SEXP ext) {    
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGDMatrixFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent) {
    void *handle = XGDMatrixCreateFromFile(CHAR(asChar(fname)), asInteger(silent));
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  SEXP XGDMatrixCreateFromMat_R(SEXP mat, 
                                SEXP missing) {
    SEXP dim = getAttrib(mat, R_DimSymbol);
    int nrow = INTEGER(dim)[0];
    int ncol = INTEGER(dim)[1];    
    double *din = REAL(mat);
    std::vector<float> data(nrow * ncol);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < ncol; ++j) {
        data[i * ncol +j] = din[i + nrow * j];
      }
    }
    void *handle = XGDMatrixCreateFromMat(&data[0], nrow, ncol, asReal(missing));
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;    
  }
  SEXP XGDMatrixCreateFromCSC_R(SEXP indptr,
                                SEXP indices,
                                SEXP data) {
    const int *col_ptr = INTEGER(indptr);
    const int *row_index = INTEGER(indices);
    const double *col_data = REAL(data);
    int ncol = length(indptr) - 1;
    int ndata = length(data);
    // transform into CSR format
    std::vector<size_t> row_ptr;
    std::vector< std::pair<unsigned, float> > csr_data;
    utils::SparseCSRMBuilder< std::pair<unsigned,float> > builder(row_ptr, csr_data);
    builder.InitBudget();
    for (int i = 0; i < ncol; ++i) {
      for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
        builder.AddBudget(row_index[j]);
      }
    }
    builder.InitStorage();
    for (int i = 0; i < ncol; ++i) {
      for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j) {
        builder.PushElem(row_index[j], std::make_pair(i, col_data[j]));
      }
    }
    utils::Assert(csr_data.size() == static_cast<size_t>(ndata), "BUG CreateFromCSC");
    std::vector<float> row_data(ndata);
    std::vector<unsigned> col_index(ndata);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ndata; ++i) {
      col_index[i] = csr_data[i].first;
      row_data[i] = csr_data[i].second;      
    }
    void *handle = XGDMatrixCreateFromCSR(&row_ptr[0], &col_index[0], &row_data[0], row_ptr.size(), ndata );
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
    XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                        CHAR(asChar(fname)), asInteger(silent));
  }
  void XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
    int len = length(array);
    const char *name = CHAR(asChar(field));
    if (!strcmp("group", name)) {
      std::vector<unsigned> vec(len);
      #pragma omp parallel for schedule(static)      
      for (int i = 0; i < len; ++i) {
        vec[i] = static_cast<unsigned>(INTEGER(array)[i]);
      }
      XGDMatrixSetGroup(R_ExternalPtrAddr(handle), &vec[0], len);
      return;
    }
    {
      std::vector<float> vec(len);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < len; ++i) {
        vec[i] = REAL(array)[i];
      }
      XGDMatrixSetFloatInfo(R_ExternalPtrAddr(handle), 
                            CHAR(asChar(field)),
                            &vec[0], len);
    }
  }
  SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field) {
    size_t olen;
    const float *res = XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle),
                                             CHAR(asChar(field)), &olen);
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  // functions related to booster
  void _BoosterFinalizer(SEXP ext) {    
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGBoosterFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGBoosterCreate_R(SEXP dmats) {
    int len = length(dmats);
    std::vector<void*> dvec;
    for (int i = 0; i < len; ++i){
      dvec.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
    }
    void *handle = XGBoosterCreate(&dvec[0], dvec.size());
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
    XGBoosterSetParam(R_ExternalPtrAddr(handle),
                      CHAR(asChar(name)),
                      CHAR(asChar(val)));
  }
  void XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
    XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                           asInteger(iter),
                           R_ExternalPtrAddr(dtrain));
  }
  void XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess) {
    utils::Check(length(grad) == length(hess), "gradient and hess must have same length");
    int len = length(grad);
    std::vector<float> tgrad(len), thess(len);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < len; ++j) {
      tgrad[j] = REAL(grad)[j];
      thess[j] = REAL(hess)[j];
    }
    XGBoosterBoostOneIter(R_ExternalPtrAddr(handle),
                          R_ExternalPtrAddr(dtrain),
                          &tgrad[0], &thess[0], len);
  }
  SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
    utils::Check(length(dmats) == length(evnames), "dmats and evnams must have same length");
    int len = length(dmats);
    std::vector<void*> vec_dmats;
    std::vector<std::string> vec_names;
    std::vector<const char*> vec_sptr;
    for (int i = 0; i < len; ++i) {
      vec_dmats.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
      vec_names.push_back(std::string(CHAR(asChar(VECTOR_ELT(evnames, i)))));
    }
    for (int i = 0; i < len; ++i) {
      vec_sptr.push_back(vec_names[i].c_str());
    }
    return mkString(XGBoosterEvalOneIter(R_ExternalPtrAddr(handle),
                                         asInteger(iter),
                                         &vec_dmats[0], &vec_sptr[0], len));
  }
  SEXP XGBoosterPredict_R(SEXP handle, SEXP dmat, SEXP output_margin) {
    size_t olen;
    const float *res = XGBoosterPredict(R_ExternalPtrAddr(handle),
                                        R_ExternalPtrAddr(dmat),
                                        asInteger(output_margin),
                                        &olen);
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
    XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname)));
  }
  void XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
    XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname)));
  }
  void XGBoosterDumpModel_R(SEXP handle, SEXP fname, SEXP fmap) {
    size_t olen;
    const char **res = XGBoosterDumpModel(R_ExternalPtrAddr(handle),
                                          CHAR(asChar(fmap)),
                                          &olen);
    FILE *fo = utils::FopenCheck(CHAR(asChar(fname)), "w");
    for (size_t i = 0; i < olen; ++i) {
      fprintf(fo, "booster[%lu]:\n", i);
      fprintf(fo, "%s", res[i]);
    }
    fclose(fo);
  }
}
