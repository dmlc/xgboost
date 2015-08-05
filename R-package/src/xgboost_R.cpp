// Copyright (c) 2014 by Contributors
#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <cstdio>
#include <sstream>
#include "wrapper/xgboost_wrapper.h"
#include "src/utils/utils.h"
#include "src/utils/omp.h"
#include "xgboost_R.h"

using namespace std;
using namespace xgboost;

extern "C" {
  void XGBoostAssert_R(int exp, const char *fmt, ...);
  void XGBoostCheck_R(int exp, const char *fmt, ...);
  int XGBoostSPrintf_R(char *buf, size_t size, const char *fmt, ...);
}

// implements error handling
namespace xgboost {
namespace utils {
extern "C" {
  void (*Printf)(const char *fmt, ...) = Rprintf;
  int (*SPrintf)(char *buf, size_t size, const char *fmt, ...) = XGBoostSPrintf_R;
  void (*Assert)(int exp, const char *fmt, ...) = XGBoostAssert_R;
  void (*Check)(int exp, const char *fmt, ...) = XGBoostCheck_R;
  void (*Error)(const char *fmt, ...) = error;
}
bool CheckNAN(double v) {
  return ISNAN(v);
}
double LogGamma(double v) {
  return lgammafn(v);
}
}  // namespace utils

namespace random {
void Seed(unsigned seed) {
  //  warning("parameter seed is ignored, please set random seed using set.seed");
}
double Uniform(void) {
  return unif_rand();
}
double Normal(void) {
  return norm_rand();
}
}  // namespace random
}  // namespace xgboost

// call before wrapper starts
inline void _WrapperBegin(void) {
  GetRNGstate();
}
// call after wrapper starts
inline void _WrapperEnd(void) {
  PutRNGstate();
}

// do nothing, check error
inline void CheckErr(int ret) {
}

extern "C" {
  SEXP XGCheckNullPtr_R(SEXP handle) {
    return ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
  }
  void _DMatrixFinalizer(SEXP ext) {
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGDMatrixFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent) {
    _WrapperBegin();
    DMatrixHandle handle;
    CheckErr(XGDMatrixCreateFromFile(CHAR(asChar(fname)), asInteger(silent), &handle));
    _WrapperEnd();
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  SEXP XGDMatrixCreateFromMat_R(SEXP mat,
                                SEXP missing) {
    _WrapperBegin();
    SEXP dim = getAttrib(mat, R_DimSymbol);
    size_t nrow = static_cast<size_t>(INTEGER(dim)[0]);
    size_t ncol = static_cast<size_t>(INTEGER(dim)[1]);
    double *din = REAL(mat);
    std::vector<float> data(nrow * ncol);
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < nrow; ++i) {
      for (size_t j = 0; j < ncol; ++j) {
        data[i * ncol +j] = din[i + nrow * j];
      }
    }
    DMatrixHandle handle;
    CheckErr(XGDMatrixCreateFromMat(BeginPtr(data), nrow, ncol, asReal(missing), &handle));
    _WrapperEnd();
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  SEXP XGDMatrixCreateFromCSC_R(SEXP indptr,
                                SEXP indices,
                                SEXP data) {
    _WrapperBegin();
    const int *p_indptr = INTEGER(indptr);
    const int *p_indices = INTEGER(indices);
    const double *p_data = REAL(data);
    int nindptr = length(indptr);
    int ndata = length(data);
    std::vector<bst_ulong> col_ptr_(nindptr);
    std::vector<unsigned> indices_(ndata);
    std::vector<float> data_(ndata);

    for (int i = 0; i < nindptr; ++i) {
      col_ptr_[i] = static_cast<bst_ulong>(p_indptr[i]);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ndata; ++i) {
      indices_[i] = static_cast<unsigned>(p_indices[i]);
      data_[i] = static_cast<float>(p_data[i]);
    }
    DMatrixHandle handle;
    CheckErr(XGDMatrixCreateFromCSC(BeginPtr(col_ptr_), BeginPtr(indices_),
                                    BeginPtr(data_), nindptr, ndata,
                                    &handle));
    _WrapperEnd();
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset) {
    _WrapperBegin();
    int len = length(idxset);
    std::vector<int> idxvec(len);
    for (int i = 0; i < len; ++i) {
      idxvec[i] = INTEGER(idxset)[i] - 1;
    }
    DMatrixHandle res;
    CheckErr(XGDMatrixSliceDMatrix(R_ExternalPtrAddr(handle),
                                   BeginPtr(idxvec), len,
                                   &res));
    _WrapperEnd();
    SEXP ret = PROTECT(R_MakeExternalPtr(res, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
    _WrapperBegin();
    CheckErr(XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                                 CHAR(asChar(fname)), asInteger(silent)));
    _WrapperEnd();
  }
  void XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
    _WrapperBegin();
    int len = length(array);
    const char *name = CHAR(asChar(field));
    if (!strcmp("group", name)) {
      std::vector<unsigned> vec(len);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < len; ++i) {
        vec[i] = static_cast<unsigned>(INTEGER(array)[i]);
      }
      CheckErr(XGDMatrixSetGroup(R_ExternalPtrAddr(handle), BeginPtr(vec), len));
    } else {
      std::vector<float> vec(len);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < len; ++i) {
        vec[i] = REAL(array)[i];
      }
      CheckErr(XGDMatrixSetFloatInfo(R_ExternalPtrAddr(handle),
                                     CHAR(asChar(field)),
                                     BeginPtr(vec), len));
    }
    _WrapperEnd();
  }
  SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field) {
    _WrapperBegin();
    bst_ulong olen;
    const float *res;
    CheckErr(XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle),
                                   CHAR(asChar(field)),
                                   &olen,
                                   &res));
    _WrapperEnd();
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  SEXP XGDMatrixNumRow_R(SEXP handle) {
    bst_ulong nrow;
    CheckErr(XGDMatrixNumRow(R_ExternalPtrAddr(handle), &nrow));
    return ScalarInteger(static_cast<int>(nrow));
  }
  // functions related to booster
  void _BoosterFinalizer(SEXP ext) {
    if (R_ExternalPtrAddr(ext) == NULL) return;
    CheckErr(XGBoosterFree(R_ExternalPtrAddr(ext)));
    R_ClearExternalPtr(ext);
  }
  SEXP XGBoosterCreate_R(SEXP dmats) {
    _WrapperBegin();
    int len = length(dmats);
    std::vector<void*> dvec;
    for (int i = 0; i < len; ++i) {
      dvec.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
    }
    BoosterHandle handle;
    CheckErr(XGBoosterCreate(BeginPtr(dvec), dvec.size(), &handle));
    _WrapperEnd();
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
    _WrapperBegin();
    CheckErr(XGBoosterSetParam(R_ExternalPtrAddr(handle),
                               CHAR(asChar(name)),
                               CHAR(asChar(val))));
    _WrapperEnd();
  }
  void XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
    _WrapperBegin();
    CheckErr(XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                                    asInteger(iter),
                                    R_ExternalPtrAddr(dtrain)));
    _WrapperEnd();
  }
  void XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess) {
    _WrapperBegin();
    utils::Check(length(grad) == length(hess), "gradient and hess must have same length");
    int len = length(grad);
    std::vector<float> tgrad(len), thess(len);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < len; ++j) {
      tgrad[j] = REAL(grad)[j];
      thess[j] = REAL(hess)[j];
    }
    CheckErr(XGBoosterBoostOneIter(R_ExternalPtrAddr(handle),
                                   R_ExternalPtrAddr(dtrain),
                                   BeginPtr(tgrad), BeginPtr(thess),
                                   len));
    _WrapperEnd();
  }
  SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
    _WrapperBegin();
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
    const char *ret;
    CheckErr(XGBoosterEvalOneIter(R_ExternalPtrAddr(handle),
                                  asInteger(iter),
                                  BeginPtr(vec_dmats),
                                  BeginPtr(vec_sptr),
                                  len, &ret));
    _WrapperEnd();
    return mkString(ret);
  }
  SEXP XGBoosterPredict_R(SEXP handle, SEXP dmat, SEXP option_mask, SEXP ntree_limit) {
    _WrapperBegin();
    bst_ulong olen;
    const float *res;
    CheckErr(XGBoosterPredict(R_ExternalPtrAddr(handle),
                              R_ExternalPtrAddr(dmat),
                              asInteger(option_mask),
                              asInteger(ntree_limit),
                              &olen, &res));
    _WrapperEnd();
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
    _WrapperBegin();
    CheckErr(XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
    _WrapperEnd();
  }
  void XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
    _WrapperBegin();
    CheckErr(XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
    _WrapperEnd();
  }
  void XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw) {
    _WrapperBegin();
    XGBoosterLoadModelFromBuffer(R_ExternalPtrAddr(handle),
                                 RAW(raw),
                                 length(raw));
    _WrapperEnd();
  }
  SEXP XGBoosterModelToRaw_R(SEXP handle) {
    bst_ulong olen;
    _WrapperBegin();
    const char *raw;
    CheckErr(XGBoosterGetModelRaw(R_ExternalPtrAddr(handle), &olen, &raw));
    _WrapperEnd();
    SEXP ret = PROTECT(allocVector(RAWSXP, olen));
    if (olen != 0) {
      memcpy(RAW(ret), raw, olen);
    }
    UNPROTECT(1);
    return ret;
  }
  SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats) {
    _WrapperBegin();
    bst_ulong olen;
    const char **res;
    CheckErr(XGBoosterDumpModel(R_ExternalPtrAddr(handle),
                                CHAR(asChar(fmap)),
                                asInteger(with_stats),
                                &olen, &res));
    _WrapperEnd();
    SEXP out = PROTECT(allocVector(STRSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      stringstream stream;
      stream <<  "booster[" << i <<"]\n" << res[i];
      SET_STRING_ELT(out, i, mkChar(stream.str().c_str()));
    }
    UNPROTECT(1);
    return out;
  }
}
