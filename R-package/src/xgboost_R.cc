// Copyright (c) 2014 by Contributors
#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <xgboost/c_api.h>
#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <cstdio>
#include <sstream>
#include "./xgboost_R.h"

/*!
 * \brief macro to annotate begin of api
 */
#define R_API_BEGIN()                           \
  GetRNGstate();                                \
  try {
/*!
 * \brief macro to annotate end of api
 */
#define R_API_END()                             \
  } catch(dmlc::Error& e) {                     \
    PutRNGstate();                              \
    error(e.what());                            \
  }                                             \
  PutRNGstate();

/*!
 * \brief macro to check the call.
 */
#define CHECK_CALL(x)                           \
  if ((x) != 0) {                               \
    error(XGBGetLastError());                   \
  }


using namespace dmlc;

SEXP XGCheckNullPtr_R(SEXP handle) {
  return ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
}

void _DMatrixFinalizer(SEXP ext) {
  R_API_BEGIN();
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(XGDMatrixFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
  R_API_END();
}

SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent) {
  SEXP ret;
  R_API_BEGIN();
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromFile(CHAR(asChar(fname)), asInteger(silent), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGDMatrixCreateFromMat_R(SEXP mat,
                              SEXP missing) {
  SEXP ret;
  R_API_BEGIN();
  SEXP dim = getAttrib(mat, R_DimSymbol);
  size_t nrow = static_cast<size_t>(INTEGER(dim)[0]);
  size_t ncol = static_cast<size_t>(INTEGER(dim)[1]);
  const bool is_int = TYPEOF(mat) == INTSXP;
  double *din;
  int *iin;
  if (is_int) {
    iin = INTEGER(mat);
  } else {
    din = REAL(mat);
  }
  std::vector<float> data(nrow * ncol);
  #pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < nrow; ++i) {
    for (size_t j = 0; j < ncol; ++j) {
      data[i * ncol +j] = is_int ? static_cast<float>(iin[i + nrow * j]) : din[i + nrow * j];
    }
  }
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromMat(BeginPtr(data), nrow, ncol, asReal(missing), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGDMatrixCreateFromCSC_R(SEXP indptr,
                              SEXP indices,
                              SEXP data,
                              SEXP num_row) {
  SEXP ret;
  R_API_BEGIN();
  const int *p_indptr = INTEGER(indptr);
  const int *p_indices = INTEGER(indices);
  const double *p_data = REAL(data);
  size_t nindptr = static_cast<size_t>(length(indptr));
  size_t ndata = static_cast<size_t>(length(data));
  size_t nrow = static_cast<size_t>(INTEGER(num_row)[0]);
  std::vector<size_t> col_ptr_(nindptr);
  std::vector<unsigned> indices_(ndata);
  std::vector<float> data_(ndata);

  for (size_t i = 0; i < nindptr; ++i) {
    col_ptr_[i] = static_cast<size_t>(p_indptr[i]);
  }
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < static_cast<int64_t>(ndata); ++i) {
    indices_[i] = static_cast<unsigned>(p_indices[i]);
    data_[i] = static_cast<float>(p_data[i]);
  }
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromCSCEx(BeginPtr(col_ptr_), BeginPtr(indices_),
                                      BeginPtr(data_), nindptr, ndata,
                                      nrow, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset) {
  SEXP ret;
  R_API_BEGIN();
  int len = length(idxset);
  std::vector<int> idxvec(len);
  for (int i = 0; i < len; ++i) {
    idxvec[i] = INTEGER(idxset)[i] - 1;
  }
  DMatrixHandle res;
  CHECK_CALL(XGDMatrixSliceDMatrixEx(R_ExternalPtrAddr(handle),
                                     BeginPtr(idxvec), len,
                                     &res,
                                     0));
  ret = PROTECT(R_MakeExternalPtr(res, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                                 CHAR(asChar(fname)),
                                 asInteger(silent)));
  R_API_END();
  return R_NilValue;
}

SEXP XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  int len = length(array);
  const char *name = CHAR(asChar(field));
  if (!strcmp("group", name)) {
    std::vector<unsigned> vec(len);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<unsigned>(INTEGER(array)[i]);
    }
    CHECK_CALL(XGDMatrixSetUIntInfo(R_ExternalPtrAddr(handle),
                                    CHAR(asChar(field)),
                                    BeginPtr(vec), len));
  } else {
    std::vector<float> vec(len);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
      vec[i] = REAL(array)[i];
    }
    CHECK_CALL(XGDMatrixSetFloatInfo(R_ExternalPtrAddr(handle),
                                     CHAR(asChar(field)),
                                     BeginPtr(vec), len));
  }
  R_API_END();
  return R_NilValue;
}

SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const float *res;
  CHECK_CALL(XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle),
                                   CHAR(asChar(field)),
                                 &olen,
                                 &res));
  ret = PROTECT(allocVector(REALSXP, olen));
  for (size_t i = 0; i < olen; ++i) {
    REAL(ret)[i] = res[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGDMatrixNumRow_R(SEXP handle) {
  bst_ulong nrow;
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixNumRow(R_ExternalPtrAddr(handle), &nrow));
  R_API_END();
  return ScalarInteger(static_cast<int>(nrow));
}

SEXP XGDMatrixNumCol_R(SEXP handle) {
  bst_ulong ncol;
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixNumCol(R_ExternalPtrAddr(handle), &ncol));
  R_API_END();
  return ScalarInteger(static_cast<int>(ncol));
}

// functions related to booster
void _BoosterFinalizer(SEXP ext) {
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(XGBoosterFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
}

SEXP XGBoosterCreate_R(SEXP dmats) {
  SEXP ret;
  R_API_BEGIN();
  int len = length(dmats);
  std::vector<void*> dvec;
  for (int i = 0; i < len; ++i) {
    dvec.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
  }
  BoosterHandle handle;
  CHECK_CALL(XGBoosterCreate(BeginPtr(dvec), dvec.size(), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterSetParam(R_ExternalPtrAddr(handle),
                               CHAR(asChar(name)),
                               CHAR(asChar(val))));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                                  asInteger(iter),
                                  R_ExternalPtrAddr(dtrain)));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess) {
  R_API_BEGIN();
  CHECK_EQ(length(grad), length(hess))
      << "gradient and hess must have same length";
  int len = length(grad);
  std::vector<float> tgrad(len), thess(len);
  #pragma omp parallel for schedule(static)
  for (int j = 0; j < len; ++j) {
    tgrad[j] = REAL(grad)[j];
    thess[j] = REAL(hess)[j];
  }
  CHECK_CALL(XGBoosterBoostOneIter(R_ExternalPtrAddr(handle),
                                 R_ExternalPtrAddr(dtrain),
                                 BeginPtr(tgrad), BeginPtr(thess),
                                 len));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
  const char *ret;
  R_API_BEGIN();
  CHECK_EQ(length(dmats), length(evnames))
      << "dmats and evnams must have same length";
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
  CHECK_CALL(XGBoosterEvalOneIter(R_ExternalPtrAddr(handle),
                                asInteger(iter),
                                BeginPtr(vec_dmats),
                                BeginPtr(vec_sptr),
                                len, &ret));
  R_API_END();
  return mkString(ret);
}

SEXP XGBoosterPredict_R(SEXP handle, SEXP dmat, SEXP option_mask, SEXP ntree_limit) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const float *res;
  CHECK_CALL(XGBoosterPredict(R_ExternalPtrAddr(handle),
                            R_ExternalPtrAddr(dmat),
                            asInteger(option_mask),
                            asInteger(ntree_limit),
                            &olen, &res));
  ret = PROTECT(allocVector(REALSXP, olen));
  for (size_t i = 0; i < olen; ++i) {
    REAL(ret)[i] = res[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModelFromBuffer(R_ExternalPtrAddr(handle),
                                          RAW(raw),
                                          length(raw)));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterModelToRaw_R(SEXP handle) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const char *raw;
  CHECK_CALL(XGBoosterGetModelRaw(R_ExternalPtrAddr(handle), &olen, &raw));
  ret = PROTECT(allocVector(RAWSXP, olen));
  if (olen != 0) {
    memcpy(RAW(ret), raw, olen);
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats, SEXP dump_format) {
  SEXP out;
  R_API_BEGIN();
  bst_ulong olen;
  const char **res;
  const char *fmt = CHAR(asChar(dump_format));
  CHECK_CALL(XGBoosterDumpModelEx(R_ExternalPtrAddr(handle),
                                CHAR(asChar(fmap)),
                                asInteger(with_stats),
                                fmt,
                                &olen, &res));
  out = PROTECT(allocVector(STRSXP, olen));
  if (!strcmp("json", fmt)) {
    std::stringstream stream;
    stream <<  "[\n";
    for (size_t i = 0; i < olen; ++i) {
      stream << res[i];
      if (i < olen - 1) {
        stream << ",\n";
      } else {
        stream << "\n";
      }
    }
    stream <<  "]";
    SET_STRING_ELT(out, 0, mkChar(stream.str().c_str()));
  } else {
    for (size_t i = 0; i < olen; ++i) {
      std::stringstream stream;
      stream <<  "booster[" << i <<"]\n" << res[i];
      SET_STRING_ELT(out, i, mkChar(stream.str().c_str()));
    }
  }
  R_API_END();
  UNPROTECT(1);
  return out;
}

SEXP XGBoosterGetAttr_R(SEXP handle, SEXP name) {
  SEXP out;
  R_API_BEGIN();
  int success;
  const char *val;
  CHECK_CALL(XGBoosterGetAttr(R_ExternalPtrAddr(handle),
                              CHAR(asChar(name)),
                              &val,
                              &success));
  if (success) {
    out = PROTECT(allocVector(STRSXP, 1));
    SET_STRING_ELT(out, 0, mkChar(val));
  } else {
    out = PROTECT(R_NilValue);
  }
  R_API_END();
  UNPROTECT(1);
  return out;
}

SEXP XGBoosterSetAttr_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  const char *v = isNull(val) ? nullptr : CHAR(asChar(val));
  CHECK_CALL(XGBoosterSetAttr(R_ExternalPtrAddr(handle),
                              CHAR(asChar(name)), v));
  R_API_END();
  return R_NilValue;
}

SEXP XGBoosterGetAttrNames_R(SEXP handle) {
  SEXP out;
  R_API_BEGIN();
  bst_ulong len;
  const char **res;
  CHECK_CALL(XGBoosterGetAttrNames(R_ExternalPtrAddr(handle),
                                   &len, &res));
  if (len > 0) {
    out = PROTECT(allocVector(STRSXP, len));
    for (size_t i = 0; i < len; ++i) {
      SET_STRING_ELT(out, i, mkChar(res[i]));
    }
  } else {
    out = PROTECT(R_NilValue);
  }
  R_API_END();
  UNPROTECT(1);
  return out;
}
