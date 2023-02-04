/**
 * Copyright 2014-2023 by XGBoost Contributors
 */
#include <dmlc/common.h>
#include <dmlc/omp.h>
#include <xgboost/c_api.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../src/c_api/c_api_error.h"
#include "../../src/c_api/c_api_utils.h"  // MakeSparseFromPtr
#include "../../src/common/threading_utils.h"

#include "./xgboost_R.h"  // Must follow other includes.
#include "Rinternals.h"

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

using dmlc::BeginPtr;

xgboost::Context const *BoosterCtx(BoosterHandle handle) {
  CHECK_HANDLE();
  auto *learner = static_cast<xgboost::Learner *>(handle);
  CHECK(learner);
  return learner->Ctx();
}

xgboost::Context const *DMatrixCtx(DMatrixHandle handle) {
  CHECK_HANDLE();
  auto p_m = static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  CHECK(p_m);
  return p_m->get()->Ctx();
}

XGB_DLL SEXP XGCheckNullPtr_R(SEXP handle) {
  return ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
}

XGB_DLL void _DMatrixFinalizer(SEXP ext) {
  R_API_BEGIN();
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(XGDMatrixFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
  R_API_END();
}

XGB_DLL SEXP XGBSetGlobalConfig_R(SEXP json_str) {
  R_API_BEGIN();
  CHECK_CALL(XGBSetGlobalConfig(CHAR(asChar(json_str))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBGetGlobalConfig_R() {
  const char* json_str;
  R_API_BEGIN();
  CHECK_CALL(XGBGetGlobalConfig(&json_str));
  R_API_END();
  return mkString(json_str);
}

XGB_DLL SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent) {
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

XGB_DLL SEXP XGDMatrixCreateFromMat_R(SEXP mat, SEXP missing, SEXP n_threads) {
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
  xgboost::Context ctx;
  ctx.nthread = asInteger(n_threads);
  std::int32_t threads = ctx.Threads();

  xgboost::common::ParallelFor(nrow, threads, [&](xgboost::omp_ulong i) {
    for (size_t j = 0; j < ncol; ++j) {
      data[i * ncol + j] = is_int ? static_cast<float>(iin[i + nrow * j]) : din[i + nrow * j];
    }
  });
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromMat_omp(BeginPtr(data), nrow, ncol,
                                        asReal(missing), &handle, threads));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

namespace {
void CreateFromSparse(SEXP indptr, SEXP indices, SEXP data, std::string *indptr_str,
                      std::string *indices_str, std::string *data_str) {
  const int *p_indptr = INTEGER(indptr);
  const int *p_indices = INTEGER(indices);
  const double *p_data = REAL(data);

  auto nindptr = static_cast<std::size_t>(length(indptr));
  auto ndata = static_cast<std::size_t>(length(data));
  CHECK_EQ(ndata, p_indptr[nindptr - 1]);
  xgboost::detail::MakeSparseFromPtr(p_indptr, p_indices, p_data, nindptr, indptr_str, indices_str,
                                     data_str);
}
}  // namespace

XGB_DLL SEXP XGDMatrixCreateFromCSC_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_row,
                                      SEXP missing, SEXP n_threads) {
  SEXP ret;
  R_API_BEGIN();
  std::int32_t threads = asInteger(n_threads);

  using xgboost::Integer;
  using xgboost::Json;
  using xgboost::Object;

  std::string sindptr, sindices, sdata;
  CreateFromSparse(indptr, indices, data, &sindptr, &sindices, &sdata);
  auto nrow = static_cast<std::size_t>(INTEGER(num_row)[0]);

  DMatrixHandle handle;
  Json jconfig{Object{}};
  // Construct configuration
  jconfig["nthread"] = Integer{threads};
  jconfig["missing"] = xgboost::Number{asReal(missing)};
  std::string config;
  Json::Dump(jconfig, &config);
  CHECK_CALL(XGDMatrixCreateFromCSC(sindptr.c_str(), sindices.c_str(), sdata.c_str(), nrow,
                                    config.c_str(), &handle));

  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));

  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromCSR_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_col,
                                      SEXP missing, SEXP n_threads) {
  SEXP ret;
  R_API_BEGIN();
  std::int32_t threads = asInteger(n_threads);

  using xgboost::Integer;
  using xgboost::Json;
  using xgboost::Object;

  std::string sindptr, sindices, sdata;
  CreateFromSparse(indptr, indices, data, &sindptr, &sindices, &sdata);
  auto ncol = static_cast<std::size_t>(INTEGER(num_col)[0]);

  DMatrixHandle handle;
  Json jconfig{Object{}};
  // Construct configuration
  jconfig["nthread"] = Integer{threads};
  jconfig["missing"] = xgboost::Number{asReal(missing)};
  std::string config;
  Json::Dump(jconfig, &config);
  CHECK_CALL(XGDMatrixCreateFromCSR(sindptr.c_str(), sindices.c_str(), sdata.c_str(), ncol,
                                    config.c_str(), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));

  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset) {
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

XGB_DLL SEXP XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                                 CHAR(asChar(fname)),
                                 asInteger(silent)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  int len = length(array);
  const char *name = CHAR(asChar(field));
  auto ctx = DMatrixCtx(R_ExternalPtrAddr(handle));
  if (!strcmp("group", name)) {
    std::vector<unsigned> vec(len);
    xgboost::common::ParallelFor(len, ctx->Threads(), [&](xgboost::omp_ulong i) {
      vec[i] = static_cast<unsigned>(INTEGER(array)[i]);
    });
    CHECK_CALL(
        XGDMatrixSetUIntInfo(R_ExternalPtrAddr(handle), CHAR(asChar(field)), BeginPtr(vec), len));
  } else {
    std::vector<float> vec(len);
    xgboost::common::ParallelFor(len, ctx->Threads(),
                                 [&](xgboost::omp_ulong i) { vec[i] = REAL(array)[i]; });
    CHECK_CALL(
        XGDMatrixSetFloatInfo(R_ExternalPtrAddr(handle), CHAR(asChar(field)), BeginPtr(vec), len));
  }
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixSetStrFeatureInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  size_t len{0};
  if (!isNull(array)) {
    len = length(array);
  }

  const char *name = CHAR(asChar(field));
  std::vector<std::string> str_info;
  for (size_t i = 0; i < len; ++i) {
    str_info.emplace_back(CHAR(asChar(VECTOR_ELT(array, i))));
  }
  std::vector<char const*> vec(len);
  std::transform(str_info.cbegin(), str_info.cend(), vec.begin(),
                 [](std::string const &str) { return str.c_str(); });
  CHECK_CALL(XGDMatrixSetStrFeatureInfo(R_ExternalPtrAddr(handle), name, vec.data(), len));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixGetStrFeatureInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  char const **out_features{nullptr};
  bst_ulong len{0};
  const char *name = CHAR(asChar(field));
  XGDMatrixGetStrFeatureInfo(R_ExternalPtrAddr(handle), name, &len, &out_features);

  if (len > 0) {
    ret = PROTECT(allocVector(STRSXP, len));
    for (size_t i = 0; i < len; ++i) {
      SET_STRING_ELT(ret, i, mkChar(out_features[i]));
    }
  } else {
    ret = PROTECT(R_NilValue);
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const float *res;
  CHECK_CALL(XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle), CHAR(asChar(field)), &olen, &res));
  ret = PROTECT(allocVector(REALSXP, olen));
  for (size_t i = 0; i < olen; ++i) {
    REAL(ret)[i] = res[i];
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixNumRow_R(SEXP handle) {
  bst_ulong nrow;
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixNumRow(R_ExternalPtrAddr(handle), &nrow));
  R_API_END();
  return ScalarInteger(static_cast<int>(nrow));
}

XGB_DLL SEXP XGDMatrixNumCol_R(SEXP handle) {
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

XGB_DLL SEXP XGBoosterCreate_R(SEXP dmats) {
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

XGB_DLL SEXP XGBoosterCreateInEmptyObj_R(SEXP dmats, SEXP R_handle) {
  R_API_BEGIN();
  int len = length(dmats);
  std::vector<void*> dvec;
  for (int i = 0; i < len; ++i) {
    dvec.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
  }
  BoosterHandle handle;
  CHECK_CALL(XGBoosterCreate(BeginPtr(dvec), dvec.size(), &handle));
  R_SetExternalPtrAddr(R_handle, handle);
  R_RegisterCFinalizerEx(R_handle, _BoosterFinalizer, TRUE);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterSetParam(R_ExternalPtrAddr(handle),
                               CHAR(asChar(name)),
                               CHAR(asChar(val))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                                  asInteger(iter),
                                  R_ExternalPtrAddr(dtrain)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess) {
  R_API_BEGIN();
  CHECK_EQ(length(grad), length(hess))
      << "gradient and hess must have same length";
  int len = length(grad);
  std::vector<float> tgrad(len), thess(len);
  auto ctx = BoosterCtx(R_ExternalPtrAddr(handle));
  xgboost::common::ParallelFor(len, ctx->Threads(), [&](xgboost::omp_ulong j) {
    tgrad[j] = REAL(grad)[j];
    thess[j] = REAL(hess)[j];
  });
  CHECK_CALL(XGBoosterBoostOneIter(R_ExternalPtrAddr(handle),
                                 R_ExternalPtrAddr(dtrain),
                                 BeginPtr(tgrad), BeginPtr(thess),
                                 len));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
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

XGB_DLL SEXP XGBoosterPredictFromDMatrix_R(SEXP handle, SEXP dmat, SEXP json_config)  {
  SEXP r_out_shape;
  SEXP r_out_result;
  SEXP r_out;

  R_API_BEGIN();
  char const *c_json_config = CHAR(asChar(json_config));

  bst_ulong out_dim;
  bst_ulong const *out_shape;
  float const *out_result;
  CHECK_CALL(XGBoosterPredictFromDMatrix(R_ExternalPtrAddr(handle),
                                         R_ExternalPtrAddr(dmat), c_json_config,
                                         &out_shape, &out_dim, &out_result));

  r_out_shape = PROTECT(allocVector(INTSXP, out_dim));
  size_t len = 1;
  for (size_t i = 0; i < out_dim; ++i) {
    INTEGER(r_out_shape)[i] = out_shape[i];
    len *= out_shape[i];
  }
  r_out_result = PROTECT(allocVector(REALSXP, len));
  auto ctx = BoosterCtx(R_ExternalPtrAddr(handle));
  xgboost::common::ParallelFor(len, ctx->Threads(), [&](xgboost::omp_ulong i) {
    REAL(r_out_result)[i] = out_result[i];
  });

  r_out = PROTECT(allocVector(VECSXP, 2));

  SET_VECTOR_ELT(r_out, 0, r_out_shape);
  SET_VECTOR_ELT(r_out, 1, r_out_result);

  R_API_END();
  UNPROTECT(3);

  return r_out;
}

XGB_DLL SEXP XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModelFromBuffer(R_ExternalPtrAddr(handle),
                                          RAW(raw),
                                          length(raw)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSaveModelToRaw_R(SEXP handle, SEXP json_config) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  char const *c_json_config = CHAR(asChar(json_config));
  char const *raw;
  CHECK_CALL(XGBoosterSaveModelToBuffer(R_ExternalPtrAddr(handle), c_json_config, &olen, &raw))
  ret = PROTECT(allocVector(RAWSXP, olen));
  if (olen != 0) {
    std::memcpy(RAW(ret), raw, olen);
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGBoosterSaveJsonConfig_R(SEXP handle) {
  const char* ret;
  R_API_BEGIN();
  bst_ulong len {0};
  CHECK_CALL(XGBoosterSaveJsonConfig(R_ExternalPtrAddr(handle),
                                     &len,
                                     &ret));
  R_API_END();
  return mkString(ret);
}

XGB_DLL SEXP XGBoosterLoadJsonConfig_R(SEXP handle, SEXP value) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadJsonConfig(R_ExternalPtrAddr(handle), CHAR(asChar(value))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSerializeToBuffer_R(SEXP handle) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong out_len;
  const char *raw;
  CHECK_CALL(XGBoosterSerializeToBuffer(R_ExternalPtrAddr(handle), &out_len, &raw));
  ret = PROTECT(allocVector(RAWSXP, out_len));
  if (out_len != 0) {
    memcpy(RAW(ret), raw, out_len);
  }
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGBoosterUnserializeFromBuffer_R(SEXP handle, SEXP raw) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterUnserializeFromBuffer(R_ExternalPtrAddr(handle),
                                 RAW(raw),
                                 length(raw)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats, SEXP dump_format) {
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

XGB_DLL SEXP XGBoosterGetAttr_R(SEXP handle, SEXP name) {
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

XGB_DLL SEXP XGBoosterSetAttr_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  const char *v = isNull(val) ? nullptr : CHAR(asChar(val));
  CHECK_CALL(XGBoosterSetAttr(R_ExternalPtrAddr(handle),
                              CHAR(asChar(name)), v));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterGetAttrNames_R(SEXP handle) {
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

XGB_DLL SEXP XGBoosterFeatureScore_R(SEXP handle, SEXP json_config) {
  SEXP out_features_sexp;
  SEXP out_scores_sexp;
  SEXP out_shape_sexp;
  SEXP r_out;

  R_API_BEGIN();
  char const *c_json_config = CHAR(asChar(json_config));
  bst_ulong out_n_features;
  char const **out_features;

  bst_ulong out_dim;
  bst_ulong const *out_shape;
  float const *out_scores;

  CHECK_CALL(XGBoosterFeatureScore(R_ExternalPtrAddr(handle), c_json_config,
                                   &out_n_features, &out_features,
                                   &out_dim, &out_shape, &out_scores));
  out_shape_sexp = PROTECT(allocVector(INTSXP, out_dim));
  size_t len = 1;
  for (size_t i = 0; i < out_dim; ++i) {
    INTEGER(out_shape_sexp)[i] = out_shape[i];
    len *= out_shape[i];
  }

  out_scores_sexp = PROTECT(allocVector(REALSXP, len));
  auto ctx = BoosterCtx(R_ExternalPtrAddr(handle));
  xgboost::common::ParallelFor(len, ctx->Threads(), [&](xgboost::omp_ulong i) {
    REAL(out_scores_sexp)[i] = out_scores[i];
  });

  out_features_sexp = PROTECT(allocVector(STRSXP, out_n_features));
  for (size_t i = 0; i < out_n_features; ++i) {
    SET_STRING_ELT(out_features_sexp, i, mkChar(out_features[i]));
  }

  r_out = PROTECT(allocVector(VECSXP, 3));
  SET_VECTOR_ELT(r_out, 0, out_features_sexp);
  SET_VECTOR_ELT(r_out, 1, out_shape_sexp);
  SET_VECTOR_ELT(r_out, 2, out_scores_sexp);

  R_API_END();
  UNPROTECT(4);

  return r_out;
}
