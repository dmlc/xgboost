/**
 * Copyright 2014-2024, XGBoost Contributors
 */
#include <dmlc/common.h>
#include <dmlc/omp.h>
#include <xgboost/c_api.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../src/c_api/c_api_error.h"
#include "../../src/c_api/c_api_utils.h"  // MakeSparseFromPtr
#include "../../src/common/threading_utils.h"
#include "../../src/data/array_interface.h"  // for ArrayInterface

#include "./xgboost_R.h"  // Must follow other includes.

#ifdef _MSC_VER
#error "Compilation of R package with MSVC is not supported due to issues handling R headers"
#endif

namespace {

/* Note: this class is used as a throwable exception.
Some xgboost C functions that use callbacks will catch exceptions
that happen inside of the callback execution, hence it purposefully
doesn't inherit from 'std::exception' even if used as such. */
struct ErrorWithUnwind {};

void ThrowExceptionFromRError(void *, Rboolean jump) {
  if (jump) {
    throw ErrorWithUnwind();
  }
}

struct PtrToConstChar {
  const char *ptr;
};

SEXP WrappedMkChar(void *void_ptr) {
  return Rf_mkChar(static_cast<PtrToConstChar*>(void_ptr)->ptr);
}

SEXP SafeMkChar(const char *c_str, SEXP continuation_token) {
  PtrToConstChar ptr_struct{c_str};
  return R_UnwindProtect(
    WrappedMkChar, static_cast<void*>(&ptr_struct),
    ThrowExceptionFromRError, nullptr,
    continuation_token);
}

struct RFunAndEnv {
  SEXP R_fun;
  SEXP R_calling_env;
};

SEXP WrappedExecFun(void *void_ptr) {
  RFunAndEnv *r_fun_and_env = static_cast<RFunAndEnv*>(void_ptr);
  SEXP f_expr = Rf_protect(Rf_lang1(r_fun_and_env->R_fun));
  SEXP out = Rf_protect(Rf_eval(f_expr, r_fun_and_env->R_calling_env));
  Rf_unprotect(2);
  return out;
}

SEXP SafeExecFun(SEXP R_fun, SEXP R_calling_env, SEXP continuation_token) {
  RFunAndEnv r_fun_and_env{R_fun, R_calling_env};
  return R_UnwindProtect(
    WrappedExecFun, static_cast<void*>(&r_fun_and_env),
    ThrowExceptionFromRError, nullptr,
    continuation_token);
}

SEXP WrappedAllocReal(void *void_ptr) {
  size_t *size = static_cast<size_t*>(void_ptr);
  return Rf_allocVector(REALSXP, *size);
}

SEXP SafeAllocReal(size_t size, SEXP continuation_token) {
  return R_UnwindProtect(
    WrappedAllocReal, static_cast<void*>(&size),
    ThrowExceptionFromRError, nullptr,
    continuation_token);
}

SEXP WrappedAllocInteger(void *void_ptr) {
  size_t *size = static_cast<size_t*>(void_ptr);
  return Rf_allocVector(INTSXP, *size);
}

SEXP SafeAllocInteger(size_t size, SEXP continuation_token) {
  return R_UnwindProtect(
    WrappedAllocInteger, static_cast<void*>(&size),
    ThrowExceptionFromRError, nullptr,
    continuation_token);
}

[[nodiscard]] std::string MakeArrayInterfaceFromRMat(SEXP R_mat) {
  SEXP mat_dims = Rf_getAttrib(R_mat, R_DimSymbol);
  if (Rf_xlength(mat_dims) > 2) {
    LOG(FATAL) << "Passed input array with more than two dimensions, which is not supported.";
  }
  const int *ptr_mat_dims = INTEGER(mat_dims);

  // Lambda for type dispatch.
  auto make_matrix = [=](auto const *ptr) {
    using namespace xgboost;  // NOLINT
    using T = std::remove_pointer_t<decltype(ptr)>;

    auto m = linalg::MatrixView<T>{
        common::Span{ptr,
          static_cast<std::size_t>(ptr_mat_dims[0]) * static_cast<std::size_t>(ptr_mat_dims[1])},
        {ptr_mat_dims[0], ptr_mat_dims[1]},  // Shape
        DeviceOrd::CPU(),
        linalg::Order::kF  // R uses column-major
    };
    CHECK(m.FContiguous());
    return linalg::ArrayInterfaceStr(m);
  };

  const SEXPTYPE arr_type = TYPEOF(R_mat);
  switch (arr_type) {
    case REALSXP:
      return make_matrix(REAL(R_mat));
    case INTSXP:
      return make_matrix(INTEGER(R_mat));
    case LGLSXP:
      return make_matrix(LOGICAL(R_mat));
    default:
      LOG(FATAL) << "Array or matrix has unsupported type.";
  }

  LOG(FATAL) << "Not reachable";
  return "";
}

[[nodiscard]] std::string MakeArrayInterfaceFromRVector(SEXP R_vec) {
  const size_t vec_len = Rf_xlength(R_vec);

  // Lambda for type dispatch.
  auto make_vec = [=](auto const *ptr) {
    using namespace xgboost;  // NOLINT
    auto v = linalg::MakeVec(ptr, vec_len);
    return linalg::ArrayInterfaceStr(v);
  };

  const SEXPTYPE arr_type = TYPEOF(R_vec);
  switch (arr_type) {
    case REALSXP:
      return make_vec(REAL(R_vec));
    case INTSXP:
      return make_vec(INTEGER(R_vec));
    case LGLSXP:
      return make_vec(LOGICAL(R_vec));
    default:
      LOG(FATAL) << "Array or matrix has unsupported type.";
  }

  LOG(FATAL) << "Not reachable";
  return "";
}

[[nodiscard]] std::string MakeArrayInterfaceFromRDataFrame(SEXP R_df) {
  auto make_vec = [&](auto const *ptr, std::size_t len) {
    auto v = xgboost::linalg::MakeVec(ptr, len);
    return xgboost::linalg::ArrayInterface(v);
  };

  R_xlen_t n_features = Rf_xlength(R_df);
  std::vector<xgboost::Json> array(n_features);
  CHECK_GT(n_features, 0);
  std::size_t len = Rf_xlength(VECTOR_ELT(R_df, 0));

  // The `data.frame` in R actually converts all data into numeric. The other type
  // handlers here are not used. At the moment they are kept as a reference for when we
  // can avoid making data copies during transformation.
  for (R_xlen_t i = 0; i < n_features; ++i) {
    switch (TYPEOF(VECTOR_ELT(R_df, i))) {
      case INTSXP: {
        auto const *ptr = INTEGER(VECTOR_ELT(R_df, i));
        array[i] = make_vec(ptr, len);
        break;
      }
      case REALSXP: {
        auto const *ptr = REAL(VECTOR_ELT(R_df, i));
        array[i] = make_vec(ptr, len);
        break;
      }
      case LGLSXP: {
        auto const *ptr = LOGICAL(VECTOR_ELT(R_df, i));
        array[i] = make_vec(ptr, len);
        break;
      }
      default: {
        LOG(FATAL) << "data.frame has unsupported type.";
      }
    }
  }

  xgboost::Json jinterface{std::move(array)};
  return xgboost::Json::Dump(jinterface);
}

void AddMissingToJson(xgboost::Json *jconfig, SEXP missing, SEXPTYPE arr_type) {
  if (Rf_isNull(missing) || ISNAN(Rf_asReal(missing))) {
    // missing is not specified
    if (arr_type == REALSXP) {
      (*jconfig)["missing"] = std::numeric_limits<double>::quiet_NaN();
    } else {
      (*jconfig)["missing"] = R_NaInt;
    }
  } else {
    // missing specified
    (*jconfig)["missing"] = Rf_asReal(missing);
  }
}

[[nodiscard]] std::string MakeJsonConfigForArray(SEXP missing, SEXP n_threads, SEXPTYPE arr_type) {
  using namespace ::xgboost;  // NOLINT
  Json jconfig{Object{}};
  AddMissingToJson(&jconfig, missing, arr_type);
  jconfig["nthread"] = Rf_asInteger(n_threads);
  return Json::Dump(jconfig);
}

// Allocate a R vector and copy an array interface encoded object to it.
[[nodiscard]] SEXP CopyArrayToR(const char *array_str, SEXP ctoken) {
  xgboost::ArrayInterface<1> array{xgboost::StringView{array_str}};
  // R supports only int and double.
  bool is_int_type =
      xgboost::DispatchDType(array.type, [](auto t) { return std::is_integral_v<decltype(t)>; });
  bool is_float = xgboost::DispatchDType(
      array.type, [](auto v) { return std::is_floating_point_v<decltype(v)>; });
  CHECK(is_int_type || is_float) << "Internal error: Invalid DType.";
  CHECK(array.is_contiguous) << "Internal error: Return by XGBoost should be contiguous";

  // Note: the only case in which this will receive an integer type is
  // for the 'indptr' part of the quantile cut outputs, which comes
  // in sorted order, so the last element contains the maximum value.
  bool fits_into_C_int = xgboost::DispatchDType(array.type, [&](auto t) {
    using T = decltype(t);
    if (!std::is_integral_v<decltype(t)>) {
      return false;
    }
    auto ptr = static_cast<T const *>(array.data);
    T last_elt = ptr[array.n - 1];
    if (last_elt < 0) {
      last_elt = -last_elt;  // no std::abs overload for all possible types
    }
    return last_elt <= std::numeric_limits<int>::max();
  });
  bool use_int = is_int_type && fits_into_C_int;

  // Allocate memory in R
  SEXP out =
      Rf_protect(use_int ? SafeAllocInteger(array.n, ctoken) : SafeAllocReal(array.n, ctoken));

  xgboost::DispatchDType(array.type, [&](auto t) {
    using T = decltype(t);
    auto in_ptr = static_cast<T const *>(array.data);
    if (use_int) {
      auto out_ptr = INTEGER(out);
      std::copy_n(in_ptr, array.n, out_ptr);
    } else {
      auto out_ptr = REAL(out);
      std::copy_n(in_ptr, array.n, out_ptr);
    }
  });

  Rf_unprotect(1);
  return out;
}
}  // namespace

struct RRNGStateController {
  RRNGStateController() {
    GetRNGstate();
  }

  ~RRNGStateController() {
    PutRNGstate();
  }
};

/*!
 * \brief macro to annotate begin of api
 */
#define R_API_BEGIN()                           \
  try {                                         \
    RRNGStateController rng_controller{};

/* Note: an R error triggers a long jump, hence all C++ objects that
allocated memory through non-R allocators, including the exception
object, need to be destructed before triggering the R error.
In order to preserve the error message, it gets copied to a temporary
buffer, and the R error section is reached through a 'goto' statement
that bypasses usual function control flow. */
char cpp_ex_msg[512];
/*!
 * \brief macro to annotate end of api
 */
#define R_API_END()                             \
  } catch(std::exception &e) {                  \
    std::strncpy(cpp_ex_msg, e.what(), 512);    \
    goto throw_cpp_ex_as_R_err;                 \
  }                                             \
  if (false) {                                  \
    throw_cpp_ex_as_R_err:                      \
    Rf_error("%s", cpp_ex_msg);                 \
  }

/**
 * @brief Macro for checking XGBoost return code.
 */
#define CHECK_CALL(__rc)               \
  if ((__rc) != 0) {                   \
    Rf_error("%s", XGBGetLastError()); \
  }

using dmlc::BeginPtr;

XGB_DLL SEXP XGCheckNullPtr_R(SEXP handle) {
  return Rf_ScalarLogical(R_ExternalPtrAddr(handle) == nullptr);
}

XGB_DLL SEXP XGSetArrayDimNamesInplace_R(SEXP arr, SEXP dim_names) {
  Rf_setAttrib(arr, R_DimNamesSymbol, dim_names);
  return R_NilValue;
}

XGB_DLL SEXP XGSetVectorNamesInplace_R(SEXP arr, SEXP names) {
  Rf_setAttrib(arr, R_NamesSymbol, names);
  return R_NilValue;
}

namespace {
void _DMatrixFinalizer(SEXP ext) {
  R_API_BEGIN();
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(XGDMatrixFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
  R_API_END();
}
} /* namespace */

XGB_DLL SEXP XGBSetGlobalConfig_R(SEXP json_str) {
  R_API_BEGIN();
  CHECK_CALL(XGBSetGlobalConfig(CHAR(Rf_asChar(json_str))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBGetGlobalConfig_R() {
  const char* json_str;
  R_API_BEGIN();
  CHECK_CALL(XGBGetGlobalConfig(&json_str));
  R_API_END();
  return Rf_mkString(json_str);
}

XGB_DLL SEXP XGDMatrixCreateFromURI_R(SEXP uri, SEXP silent, SEXP data_split_mode) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  SEXP uri_char = Rf_protect(Rf_asChar(uri));
  const char *uri_ptr = CHAR(uri_char);
  R_API_BEGIN();
  xgboost::Json jconfig{xgboost::Object{}};
  jconfig["uri"] = std::string(uri_ptr);
  jconfig["silent"] = Rf_asLogical(silent);
  jconfig["data_split_mode"] = Rf_asInteger(data_split_mode);
  const std::string sconfig = xgboost::Json::Dump(jconfig);
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromURI(sconfig.c_str(), &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(2);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromMat_R(SEXP mat, SEXP missing, SEXP n_threads) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();

  DMatrixHandle handle;
  int res_code;
  {
    auto array_str = MakeArrayInterfaceFromRMat(mat);
    auto config_str = MakeJsonConfigForArray(missing, n_threads, TYPEOF(mat));

    res_code = XGDMatrixCreateFromDense(array_str.c_str(), config_str.c_str(), &handle);
  }
  CHECK_CALL(res_code);
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromDF_R(SEXP df, SEXP missing, SEXP n_threads) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();

  DMatrixHandle handle;
  std::int32_t rc{0};
  {
    const std::string sinterface = MakeArrayInterfaceFromRDataFrame(df);
    xgboost::Json jconfig{xgboost::Object{}};
    jconfig["missing"] = Rf_asReal(missing);
    jconfig["nthread"] = Rf_asInteger(n_threads);
    std::string sconfig = xgboost::Json::Dump(jconfig);

    rc = XGDMatrixCreateFromColumnar(sinterface.c_str(), sconfig.c_str(), &handle);
  }

  CHECK_CALL(rc);
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(1);

  return ret;
}

namespace {
void CreateFromSparse(SEXP indptr, SEXP indices, SEXP data, std::string *indptr_str,
                      std::string *indices_str, std::string *data_str) {
  const int *p_indptr = INTEGER(indptr);
  const int *p_indices = INTEGER(indices);
  const double *p_data = REAL(data);

  auto nindptr = static_cast<std::size_t>(Rf_xlength(indptr));
  auto ndata = static_cast<std::size_t>(Rf_xlength(data));
  CHECK_EQ(ndata, p_indptr[nindptr - 1]);
  xgboost::detail::MakeSparseFromPtr(p_indptr, p_indices, p_data, nindptr, indptr_str, indices_str,
                                     data_str);
}
}  // namespace

XGB_DLL SEXP XGDMatrixCreateFromCSC_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_row,
                                      SEXP missing, SEXP n_threads) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  std::int32_t threads = Rf_asInteger(n_threads);
  DMatrixHandle handle;

  int res_code;
  {
    using xgboost::Integer;
    using xgboost::Json;
    using xgboost::Object;
    std::string sindptr, sindices, sdata;
    CreateFromSparse(indptr, indices, data, &sindptr, &sindices, &sdata);
    auto nrow = static_cast<std::size_t>(INTEGER(num_row)[0]);

    Json jconfig{Object{}};
    // Construct configuration
    jconfig["nthread"] = Integer{threads};
    AddMissingToJson(&jconfig, missing, TYPEOF(data));
    std::string config;
    Json::Dump(jconfig, &config);
    res_code = XGDMatrixCreateFromCSC(sindptr.c_str(), sindices.c_str(), sdata.c_str(), nrow,
                                      config.c_str(), &handle);
  }
  CHECK_CALL(res_code);

  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromCSR_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_col,
                                      SEXP missing, SEXP n_threads) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  std::int32_t threads = Rf_asInteger(n_threads);
  DMatrixHandle handle;

  int res_code;
  {
    using xgboost::Integer;
    using xgboost::Json;
    using xgboost::Object;

    std::string sindptr, sindices, sdata;
    CreateFromSparse(indptr, indices, data, &sindptr, &sindices, &sdata);
    auto ncol = static_cast<std::size_t>(INTEGER(num_col)[0]);

    Json jconfig{Object{}};
    // Construct configuration
    jconfig["nthread"] = Integer{threads};
    AddMissingToJson(&jconfig, missing, TYPEOF(data));
    std::string config;
    Json::Dump(jconfig, &config);
    res_code = XGDMatrixCreateFromCSR(sindptr.c_str(), sindices.c_str(), sdata.c_str(), ncol,
                                      config.c_str(), &handle);
  }
  CHECK_CALL(res_code);
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset, SEXP allow_groups) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  R_xlen_t len = Rf_xlength(idxset);
  const int *idxset_ = INTEGER(idxset);
  DMatrixHandle res;

  int res_code;
  {
    std::vector<int> idxvec(len);
    #ifndef _MSC_VER
    #pragma omp simd
    #endif
    for (R_xlen_t i = 0; i < len; ++i) {
      idxvec[i] = idxset_[i] - 1;
    }
    res_code = XGDMatrixSliceDMatrixEx(R_ExternalPtrAddr(handle),
                                       BeginPtr(idxvec), len,
                                       &res,
                                       Rf_asLogical(allow_groups));
  }
  CHECK_CALL(res_code);
  R_SetExternalPtrAddr(ret, res);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                                 CHAR(Rf_asChar(fname)),
                                 Rf_asInteger(silent)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  SEXP field_ = Rf_protect(Rf_asChar(field));
  SEXP arr_dim = Rf_getAttrib(array, R_DimSymbol);
  int res_code;
  {
    const std::string array_str = Rf_isNull(arr_dim)?
      MakeArrayInterfaceFromRVector(array) : MakeArrayInterfaceFromRMat(array);
    res_code = XGDMatrixSetInfoFromInterface(
      R_ExternalPtrAddr(handle), CHAR(field_), array_str.c_str());
  }
  CHECK_CALL(res_code);
  Rf_unprotect(1);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixSetStrFeatureInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  size_t len{0};
  if (!Rf_isNull(array)) {
    len = Rf_xlength(array);
  }

  SEXP str_info_holder = Rf_protect(Rf_allocVector(VECSXP, len));
  if (TYPEOF(array) == STRSXP) {
    for (size_t i = 0; i < len; ++i) {
      SET_VECTOR_ELT(str_info_holder, i, STRING_ELT(array, i));
    }
  } else {
    for (size_t i = 0; i < len; ++i) {
      SET_VECTOR_ELT(str_info_holder, i, Rf_asChar(VECTOR_ELT(array, i)));
    }
  }

  SEXP field_ = Rf_protect(Rf_asChar(field));
  const char *name = CHAR(field_);
  int res_code;
  {
    std::vector<std::string> str_info;
    str_info.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      str_info.emplace_back(CHAR(VECTOR_ELT(str_info_holder, i)));
    }
    std::vector<char const*> vec(len);
    std::transform(str_info.cbegin(), str_info.cend(), vec.begin(),
                   [](std::string const &str) { return str.c_str(); });
    res_code = XGDMatrixSetStrFeatureInfo(R_ExternalPtrAddr(handle), name, vec.data(), len);
  }
  CHECK_CALL(res_code);
  Rf_unprotect(2);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixGetStrFeatureInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  char const **out_features{nullptr};
  bst_ulong len{0};
  const char *name = CHAR(Rf_asChar(field));
  XGDMatrixGetStrFeatureInfo(R_ExternalPtrAddr(handle), name, &len, &out_features);

  if (len > 0) {
    ret = Rf_protect(Rf_allocVector(STRSXP, len));
    for (size_t i = 0; i < len; ++i) {
      SET_STRING_ELT(ret, i, Rf_mkChar(out_features[i]));
    }
  } else {
    ret = Rf_protect(R_NilValue);
  }
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixGetFloatInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const float *res;
  CHECK_CALL(XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(field)), &olen, &res));
  ret = Rf_protect(Rf_allocVector(REALSXP, olen));
  std::copy(res, res + olen, REAL(ret));
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixGetUIntInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const unsigned *res;
  CHECK_CALL(XGDMatrixGetUIntInfo(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(field)), &olen, &res));
  ret = Rf_protect(Rf_allocVector(INTSXP, olen));
  std::copy(res, res + olen, INTEGER(ret));
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixNumRow_R(SEXP handle) {
  bst_ulong nrow;
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixNumRow(R_ExternalPtrAddr(handle), &nrow));
  R_API_END();
  return Rf_ScalarInteger(static_cast<int>(nrow));
}

XGB_DLL SEXP XGDMatrixNumCol_R(SEXP handle) {
  bst_ulong ncol;
  R_API_BEGIN();
  CHECK_CALL(XGDMatrixNumCol(R_ExternalPtrAddr(handle), &ncol));
  R_API_END();
  return Rf_ScalarInteger(static_cast<int>(ncol));
}

XGB_DLL SEXP XGProxyDMatrixCreate_R() {
  SEXP out = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  DMatrixHandle proxy_dmat_handle;
  CHECK_CALL(XGProxyDMatrixCreate(&proxy_dmat_handle));
  R_SetExternalPtrAddr(out, proxy_dmat_handle);
  R_RegisterCFinalizerEx(out, _DMatrixFinalizer, TRUE);
  Rf_unprotect(1);
  R_API_END();
  return out;
}

XGB_DLL SEXP XGProxyDMatrixSetDataDense_R(SEXP handle, SEXP R_mat) {
  R_API_BEGIN();
  DMatrixHandle proxy_dmat = R_ExternalPtrAddr(handle);
  int res_code;
  {
    std::string array_str = MakeArrayInterfaceFromRMat(R_mat);
    res_code = XGProxyDMatrixSetDataDense(proxy_dmat, array_str.c_str());
  }
  CHECK_CALL(res_code);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGProxyDMatrixSetDataCSR_R(SEXP handle, SEXP lst) {
  R_API_BEGIN();
  DMatrixHandle proxy_dmat = R_ExternalPtrAddr(handle);
  int res_code;
  {
    std::string array_str_indptr = MakeArrayInterfaceFromRVector(VECTOR_ELT(lst, 0));
    std::string array_str_indices = MakeArrayInterfaceFromRVector(VECTOR_ELT(lst, 1));
    std::string array_str_data = MakeArrayInterfaceFromRVector(VECTOR_ELT(lst, 2));
    const int ncol = Rf_asInteger(VECTOR_ELT(lst, 3));
    res_code = XGProxyDMatrixSetDataCSR(proxy_dmat,
                                        array_str_indptr.c_str(),
                                        array_str_indices.c_str(),
                                        array_str_data.c_str(),
                                        ncol);
  }
  CHECK_CALL(res_code);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGProxyDMatrixSetDataColumnar_R(SEXP handle, SEXP lst) {
  R_API_BEGIN();
  DMatrixHandle proxy_dmat = R_ExternalPtrAddr(handle);
  int res_code;
  {
    std::string sinterface = MakeArrayInterfaceFromRDataFrame(lst);
    res_code = XGProxyDMatrixSetDataColumnar(proxy_dmat, sinterface.c_str());
  }
  CHECK_CALL(res_code);
  R_API_END();
  return R_NilValue;
}

namespace {

struct _RDataIterator {
  SEXP f_next;
  SEXP f_reset;
  SEXP calling_env;
  SEXP continuation_token;

  _RDataIterator(
    SEXP f_next, SEXP f_reset, SEXP calling_env, SEXP continuation_token) :
  f_next(f_next), f_reset(f_reset), calling_env(calling_env),
  continuation_token(continuation_token) {}

  void reset() {
    SafeExecFun(this->f_reset, this->calling_env, this->continuation_token);
  }

  int next() {
    SEXP R_res = Rf_protect(
      SafeExecFun(this->f_next, this->calling_env, this->continuation_token));
    int res = Rf_asInteger(R_res);
    Rf_unprotect(1);
    return res;
  }
};

void _reset_RDataIterator(DataIterHandle iter) {
  static_cast<_RDataIterator*>(iter)->reset();
}

int _next_RDataIterator(DataIterHandle iter) {
  return static_cast<_RDataIterator*>(iter)->next();
}

SEXP XGDMatrixCreateFromCallbackGeneric_R(
  SEXP f_next, SEXP f_reset, SEXP calling_env, SEXP proxy_dmat,
  SEXP n_threads, SEXP missing, SEXP max_bin, SEXP ref_dmat,
  SEXP cache_prefix, bool as_quantile_dmatrix) {
  SEXP continuation_token = Rf_protect(R_MakeUnwindCont());
  SEXP out = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  DMatrixHandle out_dmat;

  int res_code;
  try {
    _RDataIterator data_iterator(f_next, f_reset, calling_env, continuation_token);

    std::string str_cache_prefix;
    xgboost::Json jconfig{xgboost::Object{}};
    jconfig["missing"] = Rf_asReal(missing);
    if (!Rf_isNull(n_threads)) {
      jconfig["nthread"] = Rf_asInteger(n_threads);
    }
    if (as_quantile_dmatrix) {
      if (!Rf_isNull(max_bin)) {
        jconfig["max_bin"] = Rf_asInteger(max_bin);
      }
    } else {
      str_cache_prefix = std::string(CHAR(Rf_asChar(cache_prefix)));
      jconfig["cache_prefix"] = str_cache_prefix;
    }
    std::string json_str = xgboost::Json::Dump(jconfig);

    DMatrixHandle ref_dmat_handle = nullptr;
    if (as_quantile_dmatrix && !Rf_isNull(ref_dmat)) {
      ref_dmat_handle = R_ExternalPtrAddr(ref_dmat);
    }

    if (as_quantile_dmatrix) {
      res_code = XGQuantileDMatrixCreateFromCallback(
        &data_iterator,
        R_ExternalPtrAddr(proxy_dmat),
        ref_dmat_handle,
        _reset_RDataIterator,
        _next_RDataIterator,
        json_str.c_str(),
        &out_dmat);
    } else {
      res_code = XGDMatrixCreateFromCallback(
        &data_iterator,
        R_ExternalPtrAddr(proxy_dmat),
        _reset_RDataIterator,
        _next_RDataIterator,
        json_str.c_str(),
        &out_dmat);
    }
  } catch (ErrorWithUnwind &e) {
    R_ContinueUnwind(continuation_token);
  }
  CHECK_CALL(res_code);

  R_SetExternalPtrAddr(out, out_dmat);
  R_RegisterCFinalizerEx(out, _DMatrixFinalizer, TRUE);
  Rf_unprotect(2);
  R_API_END();
  return out;
}

} /* namespace */

XGB_DLL SEXP XGQuantileDMatrixCreateFromCallback_R(
  SEXP f_next, SEXP f_reset, SEXP calling_env, SEXP proxy_dmat,
  SEXP n_threads, SEXP missing, SEXP max_bin, SEXP ref_dmat) {
  return XGDMatrixCreateFromCallbackGeneric_R(
    f_next, f_reset, calling_env, proxy_dmat,
    n_threads, missing, max_bin, ref_dmat,
    R_NilValue, true);
}

XGB_DLL SEXP XGDMatrixCreateFromCallback_R(
  SEXP f_next, SEXP f_reset, SEXP calling_env, SEXP proxy_dmat,
  SEXP n_threads, SEXP missing, SEXP cache_prefix) {
  return XGDMatrixCreateFromCallbackGeneric_R(
    f_next, f_reset, calling_env, proxy_dmat,
    n_threads, missing, R_NilValue, R_NilValue,
    cache_prefix, false);
}

XGB_DLL SEXP XGDMatrixFree_R(SEXP proxy_dmat) {
  _DMatrixFinalizer(proxy_dmat);
  return R_NilValue;
}

XGB_DLL SEXP XGGetRNAIntAsDouble() {
  double sentinel_as_double = static_cast<double>(R_NaInt);
  return Rf_ScalarReal(sentinel_as_double);
}

XGB_DLL SEXP XGDuplicate_R(SEXP obj) {
  return Rf_duplicate(obj);
}

XGB_DLL SEXP XGPointerEqComparison_R(SEXP obj1, SEXP obj2) {
  return Rf_ScalarLogical(R_ExternalPtrAddr(obj1) == R_ExternalPtrAddr(obj2));
}

XGB_DLL SEXP XGDMatrixGetQuantileCut_R(SEXP handle) {
  const char *out_names[] = {"indptr", "data", ""};
  SEXP continuation_token = Rf_protect(R_MakeUnwindCont());
  SEXP out = Rf_protect(Rf_mkNamed(VECSXP, out_names));
  R_API_BEGIN();
  const char *out_indptr;
  const char *out_data;
  CHECK_CALL(XGDMatrixGetQuantileCut(R_ExternalPtrAddr(handle), "{}", &out_indptr, &out_data));
  try {
    SET_VECTOR_ELT(out, 0, CopyArrayToR(out_indptr, continuation_token));
    SET_VECTOR_ELT(out, 1, CopyArrayToR(out_data, continuation_token));
  } catch (ErrorWithUnwind &e) {
    R_ContinueUnwind(continuation_token);
  }
  R_API_END();
  Rf_unprotect(2);
  return out;
}

XGB_DLL SEXP XGDMatrixNumNonMissing_R(SEXP handle) {
  SEXP out = Rf_protect(Rf_allocVector(REALSXP, 1));
  R_API_BEGIN();
  bst_ulong out_;
  CHECK_CALL(XGDMatrixNumNonMissing(R_ExternalPtrAddr(handle), &out_));
  REAL(out)[0] = static_cast<double>(out_);
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGDMatrixGetDataAsCSR_R(SEXP handle) {
  const char *out_names[] = {"indptr", "indices", "data", "ncols", ""};
  SEXP out = Rf_protect(Rf_mkNamed(VECSXP, out_names));
  R_API_BEGIN();

  bst_ulong nrows, ncols, nnz;
  CHECK_CALL(XGDMatrixNumRow(R_ExternalPtrAddr(handle), &nrows));
  CHECK_CALL(XGDMatrixNumCol(R_ExternalPtrAddr(handle), &ncols));
  CHECK_CALL(XGDMatrixNumNonMissing(R_ExternalPtrAddr(handle), &nnz));
  if (std::max(nrows, ncols) > std::numeric_limits<int>::max()) {
    Rf_error("%s", "Error: resulting DMatrix data does not fit into R 'dgRMatrix'.");
  }

  SET_VECTOR_ELT(out, 0, Rf_allocVector(INTSXP, nrows + 1));
  SET_VECTOR_ELT(out, 1, Rf_allocVector(INTSXP, nnz));
  SET_VECTOR_ELT(out, 2, Rf_allocVector(REALSXP, nnz));
  SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(ncols));

  std::unique_ptr<bst_ulong[]> indptr(new bst_ulong[nrows + 1]);
  std::unique_ptr<unsigned[]> indices(new unsigned[nnz]);
  std::unique_ptr<float[]> data(new float[nnz]);

  CHECK_CALL(XGDMatrixGetDataAsCSR(R_ExternalPtrAddr(handle),
                                   "{}",
                                   indptr.get(),
                                   indices.get(),
                                   data.get()));

  std::copy(indptr.get(), indptr.get() + nrows + 1, INTEGER(VECTOR_ELT(out, 0)));
  std::copy(indices.get(), indices.get() + nnz, INTEGER(VECTOR_ELT(out, 1)));
  std::copy(data.get(), data.get() + nnz, REAL(VECTOR_ELT(out, 2)));

  R_API_END();
  Rf_unprotect(1);
  return out;
}

// functions related to booster
namespace {
void _BoosterFinalizer(SEXP R_ptr) {
  if (R_ExternalPtrAddr(R_ptr) == NULL) return;
  CHECK_CALL(XGBoosterFree(R_ExternalPtrAddr(R_ptr)));
  R_ClearExternalPtr(R_ptr);
}

/* Booster is represented as an altrep list with one element which
corresponds to an 'externalptr' holding the C object, forbidding
modification by not implementing setters, and adding custom serialization. */
R_altrep_class_t XGBAltrepPointerClass;

R_xlen_t XGBAltrepPointerLength_R(SEXP R_altrepped_obj) {
  return 1;
}

SEXP XGBAltrepPointerGetElt_R(SEXP R_altrepped_obj, R_xlen_t idx) {
  return R_altrep_data1(R_altrepped_obj);
}

SEXP XGBMakeEmptyAltrep() {
  SEXP class_name = Rf_protect(Rf_mkString("xgb.Booster"));
  SEXP elt_names = Rf_protect(Rf_mkString("ptr"));
  SEXP R_ptr = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  SEXP R_altrepped_obj = Rf_protect(R_new_altrep(XGBAltrepPointerClass, R_ptr, R_NilValue));
  Rf_setAttrib(R_altrepped_obj, R_NamesSymbol, elt_names);
  Rf_setAttrib(R_altrepped_obj, R_ClassSymbol, class_name);
  Rf_unprotect(4);
  return R_altrepped_obj;
}

/* Note: the idea for separating this function from the one above is to be
able to trigger all R allocations first before doing non-R allocations. */
void XGBAltrepSetPointer(SEXP R_altrepped_obj, BoosterHandle handle) {
  SEXP R_ptr = R_altrep_data1(R_altrepped_obj);
  R_SetExternalPtrAddr(R_ptr, handle);
  R_RegisterCFinalizerEx(R_ptr, _BoosterFinalizer, TRUE);
}

SEXP XGBAltrepSerializer_R(SEXP R_altrepped_obj) {
  R_API_BEGIN();
  BoosterHandle handle = R_ExternalPtrAddr(R_altrep_data1(R_altrepped_obj));
  char const *serialized_bytes;
  bst_ulong serialized_length;
  CHECK_CALL(XGBoosterSerializeToBuffer(
    handle, &serialized_length, &serialized_bytes));
  SEXP R_state = Rf_protect(Rf_allocVector(RAWSXP, serialized_length));
  if (serialized_length != 0) {
    std::memcpy(RAW(R_state), serialized_bytes, serialized_length);
  }
  Rf_unprotect(1);
  return R_state;
  R_API_END();
  return R_NilValue; /* <- should not be reached */
}

SEXP XGBAltrepDeserializer_R(SEXP unused, SEXP R_state) {
  SEXP R_altrepped_obj = Rf_protect(XGBMakeEmptyAltrep());
  R_API_BEGIN();
  BoosterHandle handle = nullptr;
  CHECK_CALL(XGBoosterCreate(nullptr, 0, &handle));
  int res_code = XGBoosterUnserializeFromBuffer(handle,
                                                RAW(R_state),
                                                Rf_xlength(R_state));
  if (res_code != 0) {
    XGBoosterFree(handle);
  }
  CHECK_CALL(res_code);
  XGBAltrepSetPointer(R_altrepped_obj, handle);
  R_API_END();
  Rf_unprotect(1);
  return R_altrepped_obj;
}

// https://purrple.cat/blog/2018/10/14/altrep-and-cpp/
Rboolean XGBAltrepInspector_R(
  SEXP x, int pre, int deep, int pvec,
  void (*inspect_subtree)(SEXP, int, int, int)) {
  Rprintf("Altrepped external pointer [address:%p]\n",
          R_ExternalPtrAddr(R_altrep_data1(x)));
  return TRUE;
}

SEXP XGBAltrepDuplicate_R(SEXP R_altrepped_obj, Rboolean deep) {
  R_API_BEGIN();
  if (!deep) {
    SEXP out = Rf_protect(XGBMakeEmptyAltrep());
    R_set_altrep_data1(out, R_altrep_data1(R_altrepped_obj));
    Rf_unprotect(1);
    return out;
  } else {
    SEXP out = Rf_protect(XGBMakeEmptyAltrep());
    char const *serialized_bytes;
    bst_ulong serialized_length;
    CHECK_CALL(XGBoosterSerializeToBuffer(
      R_ExternalPtrAddr(R_altrep_data1(R_altrepped_obj)),
      &serialized_length, &serialized_bytes));
    BoosterHandle new_handle = nullptr;
    CHECK_CALL(XGBoosterCreate(nullptr, 0, &new_handle));
    int res_code = XGBoosterUnserializeFromBuffer(new_handle,
                                                  serialized_bytes,
                                                  serialized_length);
    if (res_code != 0) {
      XGBoosterFree(new_handle);
    }
    CHECK_CALL(res_code);
    XGBAltrepSetPointer(out, new_handle);
    Rf_unprotect(1);
    return out;
  }
  R_API_END();
  return R_NilValue; /* <- should not be reached */
}

} /* namespace */

XGB_DLL void XGBInitializeAltrepClass_R(DllInfo *dll) {
  XGBAltrepPointerClass = R_make_altlist_class("XGBAltrepPointerClass", "xgboost", dll);
  R_set_altrep_Length_method(XGBAltrepPointerClass, XGBAltrepPointerLength_R);
  R_set_altlist_Elt_method(XGBAltrepPointerClass, XGBAltrepPointerGetElt_R);
  R_set_altrep_Inspect_method(XGBAltrepPointerClass, XGBAltrepInspector_R);
  R_set_altrep_Serialized_state_method(XGBAltrepPointerClass, XGBAltrepSerializer_R);
  R_set_altrep_Unserialize_method(XGBAltrepPointerClass, XGBAltrepDeserializer_R);
  R_set_altrep_Duplicate_method(XGBAltrepPointerClass, XGBAltrepDuplicate_R);
}

XGB_DLL SEXP XGBoosterCreate_R(SEXP dmats) {
  SEXP out = Rf_protect(XGBMakeEmptyAltrep());
  R_API_BEGIN();
  R_xlen_t len = Rf_xlength(dmats);
  BoosterHandle handle;

  int res_code;
  {
    std::vector<void*> dvec(len);
    for (R_xlen_t i = 0; i < len; ++i) {
      dvec[i] = R_ExternalPtrAddr(VECTOR_ELT(dmats, i));
    }
    res_code = XGBoosterCreate(BeginPtr(dvec), dvec.size(), &handle);
  }
  CHECK_CALL(res_code);
  XGBAltrepSetPointer(out, handle);
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGBoosterCopyInfoFromDMatrix_R(SEXP booster, SEXP dmat) {
  R_API_BEGIN();
  char const **feature_names;
  bst_ulong len_feature_names = 0;
  CHECK_CALL(XGDMatrixGetStrFeatureInfo(R_ExternalPtrAddr(dmat),
                                        "feature_name",
                                        &len_feature_names,
                                        &feature_names));
  if (len_feature_names) {
    CHECK_CALL(XGBoosterSetStrFeatureInfo(R_ExternalPtrAddr(booster),
                                          "feature_name",
                                          feature_names,
                                          len_feature_names));
  }

  char const **feature_types;
  bst_ulong len_feature_types = 0;
  CHECK_CALL(XGDMatrixGetStrFeatureInfo(R_ExternalPtrAddr(dmat),
                                        "feature_type",
                                        &len_feature_types,
                                        &feature_types));
  if (len_feature_types) {
    CHECK_CALL(XGBoosterSetStrFeatureInfo(R_ExternalPtrAddr(booster),
                                          "feature_type",
                                          feature_types,
                                          len_feature_types));
  }
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSetStrFeatureInfo_R(SEXP handle, SEXP field, SEXP features) {
  R_API_BEGIN();
  SEXP field_char = Rf_protect(Rf_asChar(field));
  bst_ulong len_features = Rf_xlength(features);

  int res_code;
  {
    std::vector<const char*> str_arr(len_features);
    for (bst_ulong idx = 0; idx < len_features; idx++) {
      str_arr[idx] = CHAR(STRING_ELT(features, idx));
    }
    res_code = XGBoosterSetStrFeatureInfo(R_ExternalPtrAddr(handle),
                                          CHAR(field_char),
                                          str_arr.data(),
                                          len_features);
  }
  CHECK_CALL(res_code);
  Rf_unprotect(1);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterGetStrFeatureInfo_R(SEXP handle, SEXP field) {
  R_API_BEGIN();
  bst_ulong len;
  const char **out_features;
  SEXP field_char = Rf_protect(Rf_asChar(field));
  CHECK_CALL(XGBoosterGetStrFeatureInfo(R_ExternalPtrAddr(handle),
                                        CHAR(field_char), &len, &out_features));
  SEXP out = Rf_protect(Rf_allocVector(STRSXP, len));
  for (bst_ulong idx = 0; idx < len; idx++) {
    SET_STRING_ELT(out, idx, Rf_mkChar(out_features[idx]));
  }
  Rf_unprotect(2);
  return out;
  R_API_END();
  return R_NilValue; /* <- should not be reached */
}

XGB_DLL SEXP XGBoosterBoostedRounds_R(SEXP handle) {
  SEXP out = Rf_protect(Rf_allocVector(INTSXP, 1));
  R_API_BEGIN();
  CHECK_CALL(XGBoosterBoostedRounds(R_ExternalPtrAddr(handle), INTEGER(out)));
  R_API_END();
  Rf_unprotect(1);
  return out;
}

/* Note: R's integer class is 32-bit-and-signed only, while xgboost
supports more, so it returns it as a floating point instead */
XGB_DLL SEXP XGBoosterGetNumFeature_R(SEXP handle) {
  SEXP out = Rf_protect(Rf_allocVector(REALSXP, 1));
  R_API_BEGIN();
  bst_ulong res;
  CHECK_CALL(XGBoosterGetNumFeature(R_ExternalPtrAddr(handle), &res));
  REAL(out)[0] = static_cast<double>(res);
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  SEXP name_ = Rf_protect(Rf_asChar(name));
  SEXP val_ = Rf_protect(Rf_asChar(val));
  CHECK_CALL(XGBoosterSetParam(R_ExternalPtrAddr(handle),
                               CHAR(name_),
                               CHAR(val_)));
  Rf_unprotect(2);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                                    Rf_asInteger(iter),
                                    R_ExternalPtrAddr(dtrain)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterTrainOneIter_R(SEXP handle, SEXP dtrain, SEXP iter, SEXP grad, SEXP hess) {
  R_API_BEGIN();
  CHECK_EQ(Rf_xlength(grad), Rf_xlength(hess)) << "gradient and hess must have same length.";
  SEXP gdim = Rf_getAttrib(grad, R_DimSymbol);
  SEXP hdim = Rf_getAttrib(hess, R_DimSymbol);

  int res_code;
  {
    const std::string s_grad = Rf_isNull(gdim)?
      MakeArrayInterfaceFromRVector(grad) : MakeArrayInterfaceFromRMat(grad);
    const std::string s_hess = Rf_isNull(hdim)?
      MakeArrayInterfaceFromRVector(hess) : MakeArrayInterfaceFromRMat(hess);
    res_code = XGBoosterTrainOneIter(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(dtrain),
                                     Rf_asInteger(iter), s_grad.c_str(), s_hess.c_str());
  }
  CHECK_CALL(res_code);

  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
  const char *ret;
  R_API_BEGIN();
  CHECK_EQ(Rf_xlength(dmats), Rf_xlength(evnames))
      << "dmats and evnams must have same length";
  R_xlen_t len = Rf_xlength(dmats);
  SEXP evnames_lst = Rf_protect(Rf_allocVector(VECSXP, len));
  for (R_xlen_t i = 0; i < len; i++) {
    SET_VECTOR_ELT(evnames_lst, i, Rf_asChar(VECTOR_ELT(evnames, i)));
  }

  int res_code;
  {
    std::vector<void*> vec_dmats(len);
    std::vector<std::string> vec_names;
    vec_names.reserve(len);
    std::vector<const char*> vec_sptr(len);
    for (R_xlen_t i = 0; i < len; ++i) {
      vec_dmats[i] = R_ExternalPtrAddr(VECTOR_ELT(dmats, i));
      vec_names.emplace_back(CHAR(VECTOR_ELT(evnames_lst, i)));
    }
    for (R_xlen_t i = 0; i < len; ++i) {
      vec_sptr[i] = vec_names[i].c_str();
    }
    res_code = XGBoosterEvalOneIter(R_ExternalPtrAddr(handle),
                                    Rf_asInteger(iter),
                                    BeginPtr(vec_dmats),
                                    BeginPtr(vec_sptr),
                                    len, &ret);
  }
  CHECK_CALL(res_code);
  Rf_unprotect(1);
  R_API_END();
  return Rf_mkString(ret);
}

namespace {

struct ProxyDmatrixError : public std::exception {};

struct ProxyDmatrixWrapper {
  DMatrixHandle proxy_dmat_handle;

  ProxyDmatrixWrapper() {
    int res_code = XGProxyDMatrixCreate(&this->proxy_dmat_handle);
    if (res_code != 0) {
      throw ProxyDmatrixError();
    }
  }

  ~ProxyDmatrixWrapper() {
    if (this->proxy_dmat_handle) {
      XGDMatrixFree(this->proxy_dmat_handle);
      this->proxy_dmat_handle = nullptr;
    }
  }

  DMatrixHandle get_handle() {
    return this->proxy_dmat_handle;
  }
};

std::unique_ptr<ProxyDmatrixWrapper> GetProxyDMatrixWithBaseMargin(SEXP base_margin) {
  if (Rf_isNull(base_margin)) {
    return std::unique_ptr<ProxyDmatrixWrapper>(nullptr);
  }

  SEXP base_margin_dim = Rf_getAttrib(base_margin, R_DimSymbol);
  int res_code;
  try {
    const std::string array_str = Rf_isNull(base_margin_dim)?
      MakeArrayInterfaceFromRVector(base_margin) : MakeArrayInterfaceFromRMat(base_margin);
    std::unique_ptr<ProxyDmatrixWrapper> proxy_dmat(new ProxyDmatrixWrapper());
    res_code = XGDMatrixSetInfoFromInterface(proxy_dmat->get_handle(),
                                             "base_margin",
                                             array_str.c_str());
    if (res_code != 0) {
      throw ProxyDmatrixError();
    }
    return proxy_dmat;
  } catch(ProxyDmatrixError &err) {
    Rf_error("%s", XGBGetLastError());
  }
}

enum class PredictionInputType {DMatrix, DenseMatrix, CSRMatrix, DataFrame};

SEXP XGBoosterPredictGeneric(SEXP handle, SEXP input_data, SEXP json_config,
                                    PredictionInputType input_type, SEXP missing,
                                    SEXP base_margin) {
  SEXP r_out_result = R_NilValue;
  R_API_BEGIN();
  SEXP json_config_ = Rf_protect(Rf_asChar(json_config));
  char const *c_json_config = CHAR(json_config_);

  bst_ulong out_dim;
  bst_ulong const *out_shape;
  float const *out_result;

  int res_code;
  {
    switch (input_type) {
      case PredictionInputType::DMatrix: {
        res_code = XGBoosterPredictFromDMatrix(R_ExternalPtrAddr(handle),
                                               R_ExternalPtrAddr(input_data), c_json_config,
                                               &out_shape, &out_dim, &out_result);
        break;
      }

      case PredictionInputType::CSRMatrix: {
        std::unique_ptr<ProxyDmatrixWrapper> proxy_dmat = GetProxyDMatrixWithBaseMargin(
          base_margin);
        DMatrixHandle proxy_dmat_handle = proxy_dmat.get()? proxy_dmat->get_handle() : nullptr;

        SEXP indptr = VECTOR_ELT(input_data, 0);
        SEXP indices = VECTOR_ELT(input_data, 1);
        SEXP data = VECTOR_ELT(input_data, 2);
        const int ncol_csr = Rf_asInteger(VECTOR_ELT(input_data, 3));
        const SEXPTYPE type_data = TYPEOF(data);
        CHECK_EQ(type_data, REALSXP);
        std::string sindptr, sindices, sdata;
        CreateFromSparse(indptr, indices, data, &sindptr, &sindices, &sdata);

        xgboost::StringView json_str(c_json_config);
        xgboost::Json new_json = xgboost::Json::Load(json_str);
        AddMissingToJson(&new_json, missing, type_data);
        const std::string new_c_json = xgboost::Json::Dump(new_json);

        res_code = XGBoosterPredictFromCSR(
          R_ExternalPtrAddr(handle), sindptr.c_str(), sindices.c_str(), sdata.c_str(),
          ncol_csr, new_c_json.c_str(), proxy_dmat_handle, &out_shape, &out_dim, &out_result);
        break;
      }

      case PredictionInputType::DenseMatrix: {
        std::unique_ptr<ProxyDmatrixWrapper> proxy_dmat = GetProxyDMatrixWithBaseMargin(
          base_margin);
        DMatrixHandle proxy_dmat_handle = proxy_dmat.get()? proxy_dmat->get_handle() : nullptr;
        const std::string array_str = MakeArrayInterfaceFromRMat(input_data);

        xgboost::StringView json_str(c_json_config);
        xgboost::Json new_json = xgboost::Json::Load(json_str);
        AddMissingToJson(&new_json, missing, TYPEOF(input_data));
        const std::string new_c_json = xgboost::Json::Dump(new_json);

        res_code = XGBoosterPredictFromDense(
          R_ExternalPtrAddr(handle), array_str.c_str(), new_c_json.c_str(),
          proxy_dmat_handle, &out_shape, &out_dim, &out_result);
        break;
      }

      case PredictionInputType::DataFrame: {
        std::unique_ptr<ProxyDmatrixWrapper> proxy_dmat = GetProxyDMatrixWithBaseMargin(
          base_margin);
        DMatrixHandle proxy_dmat_handle = proxy_dmat.get()? proxy_dmat->get_handle() : nullptr;

        const std::string df_str = MakeArrayInterfaceFromRDataFrame(input_data);

        xgboost::StringView json_str(c_json_config);
        xgboost::Json new_json = xgboost::Json::Load(json_str);
        AddMissingToJson(&new_json, missing, REALSXP);
        const std::string new_c_json = xgboost::Json::Dump(new_json);

        res_code = XGBoosterPredictFromColumnar(
          R_ExternalPtrAddr(handle), df_str.c_str(), new_c_json.c_str(),
          proxy_dmat_handle, &out_shape, &out_dim, &out_result);
        break;
      }
    }
  }
  CHECK_CALL(res_code);

  SEXP r_out_shape = Rf_protect(Rf_allocVector(INTSXP, out_dim));
  size_t len = 1;
  int *r_out_shape_ = INTEGER(r_out_shape);
  for (size_t i = 0; i < out_dim; ++i) {
    r_out_shape_[out_dim - i - 1] = out_shape[i];
    len *= out_shape[i];
  }
  r_out_result = Rf_protect(Rf_allocVector(REALSXP, len));
  std::copy(out_result, out_result + len, REAL(r_out_result));

  if (out_dim > 1) {
    Rf_setAttrib(r_out_result, R_DimSymbol, r_out_shape);
  }

  R_API_END();
  Rf_unprotect(3);

  return r_out_result;
}

}  // namespace

XGB_DLL SEXP XGBoosterPredictFromDMatrix_R(SEXP handle, SEXP dmat, SEXP json_config)  {
  return XGBoosterPredictGeneric(handle, dmat, json_config,
                                 PredictionInputType::DMatrix, R_NilValue, R_NilValue);
}

XGB_DLL SEXP XGBoosterPredictFromDense_R(SEXP handle, SEXP R_mat, SEXP missing,
                                         SEXP json_config, SEXP base_margin) {
  return XGBoosterPredictGeneric(handle, R_mat, json_config,
                                 PredictionInputType::DenseMatrix, missing, base_margin);
}

XGB_DLL SEXP XGBoosterPredictFromCSR_R(SEXP handle, SEXP lst, SEXP missing,
                                       SEXP json_config, SEXP base_margin) {
  return XGBoosterPredictGeneric(handle, lst, json_config,
                                 PredictionInputType::CSRMatrix, missing, base_margin);
}

XGB_DLL SEXP XGBoosterPredictFromColumnar_R(SEXP handle, SEXP R_df, SEXP missing,
                                            SEXP json_config, SEXP base_margin) {
  return XGBoosterPredictGeneric(handle, R_df, json_config,
                                 PredictionInputType::DataFrame, missing, base_margin);
}

XGB_DLL SEXP XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(fname))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(fname))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadModelFromBuffer(R_ExternalPtrAddr(handle),
                                          RAW(raw),
                                          Rf_xlength(raw)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSaveModelToRaw_R(SEXP handle, SEXP json_config) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  char const *c_json_config = CHAR(Rf_asChar(json_config));
  char const *raw;
  CHECK_CALL(XGBoosterSaveModelToBuffer(R_ExternalPtrAddr(handle), c_json_config, &olen, &raw))
  ret = Rf_protect(Rf_allocVector(RAWSXP, olen));
  if (olen != 0) {
    std::memcpy(RAW(ret), raw, olen);
  }
  R_API_END();
  Rf_unprotect(1);
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
  return Rf_mkString(ret);
}

XGB_DLL SEXP XGBoosterLoadJsonConfig_R(SEXP handle, SEXP value) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterLoadJsonConfig(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(value))));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSerializeToBuffer_R(SEXP handle) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong out_len;
  const char *raw;
  CHECK_CALL(XGBoosterSerializeToBuffer(R_ExternalPtrAddr(handle), &out_len, &raw));
  ret = Rf_protect(Rf_allocVector(RAWSXP, out_len));
  if (out_len != 0) {
    memcpy(RAW(ret), raw, out_len);
  }
  R_API_END();
  Rf_unprotect(1);
  return ret;
}

XGB_DLL SEXP XGBoosterUnserializeFromBuffer_R(SEXP handle, SEXP raw) {
  R_API_BEGIN();
  CHECK_CALL(XGBoosterUnserializeFromBuffer(R_ExternalPtrAddr(handle),
                                 RAW(raw),
                                 Rf_xlength(raw)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats, SEXP dump_format) {
  SEXP out;
  SEXP continuation_token = Rf_protect(R_MakeUnwindCont());
  SEXP dump_format_ = Rf_protect(Rf_asChar(dump_format));
  SEXP fmap_ = Rf_protect(Rf_asChar(fmap));
  R_API_BEGIN();
  bst_ulong olen;
  const char **res;
  const char *fmt = CHAR(dump_format_);
  CHECK_CALL(XGBoosterDumpModelEx(R_ExternalPtrAddr(handle),
                                CHAR(fmap_),
                                Rf_asInteger(with_stats),
                                fmt,
                                &olen, &res));
  out = Rf_protect(Rf_allocVector(STRSXP, olen));
  try {
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
      const std::string temp_str = stream.str();
      SET_STRING_ELT(out, 0, SafeMkChar(temp_str.c_str(), continuation_token));
    } else {
      for (size_t i = 0; i < olen; ++i) {
        std::stringstream stream;
        stream <<  "booster[" << i <<"]\n" << res[i];
        const std::string temp_str = stream.str();
        SET_STRING_ELT(out, i, SafeMkChar(temp_str.c_str(), continuation_token));
      }
    }
  } catch (ErrorWithUnwind &e) {
    R_ContinueUnwind(continuation_token);
  }
  R_API_END();
  Rf_unprotect(4);
  return out;
}

XGB_DLL SEXP XGBoosterGetAttr_R(SEXP handle, SEXP name) {
  SEXP out;
  R_API_BEGIN();
  int success;
  const char *val;
  CHECK_CALL(XGBoosterGetAttr(R_ExternalPtrAddr(handle),
                              CHAR(Rf_asChar(name)),
                              &val,
                              &success));
  if (success) {
    out = Rf_protect(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(out, 0, Rf_mkChar(val));
  } else {
    out = Rf_protect(R_NilValue);
  }
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGBoosterSetAttr_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  const char *v = nullptr;
  SEXP name_ = Rf_protect(Rf_asChar(name));
  SEXP val_;
  int n_protected = 1;
  if (!Rf_isNull(val)) {
    val_ = Rf_protect(Rf_asChar(val));
    n_protected++;
    v = CHAR(val_);
  }

  CHECK_CALL(XGBoosterSetAttr(R_ExternalPtrAddr(handle),
                              CHAR(name_), v));
  Rf_unprotect(n_protected);
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
    out = Rf_protect(Rf_allocVector(STRSXP, len));
    for (size_t i = 0; i < len; ++i) {
      SET_STRING_ELT(out, i, Rf_mkChar(res[i]));
    }
  } else {
    out = Rf_protect(R_NilValue);
  }
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGBoosterFeatureScore_R(SEXP handle, SEXP json_config) {
  SEXP out_features_sexp;
  SEXP out_scores_sexp;
  SEXP out_shape_sexp;
  SEXP r_out = Rf_protect(Rf_allocVector(VECSXP, 3));

  R_API_BEGIN();
  char const *c_json_config = CHAR(Rf_asChar(json_config));
  bst_ulong out_n_features;
  char const **out_features;

  bst_ulong out_dim;
  bst_ulong const *out_shape;
  float const *out_scores;

  CHECK_CALL(XGBoosterFeatureScore(R_ExternalPtrAddr(handle), c_json_config,
                                   &out_n_features, &out_features,
                                   &out_dim, &out_shape, &out_scores));
  out_shape_sexp = Rf_protect(Rf_allocVector(INTSXP, out_dim));
  size_t len = 1;
  int *out_shape_sexp_ = INTEGER(out_shape_sexp);
  for (size_t i = 0; i < out_dim; ++i) {
    out_shape_sexp_[i] = out_shape[i];
    len *= out_shape[i];
  }

  out_features_sexp = Rf_protect(Rf_allocVector(STRSXP, out_n_features));
  for (size_t i = 0; i < out_n_features; ++i) {
    SET_STRING_ELT(out_features_sexp, i, Rf_mkChar(out_features[i]));
  }

  out_scores_sexp = Rf_protect(Rf_allocVector(REALSXP, len));
  std::copy(out_scores, out_scores + len, REAL(out_scores_sexp));

  SET_VECTOR_ELT(r_out, 0, out_features_sexp);
  SET_VECTOR_ELT(r_out, 1, out_shape_sexp);
  SET_VECTOR_ELT(r_out, 2, out_scores_sexp);

  R_API_END();
  Rf_unprotect(4);

  return r_out;
}

XGB_DLL SEXP XGBoosterSlice_R(SEXP handle, SEXP begin_layer, SEXP end_layer, SEXP step) {
  SEXP out = Rf_protect(XGBMakeEmptyAltrep());
  R_API_BEGIN();
  BoosterHandle handle_out = nullptr;
  CHECK_CALL(XGBoosterSlice(R_ExternalPtrAddr(handle),
                            Rf_asInteger(begin_layer),
                            Rf_asInteger(end_layer),
                            Rf_asInteger(step),
                            &handle_out));
  XGBAltrepSetPointer(out, handle_out);
  R_API_END();
  Rf_unprotect(1);
  return out;
}

XGB_DLL SEXP XGBoosterSliceAndReplace_R(SEXP handle, SEXP begin_layer, SEXP end_layer, SEXP step) {
  R_API_BEGIN();
  BoosterHandle old_handle = R_ExternalPtrAddr(handle);
  BoosterHandle new_handle = nullptr;
  CHECK_CALL(XGBoosterSlice(old_handle,
                            Rf_asInteger(begin_layer),
                            Rf_asInteger(end_layer),
                            Rf_asInteger(step),
                            &new_handle));
  R_SetExternalPtrAddr(handle, new_handle);
  CHECK_CALL(XGBoosterFree(old_handle));
  R_API_END();
  return R_NilValue;
}
