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

namespace {
struct ErrorWithUnwind : public std::exception {};

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

[[nodiscard]] std::string MakeJsonConfigForArray(SEXP missing, SEXP n_threads, SEXPTYPE arr_type) {
  using namespace ::xgboost;  // NOLINT
  Json jconfig{Object{}};

  const SEXPTYPE missing_type = TYPEOF(missing);
  if (Rf_isNull(missing) || (missing_type == REALSXP && ISNAN(Rf_asReal(missing))) ||
      (missing_type == LGLSXP && Rf_asLogical(missing) == R_NaInt) ||
      (missing_type == INTSXP && Rf_asInteger(missing) == R_NaInt)) {
    // missing is not specified
    if (arr_type == REALSXP) {
      jconfig["missing"] = std::numeric_limits<double>::quiet_NaN();
    } else {
      jconfig["missing"] = R_NaInt;
    }
  } else {
    // missing specified
    jconfig["missing"] = Rf_asReal(missing);
  }

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
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  DMatrixHandle handle;
  CHECK_CALL(XGDMatrixCreateFromFile(CHAR(asChar(fname)), asInteger(silent), &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromMat_R(SEXP mat, SEXP missing, SEXP n_threads) {
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
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
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromDF_R(SEXP df, SEXP missing, SEXP n_threads) {
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();

  DMatrixHandle handle;

  auto make_vec = [&](auto const *ptr, std::int32_t len) {
    auto v = xgboost::linalg::MakeVec(ptr, len);
    return xgboost::linalg::ArrayInterface(v);
  };

  std::int32_t rc{0};
  {
    using xgboost::Json;
    auto n_features = Rf_xlength(df);
    std::vector<Json> array(n_features);
    CHECK_GT(n_features, 0);
    auto len = Rf_xlength(VECTOR_ELT(df, 0));
    // The `data.frame` in R actually converts all data into numeric. The other type
    // handlers here are not used. At the moment they are kept as a reference for when we
    // can avoid making data copies during transformation.
    for (decltype(n_features) i = 0; i < n_features; ++i) {
      switch (TYPEOF(VECTOR_ELT(df, i))) {
        case INTSXP: {
          auto const *ptr = INTEGER(VECTOR_ELT(df, i));
          array[i] = make_vec(ptr, len);
          break;
        }
        case REALSXP: {
          auto const *ptr = REAL(VECTOR_ELT(df, i));
          array[i] = make_vec(ptr, len);
          break;
        }
        case LGLSXP: {
          auto const *ptr = LOGICAL(VECTOR_ELT(df, i));
          array[i] = make_vec(ptr, len);
          break;
        }
        default: {
          LOG(FATAL) << "data.frame has unsupported type.";
        }
      }
    }

    Json jinterface{std::move(array)};
    auto sinterface = Json::Dump(jinterface);
    Json jconfig{xgboost::Object{}};
    jconfig["missing"] = asReal(missing);
    jconfig["nthread"] = asInteger(n_threads);
    auto sconfig = Json::Dump(jconfig);

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
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  std::int32_t threads = asInteger(n_threads);
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
    jconfig["missing"] = xgboost::Number{asReal(missing)};
    std::string config;
    Json::Dump(jconfig, &config);
    res_code = XGDMatrixCreateFromCSC(sindptr.c_str(), sindices.c_str(), sdata.c_str(), nrow,
                                      config.c_str(), &handle);
  }
  CHECK_CALL(res_code);

  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixCreateFromCSR_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_col,
                                      SEXP missing, SEXP n_threads) {
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  R_API_BEGIN();
  std::int32_t threads = asInteger(n_threads);
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
    jconfig["missing"] = xgboost::Number{asReal(missing)};
    std::string config;
    Json::Dump(jconfig, &config);
    res_code = XGDMatrixCreateFromCSR(sindptr.c_str(), sindices.c_str(), sdata.c_str(), ncol,
                                      config.c_str(), &handle);
  }
  CHECK_CALL(res_code);
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset) {
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
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
                                       0);
  }
  CHECK_CALL(res_code);
  R_SetExternalPtrAddr(ret, res);
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
  SEXP field_ = PROTECT(Rf_asChar(field));
  SEXP arr_dim = Rf_getAttrib(array, R_DimSymbol);
  int res_code;
  {
    const std::string array_str = Rf_isNull(arr_dim)?
      MakeArrayInterfaceFromRVector(array) : MakeArrayInterfaceFromRMat(array);
    res_code = XGDMatrixSetInfoFromInterface(
      R_ExternalPtrAddr(handle), CHAR(field_), array_str.c_str());
  }
  CHECK_CALL(res_code);
  UNPROTECT(1);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGDMatrixSetStrFeatureInfo_R(SEXP handle, SEXP field, SEXP array) {
  R_API_BEGIN();
  size_t len{0};
  if (!isNull(array)) {
    len = Rf_xlength(array);
  }

  SEXP str_info_holder = PROTECT(Rf_allocVector(VECSXP, len));
  for (size_t i = 0; i < len; ++i) {
    SET_VECTOR_ELT(str_info_holder, i, Rf_asChar(VECTOR_ELT(array, i)));
  }

  SEXP field_ = PROTECT(Rf_asChar(field));
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
  UNPROTECT(2);
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

XGB_DLL SEXP XGDMatrixGetFloatInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const float *res;
  CHECK_CALL(XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle), CHAR(asChar(field)), &olen, &res));
  ret = PROTECT(allocVector(REALSXP, olen));
  std::copy(res, res + olen, REAL(ret));
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGDMatrixGetUIntInfo_R(SEXP handle, SEXP field) {
  SEXP ret;
  R_API_BEGIN();
  bst_ulong olen;
  const unsigned *res;
  CHECK_CALL(XGDMatrixGetUIntInfo(R_ExternalPtrAddr(handle), CHAR(asChar(field)), &olen, &res));
  ret = PROTECT(allocVector(INTSXP, olen));
  std::copy(res, res + olen, INTEGER(ret));
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
void _BoosterFinalizer(SEXP ext) {
  if (R_ExternalPtrAddr(ext) == NULL) return;
  CHECK_CALL(XGBoosterFree(R_ExternalPtrAddr(ext)));
  R_ClearExternalPtr(ext);
}

XGB_DLL SEXP XGBoosterCreate_R(SEXP dmats) {
  SEXP ret = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
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
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  R_API_END();
  UNPROTECT(1);
  return ret;
}

XGB_DLL SEXP XGBoosterCreateInEmptyObj_R(SEXP dmats, SEXP R_handle) {
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
  R_SetExternalPtrAddr(R_handle, handle);
  R_RegisterCFinalizerEx(R_handle, _BoosterFinalizer, TRUE);
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
  R_API_BEGIN();
  SEXP name_ = PROTECT(Rf_asChar(name));
  SEXP val_ = PROTECT(Rf_asChar(val));
  CHECK_CALL(XGBoosterSetParam(R_ExternalPtrAddr(handle),
                               CHAR(name_),
                               CHAR(val_)));
  UNPROTECT(2);
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

XGB_DLL SEXP XGBoosterTrainOneIter_R(SEXP handle, SEXP dtrain, SEXP iter, SEXP grad, SEXP hess) {
  R_API_BEGIN();
  CHECK_EQ(Rf_xlength(grad), Rf_xlength(hess)) << "gradient and hess must have same length.";
  SEXP gdim = getAttrib(grad, R_DimSymbol);
  SEXP hdim = getAttrib(hess, R_DimSymbol);

  int res_code;
  {
    const std::string s_grad = Rf_isNull(gdim)?
      MakeArrayInterfaceFromRVector(grad) : MakeArrayInterfaceFromRMat(grad);
    const std::string s_hess = Rf_isNull(hdim)?
      MakeArrayInterfaceFromRVector(hess) : MakeArrayInterfaceFromRMat(hess);
    res_code = XGBoosterTrainOneIter(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(dtrain),
                                     asInteger(iter), s_grad.c_str(), s_hess.c_str());
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
  SEXP evnames_lst = PROTECT(Rf_allocVector(VECSXP, len));
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
                                    asInteger(iter),
                                    BeginPtr(vec_dmats),
                                    BeginPtr(vec_sptr),
                                    len, &ret);
  }
  CHECK_CALL(res_code);
  UNPROTECT(1);
  R_API_END();
  return mkString(ret);
}

XGB_DLL SEXP XGBoosterPredictFromDMatrix_R(SEXP handle, SEXP dmat, SEXP json_config)  {
  SEXP r_out_shape;
  SEXP r_out_result;
  SEXP r_out = PROTECT(allocVector(VECSXP, 2));
  SEXP json_config_ = PROTECT(Rf_asChar(json_config));

  R_API_BEGIN();
  char const *c_json_config = CHAR(json_config_);

  bst_ulong out_dim;
  bst_ulong const *out_shape;
  float const *out_result;
  CHECK_CALL(XGBoosterPredictFromDMatrix(R_ExternalPtrAddr(handle),
                                         R_ExternalPtrAddr(dmat), c_json_config,
                                         &out_shape, &out_dim, &out_result));

  r_out_shape = PROTECT(allocVector(INTSXP, out_dim));
  size_t len = 1;
  int *r_out_shape_ = INTEGER(r_out_shape);
  for (size_t i = 0; i < out_dim; ++i) {
    r_out_shape_[i] = out_shape[i];
    len *= out_shape[i];
  }
  r_out_result = PROTECT(allocVector(REALSXP, len));
  std::copy(out_result, out_result + len, REAL(r_out_result));

  SET_VECTOR_ELT(r_out, 0, r_out_shape);
  SET_VECTOR_ELT(r_out, 1, r_out_result);

  R_API_END();
  UNPROTECT(4);

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
                                          Rf_xlength(raw)));
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
                                 Rf_xlength(raw)));
  R_API_END();
  return R_NilValue;
}

XGB_DLL SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats, SEXP dump_format) {
  SEXP out;
  SEXP continuation_token = PROTECT(R_MakeUnwindCont());
  SEXP dump_format_ = PROTECT(Rf_asChar(dump_format));
  SEXP fmap_ = PROTECT(Rf_asChar(fmap));
  R_API_BEGIN();
  bst_ulong olen;
  const char **res;
  const char *fmt = CHAR(dump_format_);
  CHECK_CALL(XGBoosterDumpModelEx(R_ExternalPtrAddr(handle),
                                CHAR(fmap_),
                                asInteger(with_stats),
                                fmt,
                                &olen, &res));
  out = PROTECT(allocVector(STRSXP, olen));
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
  UNPROTECT(4);
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
  const char *v = nullptr;
  SEXP name_ = PROTECT(Rf_asChar(name));
  SEXP val_;
  int n_protected = 1;
  if (!Rf_isNull(val)) {
    val_ = PROTECT(Rf_asChar(val));
    n_protected++;
    v = CHAR(val_);
  }

  CHECK_CALL(XGBoosterSetAttr(R_ExternalPtrAddr(handle),
                              CHAR(name_), v));
  UNPROTECT(n_protected);
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
  SEXP r_out = PROTECT(allocVector(VECSXP, 3));

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
  int *out_shape_sexp_ = INTEGER(out_shape_sexp);
  for (size_t i = 0; i < out_dim; ++i) {
    out_shape_sexp_[i] = out_shape[i];
    len *= out_shape[i];
  }

  out_features_sexp = PROTECT(allocVector(STRSXP, out_n_features));
  for (size_t i = 0; i < out_n_features; ++i) {
    SET_STRING_ELT(out_features_sexp, i, mkChar(out_features[i]));
  }

  out_scores_sexp = PROTECT(allocVector(REALSXP, len));
  std::copy(out_scores, out_scores + len, REAL(out_scores_sexp));

  SET_VECTOR_ELT(r_out, 0, out_features_sexp);
  SET_VECTOR_ELT(r_out, 1, out_shape_sexp);
  SET_VECTOR_ELT(r_out, 2, out_scores_sexp);

  R_API_END();
  UNPROTECT(4);

  return r_out;
}
