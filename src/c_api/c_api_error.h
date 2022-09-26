/*!
 *  Copyright (c) 2015-2022 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef XGBOOST_C_API_C_API_ERROR_H_
#define XGBOOST_C_API_C_API_ERROR_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>

#include "c_api_utils.h"

/*! \brief  macro to guard beginning and end section of all functions */
#ifdef LOG_CAPI_INVOCATION
#define API_BEGIN()                                                            \
  LOG(CONSOLE) << "[XGBoost C API invocation] " << __PRETTY_FUNCTION__;        \
  try {                                                                        \
    auto __guard = ::xgboost::XGBoostAPIGuard();
#else  // LOG_CAPI_INVOCATION
#define API_BEGIN()                                                            \
  try {                                                                        \
    auto __guard = ::xgboost::XGBoostAPIGuard();

#define API_BEGIN_UNGUARD() try {
#endif  // LOG_CAPI_INVOCATION

/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() */
#define API_END()                                                              \
  } catch (dmlc::Error & _except_) {                                           \
    return XGBAPIHandleException(_except_);                                    \
  } catch (std::exception const &_except_) {                                   \
    return XGBAPIHandleException(dmlc::Error(_except_.what()));                \
  }                                                                            \
  return 0; // NOLINT(*)

#define CHECK_HANDLE() if (handle == nullptr) \
  LOG(FATAL) << "DMatrix/Booster has not been initialized or has already been disposed.";

/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void XGBAPISetLastError(const char* msg);
/*!
 * \brief handle exception thrown out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int XGBAPIHandleException(const dmlc::Error &e) {
  XGBAPISetLastError(e.what());
  return -1;
}

#define xgboost_CHECK_C_ARG_PTR(out_ptr)                      \
  do {                                                        \
    if (XGBOOST_EXPECT(!(out_ptr), false)) {                  \
      LOG(FATAL) << "Invalid pointer argument: " << #out_ptr; \
    }                                                         \
  } while (0)

#endif  // XGBOOST_C_API_C_API_ERROR_H_
