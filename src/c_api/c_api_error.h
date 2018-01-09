/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef XGBOOST_C_API_C_API_ERROR_H_
#define XGBOOST_C_API_C_API_ERROR_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <xgboost/c_api.h>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return XGBAPIHandleException(_except_); } return 0;  // NOLINT(*)
#define CHECK_HANDLE() if (handle == nullptr) \
  LOG(FATAL) << "DMatrix/Booster has not been intialized or has already been disposed.";
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &_except_) { Finalize; return XGBAPIHandleException(_except_); } return 0; // NOLINT(*)

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
#endif  // XGBOOST_C_API_C_API_ERROR_H_
