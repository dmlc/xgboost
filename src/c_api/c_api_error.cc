/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include <dmlc/thread_local.h>
#include "xgboost/c_api.h"
#include "./c_api_error.h"

struct XGBAPIErrorEntry {
  std::string last_error;
};

using XGBAPIErrorStore = dmlc::ThreadLocalStore<XGBAPIErrorEntry>;

XGB_DLL const char *XGBGetLastError() {
  return XGBAPIErrorStore::Get()->last_error.c_str();
}

void XGBAPISetLastError(const char* msg) {
  XGBAPIErrorStore::Get()->last_error = msg;
}
