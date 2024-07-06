/**
 *  Copyright 2015-2023, XGBoost Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include "./c_api_error.h"

#include <dmlc/thread_local.h>

#include "xgboost/c_api.h"
#include "../collective/comm.h"
#include "../collective/comm_group.h"

struct XGBAPIErrorEntry {
  std::string last_error;
  std::int32_t code{-1};
};

using XGBAPIErrorStore = dmlc::ThreadLocalStore<XGBAPIErrorEntry>;

XGB_DLL const char* XGBGetLastError() { return XGBAPIErrorStore::Get()->last_error.c_str(); }

void XGBAPISetLastError(const char* msg) {
  XGBAPIErrorStore::Get()->last_error = msg;
  XGBAPIErrorStore::Get()->code = -1;
}

XGB_DLL int XGBGetLastErrorCode() { return XGBAPIErrorStore::Get()->code; }
