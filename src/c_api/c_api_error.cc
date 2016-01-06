/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include "./c_api_error.h"
#include "../common/thread_local.h"

struct XGBAPIErrorEntry {
  std::string last_error;
};

typedef xgboost::common::ThreadLocalStore<XGBAPIErrorEntry> XGBAPIErrorStore;

const char *XGBGetLastError() {
  return XGBAPIErrorStore::Get()->last_error.c_str();
}

void XGBAPISetLastError(const char* msg) {
  XGBAPIErrorStore::Get()->last_error = msg;
}
