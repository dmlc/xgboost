// Copyright (c) 2019-2020 by Contributors
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "xgboost/learner.h"
#include "c_api_error.h"
#include "../data/device_adapter.cuh"

using namespace xgboost;  // NOLINT

XGB_DLL int XGDMatrixCreateFromArrayInterfaceColumns(char const* c_json_strs,
                                                     bst_float missing,
                                                     int nthread,
                                                     DMatrixHandle* out) {
  API_BEGIN();
  std::string json_str{c_json_strs};
  data::CudfAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, nthread));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromArrayInterface(char const* c_json_strs,
                                              bst_float missing, int nthread,
                                              DMatrixHandle* out) {
  API_BEGIN();
  std::string json_str{c_json_strs};
  data::CupyAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, nthread));
  API_END();
}
