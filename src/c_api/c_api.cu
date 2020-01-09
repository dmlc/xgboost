// Copyright (c) 2014-2019 by Contributors

#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "c_api_error.h"
#include "../data/simple_csr_source.h"
#include "../data/device_adapter.cuh"

namespace xgboost {
XGB_DLL int XGDMatrixCreateFromArrayInterfaces(char const* c_json_strs,
                                               bst_int has_missing,
                                               bst_float missing,
                                               DMatrixHandle* out) {
  API_BEGIN();
  std::string json_str{c_json_strs};
  data::CudfAdapter adapter(json_str);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, 1));
  API_END();
}
}  // namespace xgboost
