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

// A hidden API as cache id is not being supported yet.
XGB_DLL int XGBoosterPredictFromArrayInterfaceColumns(BoosterHandle handle,
                                                      char const* c_json_strs,
                                                      float missing,
                                                      unsigned iteration_begin,
                                                      unsigned iteration_end,
                                                      char const* c_type,
                                                      xgboost::bst_ulong cache_id,
                                                      xgboost::bst_ulong *out_len,
                                                      float const** out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  CHECK_EQ(cache_id, 0) << "Cache ID is not supported yet";
  auto *learner = static_cast<Learner*>(handle);

  std::string json_str{c_json_strs};
  auto x = std::make_shared<data::CudfAdapter>(json_str);
  HostDeviceVector<float>* p_predt { nullptr };
  std::string type { c_type };
  learner->InplacePredict(x, type, missing, &p_predt, iteration_begin, iteration_end);
  CHECK(p_predt);
  CHECK(p_predt->DeviceCanRead());

  *out_result = p_predt->ConstDevicePointer();
  *out_len = static_cast<xgboost::bst_ulong>(p_predt->Size());

  API_END();
}
// A hidden API as cache id is not being supported yet.
XGB_DLL int XGBoosterPredictFromArrayInterface(BoosterHandle handle,
                                               char const* c_json_strs,
                                               float missing,
                                               unsigned iteration_begin,
                                               unsigned iteration_end,
                                               char const* c_type,
                                               xgboost::bst_ulong cache_id,
                                               xgboost::bst_ulong *out_len,
                                               float const** out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  CHECK_EQ(cache_id, 0) << "Cache ID is not supported yet";
  auto *learner = static_cast<Learner*>(handle);

  std::string json_str{c_json_strs};
  auto x = std::make_shared<data::CupyAdapter>(json_str);
  HostDeviceVector<float>* p_predt { nullptr };
  std::string type { c_type };
  learner->InplacePredict(x, type, missing, &p_predt, iteration_begin, iteration_end);
  CHECK(p_predt);
  CHECK(p_predt->DeviceCanRead());

  *out_result = p_predt->ConstDevicePointer();
  *out_len = static_cast<xgboost::bst_ulong>(p_predt->Size());

  API_END();
}
