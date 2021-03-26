// Copyright (c) 2019-2021 by Contributors
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "xgboost/learner.h"
#include "c_api_error.h"
#include "c_api_utils.h"
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

template <typename T>
int InplacePreidctCuda(BoosterHandle handle, char const *c_json_strs,
                       char const *c_json_config,
                       std::shared_ptr<DMatrix> p_m,
                       xgboost::bst_ulong const **out_shape,
                       xgboost::bst_ulong *out_dim, const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  auto config = Json::Load(StringView{c_json_config});
  CHECK_EQ(get<Integer const>(config["cache_id"]), 0)
      << "Cache ID is not supported yet";
  auto *learner = static_cast<Learner *>(handle);

  std::string json_str{c_json_strs};
  auto x = std::make_shared<T>(json_str);
  HostDeviceVector<float> *p_predt{nullptr};
  auto type = PredictionType(get<Integer const>(config["type"]));
  float missing = GetMissing(config);

  learner->InplacePredict(x, p_m, type, missing, &p_predt,
                          get<Integer const>(config["iteration_begin"]),
                          get<Integer const>(config["iteration_end"]));
  CHECK(p_predt);
  CHECK(p_predt->DeviceCanRead() && !p_predt->HostCanRead());

  auto &shape = learner->GetThreadLocal().prediction_shape;
  auto chunksize = x->NumRows() == 0 ? 0 : p_predt->Size() / x->NumRows();
  bool strict_shape = get<Boolean const>(config["strict_shape"]);
  CalcPredictShape(strict_shape, type, x->NumRows(), x->NumColumns(), chunksize,
                   learner->Groups(), learner->BoostedRounds(), &shape,
                   out_dim);
  *out_shape = dmlc::BeginPtr(shape);
  *out_result = p_predt->ConstDevicePointer();
  API_END();
}

XGB_DLL int XGBoosterPredictFromCudaColumnar(
    BoosterHandle handle, char const *c_json_strs, char const *c_json_config,
    DMatrixHandle m, xgboost::bst_ulong const **out_shape,
    xgboost::bst_ulong *out_dim, const float **out_result) {
  std::shared_ptr<DMatrix> p_m {nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  return InplacePreidctCuda<data::CudfAdapter>(
      handle, c_json_strs, c_json_config, p_m, out_shape, out_dim, out_result);
}

XGB_DLL int XGBoosterPredictFromCudaArray(
    BoosterHandle handle, char const *c_json_strs, char const *c_json_config,
    DMatrixHandle m, xgboost::bst_ulong const **out_shape,
    xgboost::bst_ulong *out_dim, const float **out_result) {
  std::shared_ptr<DMatrix> p_m {nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  return InplacePreidctCuda<data::CupyAdapter>(
      handle, c_json_strs, c_json_config, p_m, out_shape, out_dim, out_result);
}
