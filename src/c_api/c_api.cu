/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include "../common/api_entry.h"  // XGBAPIThreadLocalEntry
#include "../common/threading_utils.h"
#include "../data/device_adapter.cuh"
#include "../data/proxy_dmatrix.h"
#include "c_api_error.h"
#include "c_api_utils.h"
#include "xgboost/c_api.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"

namespace xgboost {

void XGBBuildInfoDevice(Json *p_info) {
  auto &info = *p_info;

  info["USE_CUDA"] = true;

  std::vector<Json> v{Json{Integer{THRUST_MAJOR_VERSION}}, Json{Integer{THRUST_MINOR_VERSION}},
                      Json{Integer{THRUST_SUBMINOR_VERSION}}};
  info["THRUST_VERSION"] = v;

  v = {Json{Integer{dh::CUDAVersion().first}}, Json{Integer{dh::CUDAVersion().second}}};
  info["CUDA_VERSION"] = v;

#if defined(XGBOOST_USE_NCCL)
  info["USE_NCCL"] = Boolean{true};
  v = {Json{Integer{NCCL_MAJOR}}, Json{Integer{NCCL_MINOR}}, Json{Integer{NCCL_PATCH}}};
  info["NCCL_VERSION"] = v;
#else
  info["USE_NCCL"] = Boolean{false};
#endif

#if defined(XGBOOST_USE_RMM)
  info["USE_RMM"] = Boolean{true};
  v = {Json{Integer{RMM_VERSION_MAJOR}}, Json{Integer{RMM_VERSION_MINOR}},
       Json{Integer{RMM_VERSION_PATCH}}};
  info["RMM_VERSION"] = v;
#else
  info["USE_RMM"] = Boolean{false};
#endif
}

void XGBoostAPIGuard::SetGPUAttribute() {
  // Not calling `safe_cuda` to avoid unnecessary exception handling overhead.
  // If errors, do nothing, assuming running on CPU only machine.
  cudaGetDevice(&device_id_);
}

void XGBoostAPIGuard::RestoreGPUAttribute() {
  // Not calling `safe_cuda` to avoid unnecessary exception handling overhead.
  // If errors, do nothing, assuming running on CPU only machine.
  cudaSetDevice(device_id_);
}
}                        // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGDMatrixCreateFromCudaColumnar(char const *data,
                                            char const* c_json_config,
                                            DMatrixHandle *out) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(c_json_config);
  xgboost_CHECK_C_ARG_PTR(data);

  std::string json_str{data};
  auto config = Json::Load(StringView{c_json_config});

  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, std::int64_t>(config, "nthread", 0);
  data::CudfAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCudaArrayInterface(char const *data,
                                                  char const* c_json_config,
                                                  DMatrixHandle *out) {
  API_BEGIN();
  std::string json_str{data};
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, std::int64_t>(config, "nthread", 0);
  data::CupyAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

int InplacePreidctCuda(BoosterHandle handle, char const *c_array_interface,
                       char const *c_json_config, std::shared_ptr<DMatrix> p_m,
                       xgboost::bst_ulong const **out_shape, xgboost::bst_ulong *out_dim,
                       const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  if (!p_m) {
    p_m.reset(new data::DMatrixProxy);
  }
  auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
  CHECK(proxy) << "Invalid input type for inplace predict.";

  proxy->SetCUDAArray(c_array_interface);

  auto config = Json::Load(StringView{c_json_config});
  CHECK_EQ(get<Integer const>(config["cache_id"]), 0) << "Cache ID is not supported yet";
  auto *learner = static_cast<Learner *>(handle);

  HostDeviceVector<float> *p_predt{nullptr};
  auto type = PredictionType(RequiredArg<Integer>(config, "type", __func__));
  float missing = GetMissing(config);

  learner->InplacePredict(p_m, type, missing, &p_predt,
                          RequiredArg<Integer>(config, "iteration_begin", __func__),
                          RequiredArg<Integer>(config, "iteration_end", __func__));
  CHECK(p_predt);
  CHECK(p_predt->DeviceCanRead() && !p_predt->HostCanRead());

  auto &shape = learner->GetThreadLocal().prediction_shape;
  size_t n_samples = p_m->Info().num_row_;
  auto chunksize = n_samples == 0 ? 0 : p_predt->Size() / n_samples;
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);

  xgboost_CHECK_C_ARG_PTR(out_result);
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_dim);

  CalcPredictShape(strict_shape, type, n_samples, p_m->Info().num_col_, chunksize,
                   learner->Groups(), learner->BoostedRounds(), &shape, out_dim);
  *out_shape = dmlc::BeginPtr(shape);
  *out_result = p_predt->ConstDevicePointer();
  API_END();
}

XGB_DLL int XGBoosterPredictFromCudaColumnar(BoosterHandle handle, char const *c_json_strs,
                                             char const *c_json_config, DMatrixHandle m,
                                             xgboost::bst_ulong const **out_shape,
                                             xgboost::bst_ulong *out_dim,
                                             const float **out_result) {
  std::shared_ptr<DMatrix> p_m{nullptr};
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  return InplacePreidctCuda(handle, c_json_strs, c_json_config, p_m, out_shape, out_dim,
                            out_result);
}

XGB_DLL int XGBoosterPredictFromCudaArray(BoosterHandle handle, char const *c_json_strs,
                                          char const *c_json_config, DMatrixHandle m,
                                          xgboost::bst_ulong const **out_shape,
                                          xgboost::bst_ulong *out_dim, const float **out_result) {
  std::shared_ptr<DMatrix> p_m{nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  xgboost_CHECK_C_ARG_PTR(out_result);
  return InplacePreidctCuda(handle, c_json_strs, c_json_config, p_m, out_shape, out_dim,
                            out_result);
}
